"""Release LoRA GPU slots when the pinned SGLang 0.5.10 engine unloads an adapter.

Upstream issue https://github.com/sgl-project/sglang/issues/21380 describes the
missing cleanup: CPU weights disappear, but the GPU pool keeps the old UID.
Repeated load/unload cycles then exhaust the pool. The CUDA 13 bundle's
SGLang 0.5.20 already implements this cleanup and must keep its native path.

Only the SGLang child opts into this hook. It waits for the engine's normal
import rather than importing CUDA or SGLang during Python site initialization.
The engine already drains requests before calling the method wrapped here.
"""

from __future__ import annotations

import builtins
import importlib.metadata
import logging
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any

_AFFECTED_VERSIONS = frozenset({"0.5.10", "0.5.10.post1"})
_MANAGER_MODULE = "sglang.srt.lora.lora_manager"
_POOL_MODULE = "sglang.srt.lora.mem_pool"
_CLASS_MARKER = "_sie_lora_slot_release_compat"
_IMPORT_MARKER = "_sie_lora_slot_release_deferred"
logger = logging.getLogger(__name__)


def _release_slot(pool: Any, lora_id: str, empty_slot: Any) -> None:
    """Forget one drained adapter, preserving the base model's ``None`` UID."""
    if lora_id is None:
        raise ValueError("a LoRA unload must never release the base model slot")
    slot = pool.uid_to_buffer_id.get(lora_id)
    if slot is not None and pool.buffer_id_to_uid[slot] != lora_id:
        raise RuntimeError("LoRA GPU slot mapping is inconsistent")
    # The policy can remember an adapter even when it is not currently in a
    # GPU slot. remove() is idempotent in both pinned upstream policies.
    pool.eviction_policy.remove(lora_id)
    if slot is not None:
        pool.uid_to_buffer_id.pop(lora_id)
        pool.buffer_id_to_uid[slot] = empty_slot


def _patch_manager_module(module: ModuleType) -> None:
    manager = getattr(module, "LoRAManager", None)
    if manager is None:
        raise RuntimeError("SGLang does not expose the expected LoRAManager")
    if getattr(manager, _CLASS_MARKER, False):
        return
    pool_module = sys.modules.get(_POOL_MODULE)
    if pool_module is None or not hasattr(pool_module, "EMPTY_SLOT"):
        raise RuntimeError("SGLang does not expose the expected LoRA pool sentinel")
    empty_slot = pool_module.EMPTY_SLOT
    original_unload = manager.unload_lora_adapter

    @wraps(original_unload)
    def unload_with_slot_release(self: Any, lora_ref: Any) -> Any:
        # Do not alter a pool for an upstream operation that did not succeed.
        result = original_unload(self, lora_ref)
        if result.success:
            try:
                _release_slot(self.memory_pool, lora_ref.lora_id, empty_slot)
            except Exception:
                logger.exception("SIE could not release an unloaded LoRA GPU slot")
                return self.create_lora_update_result(
                    success=False,
                    error_message="LoRA GPU slot cleanup failed; reload this model before loading more adapters",
                )
        return result

    manager.unload_lora_adapter = unload_with_slot_release
    setattr(manager, _CLASS_MARKER, True)


def install_lora_compat() -> bool:
    """Install the exact-version backport; return False for unaffected engines."""
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return False
    if version not in _AFFECTED_VERSIONS:
        return False
    loaded = sys.modules.get(_MANAGER_MODULE)
    if isinstance(loaded, ModuleType) and hasattr(loaded, "LoRAManager"):
        _patch_manager_module(loaded)
        return True

    current_import = builtins.__import__
    if getattr(current_import, _IMPORT_MARKER, False):
        return True

    def deferred_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        module = current_import(name, globals, locals, fromlist, level)
        loaded_module = sys.modules.get(_MANAGER_MODULE)
        if not isinstance(loaded_module, ModuleType) or not hasattr(loaded_module, "LoRAManager"):
            return module
        try:
            _patch_manager_module(loaded_module)
        finally:
            if builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
        return module

    setattr(deferred_import, _IMPORT_MARKER, True)
    setattr(builtins, "__import__", deferred_import)  # noqa: B010
    return True
