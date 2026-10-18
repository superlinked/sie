"""The pinned engine must reuse GPU slots across repeated LoRA replacements."""

from __future__ import annotations

import builtins
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_HOOK_PATH = Path(__file__).parents[2] / "src/sie_server/adapters/sglang/_compat/sie_lora_compat.py"


@pytest.fixture
def hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sie_lora_compat_test", _HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """CPU metadata contract of the pinned manager and two-slot GPU pool."""
    empty_slot = object()

    class Manager:
        def __init__(self) -> None:
            self.used: set[str | None] = {None}
            self.memory_pool = SimpleNamespace(
                uid_to_buffer_id={None: 1},
                buffer_id_to_uid=[empty_slot, None],
                eviction_policy=SimpleNamespace(remove=self.used.discard),
            )
            self.loras: dict[str, object] = {}
            self.fail_unload = False

        def add(self, name: str) -> SimpleNamespace:
            slot = self.memory_pool.buffer_id_to_uid.index(empty_slot)
            self.memory_pool.buffer_id_to_uid[slot] = name
            self.memory_pool.uid_to_buffer_id[name] = slot
            self.used.add(name)
            self.loras[name] = object()
            return SimpleNamespace(lora_id=name)

        def create_lora_update_result(self, success: bool, error_message: str = "") -> SimpleNamespace:
            return SimpleNamespace(success=success, error_message=error_message)

        def unload_lora_adapter(self, ref: Any) -> SimpleNamespace:
            if self.fail_unload:
                return self.create_lora_update_result(False, "upstream refused")
            del self.loras[ref.lora_id]
            return self.create_lora_update_result(True)

    manager_module = ModuleType("sglang.srt.lora.lora_manager")
    manager_module.__dict__["LoRAManager"] = Manager
    pool_module = ModuleType("sglang.srt.lora.mem_pool")
    pool_module.__dict__["EMPTY_SLOT"] = empty_slot
    monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    return SimpleNamespace(manager=Manager, module=manager_module, empty=empty_slot)


def test_repeated_unload_reuses_slot_without_evicting_base(hook: ModuleType, engine: SimpleNamespace) -> None:
    broken = engine.manager()
    assert broken.unload_lora_adapter(broken.add("first")).success
    with pytest.raises(ValueError, match="not in list"):
        broken.add("second")

    hook._patch_manager_module(engine.module)
    fixed = engine.manager()
    for index in range(20):
        ref = fixed.add(f"adapter-{index}")
        assert fixed.unload_lora_adapter(ref).success
        assert fixed.loras == {}
        assert fixed.memory_pool.uid_to_buffer_id == {None: 1}
        assert fixed.memory_pool.buffer_id_to_uid == [engine.empty, None]
        assert fixed.used == {None}


def test_upstream_failure_preserves_residency(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    ref = manager.add("busy")
    manager.fail_unload = True
    assert not manager.unload_lora_adapter(ref).success
    assert "busy" in manager.loras
    assert manager.memory_pool.uid_to_buffer_id == {None: 1, "busy": 0}
    assert "busy" in manager.used


def test_nonresident_adapter_cleans_only_policy(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    manager.loras["cpu-only"] = object()
    manager.used.add("cpu-only")
    assert manager.unload_lora_adapter(SimpleNamespace(lora_id="cpu-only")).success
    assert manager.used == {None}
    assert manager.memory_pool.uid_to_buffer_id == {None: 1}


def test_cleanup_failure_cannot_be_reported_as_success(hook: ModuleType, engine: SimpleNamespace) -> None:
    hook._patch_manager_module(engine.module)
    manager = engine.manager()
    ref = manager.add("adapter")
    manager.memory_pool.buffer_id_to_uid[0] = "different-adapter"
    result = manager.unload_lora_adapter(ref)
    assert not result.success
    assert "reload this model" in result.error_message
    assert manager.memory_pool.buffer_id_to_uid[0] == "different-adapter"


def test_base_slot_is_never_released(hook: ModuleType, engine: SimpleNamespace) -> None:
    manager = engine.manager()
    with pytest.raises(ValueError, match="base model"):
        hook._release_slot(manager.memory_pool, None, engine.empty)
    assert manager.memory_pool.uid_to_buffer_id == {None: 1}


@pytest.mark.parametrize("version", ["0.5.20", "0.6.0", "0.5.11"])
def test_other_engine_versions_keep_native_method(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: version)
    original = engine.manager.unload_lora_adapter
    assert hook.install_lora_compat() is False
    assert engine.manager.unload_lora_adapter is original


def test_exact_version_install_is_idempotent(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: "0.5.10.post1")
    assert hook.install_lora_compat()
    first = engine.manager.unload_lora_adapter
    assert hook.install_lora_compat()
    assert engine.manager.unload_lora_adapter is first


def test_deferred_import_restores_existing_hook(
    hook: ModuleType, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook.importlib.metadata, "version", lambda _: "0.5.10.post1")
    monkeypatch.delitem(sys.modules, engine.module.__name__)
    previous_import = builtins.__import__
    # Register restoration before installing the hook, including on assertion failure.
    monkeypatch.setattr(builtins, "__import__", previous_import)
    assert hook.install_lora_compat()
    installed = builtins.__import__
    assert installed is not previous_import
    assert hook.install_lora_compat()
    assert builtins.__import__ is installed
    monkeypatch.setitem(sys.modules, engine.module.__name__, engine.module)
    builtins.__import__("math")
    assert builtins.__import__ is previous_import
    assert getattr(engine.manager, hook._CLASS_MARKER)


def test_child_sitecustomize_installs_the_hook(tmp_path: Path) -> None:
    for package in ("sglang", "sglang/srt", "sglang/srt/lora"):
        directory = tmp_path / package
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    metadata = tmp_path / "sglang-0.5.10.post1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: sglang\nVersion: 0.5.10.post1\n")
    (tmp_path / "sglang/srt/lora/mem_pool.py").write_text("EMPTY_SLOT = object()\n")
    (tmp_path / "sglang/srt/lora/lora_manager.py").write_text(
        "from sglang.srt.lora.mem_pool import EMPTY_SLOT\n"
        "class LoRAManager:\n"
        "    def unload_lora_adapter(self, ref):\n"
        "        raise NotImplementedError\n"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SIE_SGLANG_")}
    environment["PYTHONPATH"] = os.pathsep.join([str(_HOOK_PATH.parent), str(tmp_path)])
    environment["SIE_SGLANG_LORA_COMPAT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "assert 'sglang.srt.lora.lora_manager' not in sys.modules\n"
            "from sglang.srt.lora.lora_manager import LoRAManager\n"
            "assert LoRAManager._sie_lora_slot_release_compat\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "Error in sitecustomize" not in result.stderr
