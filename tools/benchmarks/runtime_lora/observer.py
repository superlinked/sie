"""Observe successful SGLang forwards without changing scheduling or tensors."""

from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

MODULE = "sglang.srt.managers.scheduler"
LORA_MODULE = "sglang.srt.lora.lora_manager"
COMPAT_MARKER = "_sie_lora_slot_release_compat"
MARKER = "_sie_runtime_lora_benchmark"
TRACE_ENV = "SIE_RUNTIME_LORA_BATCH_TRACE"


def slot_uid(uid: Any) -> str | None | dict[str, bool]:
    if uid is None or isinstance(uid, str):
        return uid
    module = sys.modules.get("sglang.srt.lora.mem_pool")
    if module is not None and uid is getattr(module, "EMPTY_SLOT", None):
        return {"empty_slot": True}
    raise ValueError("Unexpected value in the SGLang LoRA GPU slot map")


def append(path: str, row: dict[str, Any]) -> None:
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def patch_scheduler(module: ModuleType, path: str) -> None:
    scheduler = module.Scheduler
    if getattr(scheduler, MARKER, False):
        return
    original = scheduler.run_batch

    @wraps(original)
    def observed(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        manager = self.tp_worker.model_runner.lora_manager

        def alias(uid: str | None) -> str | None:
            return manager.lora_refs[uid].lora_name if uid is not None else None

        row: dict[str, Any] = {
            "event": "forward",
            "pid": os.getpid(),
            "start_ns": time.time_ns(),
            "mode": str(batch.forward_mode),
            "requests": [
                {
                    "rid": req.rid,
                    "uid": req.lora_id,
                    "lora": alias(req.lora_id),
                    "cache_key": req.extra_key,
                    "cached_tokens": req.cached_tokens,
                    "prefix_length": len(req.prefix_indices),
                    "extend_input_length": req.extend_input_len,
                    "input_length": len(req.origin_input_ids),
                    "output_length": len(req.output_ids),
                }
                for req in batch.reqs
            ],
        }
        result = original(self, batch, *args, **kwargs)
        row["end_ns"] = time.time_ns()
        pool = manager.memory_pool
        row["gpu_slot_count"] = len(pool.buffer_id_to_uid)
        row["gpu_slot_uids"] = [slot_uid(uid) for uid in pool.buffer_id_to_uid]
        row["gpu_slots"] = [
            {"uid": uid, "lora": alias(uid), "slot": slot} for uid, slot in pool.uid_to_buffer_id.items()
        ]
        append(path, row)
        return result

    scheduler.run_batch = observed
    setattr(scheduler, MARKER, True)


def install() -> None:
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    version = importlib.metadata.version("sglang")
    if version != "0.5.10.post1":
        raise RuntimeError(f"The benchmark observer is qualified only for SGLang 0.5.10.post1; found {version}")
    previous = builtins.__import__
    installed = False

    def imported(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        nonlocal installed
        result = previous(name, globals, locals, fromlist, level)
        module = sys.modules.get(MODULE)
        if not installed and isinstance(module, ModuleType) and hasattr(module, "Scheduler"):
            if module.__file__ is None:
                raise RuntimeError("The scheduler source file is unavailable for provenance")
            manager = getattr(sys.modules.get(LORA_MODULE), "LoRAManager", None)
            slot_release_compat = bool(getattr(manager, COMPAT_MARKER, False))
            if not slot_release_compat:
                raise RuntimeError("The production LoRA slot-release compatibility hook is absent")
            installed = True
            patch_scheduler(module, path)
            append(
                path,
                {
                    "event": "observer_installed",
                    "pid": os.getpid(),
                    "version": version,
                    "slot_release_compat": slot_release_compat,
                    "scheduler_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                },
            )
            if builtins.__import__ is imported:
                setattr(builtins, "__import__", previous)  # noqa: B010
        return result

    setattr(builtins, "__import__", imported)  # noqa: B010
