"""Observe the actual child chat stream without changing requests or tensors."""

from __future__ import annotations

import builtins
import contextlib
import contextvars
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

MODULE = "sglang.srt.managers.tokenizer_manager"
VERSION = "0.5.20"
SOURCE_SHA256 = "952fdfcf54b13af6851993d4b7599d4b1efeb6aa630a7554584a1a047eda94f2"
SOURCE_FILES = {
    "managers/tokenizer_manager.py": SOURCE_SHA256,
    "entrypoints/openai/serving_chat.py": "047e4391dc87816a62da34392eea9379fcb1f93025e70c3b327de8b596f59b09",
    "sampling/sampling_params.py": "7295f275a1a286ae5a4414bcedd1d48a3a9824a0c2a5c915e091ed06b6f74730",
}
TRACE_ENV = "SIE_STREAM_DECODE_TRACE"
MARKER = "_sie_stream_decode_observer"
MAX_EVENTS = 100_000
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "sampling_seed",
    "min_new_tokens",
    "max_new_tokens",
    "n",
    "ignore_eos",
    "skip_special_tokens",
    "stop_strs",
    "stop_token_ids",
    "stop_regex_strs",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
    "custom_params",
    "logit_bias",
)


def clock_domain() -> str | None:
    """Bind monotonic timestamps to a Linux boot without publishing the boot ID."""
    path = Path("/proc/sys/kernel/random/boot_id")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def patch_manager(module: ModuleType, output: Path, manifest: dict[str, Any]) -> Callable[[], None]:
    """Observe input tokenization and output of the same child request.

    Chat bypasses SIE's native generate/parser methods. This boundary is inside
    the SGLang child, before serving_chat formats SSE. Records stay in memory
    until the stream closes. No per-event tokenization or disk writes occur.
    """
    manager = module.TokenizerManager
    if getattr(manager, MARKER, False):
        raise RuntimeError("stream observer already installed")
    generate, tokenize = manager.generate_request, manager._tokenize_one_request
    current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("stream_decode", default=None)

    @wraps(tokenize)
    async def tokenized(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await tokenize(self, *args, **kwargs)
        record = current.get()
        if record is not None:
            if "tokenized" in record:
                raise RuntimeError("benchmark requires one tokenized input per request")
            sampling = {}
            for name in SAMPLING_FIELDS:
                value = getattr(result.sampling_params, name, None)
                sampling[name] = sorted(value) if isinstance(value, set) else copy.deepcopy(value)
            record["tokenized"] = {
                "input_ids": list(result.input_ids),
                "sampling": sampling,
                "rid": result.rid,
                "has_multimodal_inputs": bool(getattr(result, "mm_inputs", None)),
            }
        return result

    @wraps(generate)
    async def generated(self: Any, obj: Any, request: Any = None) -> Any:
        route = request.url.path if request is not None else None
        if route != "/v1/chat/completions" or obj.stream is not True:
            # SGLang performs native generation during startup and health checks.
            # Preserve those calls, recording only their occupancy so analysis
            # can reject any that overlap the benchmark schedule.
            background = {
                "id": uuid.uuid4().hex,
                "pid": os.getpid(),
                "route": route,
                "started_ns": time.perf_counter_ns(),
                "finished_ns": None,
                "clock_domain": clock_domain(),
                "completed": False,
            }
            path = output / f"background-{background['id']}.json"
            save(path, background)
            try:
                async with contextlib.aclosing(generate(self, obj, request)) as stream:
                    async for event in stream:
                        yield event
                background["completed"] = True
            finally:
                background["finished_ns"] = time.perf_counter_ns()
                save(path, background)
            return
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "started_ns": time.perf_counter_ns(),
            "finished_ns": None,
            "events": [],
            "completed": False,
            "cohort": manifest["cohort"],
            "server_manifest_sha256": digest(manifest),
            "clock_domain": clock_domain(),
            "version": VERSION,
            "source_sha256": SOURCE_SHA256,
            "model_path": self.model_path,
            "served_model_name": self.served_model_name,
            "incremental_streaming_output": self.incremental_streaming_output,
        }
        path = output / f"stream-{record['id']}.json"
        # A second stream can remain open when the measured client finishes.
        # Persist its occupancy once so offline analysis cannot miss it.
        save(path, record)
        try:
            # The ordinary route already parsed this body. Never record headers,
            # credentials, or arbitrary environment values.
            record["chat_body"] = copy.deepcopy(await request.json())
            record["adapted_sampling"] = copy.deepcopy(obj.sampling_params)
            async with contextlib.aclosing(generate(self, obj, request)) as stream:
                while True:
                    # Consumers may resume an async generator in another task.
                    # Bind only the engine await; never hold a ContextVar token
                    # across a yield into the HTTP/SSE consumer.
                    token = current.set(record)
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        break
                    finally:
                        current.reset(token)
                    at_ns = time.perf_counter_ns()
                    if len(record["events"]) >= MAX_EVENTS:
                        raise RuntimeError("benchmark exceeded its fixed event bound")
                    record["events"].append({"at_ns": at_ns, "event": copy.deepcopy(event)})
                    yield event
            record["completed"] = True
        except BaseException as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["finished_ns"] = time.perf_counter_ns()
            save(path, record)

    manager.generate_request, manager._tokenize_one_request = generated, tokenized
    setattr(manager, MARKER, True)

    def restore() -> None:
        manager.generate_request, manager._tokenize_one_request = generate, tokenize
        delattr(manager, MARKER)

    return restore


def install() -> None:
    """Chain after production sitecustomize; refuse unreviewed engine source."""
    directory = os.environ.get(TRACE_ENV)
    if not directory:
        return
    output = Path(directory)
    manifest = json.loads((output / "server.json").read_text())
    if importlib.metadata.version("sglang") != VERSION:
        raise RuntimeError(f"stream observer requires SGLang {VERSION}")
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
        if not installed and isinstance(module, ModuleType) and hasattr(module, "TokenizerManager"):
            if module.__file__ is None:
                raise RuntimeError("SGLang tokenizer manager source is absent")
            source_root = Path(module.__file__).parents[1]
            actual = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
            if actual != SOURCE_FILES:
                raise RuntimeError("unreviewed SGLang stream source")
            installed = True
            patch_manager(module, output, manifest)
            save(
                output / f"installed-{os.getpid()}.json",
                {
                    "pid": os.getpid(),
                    "version": VERSION,
                    "source_sha256": SOURCE_SHA256,
                    "source_files": actual,
                    "cohort": manifest["cohort"],
                    "server_manifest_sha256": digest(manifest),
                },
            )
            if builtins.__import__ is imported:
                setattr(builtins, "__import__", previous)  # noqa: B010
        return result

    setattr(builtins, "__import__", imported)  # noqa: B010
