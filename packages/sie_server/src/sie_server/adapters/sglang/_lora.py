"""Declared-source LoRA loading and request leases for an isolated SGLang engine."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import tempfile
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, cast

import requests
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sie_server.adapters._generation_base import (
    GenerationChunk,
    GenerationInvalidRequestError,
    GenerationLoraLoadingError,
)
from sie_server.adapters.lora import LoraRuntimeConfig, LoraStateUncertainError, LoraUnavailableError
from sie_server.adapters.peft_lora_mixin import is_lora_weight_key

_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def leased_generation(
    method: Callable[..., AsyncIterator[GenerationChunk]],
) -> Callable[..., AsyncIterator[GenerationChunk]]:
    """Hold a LoRA lease until completion, cancellation, exception, or iterator close."""

    @functools.wraps(method)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        name = kwargs.get("lora_path")
        with self.lora_request(name, n=kwargs.get("n"), best_of=kwargs.get("best_of")) as engine_name:
            kwargs["lora_path"] = engine_name
            iterator = cast("AsyncGenerator[GenerationChunk, None]", method(self, *args, **kwargs))
            async with contextlib.aclosing(iterator) as chunks:
                async for chunk in chunks:
                    yield chunk

    return wrapped


def leased_encode(method: Callable[..., Any]) -> Callable[..., Any]:
    """Keep an embedding request's selected adapter resident until its backend call returns."""

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lora.lease(self._active_lora):
            return method(self, *args, **kwargs)

    return wrapped


class SGLangLoraMixin:
    """Expose the shared engine lifecycle through ModelAdapter's capability seam."""

    _lora: SGLangLoraManager

    @contextlib.contextmanager
    def lora_request(
        self, name: str | None, *, n: int | None = None, best_of: int | None = None
    ) -> Iterator[str | None]:
        """Hold one selection through the entire native or proxied request lifetime."""
        if name is not None and self._lora.deferred and any((value or 1) > 1 for value in (n, best_of)):
            raise GenerationInvalidRequestError(
                "lora_adapter", "Deferred LoRA generation supports only n=1 and best_of=1"
            )
        try:
            with self._lora.lease(name) as engine_name:
                yield engine_name
        except LoraUnavailableError as error:
            raise GenerationLoraLoadingError from error

    def supports_lora(self) -> bool:
        return bool(self._lora.aliases)

    def supports_hot_lora_reload(self) -> bool:
        return True

    def lora_runtime_config(self) -> LoraRuntimeConfig:
        return self._lora.runtime_config()

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        return self._lora.fetch(lora_path, revision)

    def discard_lora_source(self, lora_path: str) -> None:
        self._lora.discard_source(lora_path)

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        return self._lora.load(lora_path, revision, source=source)

    def unload_lora(self, lora_name: str) -> None:
        self._lora.unload(lora_name)


def validate_preload_loras(paths: Any, preload: Any) -> None:
    """Validate the optional selector against the resolved profile's source map."""
    if preload is None:
        return
    if not isinstance(paths, dict) or not paths:
        raise ValueError("preload_loras requires a nonempty lora_paths map")
    if not isinstance(preload, list) or any(not isinstance(name, str) or not name for name in preload):
        raise ValueError("preload_loras must be a list of declared served names")
    if len(preload) != len(set(preload)) or not set(preload).issubset(paths):
        raise ValueError("preload_loras must contain unique names declared in lora_paths")


class SGLangLoraManager:
    """Keep source authorization, engine names, and residency in one place.

    Downloads happen before the worker's mutation lock. HTTP mutation is
    synchronous and bounded; SGLang may pause scheduling while attaching.
    Generation streams acquire per-adapter leases, because they bypass the
    embedding worker's dispatch lock. An eviction stops new leases before
    draining old ones, then asks the engine to remove the adapter.
    """

    def __init__(
        self,
        paths: dict[str, str] | None,
        *,
        preload: list[str] | None,
        revisions: dict[str, str | None] | None,
        max_rank: int | None,
        target_modules: list[str] | None,
        max_per_batch: int,
        server_url: Callable[[], str | None],
        max_loaded: int | None = None,
        mutation_timeout_s: float = 60,
    ) -> None:
        validate_preload_loras(paths, preload)
        self.aliases = dict(paths or {})
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.aliases.items()
        ):
            raise ValueError("lora_paths must map nonempty served names to declared source ids")
        if any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", alias) for alias in self.aliases):
            raise ValueError("LoRA served names may contain only letters, digits, dots, underscores, and hyphens")
        # Reject ambiguous aliases before source-id requests and served-name
        # requests can select different adapters.
        if any(alias in self.aliases.values() and alias != source for alias, source in self.aliases.items()):
            raise ValueError("A LoRA served name cannot shadow another declared source id")
        self.revisions = revisions or {}
        self.preloaded = frozenset(self.aliases[name] for name in (self.aliases if preload is None else preload))
        self.deferred = set(self.aliases.values()) - self.preloaded
        configured = max_loaded if max_loaded is not None else int(os.environ.get("SIE_MAX_LORAS_PER_MODEL", "10"))
        if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
            raise ValueError("SIE_MAX_LORAS_PER_MODEL must be positive")
        if isinstance(max_per_batch, bool) or not isinstance(max_per_batch, int) or max_per_batch < 1:
            raise ValueError("max_loras_per_batch must be a positive integer")
        # Startup-only profiles historically load every declared adapter. The
        # runtime residency limit must not reject their pinned set or batch size.
        self.max_loaded = (
            max(configured, len(self.preloaded), max_per_batch) if self.aliases and not self.deferred else configured
        )
        if self.aliases and (max_per_batch > self.max_loaded or len(self.preloaded) > self.max_loaded):
            raise ValueError("LoRA batch capacity and preload count must not exceed SIE_MAX_LORAS_PER_MODEL")
        if self.deferred and len(self.preloaded) == self.max_loaded:
            raise ValueError("Deferred LoRAs require at least one slot not occupied by pinned preloads")
        if self.deferred and (max_rank is None or not target_modules):
            raise ValueError("Deferred LoRAs require max_lora_rank and lora_target_modules before engine launch")
        if max_rank is not None and (isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 1):
            raise ValueError("max_lora_rank must be a positive integer")
        if target_modules is not None and (
            not isinstance(target_modules, list)
            or not target_modules
            or any(not isinstance(name, str) or not name for name in target_modules)
            or len(set(target_modules)) != len(target_modules)
            or ("all" in target_modules and target_modules != ["all"])
        ):
            raise ValueError("lora_target_modules must contain unique module names, or only 'all'")
        self.max_rank = max_rank
        self.target_modules = target_modules
        self.max_per_batch = max_per_batch
        self._server_url = server_url
        self._timeout = mutation_timeout_s
        self._names = {source: name for name, source in reversed(list(self.aliases.items()))}
        self._condition = threading.Condition()
        self._mutation_lock = threading.Lock()
        self._loaded: set[str] = set()
        self._draining: set[str] = set()
        self._uncertain: set[str] = set()
        self._leases: dict[str, int] = {}
        self._snapshots: dict[str, tempfile.TemporaryDirectory[str]] = {}
        self._closed = False

    def runtime_config(self) -> LoraRuntimeConfig:
        return LoraRuntimeConfig(self.aliases, self.preloaded, self.preloaded, self.max_loaded)

    def source(self, name: str) -> str:
        source = self.aliases.get(name, name)
        if source not in self._names:
            raise ValueError("This LoRA adapter is not declared for the loaded profile")
        return source

    def engine_name(self, name: str) -> str:
        return self._names[self.source(name)]

    @property
    def available(self) -> list[str]:
        with self._condition:
            return [
                name
                for name, source in self.aliases.items()
                if source in self._loaded and source not in self._draining and source not in self._uncertain
            ]

    def fetch(self, name: str, revision: str | None = None) -> str:
        source = self.source(name)
        declared_revision = self.revisions.get(source)
        if declared_revision is not None and revision not in (None, declared_revision):
            raise ValueError("LoRA revision does not match the declared immutable revision")
        with self._condition:
            if self._closed:
                raise RuntimeError("The LoRA engine was unloaded during fetch")
            if source in self._snapshots:
                return self._snapshots[source].name
        directory = Path(source)
        if not directory.is_dir():
            directory = Path(snapshot_download(source, revision=declared_revision or revision, allow_patterns=_FILES))
        config = json.loads((directory / _FILES[0]).read_text())
        if not isinstance(config, dict) or config.get("peft_type") != "LORA":
            raise ValueError("SGLang requires a PEFT LoRA adapter")
        # Runtime adapters must not replace shared base weights or rely on
        # training-time transforms that SGLang does not implement.
        if (
            any(
                config.get(key)
                for key in (
                    "modules_to_save",
                    "layer_replication",
                    "megatron_config",
                    "trainable_token_indices",
                    "use_dora",
                    "rank_pattern",
                    "alpha_pattern",
                    "use_rslora",
                )
            )
            or config.get("bias", "none") != "none"
        ):
            raise ValueError("SGLang runtime LoRA requires adapter-only weights without base-model transforms")
        initializer = config.get("init_lora_weights", True)
        if not isinstance(initializer, bool) and initializer not in ("gaussian", "eva", "orthogonal"):
            raise ValueError("LoRA initialization rewrites base weights that SGLang cannot reproduce")
        rank = config.get("r")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or (self.max_rank and rank > self.max_rank):
            raise ValueError("LoRA rank exceeds the engine's configured capacity")
        modules = config.get("target_modules")
        if not isinstance(modules, list) or not modules or any(not isinstance(module, str) for module in modules):
            raise ValueError("SGLang LoRA target_modules must be an explicit list")
        if self.target_modules and self.target_modules != ["all"] and not set(modules).issubset(self.target_modules):
            raise ValueError("LoRA targets modules outside the engine's configured capacity")
        with safe_open(str(directory / _FILES[1]), framework="pt") as handle:
            names = handle.keys()
            if not names or any(
                not is_lora_weight_key(name) or not name.endswith(("lora_A.weight", "lora_B.weight")) for name in names
            ):
                raise ValueError("LoRA weights contain parameters outside the adapter")
        # SGLang's generic loader globs the whole directory. A Hub cache or
        # operator directory can also contain unrelated weights. Expose only
        # the validated pair, and keep it alive until confirmed engine removal.
        snapshot = tempfile.TemporaryDirectory(prefix="sie-sglang-lora-")
        try:
            for filename in _FILES:
                (Path(snapshot.name) / filename).symlink_to((directory / filename).resolve())
            with self._condition:
                if self._closed:
                    raise RuntimeError("The LoRA engine was unloaded during fetch")
                existing = self._snapshots.get(source)
                if existing is None:
                    self._snapshots[source] = snapshot
                    return snapshot.name
            snapshot.cleanup()
            return existing.name
        except BaseException:
            snapshot.cleanup()
            raise

    def _release_source(self, source: str) -> None:
        with self._condition:
            snapshot = self._snapshots.pop(source, None)
        if snapshot is not None:
            snapshot.cleanup()

    def discard_source(self, name: str) -> None:
        """Drop an unused fetched view after a completed failure or stale load."""
        source = self.source(name)
        with self._condition:
            if source in self._loaded or source in self._uncertain:
                return
            snapshot = self._snapshots.pop(source, None)
        if snapshot is not None:
            snapshot.cleanup()

    def close(self) -> None:
        """Release validated source views after the engine process has stopped."""
        self.stop_mutations()
        with self._condition:
            snapshots = list(self._snapshots.values())
            self._snapshots.clear()
            self._loaded.clear()
        for snapshot in snapshots:
            snapshot.cleanup()

    def stop_mutations(self) -> None:
        """Join bounded HTTP mutations before the child releases its port.

        Generation has no worker executor to join at model teardown. Closing
        this gate before stopping the child prevents an old POST from reaching
        a replacement engine that reuses the port.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        with self._mutation_lock:
            pass

    def launch_args(self) -> list[str]:
        with self._condition:
            self._closed = False
        if not self.aliases:
            return []
        args = [
            "--enable-lora",
            "--max-loras-per-batch",
            str(self.max_per_batch),
            "--max-loaded-loras",
            str(self.max_loaded),
        ]
        if self.max_rank is not None:
            args.extend(["--max-lora-rank", str(self.max_rank)])
        if self.target_modules:
            args.extend(["--lora-target-modules", *self.target_modules])
        if self.preloaded:
            args.append("--lora-paths")
            args.extend(f"{self._names[source]}={self.fetch(source)}" for source in sorted(self.preloaded))
        return args

    def started(self) -> None:
        with self._condition:
            self._loaded = set(self.preloaded)
            self._draining.clear()
            self._uncertain.clear()

    def _mutate(self, endpoint: str, payload: dict[str, Any]) -> None:
        with self._mutation_lock:
            with self._condition:
                if self._closed:
                    raise RuntimeError("The LoRA engine is unloading")
            url = self._server_url()
            if url is None:
                raise RuntimeError("SGLang is not running")
            response = requests.post(f"{url}/{endpoint}", json=payload, timeout=self._timeout)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or body.get("success") is not True:
                # Backend diagnostics may contain private filesystem or repo paths.
                raise RuntimeError("SGLang refused the LoRA mutation")

    def load(self, name: str, revision: str | None = None, *, source: str | None = None) -> int:
        declared = self.source(name)
        directory = source or self.fetch(declared, revision)
        with self._condition:
            if declared in self._uncertain:
                raise LoraStateUncertainError("LoRA state requires a model reload")
            if declared in self._loaded:
                return 0
        try:
            self._mutate("load_lora_adapter", {"lora_name": self._names[declared], "lora_path": directory})
        except Exception:
            try:
                self._mutate("unload_lora_adapter", {"lora_name": self._names[declared]})
            except Exception as rollback_error:
                with self._condition:
                    self._uncertain.add(declared)
                raise LoraStateUncertainError(
                    "LoRA attach rollback could not be confirmed; reload the model"
                ) from rollback_error
            self._release_source(declared)
            raise
        with self._condition:
            self._loaded.add(declared)
        # Engine preallocates GPU LoRA buffers at startup; do not double count.
        return 0

    @contextlib.contextmanager
    def lease(self, name: str | None) -> Iterator[str | None]:
        if name is None:
            yield None
            return
        declared = self.source(name)
        with self._condition:
            if (
                self._closed
                or declared not in self._loaded
                or declared in self._draining
                or declared in self._uncertain
            ):
                raise LoraUnavailableError("The declared LoRA is loading or draining; retry the request")
            self._leases[declared] = self._leases.get(declared, 0) + 1
        try:
            yield self._names[declared]
        finally:
            with self._condition:
                self._leases[declared] -= 1
                self._condition.notify_all()

    def unload(self, name: str) -> None:
        declared = self.source(name)
        if declared in self.preloaded:
            raise ValueError("Startup LoRAs are pinned and cannot be evicted")
        with self._condition:
            self._draining.add(declared)
            self._condition.notify_all()
            if not self._condition.wait_for(lambda: self._leases.get(declared, 0) == 0, timeout=self._timeout):
                self._draining.discard(declared)
                raise TimeoutError("Active LoRA requests did not drain before eviction")
        try:
            self._mutate("unload_lora_adapter", {"lora_name": self._names[declared]})
        except Exception as error:
            with self._condition:
                self._uncertain.add(declared)
            raise LoraStateUncertainError("LoRA removal could not be confirmed; reload the model") from error
        with self._condition:
            self._loaded.discard(declared)
            self._draining.discard(declared)
        self._release_source(declared)
