"""Deterministic weightless fake adapters (Fake Engine, #1836 / #1847).

The fake family exercises the whole orchestration stack — load, batch,
schedule, evict, deliver — with zero model downloads and zero GPUs. Outputs
are derived deterministically from input hashes.

Determinism contract (properties, not golden values):

- Same input → identical output within a running version.
- Distinct inputs → distinct outputs.
- Shapes/dims always match the catalog entry.
- Score outputs are totally ordered floats in ``[0, 1)``.

The exact hash→output derivation is NOT a stable contract and may change at
any time; tests must assert self-equality, shape, and ordering — never pinned
vectors or token strings.

The whole family is ONE catalog entry — ``models/sie-fake.yaml``
(``package_backed``, no HF weights) — served by the combined
:class:`FakeAdapter` (encode + score + generate). Different cases are
profiles of that one model, addressable as ``sie-fake:<profile>`` via the
loader's variant expansion. Chat completions return ``invalid_request`` —
chat-template rendering needs a real tokenizer source; ``/v1/generate``
(raw prompt) is the supported generation surface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.core.inference_output import EncodeOutput, ScoreOutput
from sie_server.core.oom import ResourceExhausted, ResourceExhaustedError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sie_server.types.grammar import GrammarSpec
    from sie_server.types.inputs import ImageInput, Item

# Default declared footprint. Consumed by the MemoryManager once the
# synthetic budget tracker (#1848) makes ``estimated_bytes`` load-bearing;
# until then it is informational, matching real adapters.
_DEFAULT_FOOTPRINT_BYTES = 64 * 1024 * 1024

# Failure injection (#1849). Primary surface: a ``faults`` mapping in the
# fake model's ``adapter_options.loadtime``. Secondary: this env var — a JSON
# object ``{fault_key: {fault: value, ...}}`` merged over the YAML faults at
# adapter construction, keyed by the YAML's loadtime ``fault_key`` (each
# ``sie-fake/*`` entry carries its own sie_id there). Static per boot; there
# is deliberately no runtime-mutation surface.
SIE_FAKE_FAULTS_ENV = "SIE_FAKE_FAULTS"

_LATCH_POLL_INTERVAL_S = 0.01


@dataclass(frozen=True, slots=True)
class FakeFaults:
    """Failure switches for one fake model. All faults flow through the real
    code paths: the injected OOM string satisfies the real ``is_oom_error``
    classification, slow/latched loads hold the real ``MODEL_LOADING`` window,
    and the teardown hang exercises the real unload/drain path.

    ``oom_on_dispatch`` is the 1-based dispatch ordinal on which an OOM is
    raised; ``oom_repeat`` widens the failure window so retries keep failing
    (walks the recovery ladder past its first rung). Latch faults block until
    the sentinel file exists — the deterministic race-sequencing primitive —
    and raise ``TimeoutError`` after ``latch_timeout_s`` so an unreleased
    latch fails a test rather than wedging CI.
    """

    oom_on_dispatch: int | None = None
    oom_repeat: int = 1
    # When True the injected OOM is the typed ``ResourceExhaustedError``
    # instead of a string-indicator RuntimeError, exercising the isinstance
    # tier of ``is_oom_error`` (core/oom.py) rather than the substring tier.
    oom_typed: bool = False
    slow_load_s: float = 0.0
    fail_load: bool = False
    teardown_hang_s: float = 0.0
    load_latch_file: str | None = None
    dispatch_latch_file: str | None = None
    latch_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        """Reject non-finite / out-of-range numerics at construction.

        ``json.loads`` accepts ``NaN``/``Infinity``, and a NaN
        ``latch_timeout_s`` makes ``_wait_for_latch`` never reach its
        deadline — an unreleased latch would hang forever instead of
        failing loudly. Same class of hazard for the other numeric knobs,
        so validate them all here (misconfigured injection must fail the
        load, not silently wedge it).
        """
        if self.oom_on_dispatch is not None and self.oom_on_dispatch < 1:
            msg = f"oom_on_dispatch must be >= 1, got {self.oom_on_dispatch}"
            raise ValueError(msg)
        if self.oom_repeat < 1:
            msg = f"oom_repeat must be >= 1, got {self.oom_repeat}"
            raise ValueError(msg)
        for name in ("slow_load_s", "teardown_hang_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                msg = f"{name} must be a finite number >= 0, got {value}"
                raise ValueError(msg)
        if not math.isfinite(self.latch_timeout_s) or self.latch_timeout_s <= 0:
            msg = f"latch_timeout_s must be a finite number > 0, got {self.latch_timeout_s}"
            raise ValueError(msg)


def _resolve_faults(loadtime_faults: dict[str, Any] | None, fault_key: str | None) -> FakeFaults:
    """Merge the env override (``SIE_FAKE_FAULTS``) over the YAML fault spec.

    Malformed env JSON or unknown fault names raise at adapter construction —
    a misconfigured injection must fail the load loudly, not silently run a
    healthy fake.
    """
    merged: dict[str, Any] = dict(loadtime_faults or {})
    raw = os.environ.get(SIE_FAKE_FAULTS_ENV)
    if raw is not None and fault_key is not None:
        try:
            by_model = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"invalid {SIE_FAKE_FAULTS_ENV} JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(by_model, dict):
            msg = f"{SIE_FAKE_FAULTS_ENV} must be a JSON object keyed by fault_key"
            raise ValueError(msg)
        override = by_model.get(fault_key)
        if override is not None:
            if not isinstance(override, dict):
                msg = f"{SIE_FAKE_FAULTS_ENV}[{fault_key!r}] must be a JSON object"
                raise ValueError(msg)
            merged.update(override)
    try:
        return FakeFaults(**merged)
    except TypeError as exc:
        msg = f"unknown fake fault option for {fault_key!r}: {exc}"
        raise ValueError(msg) from exc


def _wait_for_latch(latch_file: str, timeout_s: float, *, what: str) -> None:
    """Block (sync) until ``latch_file`` exists; TimeoutError on expiry."""
    deadline = time.monotonic() + timeout_s
    path = Path(latch_file)
    while not path.exists():
        if time.monotonic() >= deadline:
            msg = f"sie-fake {what} latch '{latch_file}' not released within {timeout_s}s"
            raise TimeoutError(msg)
        time.sleep(_LATCH_POLL_INTERVAL_S)


async def _await_latch(latch_file: str, timeout_s: float, *, what: str) -> None:
    """Async variant of :func:`_wait_for_latch` for the generation path.

    Delegates the poll loop to a worker thread so the event loop never
    blocks on filesystem checks.
    """
    await asyncio.to_thread(_wait_for_latch, latch_file, timeout_s, what=what)


class _FaultRuntime:
    """Per-adapter-instance fault state: dispatch counter + hook helpers."""

    def __init__(self, faults: FakeFaults) -> None:
        self.faults = faults
        self._dispatches = 0

    def on_load(self) -> None:
        if self.faults.fail_load:
            msg = "sie-fake injected load failure"
            raise RuntimeError(msg)
        if self.faults.slow_load_s > 0:
            time.sleep(self.faults.slow_load_s)
        if self.faults.load_latch_file is not None:
            _wait_for_latch(self.faults.load_latch_file, self.faults.latch_timeout_s, what="load")

    def _next_dispatch_ordinal(self) -> int:
        self._dispatches += 1
        return self._dispatches

    def _maybe_raise_oom(self, ordinal: int) -> None:
        first = self.faults.oom_on_dispatch
        if first is not None and first <= ordinal < first + self.faults.oom_repeat:
            if self.faults.oom_typed:
                # The typed terminal the worker's BatchExecutor raises —
                # exercises ``is_oom_error``'s isinstance tier.
                msg = f"sie-fake injected typed OOM (dispatch {ordinal})"
                raise ResourceExhaustedError(
                    msg,
                    ResourceExhausted(operation="fake-dispatch", attempts=1, original_message=msg),
                )
            # Phrase chosen to satisfy the real ``is_oom_error`` substring
            # classification (core/oom.py) — no test-only error types.
            msg = f"out of memory (sie-fake injected fault, dispatch {ordinal})"
            raise RuntimeError(msg)

    def on_dispatch(self) -> None:
        if self.faults.dispatch_latch_file is not None:
            _wait_for_latch(self.faults.dispatch_latch_file, self.faults.latch_timeout_s, what="dispatch")
        self._maybe_raise_oom(self._next_dispatch_ordinal())

    async def on_dispatch_async(self) -> None:
        if self.faults.dispatch_latch_file is not None:
            await _await_latch(self.faults.dispatch_latch_file, self.faults.latch_timeout_s, what="dispatch")
        self._maybe_raise_oom(self._next_dispatch_ordinal())

    def on_unload(self) -> None:
        if self.faults.teardown_hang_s > 0:
            time.sleep(self.faults.teardown_hang_s)


def _item_key(item: Item) -> str:
    """Stable per-item hash key: text first, raw payload bytes otherwise."""
    if item.text is not None:
        return item.text
    hasher = hashlib.sha256()
    for image in item.images or []:
        hasher.update(image["data"])
    for payload in (item.audio, item.video, item.document):
        if payload is not None:
            hasher.update(payload["data"])
    if item.id is not None:
        hasher.update(item.id.encode())
    return hasher.hexdigest()


def _hash_unit_floats(seed: str, n: int) -> np.ndarray:
    """Expand ``seed`` into ``n`` deterministic float32 values in ``[-1, 1)``."""
    out = np.empty(n, dtype=np.float32)
    filled = 0
    counter = 0
    while filled < n:
        digest = hashlib.sha256(f"{seed}\x00{counter}".encode()).digest()
        words = np.frombuffer(digest, dtype=np.uint32)
        take = min(len(words), n - filled)
        out[filled : filled + take] = words[:take].astype(np.float64) / 2**31 - 1.0
        filled += take
        counter += 1
    return out


def _hash_unit_interval(seed: str) -> float:
    """Map ``seed`` to a deterministic float in ``[0, 1)``."""
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeAdapter(BaseAdapter, GenerationAdapter):
    """One weightless fake serving every supported surface: hash-derived
    dense embeddings (``encode``), pair scores in ``[0, 1)`` (``score``), and
    a deterministic token stream (``generate``).

    All fake catalog cases are profiles of the single ``sie-fake`` model
    (``models/sie-fake.yaml``); non-default profiles are addressable as
    ``sie-fake:<profile>`` via the loader's variant expansion.

    MRO note: ``BaseAdapter`` leads so its spec-driven ``unload``, batched
    ``score_pairs`` delegate, and ``CharCountPreprocessor`` apply, while
    ``GenerationAdapter`` membership routes the worker's generation dispatch
    (``isinstance(adapter, GenerationAdapter)``) and validates the streaming
    contract.

    Generation streams ``min(default_completion_tokens, max_new_tokens)``
    tokens from the prompt hash. ``stop`` strings, ``seed``, sampling knobs,
    and ``logprobs`` are accepted and ignored — determinism comes from the
    input alone. ``finish_reason`` is ``"length"`` when truncated by
    ``max_new_tokens``, else ``"stop"``.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("dense", "score", "tokens"),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,  # unused; fakes are package-backed
        *,
        dense_dim: int = 384,
        max_seq_length: int | None = None,
        compute_precision: str | None = None,  # unused; outputs are float32
        memory_footprint_bytes: int = _DEFAULT_FOOTPRINT_BYTES,
        request_latency_s: float = 0.0,
        default_completion_tokens: int = 32,
        inter_token_latency_s: float = 0.0,
        faults: dict[str, Any] | None = None,
        fault_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, max_seq_length, compute_precision, kwargs)
        self._dense_dim = dense_dim
        self._request_latency_s = request_latency_s
        self._default_completion_tokens = default_completion_tokens
        self._inter_token_latency_s = inter_token_latency_s
        self._memory_footprint_bytes = memory_footprint_bytes
        self._fault_runtime = _FaultRuntime(_resolve_faults(faults, fault_key))
        self._loaded = False
        self._device: str | None = None

    # -- Lifecycle -----------------------------------------------------------

    def load(self, device: str) -> None:
        self._fault_runtime.on_load()
        self._device = device
        self._loaded = True

    def unload(self) -> None:
        self._fault_runtime.on_unload()
        self._loaded = False
        super().unload()

    def memory_footprint(self) -> int:
        return self._memory_footprint_bytes

    def load_required_memory_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        _ = (device_type, device_total_bytes)
        return self._memory_footprint_bytes

    # -- Surfaces --------------------------------------------------------------

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        _ = (output_types, instruction, prepared_items, options)
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        self._fault_runtime.on_dispatch()
        if self._request_latency_s > 0:
            time.sleep(self._request_latency_s)
        dense = np.stack([_hash_unit_floats(_item_key(item), self._dense_dim) for item in items])
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = (dense / np.maximum(norms, 1e-12)).astype(np.float32)
        return EncodeOutput(dense=dense, is_query=is_query, dense_dim=self._dense_dim)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        _ = (instruction, options)
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        self._fault_runtime.on_dispatch()
        if self._request_latency_s > 0:
            time.sleep(self._request_latency_s)
        query_key = _item_key(query)
        return [_hash_unit_interval(f"{query_key}\x00{_item_key(item)}") for item in items]

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score pairs with exact synthetic-token usage.

        The fake engine defines one Unicode code point as one synthetic input
        token. This keeps fake-stack usage deterministic and authoritative
        without presenting the shared character-count reserve estimate as a
        real tokenizer measurement.
        """
        output = super().score_pairs(
            queries,
            docs,
            instruction=instruction,
            options=options,
        )
        output.input_token_counts = [
            len(query.text or "") + len(doc.text or "") for query, doc in zip(queries, docs, strict=True)
        ]
        return output

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (
            temperature,
            top_p,
            stop,
            frequency_penalty,
            presence_penalty,
            top_k,
            repetition_penalty,
            grammar,
            seed,
            logit_bias,
            min_new_tokens,
            logprobs,
            top_logprobs,
            images,
        )
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        await self._fault_runtime.on_dispatch_async()
        count = min(self._default_completion_tokens, max_new_tokens)
        finish_reason = "length" if max_new_tokens < self._default_completion_tokens else "stop"
        # Char-count proxy, mirroring the reserve-estimate basis used when no
        # real tokenizer exists.
        prompt_tokens = max(1, len(prompt) // 4)
        for i in range(count):
            if self._inter_token_latency_s > 0:
                await asyncio.sleep(self._inter_token_latency_s)
            fragment = hashlib.sha256(f"{prompt}\x00{i}".encode()).hexdigest()[:8]
            separator = "" if i == count - 1 else " "
            yield GenerationChunk(text_delta=f"tok_{fragment}{separator}", is_first=(i == 0))
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=count,
        )
