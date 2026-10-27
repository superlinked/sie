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
loader's variant expansion. The fake has no chat template: the single-node
``/v1/chat/completions`` route gets a scripted answer from
:meth:`FakeAdapter.chat_completion`, and queued chat, which renders a chat
template, is refused. ``/v1/generate`` (raw prompt) works on every path.

A model config that declares sparse or multivector dimensions, or an extract
task, also gets those outputs from the same adapter, and image items are
accepted wherever text is. With ``synthetic_usage`` set, encode and extract
report one synthetic token per Unicode code point of an item's text, as score
always does, and encode reports the images it received.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationChunk,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.base import HostOffload
from sie_server.core.inference_output import (
    EncodeOutput,
    ExtractItemError,
    ExtractOutput,
    ScoreOutput,
    SparseVector,
)
from sie_server.core.oom import ResourceExhausted, ResourceExhaustedError
from sie_server.types.responses import Classification, DetectedObject, Entity, ErrorCode, Relation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sie_server.types.grammar import GrammarSpec
    from sie_server.types.inputs import ImageInput, Item, VideoInput

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


def _hash_tokens(seed: str, count: int) -> list[str]:
    """``count`` hash-derived tokens, each but the last followed by a space."""
    tokens = []
    for i in range(count):
        fragment = hashlib.sha256(f"{seed}\x00{i}".encode()).hexdigest()[:8]
        tokens.append(f"tok_{fragment}" + ("" if i == count - 1 else " "))
    return tokens


_WORD = re.compile(r"\S+")
_PIECE = re.compile(r"\S+\s*")
_SPARSE_NONZEROS = 8
_TOKENS_PER_IMAGE = 4


def _hash_sparse(seed: str, sparse_dim: int) -> SparseVector:
    """Up to eight distinct hash-chosen indices below ``sparse_dim``, ascending, with positive values."""
    digest = hashlib.sha256(f"{seed}\x00sparse".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    count = min(_SPARSE_NONZEROS, sparse_dim)
    indices = np.sort(rng.choice(sparse_dim, size=count, replace=False)).astype(np.int32)
    return SparseVector(indices=indices, values=(rng.random(count) + 0.01).astype(np.float32))


def _hash_multivector(item: Item, multivector_dim: int) -> np.ndarray:
    """One unit vector per word of the item's text and four per image, at least one."""
    key = _item_key(item)
    rows = max(1, len(_WORD.findall(item.text or "")) + _TOKENS_PER_IMAGE * len(item.images or []))
    vectors = np.stack([_hash_unit_floats(f"{key}\x00token\x00{row}", multivector_dim) for row in range(rows)])
    return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class _FakeExtraction:
    entities: list[Entity]
    classifications: list[Classification]
    relations: list[Relation]
    objects: list[DetectedObject]
    data: dict[str, Any]
    error: ExtractItemError | None


def _hash_extraction(item: Item, labels: list[str]) -> _FakeExtraction:
    text = item.text or ""
    images = item.images or []
    if not text and not images:
        error = ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message="sie-fake extracts from text or images")
        return _FakeExtraction(entities=[], classifications=[], relations=[], objects=[], data={}, error=error)
    key = _item_key(item)
    words = list(_WORD.finditer(text))
    entities: list[Entity] = []
    for label in labels if words else []:
        word = words[int(_hash_unit_interval(f"{key}\x00{label}") * len(words))]
        entities.append(
            Entity(
                text=word.group(),
                label=label,
                score=_hash_unit_interval(f"{key}\x00{label}\x00score"),
                start=word.start(),
                end=word.end(),
                bbox=None,
            )
        )
    relations: list[Relation] = []
    if len(entities) > 1:
        relations.append(
            Relation(
                head=entities[0]["text"],
                tail=entities[1]["text"],
                relation="related_to",
                score=_hash_unit_interval(f"{key}\x00relation"),
            )
        )
    objects = [
        DetectedObject(
            label=labels[0],
            score=_hash_unit_interval(f"{key}\x00object\x00{index}"),
            bbox=[0, 0, 1 + int(64 * _hash_unit_interval(f"{key}\x00width\x00{index}")), 1 + index],
        )
        for index in range(len(images))
    ]
    return _FakeExtraction(
        entities=entities,
        classifications=[
            Classification(label=label, score=_hash_unit_interval(f"{key}\x00class\x00{label}")) for label in labels
        ],
        relations=relations,
        objects=objects,
        data={"characters": len(text), "images": len(images), "labels": list(labels)},
        error=None,
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(part.get("text") or "" for part in content)
    return content or ""


def _chat_transcript(body: dict[str, Any]) -> str:
    """Each declared tool, then each message's role, text and tool calls, one per line."""
    lines = [json.dumps(tool, sort_keys=True) for tool in body.get("tools") or []]
    for message in body["messages"]:
        calls = json.dumps(message["tool_calls"]) if message.get("tool_calls") else ""
        lines.append(f"{message['role']} {_message_text(message)} {calls}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ChatTurn:
    """One scripted answer cut to its token budget: one ``(kind, text)`` step per output token."""

    steps: list[tuple[str, str]]
    call_id: str
    call_name: str | None
    finish_reason: str
    prompt_tokens: int
    choices: int

    @property
    def usage(self) -> dict[str, int]:
        completion = len(self.steps) * self.choices
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": self.prompt_tokens + completion,
        }

    def delta(self, kind: str, text: str) -> dict[str, Any]:
        if kind == "call":
            function = {"name": self.call_name, "arguments": ""}
            return {"tool_calls": [{"index": 0, "id": self.call_id, "type": "function", "function": function}]}
        if kind == "arguments":
            return {"tool_calls": [{"index": 0, "function": {"arguments": text}}]}
        return {"reasoning_content" if kind == "reasoning" else "content": text}

    def message(self, *, keep_reasoning: bool) -> dict[str, Any]:
        text = {kind: "".join(t for k, t in self.steps if k == kind) for kind in ("reasoning", "arguments", "content")}
        message: dict[str, Any] = {"role": "assistant", "content": text["content"]}
        if keep_reasoning and text["reasoning"]:
            message["reasoning_content"] = text["reasoning"]
        if self.call_name is not None:
            function = {"name": self.call_name, "arguments": text["arguments"]}
            message["tool_calls"] = [{"id": self.call_id, "type": "function", "function": function}]
        return message


class FakeAdapter(BaseAdapter, GenerationAdapter):
    """One weightless fake serving every supported surface: hash-derived
    embeddings (``encode``), pair scores in ``[0, 1)`` (``score``),
    extractions (``extract``), and a deterministic token stream (``generate``).

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

    Chat follows a script, for wire tests only. With tools declared and
    ``tool_choice`` not ``"none"``, a turn calls one tool, the named one or
    else the first, unless its last message is a tool result and the call is
    not forced. Every required parameter gets the last user message's text.
    A turn ending in tool results answers ``Tool result: <content>`` for
    each, and any other turn answers ``default_completion_tokens`` hash
    tokens. When ``chat_template_kwargs`` enables thinking, one sentence of
    reasoning naming the rule comes first, returned only with
    ``keep_reasoning``. A prompt token is a whitespace-separated word of the
    tools and messages, and an output token is one streamed piece. A text
    answer stops at the token limit with ``"length"``. A tool call is never
    cut: a turn without room for its reasoning and the whole call is refused,
    as is a prompt that leaves no room in ``max_seq_length``. ``n`` repeats
    the turn.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image"),
        outputs=("dense", "sparse", "multivector", "score", "json", "tokens"),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,  # unused; fakes are package-backed
        *,
        dense_dim: int = 384,
        sparse_dim: int | None = None,
        multivector_dim: int | None = None,
        max_seq_length: int | None = None,
        compute_precision: str | None = None,  # unused; outputs are float32
        memory_footprint_bytes: int = _DEFAULT_FOOTPRINT_BYTES,
        request_latency_s: float = 0.0,
        default_completion_tokens: int = 32,
        inter_token_latency_s: float = 0.0,
        synthetic_usage: bool = False,
        faults: dict[str, Any] | None = None,
        fault_key: str | None = None,
        host_offload: bool = True,
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, compute_precision, kwargs)
        self._max_seq_length = max_seq_length
        self._dense_dim = dense_dim
        self._sparse_dim = sparse_dim
        self._multivector_dim = multivector_dim
        self._synthetic_usage = synthetic_usage
        self._request_latency_s = request_latency_s
        self._default_completion_tokens = default_completion_tokens
        self._inter_token_latency_s = inter_token_latency_s
        self._memory_footprint_bytes = memory_footprint_bytes
        self._fault_runtime = _FaultRuntime(_resolve_faults(faults, fault_key))
        self._loaded = False
        self._device: str | None = None
        self._host_offload_enabled = host_offload
        self._parked = False

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

    # -- Host-RAM parking (simulated: the declared footprint moves) ----------

    def host_offload_bytes(self) -> int | None:
        if not self._host_offload_enabled or not self._loaded:
            return None
        return self._memory_footprint_bytes

    def offload_to_host(self) -> HostOffload:
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        self._loaded = False
        self._parked = True
        return HostOffload(host_bytes=self._memory_footprint_bytes)

    def restore_from_host(self) -> int:
        if not self._parked:
            msg = "sie-fake model is not parked"
            raise RuntimeError(msg)
        self._parked = False
        self._loaded = True
        return 0

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
        _ = (instruction, prepared_items, options)
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        self._fault_runtime.on_dispatch()
        if self._request_latency_s > 0:
            time.sleep(self._request_latency_s)
        dense = np.stack([_hash_unit_floats(_item_key(item), self._dense_dim) for item in items])
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = (dense / np.maximum(norms, 1e-12)).astype(np.float32)
        sparse_dim, multivector_dim = self._sparse_dim, self._multivector_dim
        extra: dict[str, Any] = {}
        if self._synthetic_usage:
            extra["input_token_counts"] = [len(item.text or "") for item in items]
            extra["input_image_counts"] = [len(item.images or []) for item in items]
        return EncodeOutput(
            dense=dense,
            sparse=[_hash_sparse(_item_key(item), sparse_dim) for item in items]
            if "sparse" in output_types and sparse_dim is not None
            else None,
            multivector=[_hash_multivector(item, multivector_dim) for item in items]
            if "multivector" in output_types and multivector_dim is not None
            else None,
            is_query=is_query,
            dense_dim=self._dense_dim,
            extra=extra,
        )

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

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        """Extract hash-derived entities, classifications, relations, image objects and data.

        Each label gets one entity, a word of the item's text, and one
        classification. Two or more entities are related once. Each image gets
        one object. An item with neither text nor images is a per-item
        ``INVALID_INPUT`` failure.
        """
        _ = (output_schema, instruction, options, prepared_items)
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        self._fault_runtime.on_dispatch()
        if self._request_latency_s > 0:
            time.sleep(self._request_latency_s)
        names = list(labels or ["thing"])
        results = [_hash_extraction(item, names) for item in items]
        errors = [result.error for result in results]
        return ExtractOutput(
            entities=[result.entities for result in results],
            classifications=[result.classifications for result in results],
            relations=[result.relations for result in results],
            objects=[result.objects for result in results],
            data=[result.data for result in results],
            errors=errors if any(error is not None for error in errors) else None,
            input_token_counts=[len(item.text or "") for item in items] if self._synthetic_usage else None,
        )

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
        videos: list[VideoInput] | None = None,
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
            videos,
        )
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        await self._fault_runtime.on_dispatch_async()
        count = min(self._default_completion_tokens, max_new_tokens)
        finish_reason = "length" if max_new_tokens < self._default_completion_tokens else "stop"
        # Char-count proxy, mirroring the reserve-estimate basis used when no
        # real tokenizer exists.
        prompt_tokens = max(1, len(prompt) // 4)
        for i, token in enumerate(_hash_tokens(prompt, count)):
            if self._inter_token_latency_s > 0:
                await asyncio.sleep(self._inter_token_latency_s)
            yield GenerationChunk(text_delta=token, is_first=(i == 0))
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=count,
        )

    async def _chat_turn(self, body: dict[str, Any], requested_model: str) -> _ChatTurn:
        if not self._loaded:
            raise RuntimeError(ERR_NOT_LOADED)
        await self._fault_runtime.on_dispatch_async()
        transcript = _chat_transcript(body)
        prompt_tokens = len(transcript.split())
        budget = body.get("max_completion_tokens") or body["max_tokens"]
        if self._max_seq_length is not None:
            room = self._max_seq_length - prompt_tokens
            if room < 1:
                raise GenerationInputTooLongError(
                    f"prompt_tokens ({prompt_tokens}) exceeds context_length ({self._max_seq_length}) "
                    f"for model '{requested_model}'"
                )
            budget = min(budget, room)
        messages = body["messages"]
        tools = [tool["function"] for tool in body.get("tools") or []]
        choice = body.get("tool_choice")
        named = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
        forced = choice == "required" or named is not None
        results: list[str] = []
        for message in reversed(messages):
            if message["role"] != "tool":
                break
            results.insert(0, _message_text(message))
        call_name: str | None = None
        arguments = ""
        content: list[str] = []
        if tools and choice != "none" and (forced or not results):
            function = next((tool for tool in tools if tool["name"] == named), tools[0])
            call_name = function["name"]
            question = next((_message_text(m) for m in reversed(messages) if m["role"] == "user"), "")
            required = (function.get("parameters") or {}).get("required") or []
            arguments = json.dumps(dict.fromkeys(required, question), ensure_ascii=False)
            reason = "tool_choice requires a call" if forced else "the last message is not a tool result"
            thought = f"Tools are declared and {reason}, so call {call_name}."
        elif results:
            thought = "The last message is a tool result, so answer with it."
            content = _PIECE.findall(" ".join(f"Tool result: {result}" for result in results))
        else:
            thought = "No tool call applies, so answer in text."
            content = _hash_tokens(transcript, self._default_completion_tokens)
        steps: list[tuple[str, str]] = []
        if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is True:
            steps += [("reasoning", piece) for piece in _PIECE.findall(thought)]
        if call_name is not None:
            steps.append(("call", ""))
            steps += [("arguments", piece) for piece in _PIECE.findall(arguments)]
        steps += [("content", piece) for piece in content]
        if call_name is not None and len(steps) > budget:
            field = "max_completion_tokens" if body.get("max_completion_tokens") else "max_tokens"
            msg = f"the scripted tool call needs {len(steps)} output tokens and {budget} are available"
            raise GenerationInvalidRequestError(field, msg)
        return _ChatTurn(
            steps=steps[:budget],
            call_id="call_" + hashlib.sha256(f"{transcript}\x00call".encode()).hexdigest()[:24],
            call_name=call_name,
            finish_reason="length" if len(steps) > budget else "tool_calls" if call_name else "stop",
            prompt_tokens=prompt_tokens,
            choices=body.get("n") or 1,
        )

    async def chat_completion(
        self,
        body: dict[str, Any],
        *,
        requested_model: str,
        max_response_bytes: int = 32 << 20,
        keep_reasoning: bool = False,
    ) -> dict[str, Any]:
        _ = max_response_bytes
        turn = await self._chat_turn(body, requested_model)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": index,
                    "message": turn.message(keep_reasoning=keep_reasoning),
                    "finish_reason": turn.finish_reason,
                }
                for index in range(turn.choices)
            ],
            "usage": turn.usage,
        }

    async def chat_completion_stream(
        self,
        body: dict[str, Any],
        *,
        requested_model: str,
        max_response_bytes: int = 32 << 20,
        keep_reasoning: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        _ = max_response_bytes
        turn = await self._chat_turn(body, requested_model)
        envelope = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
        }
        for index in range(turn.choices):
            deltas = [{"role": "assistant"}]
            deltas += [turn.delta(kind, text) for kind, text in turn.steps if keep_reasoning or kind != "reasoning"]
            for delta in deltas:
                if self._inter_token_latency_s > 0:
                    await asyncio.sleep(self._inter_token_latency_s)
                yield {**envelope, "choices": [{"index": index, "delta": delta, "finish_reason": None}]}
            yield {**envelope, "choices": [{"index": index, "delta": {}, "finish_reason": turn.finish_reason}]}
        yield {**envelope, "choices": [], "usage": turn.usage}
