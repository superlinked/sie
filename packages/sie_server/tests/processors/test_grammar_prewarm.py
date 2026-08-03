"""Tests for the boot-time grammar prewarm path.

The :class:`StreamingProcessor` exposes
:meth:`prewarm_grammars_for_model` which iterates
``tasks.generate.prewarm_grammars`` and populates the LRU before the
first request arrives. These tests exercise:

* config-load validation of the ``prewarm_grammars`` field shape
* the no-op path (empty list)
* successful prewarm for JSON-schema and regex entries
* a mixed-kind list
* per-entry failure isolation (one bad entry does not break the others)
* prewarm-then-request: a request hitting a prewarmed grammar is a cache hit

Outlines is bundle-only, so the compile factory is monkey-patched the
same way :mod:`test_grammar_compile` and :mod:`test_streaming` do.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from sie_server.config.model import (
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    PrewarmGrammar,
    ProfileConfig,
    Tasks,
)
from sie_server.processors.streaming import StreamingProcessor

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    prewarm_grammars: list[PrewarmGrammar] | None = None,
    sie_id: str = "test/prewarm-model",
    grammar_backend: str | None = "outlines",
) -> ModelConfig:
    """Build a generation ``ModelConfig`` with the given prewarm list."""
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(
            generate=GenerateTask(
                context_length=4096,
                max_output_tokens=512,
                capabilities=GenerateCapabilities(
                    grammar=["json_schema", "regex"],
                    streaming=True,
                    tools=False,
                ),
                prewarm_grammars=prewarm_grammars or [],
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=4096,
                kv_budget_tokens=4096,
                adapter_options={"loadtime": {"grammar_backend": grammar_backend}},
            ),
        },
    )


def _make_registry(config: ModelConfig) -> MagicMock:
    """Registry stub that returns ``config`` for the model's id.

    The streaming processor only needs ``get_config`` for the prewarm
    path; the request-path methods (``get``, ``is_loaded``) are not
    exercised here.
    """
    registry = MagicMock()
    registry.get_config.return_value = config
    registry.model_names = [config.sie_id]
    return registry


def _patch_compile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raise_on: tuple[str, ...] = (),
) -> list[tuple[Any, Any]]:
    """Install a stub ``compile_outlines`` in ``streaming`` and a fake
    tokenizer loader.

    Returns a list of ``(tok, grammar)`` tuples — one per compile call.
    When ``grammar.label`` is in ``raise_on`` the stub raises a
    :class:`RuntimeError` so we can exercise the per-entry failure path.
    """
    calls: list[tuple[Any, Any]] = []

    def _stub_compile(tok: Any, grammar: Any) -> Any:
        calls.append((tok, grammar))
        if grammar.label in raise_on:
            msg = f"intentional failure for {grammar.label}"
            raise RuntimeError(msg)
        return True

    monkeypatch.setattr("sie_server.processors.streaming.compile_outlines", _stub_compile)

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return object()

    monkeypatch.setattr(
        StreamingProcessor,
        "_get_tokenizer",
        _fake_get_tokenizer,
    )
    return calls


def _make_processor(registry: MagicMock) -> StreamingProcessor:
    return StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")


# ---------------------------------------------------------------------------
# Config-load validation
# ---------------------------------------------------------------------------


class TestPrewarmGrammarConfig:
    def test_empty_list_is_default(self) -> None:
        """Field defaults to ``[]`` so existing configs are unaffected."""
        task = GenerateTask(context_length=4096, max_output_tokens=512)
        assert task.prewarm_grammars == []

    def test_json_schema_entry_accepted(self) -> None:
        entry = PrewarmGrammar(
            name="math_response",
            kind="json_schema",
            value={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        assert entry.kind == "json_schema"
        assert isinstance(entry.value, dict)

    def test_regex_entry_accepted(self) -> None:
        entry = PrewarmGrammar(name="ssn", kind="regex", value=r"\d{3}-\d{2}-\d{4}")
        assert entry.kind == "regex"
        assert entry.value == r"\d{3}-\d{2}-\d{4}"

    def test_invalid_kind_rejected(self) -> None:
        """``kind`` outside the GrammarKind literal is a config-load error.

        Note: ``ebnf`` is now a valid prewarm kind (added when the gateway/
        worker gained EBNF capability). This test uses a truly invalid
        kind to exercise the literal-typed rejection.
        """
        with pytest.raises(ValidationError):
            PrewarmGrammar(name="bad", kind="not-a-real-kind", value="ignored")  # type: ignore[arg-type]

    def test_json_schema_kind_with_string_value_rejected(self) -> None:
        """Cross-field validator: kind=json_schema requires a dict value."""
        with pytest.raises(ValidationError):
            PrewarmGrammar(name="mismatched", kind="json_schema", value="not a dict")

    def test_regex_kind_with_dict_value_rejected(self) -> None:
        """Cross-field validator: kind=regex requires a str value."""
        with pytest.raises(ValidationError):
            PrewarmGrammar(name="mismatched", kind="regex", value={"not": "a regex"})

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PrewarmGrammar(name="x", kind="regex", value="a", extras="boom")  # type: ignore[call-arg]

    def test_full_config_loads_with_prewarm_grammars(self) -> None:
        config = _make_config(
            prewarm_grammars=[
                PrewarmGrammar(name="a", kind="regex", value=r"\d+"),
                PrewarmGrammar(name="b", kind="json_schema", value={"type": "string"}),
            ]
        )
        assert config.tasks.generate is not None
        assert len(config.tasks.generate.prewarm_grammars) == 2


# ---------------------------------------------------------------------------
# Prewarm runtime behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_prewarm_list_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty list → no compiles, cache stays empty, no metric increments."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(prewarm_grammars=[])
    proc = _make_processor(_make_registry(config))

    await proc.prewarm_grammars_for_model(config.sie_id)

    assert calls == []
    assert len(proc._grammar_cache) == 0


@pytest.mark.asyncio
async def test_xgrammar_prewarm_skips_outlines_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """XGrammar prewarm must not call Outlines with Qwen3.5 tokenizers."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(
        grammar_backend="xgrammar",
        prewarm_grammars=[PrewarmGrammar(name="yes_no", kind="regex", value="^(yes|no)$")],
    )
    proc = _make_processor(_make_registry(config))

    await proc.prewarm_grammars_for_model(config.sie_id)

    assert calls == []
    assert len(proc._grammar_cache) == 0


@pytest.mark.asyncio
async def test_single_json_schema_prewarms(monkeypatch: pytest.MonkeyPatch) -> None:
    """One JSON-schema entry produces exactly one compile and one cache entry."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(
        prewarm_grammars=[
            PrewarmGrammar(
                name="math_response",
                kind="json_schema",
                value={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            ),
        ]
    )
    proc = _make_processor(_make_registry(config))
    await proc.prewarm_grammars_for_model(config.sie_id)

    assert len(calls) == 1
    assert calls[0][1].kind == "json_schema"
    assert len(proc._grammar_cache) == 1


@pytest.mark.asyncio
async def test_single_regex_prewarms(monkeypatch: pytest.MonkeyPatch) -> None:
    """One regex entry produces one compile and one cache entry."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(
        prewarm_grammars=[
            PrewarmGrammar(name="ssn", kind="regex", value=r"\d{3}-\d{2}-\d{4}"),
        ]
    )
    proc = _make_processor(_make_registry(config))
    await proc.prewarm_grammars_for_model(config.sie_id)

    assert len(calls) == 1
    assert calls[0][1].kind == "regex"
    assert len(proc._grammar_cache) == 1


@pytest.mark.asyncio
async def test_mixed_entries_all_prewarmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON-schema + regex in one config → both end up in the cache."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(
        prewarm_grammars=[
            PrewarmGrammar(name="a", kind="json_schema", value={"type": "object"}),
            PrewarmGrammar(name="b", kind="regex", value=r"\d+"),
        ]
    )
    proc = _make_processor(_make_registry(config))

    await proc.prewarm_grammars_for_model(config.sie_id)

    assert len(calls) == 2
    assert len(proc._grammar_cache) == 2


@pytest.mark.asyncio
async def test_failure_on_one_entry_does_not_block_others(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Compile failure on one entry is logged while model load continues."""
    calls = _patch_compile(monkeypatch, raise_on=("bad",))
    config = _make_config(
        sie_id="test/mixed-failure",
        prewarm_grammars=[
            PrewarmGrammar(name="good_a", kind="json_schema", value={"type": "object"}),
            PrewarmGrammar(name="bad", kind="regex", value=r"\d+"),
            PrewarmGrammar(name="good_b", kind="regex", value=r"[a-z]+"),
        ],
    )
    proc = _make_processor(_make_registry(config))

    import logging

    with caplog.at_level(logging.ERROR, logger="sie_server.processors.streaming"):
        # Must NOT raise — failure path absorbs and continues.
        await proc.prewarm_grammars_for_model(config.sie_id)

    # All three were attempted.
    assert len(calls) == 3
    # Two succeeded (good_a + good_b) → in cache.
    assert len(proc._grammar_cache) == 2
    # Failure surfaced as an ERROR log.
    assert any("bad" in rec.message and "compile failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_non_generation_config_is_silent_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config with no ``tasks.generate`` → method short-circuits cleanly.

    Belt-and-braces test: callers gate on
    :func:`is_generation_model`, but the method itself also has the
    guard for defence-in-depth.
    """
    calls = _patch_compile(monkeypatch)
    # Build a non-generation config (encode only) — has no
    # ``prewarm_grammars`` field at all.
    from sie_server.config.model import EmbeddingDim, EncodeTask

    config = ModelConfig(
        sie_id="test/encoder",
        hf_id="test/encoder",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=384))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.encode_dense:DenseEncoder",
                max_batch_tokens=4096,
            ),
        },
    )
    proc = _make_processor(_make_registry(config))

    await proc.prewarm_grammars_for_model(config.sie_id)

    assert calls == []
    assert len(proc._grammar_cache) == 0


@pytest.mark.asyncio
async def test_prewarmed_grammar_hits_cache_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: prewarm an entry, then a request with the same grammar
    observes a cache hit (no second compile call).

    Mirrors the cache-key derivation in :meth:`_ensure_grammar_ready` so a
    drift in either path would surface here.
    """
    telemetry = MagicMock()
    monkeypatch.setattr(
        "sie_server.processors.streaming._metrics.worker_telemetry",
        lambda: telemetry,
    )
    calls = _patch_compile(monkeypatch)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    config = _make_config(prewarm_grammars=[PrewarmGrammar(name="x_schema", kind="json_schema", value=schema)])
    registry = _make_registry(config)
    proc = _make_processor(registry)

    await proc.prewarm_grammars_for_model(config.sie_id)
    assert len(calls) == 1
    # Simulate the request-path lookup directly: it goes through
    # ``_ensure_grammar_ready`` which writes the same key shape into the
    # cache and emits a canonical cache-hit event. Calling the helper
    # without the full process() machinery is cleaner than wiring up a
    # work item just to test the cache collision.
    from sie_server.types.grammar import GrammarSpec, hash_grammar

    grammar = GrammarSpec(kind="json_schema", value=schema)
    tokenizer_hash = str(config.hf_id or config.weights_path or config.sie_id)
    key = (tokenizer_hash, hash_grammar(grammar), "outlines")
    assert proc._grammar_cache.get(key) is not None, "cache miss after prewarm"

    # And the request-path helper would return True via the cache-hit
    # branch — emulate the increment to confirm semantic parity.
    msg = AsyncMock()
    ok = await proc._ensure_grammar_ready(
        grammar,
        model_id=config.sie_id,
        reply_subject="_INBOX.test",
        request_id="r1",
        attempt_id="a1",
        msg=msg,
    )
    assert ok is True
    # No second compile happened.
    assert len(calls) == 1
    assert telemetry.grammar_compile_completed.call_count == 1
    assert telemetry.grammar_compile_completed.call_args.kwargs["phase"] == "prewarm"
    assert telemetry.grammar_compile_completed.call_args.kwargs["outcome"] == "success"
    assert [call.kwargs["result"] for call in telemetry.grammar_cache_lookup.call_args_list] == [
        "miss",
        "hit",
    ]


@pytest.mark.asyncio
async def test_unknown_model_is_silent_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling prewarm with a model id missing from the registry returns silently."""
    _patch_compile(monkeypatch)
    registry = MagicMock()
    registry.get_config.side_effect = KeyError("missing")
    proc = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")

    # Must not raise.
    await proc.prewarm_grammars_for_model("nonexistent/model")
    assert len(proc._grammar_cache) == 0


@pytest.mark.asyncio
async def test_idempotent_under_repeated_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running prewarm with the same config doesn't double-compile."""
    calls = _patch_compile(monkeypatch)
    config = _make_config(prewarm_grammars=[PrewarmGrammar(name="a", kind="regex", value=r"\d+")])
    proc = _make_processor(_make_registry(config))

    await proc.prewarm_grammars_for_model(config.sie_id)
    await proc.prewarm_grammars_for_model(config.sie_id)

    # Second call observed the cache populated and skipped the compile.
    assert len(calls) == 1
    assert len(proc._grammar_cache) == 1
