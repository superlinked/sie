"""Tests for the :func:`compile_outlines` wrapper.

Outlines is bundle-only (``packages/sie_server/bundles/sglang.yaml``)
so the dev environment does not have ``outlines.processors``. These
tests monkey-patch the module's resolved factories instead of
exercising the real Outlines compile.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from sie_server.processors import grammar_compile
from sie_server.types.grammar import OUTLINES_JSON_SCHEMA_TYPE_MESSAGE, GrammarSpec, GrammarValidationError


class _V5Tokenizer:
    """Raw transformers>=5 tokenizer stand-in.

    Has ``get_vocab()`` (still present in transformers v5) but no
    ``.vocabulary`` attribute — exactly the shape that crashed Outlines'
    processor factories before :func:`_adapt_tokenizer` wrapped it.
    """

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1}


class _AdaptedTokenizer:
    """Outlines-adapter stand-in: already exposes ``.vocabulary``."""

    vocabulary: ClassVar[dict[str, int]] = {"a": 0, "b": 1}


class _FakeOutlinesAdapter:
    """Stand-in for Outlines' ``TransformerTokenizer``.

    Wraps a raw tokenizer and fills ``.vocabulary`` from ``get_vocab()`` —
    the same contract the real adapter provides.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.vocabulary = tokenizer.get_vocab()


# Back-compat alias: a "raw" tokenizer for tests that don't care about the
# adapter wrapping (they assert on payloads, not the tokenizer object).
_StubTokenizer = _V5Tokenizer


@pytest.fixture(autouse=True)
def _reset_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level caches between tests so each test can install
    its own stubs without bleeding through.
    """
    monkeypatch.setattr(grammar_compile, "_FACTORIES", None)
    monkeypatch.setattr(grammar_compile, "_TOKENIZER_ADAPTER", None)


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    json_calls: list[Any],
    regex_calls: list[Any],
    json_raise: type[BaseException] | None = None,
    regex_raise: type[BaseException] | None = None,
    adapter_cls: type[Any] = _FakeOutlinesAdapter,
) -> None:
    def _json_factory(payload: Any, tok: Any) -> Any:
        json_calls.append((payload, tok))
        if json_raise is not None:
            raise json_raise("kaboom")
        return object()

    def _regex_factory(payload: Any, tok: Any) -> Any:
        regex_calls.append((payload, tok))
        if regex_raise is not None:
            raise regex_raise("kaboom")
        return object()

    monkeypatch.setattr(
        grammar_compile,
        "_resolve_outlines_factories",
        lambda: (_json_factory, _regex_factory),
    )
    # Outlines is bundle-only; stub the adapter resolver so the wrapping
    # path runs without importing the real ``outlines.models.transformers``.
    monkeypatch.setattr(
        grammar_compile,
        "_resolve_tokenizer_adapter",
        lambda: adapter_cls,
    )


def test_compile_json_schema_returns_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=regex_calls)

    g = GrammarSpec(kind="json_schema", value={"type": "object"})
    out = grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert out is True
    assert len(json_calls) == 1
    payload, _ = json_calls[0]
    # JSON path serialises the dict to a string first (more stable Outlines API).
    assert isinstance(payload, str)
    assert "object" in payload


def test_compile_regex_returns_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=regex_calls)

    g = GrammarSpec(kind="regex", value=r"\d{3}")
    out = grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert out is True
    assert regex_calls == [(r"\d{3}", regex_calls[0][1])]


def test_compile_wraps_outlines_exception_as_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(
        monkeypatch,
        json_calls=json_calls,
        regex_calls=regex_calls,
        json_raise=RuntimeError,
    )

    g = GrammarSpec(kind="json_schema", value={"type": "object"})
    with pytest.raises(GrammarValidationError) as ei:
        grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert ei.value.code == "grammar_compile_failed"
    assert ei.value.param == "grammar"
    assert "outlines compile failed" in str(ei.value)


def test_compile_wraps_regex_factory_exception_as_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regex the Outlines FSM can't build (e.g. an anchored ``^...$``,
    which raises ``interegular.patterns.Unsupported``) must surface as a
    clean ``grammar_compile_failed`` — this preflight is what stops such a
    regex from reaching SGLang and SIGQUIT-crashing the server.
    """
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(
        monkeypatch,
        json_calls=json_calls,
        regex_calls=regex_calls,
        regex_raise=RuntimeError,  # stand-in for interegular.patterns.Unsupported
    )

    g = GrammarSpec(kind="regex", value="^(yes|no)$")
    with pytest.raises(GrammarValidationError) as ei:
        grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert ei.value.code == "grammar_compile_failed"
    assert ei.value.param == "grammar"
    assert "outlines compile failed" in str(ei.value)


def test_compile_ebnf_skips_outlines_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """EBNF must NOT be rejected at the worker-side Outlines preflight.

    SGLang forwards ``ebnf`` to its own backend (Outlines or XGrammar);
    the bundled Outlines ``processors`` surface has no EBNF factory, so a
    preflight compile would wrongly reject the request. The wrapper should
    return the sentinel without touching the json/regex factories — and
    without even resolving them (so a missing Outlines doesn't matter for
    EBNF requests).
    """
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=regex_calls)

    # If factory resolution is attempted for EBNF this raises and fails
    # the test — proving the EBNF path short-circuits before resolving.
    def _boom() -> Any:
        raise AssertionError("factories must not be resolved for EBNF preflight")

    monkeypatch.setattr(grammar_compile, "_resolve_outlines_factories", _boom)

    g = GrammarSpec(kind="ebnf", value='root ::= "yes" | "no"')
    out = grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert out is True
    assert json_calls == []
    assert regex_calls == []


def test_compile_falls_back_to_dict_payload_on_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call (string payload) raises TypeError; the wrapper retries
    with the dict payload. Confirms the dual-API compatibility path is
    actually exercised — the factory's TypeError must not surface as
    a compile failure.
    """
    json_calls: list[Any] = []
    first = True

    def _json_factory(payload: Any, tok: Any) -> Any:
        nonlocal first
        json_calls.append((payload, tok))
        if first:
            first = False
            raise TypeError("schema must be a dict")
        return object()

    monkeypatch.setattr(
        grammar_compile,
        "_resolve_outlines_factories",
        lambda: (_json_factory, lambda *_: object()),
    )
    monkeypatch.setattr(
        grammar_compile,
        "_resolve_tokenizer_adapter",
        lambda: _FakeOutlinesAdapter,
    )

    g = GrammarSpec(kind="json_schema", value={"type": "object"})
    out = grammar_compile.compile_outlines(_StubTokenizer(), g)
    assert out is True
    assert len(json_calls) == 2
    # The second call passes the raw dict, not the serialised string.
    assert json_calls[1][0] == {"type": "object"}
    # Both retries reuse the SAME wrapped tokenizer instance.
    assert json_calls[0][1] is json_calls[1][1]


def test_raw_v5_tokenizer_is_wrapped_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transformers>=5 tokenizer (no ``.vocabulary``) must reach the
    factory wrapped in the Outlines adapter, not raw.

    This is the regression guard for the
    ``'TokenizersBackend' object has no attribute 'vocabulary'`` crash
    that pinned Qwen3.5 to XGrammar.
    """
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=regex_calls)

    raw = _V5Tokenizer()
    assert not hasattr(raw, "vocabulary")  # precondition: the failing shape

    g = GrammarSpec(kind="regex", value=r"\d{3}")
    out = grammar_compile.compile_outlines(raw, g)
    assert out is True
    # The factory received the wrapper, not the raw tokenizer.
    _, tok_arg = regex_calls[0]
    assert isinstance(tok_arg, _FakeOutlinesAdapter)
    assert tok_arg.tokenizer is raw
    assert tok_arg.vocabulary == {"a": 0, "b": 1}


def test_already_adapted_tokenizer_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tokenizer already exposing ``.vocabulary`` is forwarded as-is —
    no double-wrapping, and the adapter resolver is never invoked.
    """
    json_calls: list[Any] = []
    regex_calls: list[Any] = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=regex_calls)

    # If the adapter resolver runs for an already-adapted tokenizer this
    # raises and fails the test — proving the pass-through short-circuit.
    def _boom() -> Any:
        raise AssertionError("adapter must not be resolved for an adapted tokenizer")

    monkeypatch.setattr(grammar_compile, "_resolve_tokenizer_adapter", _boom)

    adapted = _AdaptedTokenizer()
    g = GrammarSpec(kind="regex", value=r"\d{3}")
    out = grammar_compile.compile_outlines(adapted, g)
    assert out is True
    _, tok_arg = regex_calls[0]
    assert tok_arg is adapted


@pytest.mark.parametrize("type_value", [["string", "null"], {}, None, 1, True])
def test_compile_known_nonstring_type_diagnostic_is_client_refusal(monkeypatch, type_value) -> None:
    def reject(*_):
        raise ValueError("'type' must be a string")

    monkeypatch.setattr(grammar_compile, "_FACTORIES", (reject, reject))
    with pytest.raises(GrammarValidationError) as error:
        grammar_compile.compile_outlines(
            _AdaptedTokenizer(), GrammarSpec(kind="json_schema", value={"type": type_value})
        )
    assert error.value.code == "invalid_request"
    assert error.value.param == "grammar"
    assert str(error.value) == OUTLINES_JSON_SCHEMA_TYPE_MESSAGE


@pytest.mark.parametrize("kind", ["json_schema", "regex"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("'type' must be a string secret"),
        ValueError("secret 'type' must be a string"),
        ValueError("secret" * 4096),
        RuntimeError("'type' must be a string"),
    ],
)
def test_compile_unknown_diagnostic_is_sanitized(monkeypatch, kind, failure) -> None:
    def reject(*_):
        raise failure

    monkeypatch.setattr(grammar_compile, "_FACTORIES", (reject, reject))
    grammar = GrammarSpec(kind=kind, value={} if kind == "json_schema" else ".*")
    with pytest.raises(GrammarValidationError) as error:
        grammar_compile.compile_outlines(_AdaptedTokenizer(), grammar)
    assert error.value.code == "grammar_compile_failed"
    assert str(error.value) == "outlines compile failed"


def test_regex_diagnostic_does_not_gain_json_schema_refusal(monkeypatch) -> None:
    def reject(*_):
        raise ValueError("'type' must be a string")

    monkeypatch.setattr(grammar_compile, "_FACTORIES", (reject, reject))
    with pytest.raises(GrammarValidationError) as error:
        grammar_compile.compile_outlines(_AdaptedTokenizer(), GrammarSpec(kind="regex", value=".*"))
    assert error.value.code == "grammar_compile_failed"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": "null"},
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
    ],
)
def test_compile_does_not_reject_or_rewrite_schema_shapes(monkeypatch, schema) -> None:
    json_calls = []
    _install_stubs(monkeypatch, json_calls=json_calls, regex_calls=[])
    assert grammar_compile.compile_outlines(_StubTokenizer(), GrammarSpec(kind="json_schema", value=schema)) is True
    assert json_calls[0][0] == grammar_compile.json.dumps(schema, sort_keys=True)
