"""Outlines compile wrapper for grammar validation.

The worker compiles the grammar once per ``(tokenizer, schema)`` pair
before forwarding the raw schema/regex to SGLang's ``/generate``
endpoint. SGLang itself runs the Outlines backend internally when
launched with ``--grammar-backend outlines``; the worker-side compile
serves two roles:

1. **Fast failure**: surface ``grammar_compile_failed`` as a worker-side
   chunk envelope before publishing inference work to SGLang. SGLang's
   own compile errors arrive as HTTP 500s and are harder to attribute.
2. **Cache key**: validation must succeed before the cache is
   populated, so a hit confirms the schema is well-formed for the
   tokenizer that hashed into the key.

The compile result is stored as a sentinel (``True``) — the gateway-to-SGLang
wire shape carries the **raw** schema/regex, not a compiled processor,
because SGLang owns its own per-engine cache. If a future backend
needs a pre-compiled processor object the wrapper can return it
instead; the cache value is opaque to :mod:`grammar_cache`.

Outlines API surface
--------------------
``outlines-core`` (preloaded in ``packages/sie_server/bundles/sglang.yaml``)
exposes its public API differently across versions. The wrapper tries
the modern ``outlines.processors`` entry point first and falls back to
older ``outlines_core.fsm`` names if that fails — both are documented to
build a regex-driven FSM against a tokenizer. The exact symbol used is
abstracted behind :func:`_resolve_outlines_factories` so future versions
can override without touching :class:`StreamingProcessor`.

Tokenizer adapter
-----------------
``JSONLogitsProcessor`` / ``RegexLogitsProcessor`` do **not** accept a raw
HuggingFace tokenizer — they expect an Outlines ``Tokenizer`` adapter
(``outlines.models.tokenizer.Tokenizer``) that exposes ``.vocabulary``.
The FSM build (``RegexGuide.from_regex``) reads ``tokenizer.vocabulary``
directly. A raw transformers tokenizer has no such attribute; under
``transformers>=5`` the fast tokenizer is a ``TokenizersBackend`` and the
unadapted access fails with ``'TokenizersBackend' object has no attribute
'vocabulary'`` — the historical reason Qwen3.5 was pinned to XGrammar.
:func:`_adapt_tokenizer` wraps the raw tokenizer in Outlines'
``TransformerTokenizer`` (which fills ``.vocabulary`` from ``get_vocab()``,
still present in transformers v5) before handing it to the factories —
mirroring exactly what SGLang's own Outlines backend does internally
(``OutlinesGrammarBackend`` constructs ``TransformerTokenizer(tokenizer)``).
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from sie_server.types.grammar import (
    OUTLINES_JSON_SCHEMA_TYPE_DIAGNOSTIC,
    OUTLINES_JSON_SCHEMA_TYPE_MESSAGE,
    GrammarSpec,
    GrammarValidationError,
)

logger = logging.getLogger(__name__)


def _resolve_outlines_factories() -> tuple[Any, Any]:
    """Return ``(json_factory, regex_factory)`` from the bundled Outlines.

    Each factory is a callable ``(payload, tokenizer) -> Any`` that
    compiles the grammar and raises on failure. The returned object is
    discarded for the SGLang path (kept only as a sentinel) so the
    factory's exact return shape is unimportant.

    The import is dynamic — :mod:`outlines.processors` lives in the
    SGLang worker bundle (``packages/sie_server/bundles/sglang.yaml``)
    and is not necessarily installed in the dev environment. A static
    ``import outlines.processors`` would trip the type checker; the
    :func:`importlib.import_module` indirection keeps the runtime
    contract while side-stepping that.

    Raises:
        ImportError: when Outlines is missing from the worker bundle.
            Grammar callers convert this to a startup failure rather
            than letting the first grammar request observe a confusing
            ImportError.
    """
    try:
        module = importlib.import_module("outlines.processors")
    except ImportError as exc:  # pragma: no cover — environment-specific
        msg = (
            "outlines.processors is not available on this worker — install "
            "`outlines-core` (declared in bundles/sglang.yaml). "
            f"original error: {exc}"
        )
        raise ImportError(msg) from exc
    try:
        json_factory = module.JSONLogitsProcessor
        regex_factory = module.RegexLogitsProcessor
    except AttributeError as exc:  # pragma: no cover — version skew
        msg = (
            "outlines.processors does not expose the expected factory "
            "symbols (JSONLogitsProcessor / RegexLogitsProcessor). The "
            "bundled outlines-core version may have moved its API; "
            "update _resolve_outlines_factories."
        )
        raise ImportError(msg) from exc
    return json_factory, regex_factory


# Module-level cache of the resolved factories so we hit the import
# system once. ``None`` until first use; populated lazily so non-grammar
# workloads (encode/score/extract) never pay the import cost.
_FACTORIES: tuple[Any, Any] | None = None


def _get_factories() -> tuple[Any, Any]:
    global _FACTORIES
    if _FACTORIES is None:
        _FACTORIES = _resolve_outlines_factories()
    return _FACTORIES


def _resolve_tokenizer_adapter() -> Any:
    """Return Outlines' ``TransformerTokenizer`` adapter class.

    The processor factories require an Outlines ``Tokenizer`` (it exposes
    ``.vocabulary``); a raw transformers tokenizer does not. This adapter
    wraps a HuggingFace tokenizer into that interface — the same wrap
    SGLang's ``OutlinesGrammarBackend`` performs before building its guide.
    Imported dynamically for the same reason as the factories: Outlines is
    a worker-bundle-only dependency (see module docstring).

    Raises:
        ImportError: when Outlines is missing or has moved the symbol.
    """
    try:
        module = importlib.import_module("outlines.models.transformers")
    except ImportError as exc:  # pragma: no cover — environment-specific
        msg = (
            "outlines.models.transformers is not available on this worker — "
            "install `outlines` (declared in bundles/sglang.yaml). "
            f"original error: {exc}"
        )
        raise ImportError(msg) from exc
    try:
        return module.TransformerTokenizer
    except AttributeError as exc:  # pragma: no cover — version skew
        msg = (
            "outlines.models.transformers does not expose TransformerTokenizer; "
            "the bundled Outlines version may have moved its API — update "
            "_resolve_tokenizer_adapter."
        )
        raise ImportError(msg) from exc


# Lazily-resolved adapter class, cached like ``_FACTORIES``.
_TOKENIZER_ADAPTER: Any | None = None


def _get_tokenizer_adapter() -> Any:
    global _TOKENIZER_ADAPTER
    if _TOKENIZER_ADAPTER is None:
        _TOKENIZER_ADAPTER = _resolve_tokenizer_adapter()
    return _TOKENIZER_ADAPTER


def _adapt_tokenizer(tokenizer: Any) -> Any:
    """Return an Outlines-compatible tokenizer for the processor factories.

    Pass-through when ``tokenizer`` already exposes ``.vocabulary`` (i.e. it
    is already an Outlines adapter); otherwise wrap it in
    ``TransformerTokenizer``. The wrap is what makes the FSM build work on
    ``transformers>=5`` (``TokenizersBackend`` has no ``.vocabulary``).
    """
    if hasattr(tokenizer, "vocabulary"):
        return tokenizer
    adapter_cls = _get_tokenizer_adapter()
    return adapter_cls(tokenizer)


def compile_outlines(tokenizer: Any, grammar: GrammarSpec) -> Any:
    """Validate ``grammar`` against ``tokenizer`` via Outlines.

    Blocking — callers must wrap in :func:`asyncio.to_thread` (see
    :class:`StreamingProcessor`). Raises
    :class:`GrammarValidationError` with a client-safe diagnostic.

    The returned value is a sentinel object that :class:`GrammarLRU`
    stores. Currently ``True`` for the SGLang path; a future
    direct-Outlines path could return the processor itself with no
    cache-shape change.
    """
    # EBNF: skip the worker-side Outlines preflight entirely. The SGLang
    # adapter forwards ``ebnf`` to SGLang's ``/generate`` endpoint on BOTH
    # the Outlines and XGrammar backends (see ``generation.py`` —
    # ``sampling_params["ebnf"]``), and SGLang's own backend compiles it.
    # Outlines' Python ``processors`` surface here exposes only
    # ``JSONLogitsProcessor`` / ``RegexLogitsProcessor`` (no EBNF factory in
    # the bundled version), so attempting a preflight compile would raise
    # ``grammar_invalid`` and reject a request that SGLang would have
    # happily served. Returning the sentinel lets the cache record the
    # (tokenizer, grammar) pair as "ready"; SGLang stays the source of
    # truth for EBNF validity (a genuinely malformed grammar surfaces as an
    # HTTP 500 → ``finish_reason: "error"`` chunk via the adapter).
    if grammar.kind == "ebnf":
        return True
    json_factory, regex_factory = _get_factories()
    # The factories require an Outlines ``Tokenizer`` adapter, not a raw
    # HF tokenizer (see module docstring "Tokenizer adapter"). Wrap once
    # and reuse for both the string and dict JSON retries.
    adapted = _adapt_tokenizer(tokenizer)
    try:
        if grammar.kind == "json_schema":
            # ``JSONLogitsProcessor`` accepts the schema as a JSON
            # string in some Outlines versions and a dict in others;
            # try string first (the more stable form) and fall back to
            # the dict shape on TypeError.
            schema_text = json.dumps(grammar.value, sort_keys=True)
            try:
                _ = json_factory(schema_text, adapted)
            except TypeError:
                _ = json_factory(grammar.value, adapted)
        elif grammar.kind == "regex":
            _ = regex_factory(grammar.value, adapted)
        else:  # pragma: no cover — defended by gateway capability gate
            msg = f"unsupported grammar kind: {grammar.kind!r}"
            raise GrammarValidationError(msg, code="grammar_invalid", param="grammar.kind")
    except GrammarValidationError:
        raise
    except Exception as exc:
        if (
            grammar.kind == "json_schema"
            and isinstance(exc, ValueError)
            and exc.args == (OUTLINES_JSON_SCHEMA_TYPE_DIAGNOSTIC,)
        ):
            raise GrammarValidationError(
                OUTLINES_JSON_SCHEMA_TYPE_MESSAGE, code="invalid_request", param="grammar"
            ) from exc
        raise GrammarValidationError(
            "outlines compile failed",
            code="grammar_compile_failed",
            param="grammar",
        ) from exc
    # Sentinel; see module docstring for the rationale.
    return True
