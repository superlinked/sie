"""Worker-side wire types for structured-output grammar specs.

The gateway serialises one ``GrammarSpec`` per ``/v1/generate`` or
``/v1/chat/completions`` request and forwards it to the worker over the
JetStream work envelope. The worker deserialises into the matching
:class:`GrammarSpec` and consults the per-process LRU
(:mod:`sie_server.processors.grammar_cache`) before compiling an
Outlines logits processor.

The on-the-wire shape mirrors the Rust ``GrammarSpec`` enum in
``packages/sie_gateway/src/queue/publisher.rs`` (a serde ``#[serde(tag =
"kind")]`` map). Validating the wire shape is the caller's
responsibility — :class:`GrammarSpec` is a plain dataclass, not a
parser.

Hashing
-------
:func:`hash_grammar` produces a stable, short fingerprint of the schema
payload used as the schema component of the cache key. The full cache
key (``(tokenizer_hash, schema_hash, backend)``) is assembled in
:class:`~sie_server.processors.streaming.StreamingProcessor`; this
module owns the ``schema_hash`` portion only so changes to the schema
representation don't ripple into the cache key shape.

The blake2b digest is truncated to 16 hex characters (~63 bits of
entropy). That is wildly more than enough to avoid collisions among
the ≤64 cache entries any one worker holds at a time, and keeps the
key compact enough for log lines / debug dumps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

# Backend literal kept here (and not in :mod:`grammar_cache`) so callers
# building cache keys don't import the cache module just for the tag.
GrammarKind = Literal["json_schema", "regex", "ebnf"]

OUTLINES_JSON_SCHEMA_TYPE_DIAGNOSTIC = "'type' must be a string"
OUTLINES_JSON_SCHEMA_TYPE_MESSAGE = (
    "JSON Schema 'type' must be a string for this backend. Type arrays are unsupported; "
    "express unions, including null, with 'anyOf' branches."
)


@dataclass(frozen=True)
class GrammarSpec:
    """Structured-output grammar carried from gateway to worker.

    Attributes:
        kind: Discriminator. ``"json_schema"`` means ``value`` is a JSON
            Schema object (``dict``); ``"regex"`` and ``"ebnf"`` mean
            ``value`` is a string (regex pattern or EBNF source).
            Mutual exclusivity is enforced by the gateway before this
            dataclass is built.
        value: The schema or grammar payload itself. Type depends on
            ``kind`` — ``dict`` for ``json_schema``, ``str`` for
            ``regex`` and ``ebnf``. The gateway has already enforced
            the safety caps (payload size, schema depth, regex /
            EBNF source length) so the worker can compile without
            re-validating.
        label: Optional human-readable name surfaced from the OpenAI
            ``response_format.json_schema.name`` field. Used for log
            lines and metric labels; never affects the compile result.
        strict: Native ``grammar.strict`` or OpenAI
            ``response_format.json_schema.strict``. ``True`` makes the worker
            verify the finished output of a ``json_schema`` or ``regex``
            grammar (see :mod:`sie_server.processors.strict_grammar`) and
            fail the request with ``MODEL_OUTPUT_PARSE_ERROR`` when it does
            not conform. Ingress rejects ``True`` on ``ebnf`` grammars.
    """

    kind: GrammarKind
    value: dict | str
    label: str | None = None
    strict: bool | None = None


class GrammarValidationError(Exception):
    """Raised by the worker when a grammar fails to compile or violates
    a worker-side invariant the gateway could not catch.

    Carries the wire-stable ``code`` and the offending ``param`` path so
    the chunk-envelope surface layer can build the standard error
    response without re-classifying the failure.
    """

    def __init__(self, message: str, *, code: str, param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.param = param


def hash_grammar(g: GrammarSpec) -> str:
    """Stable short fingerprint of ``(g.kind, g.value)`` for the cache key.

    The hash deliberately omits ``label`` and ``strict`` — a label change
    does not invalidate a compiled processor, and the LRU shouldn't
    multiply by label cardinality.

    ``kind`` IS folded in: a ``regex`` and an ``ebnf`` (or ``json_schema``)
    can share the same ``value`` string yet compile to entirely different
    processors. Hashing ``value`` alone made e.g. ``regex "[a-z]+"`` and
    ``ebnf "[a-z]+"`` collide, so an ebnf "ready" cache entry would satisfy
    a regex lookup and skip the regex's own Outlines preflight — admitting a
    malformed regex. Prefixing the kind keeps each kind in its own keyspace.

    JSON-schema values are serialised with ``sort_keys=True`` and the
    compact separator so logically equivalent dicts with different key
    orders or whitespace map to the same hash. Regex / EBNF strings hash
    directly.

    Returns a 16-character hex string (64 bits truncated from a
    16-byte blake2b digest).
    """
    if g.kind in ("regex", "ebnf"):
        # ``value`` must be ``str`` for regex and ebnf per the spec.
        # Defensive JSON coercion of dicts here was masking gateway-side
        # type bugs (a non-string slipping through would silently hash a
        # JSON-serialised representation that would never match a real
        # regex/ebnf payload). Fail loudly instead.
        if not isinstance(g.value, str):
            msg = f"GrammarSpec.kind={g.kind!r} requires str value, got {type(g.value).__name__}"
            raise TypeError(msg)
        value_payload = g.value
    else:
        if not isinstance(g.value, dict):
            msg = f"GrammarSpec.kind={g.kind!r} requires dict value, got {type(g.value).__name__}"
            raise TypeError(msg)
        value_payload = json.dumps(g.value, sort_keys=True, separators=(",", ":"))
    # Prefix the kind (with a separator that cannot appear in a kind
    # literal) so distinct kinds occupy distinct keyspaces and a
    # ``value`` shared across kinds can never collide.
    payload = f"{g.kind}\x00{value_payload}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return digest[:16]
