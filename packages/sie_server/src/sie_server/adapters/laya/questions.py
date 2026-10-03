# Portions of this module are adapted from laya 0.3.11 (laya/common.py and
# laya/agent.py, https://github.com/NandhaKishorM/laya), Copyright Convai
# Innovations, licensed under the Apache License, Version 2.0.
#
# Changes from the original: question validation, option rendering, sequence
# assembly, temperature calibration, and answer decoding are split into pure
# functions; request errors raise the server's InvalidInputError; each
# question's token prefix is built once and reused for every state instead of
# re-tokenizing per (state, question) pair; option counts are checked and long
# texts are cut to the UTF-8 bytes a row can use before anything is tokenized;
# answers are left unrounded and omit the act head's escalation probability;
# noul answers add a boolean ``answer``.
"""Laya typed questions: validation, token sequences, and calibrated decoding.

A Laya request pairs one *state* (text, or JSON-serialized dict/list) with
typed questions. Every (state, question) pair becomes one encoder row::

    [CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

The question part (the *prefix*) does not depend on the state, so it is
tokenized once per request and shared by every state. Each ``[MASK]`` marks one
answer option; the decision head scores the hidden state at each marker and a
softmax over a question's markers gives the answer distribution.

A row keeps only the first (or, for conversation lists, the last) few hundred
tokens of each text, so texts are cut to a budget of UTF-8 bytes per token the
row can use before they are tokenized (``STATE_BYTES_PER_TOKEN`` for states,
``QUESTION_BYTES_PER_TOKEN`` for instructions and options); texts within that
budget are tokenized whole, exactly like the reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np

from sie_server.types.inputs import InvalidInputError

QTYPES: dict[str, int] = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES: dict[int, str] = {v: k for k, v in QTYPES.items()}

# Per-option token cap and the head-budget floors used by the reference builder.
_OPTION_MAX_TOKENS = 48
_MIN_OPTION_BUDGET = 16
_MIN_OPTION_TOKENS = 4
_MIN_HEADER_TOKENS = 8

# UTF-8 bytes kept per token a row can use when a text is cut before tokenizing.
# These tokenizers emit at most one token per byte, so the cut bounds tokenizer
# work in any script. Text spends about 1-8 bytes per token (English about 5;
# CJK, Indic, and Arabic 2-8; emoji and rare CJK down to 1), so a cut at twice
# that or more past the last token the row keeps leaves those tokens as the
# whole text would produce them. Only a text made of very long tokens can lose
# tokens to the cut: ModernBERT's vocabulary merges runs of one character
# (spaces, "-", "=") into tokens of up to 512 characters. A state is tokenized
# once per item; instructions and options, up to 64 questions and 1024 options
# per request, get the tighter budget.
STATE_BYTES_PER_TOKEN = 32
QUESTION_BYTES_PER_TOKEN = 16

# Caller data quoted in an error message is cut to this many characters.
_SHOWN_CHARS = 64

# Calibration temperatures outside this range are clamped (laya 0.3.11): a fitted
# temperature below 1 sharpens rather than softens, and the shipped ``choice:11+``
# bucket (0.1006) would turn a 0.24 top probability into 0.99.
TEMP_MIN = 0.5
TEMP_MAX = 5.0

_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}
_NOUL_LABELS_ERROR = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"


@dataclass(frozen=True, slots=True)
class Question:
    """A validated, normalized question (the reference's internal form).

    Attributes:
        qid: Question id (the key in the request's question mapping).
        qtype: ``"choice"``, ``"score"``, or ``"noul"``.
        instructions: Instruction text (non-string instructions JSON-encoded).
        criteria: ``dict`` of label -> description for choice, ``list`` of
            level descriptions for score, optional ``dict`` for noul.
        labels: Optional noul label wording ``{"false": ..., "true": ...}``.
    """

    qid: str
    qtype: str
    instructions: str
    criteria: Any
    labels: Any = None


@dataclass(frozen=True, slots=True)
class QuestionPrefix:
    """Token ids of one question's state-independent row prefix.

    Attributes:
        question: The question.
        ids: ``[CLS] header [SEP] [MASK] opt0 ... [SEP]`` token ids.
        markers: Positions of the option ``[MASK]`` tokens within ``ids``.
    """

    question: Question
    ids: list[int]
    markers: list[int]


def clip_text(text: str, limit: int, *, from_end: bool = False) -> str:
    """``text``, or the whole characters within its first (``from_end``: last) ``limit`` UTF-8 bytes."""
    if len(text) * 4 <= limit:  # fits even at four bytes per character
        return text
    if len(text) > limit:  # the cut falls within ``limit`` characters: every character is at least one byte
        text = text[len(text) - limit :] if from_end else text[:limit]
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= limit:
        return text  # the whole text, or ``limit`` one-byte characters
    data = data[len(data) - limit :] if from_end else data[:limit]
    for trim in range(4):  # drop the bytes of a character the cut split
        piece = data[trim:] if from_end else data[: len(data) - trim]
        try:
            return piece.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            continue
    return ""


def shown(value: Any) -> str:
    """Caller data for an error message: the repr of a string or scalar, cut to ``_SHOWN_CHARS``."""
    if isinstance(value, str):
        text = repr(value[: _SHOWN_CHARS + 1])
    elif value is None or isinstance(value, (bool, int, float)):
        try:
            text = repr(value)
        except ValueError:  # an int past str()'s digit limit
            return f"<{type(value).__name__}>"
    else:
        return f"<{type(value).__name__}>"
    return text if len(text) <= _SHOWN_CHARS + 2 else text[:_SHOWN_CHARS] + "..."


def _shown_keys(mapping: Mapping[Any, Any], count: int = 3) -> str:
    """The first ``count`` keys of ``mapping`` for an error message, noting how many there are."""
    keys = ", ".join(shown(key) for key in islice(mapping, count))
    return f"[{keys}]" if len(mapping) <= count else f"[{keys}, ...] ({len(mapping)} keys)"


def _noul_key(key: Any) -> str | None:
    """A noul criteria key as the reference reads it (``str(key).lower()``), without rendering long keys."""
    if isinstance(key, str):
        return key.lower() if len(key) <= len("false") else None
    if key is None or isinstance(key, (bool, int, float)):
        return str(key).lower()
    return None


def serialize_state(
    state: str | Mapping[str, Any] | Sequence[Any], limit: int | None = None, *, from_end: bool = False
) -> str:
    """Render a state as the text the encoder reads (JSON for structured states).

    With ``limit``, returns only the whole characters within the first
    (``from_end``: last) ``limit`` UTF-8 bytes of that text, exactly as they
    appear in the full rendering, without rendering the rest.
    """
    if isinstance(state, str):
        return state if limit is None else clip_text(state, limit, from_end=from_end)
    if limit is None:
        return json.dumps(state, ensure_ascii=False)
    return _bounded_json(state, limit, from_end=from_end, default_str=False)


def render_criterion(value: Any, limit: int | None = None) -> str:
    """Render one criterion value: strings pass through, anything else becomes compact JSON.

    With ``limit``, structured values render only their first ``limit`` UTF-8 bytes.
    """
    if isinstance(value, str):
        return value
    if limit is None:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
    return _bounded_json(value, limit, from_end=False, default_str=True)


class _Raw(str):
    """JSON punctuation or an encoded key, emitted as is (not a value to encode)."""

    __slots__ = ()


_END = object()


def _bounded_json(value: Any, limit: int, *, from_end: bool, default_str: bool) -> str:
    """The first (``from_end``: last) ``limit`` UTF-8 bytes of ``json.dumps(value, ensure_ascii=False)``.

    Walks ``value`` lazily from the chosen end and stops once ``limit``
    characters (so at least ``limit`` bytes) are out, so the work is bounded by
    ``limit`` rather than by the size of ``value``. ``default_str`` encodes unsupported values as ``str()``,
    like ``json.dumps(default=str)``; otherwise they raise TypeError, as do keys
    JSON cannot encode.
    """
    parts: list[str] = []
    size = 0
    for chunk in _json_chunks(value, limit, from_end=from_end, default_str=default_str):
        parts.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    if from_end:
        parts.reverse()
    return clip_text("".join(parts), limit, from_end=from_end)


def _json_chunks(value: Any, limit: int, *, from_end: bool, default_str: bool) -> Iterator[str]:
    """``json.dumps(value, ensure_ascii=False)`` in chunks, front to back (``from_end``: back to front).

    A string longer than ``limit`` is emitted as only its ``limit`` characters
    on the reading side, which alone fills the budget.
    """
    stack: list[Iterator[Any]] = [iter((value,))]
    while stack:
        node = next(stack[-1], _END)
        if node is _END:
            stack.pop()
        elif isinstance(node, _Raw):
            yield node
        elif isinstance(node, str):
            yield _json_string(node, limit, from_end=from_end)
        elif node is None or isinstance(node, (bool, int, float)):
            yield json.dumps(node)
        elif isinstance(node, dict):
            stack.append(_dict_chunks(node, limit, from_end=from_end))
        elif isinstance(node, (list, tuple)):
            stack.append(_list_chunks(node, from_end=from_end))
        elif default_str:
            yield _json_string(str(node), limit, from_end=from_end)
        else:
            msg = f"Object of type {type(node).__name__} is not JSON serializable"
            raise TypeError(msg)


def _json_string(text: str, limit: int, *, from_end: bool) -> str:
    """A JSON string literal; past ``limit`` characters, only the cut side's quote-less part."""
    if len(text) <= limit:
        return json.dumps(text, ensure_ascii=False)
    if from_end:
        return json.dumps(text[len(text) - limit :], ensure_ascii=False)[1:]
    return json.dumps(text[:limit], ensure_ascii=False)[:-1]


def _json_key(key: Any, limit: int, *, from_end: bool) -> _Raw:
    """A dict key as JSON writes it (str, or a number, boolean, or null converted to a string)."""
    if not isinstance(key, str):
        if key is not None and not isinstance(key, (bool, int, float)):
            msg = f"keys must be str, int, float, bool or None, not {type(key).__name__}"
            raise TypeError(msg)
        key = json.dumps(key)
    return _Raw(_json_string(key, limit, from_end=from_end))


def _dict_chunks(mapping: dict[Any, Any], limit: int, *, from_end: bool) -> Iterator[Any]:
    yield _Raw("}" if from_end else "{")
    items = reversed(mapping.items()) if from_end else iter(mapping.items())
    for n, (key, item) in enumerate(items):
        if n:
            yield _Raw(", ")
        if from_end:
            yield item
            yield _Raw(": ")
            yield _json_key(key, limit, from_end=True)
        else:
            yield _json_key(key, limit, from_end=False)
            yield _Raw(": ")
            yield item
    yield _Raw("{" if from_end else "}")


def _list_chunks(values: Sequence[Any], *, from_end: bool) -> Iterator[Any]:
    yield _Raw("]" if from_end else "[")
    for n, item in enumerate(reversed(values) if from_end else values):
        if n:
            yield _Raw(", ")
        yield item
    yield _Raw("[" if from_end else "]")


def resolve_noul_labels(labels: Any = None) -> tuple[str, str]:
    """Return the (false, true) option wording for a noul question."""
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, Mapping) or len(labels) != 2 or set(labels) != {"false", "true"}:
        raise InvalidInputError(_NOUL_LABELS_ERROR)
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise InvalidInputError(_NOUL_LABELS_ERROR)
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise InvalidInputError(_NOUL_LABELS_ERROR)
    return false_label, true_label


def check_question(qid: str, qdef: Any) -> None:
    """Reject a question that cannot be answered, naming it and what to fix."""
    if not isinstance(qdef, Mapping):
        raise InvalidInputError(f"question {shown(qid)}: definition must be a dict, got {type(qdef).__name__}")
    t = qdef.get("type")
    if not isinstance(t, str) or t not in QTYPES:
        raise InvalidInputError(f"question {shown(qid)}: unknown type {shown(t)}; use one of {sorted(QTYPES)}")
    if "instructions" not in qdef:
        raise InvalidInputError(f"question {shown(qid)}: no 'instructions'; add the text the model should answer")
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (Mapping, list)):
            raise InvalidInputError(
                f"question {shown(qid)}: a choice question takes 'criteria' as a dict of "
                "label -> description, or a list of labels"
            )
        if not crit:
            raise InvalidInputError(f"question {shown(qid)}: a choice question needs at least one criterion")
        if isinstance(crit, list) and any(not isinstance(label, str) for label in crit):
            # The reference fails later with a TypeError deep in tokenization.
            raise InvalidInputError(f"question {shown(qid)}: a choice question's list criteria must be strings")
        if isinstance(crit, Mapping) and any(not isinstance(label, str) for label in crit):
            # Possible from msgpack; the label is the option text the model reads.
            raise InvalidInputError(f"question {shown(qid)}: a choice question's criteria labels must be strings")
    elif t == "score":
        if not isinstance(crit, list):
            raise InvalidInputError(
                f"question {shown(qid)}: a score question takes 'criteria' as a list of level descriptions, index 0 first"
            )
        if not crit:
            raise InvalidInputError(f"question {shown(qid)}: a score question needs at least one level")
    elif crit is not None and not isinstance(crit, Mapping):
        raise InvalidInputError(
            f"question {shown(qid)}: a noul question takes 'criteria' as a dict with optional "
            "'true'/'false' descriptions, or omits it"
        )
    elif isinstance(crit, Mapping):
        # At most two keys, checked before any key is read.
        if len(crit) > 2 or any(_noul_key(k) not in ("true", "false") for k in crit):
            raise InvalidInputError(
                f"question {shown(qid)}: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                f"or both, and omitted is fine), got {_shown_keys(crit)}. Those keys are the option texts the model "
                "reads; any other key was silently dropped and replaced with the defaults. If you "
                "want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
                "'labels' instead."
            )
    if "labels" in qdef:
        if t != "noul":
            raise InvalidInputError(f"question {shown(qid)}: 'labels' is only supported for noul questions")
        try:
            resolve_noul_labels(qdef["labels"])
        except InvalidInputError as e:
            raise InvalidInputError(f"question {shown(qid)}: {e}") from e


def normalize_question(qid: str, qdef: Mapping[str, Any]) -> Question:
    """Convert a validated question definition to its internal form."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = dict.fromkeys(crit)
    elif t == "choice" and isinstance(crit, Mapping):
        crit = dict(crit.items())
    elif t == "noul" and isinstance(crit, Mapping):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        try:
            ins = json.dumps(ins, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise InvalidInputError(
                f"question {shown(qid)}: 'instructions' must be a string or JSON-serializable"
            ) from e
    return Question(qid=qid, qtype=t, instructions=ins, criteria=crit, labels=qdef.get("labels"))


def parse_questions(questions: Mapping[str, Any]) -> list[Question]:
    """Validate and normalize a question mapping, preserving its order."""
    if not isinstance(questions, Mapping):
        raise InvalidInputError("Laya questions (output_schema) must be a mapping of question id -> definition")
    parsed: list[Question] = []
    for qid, qdef in questions.items():
        check_question(qid, qdef)
        parsed.append(normalize_question(qid, qdef))
    return parsed


def option_count(q: Question) -> int:
    """Number of answer options (markers) a question renders."""
    return 2 if q.qtype == "noul" else len(q.criteria)


def check_options_fit(q: Question, *, max_len: int, head_max_len: int) -> None:
    """Reject a question whose options can never all be marked within ``max_len``.

    ``[CLS] header [SEP]`` fills at least two positions and each option at least
    its ``[MASK]``, so more than ``max_len - 2`` options always overflow. This
    runs before anything is tokenized; the reference reports the same error
    after tokenizing every option.
    """
    if option_count(q) > max_len - 2:
        raise InvalidInputError(_options_overflow(q, head_max_len))


def _options_overflow(q: Question, head_max_len: int) -> str:
    return f"question {shown(q.qid)} options exceed head_max_len={head_max_len}"


def render_options(q: Question, *, limit: int | None = None) -> list[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true].

    With ``limit``, structured criteria render only their first ``limit`` characters.

    Raises:
        InvalidInputError: If a criterion has keys JSON cannot encode (possible from msgpack).
    """
    try:
        return _render_options(q, limit)
    except TypeError as e:
        raise InvalidInputError(f"question {shown(q.qid)}: criteria must be JSON-serializable ({e})") from e


def _render_options(q: Question, limit: int | None) -> list[str]:
    crit = q.criteria
    if q.qtype == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [k if v is None or v == "" else f"{k}: {render_criterion(v, limit)}" for k, v in crit.items()]
    if q.qtype == "score":
        return [f"level {i}: {render_criterion(c, limit)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = resolve_noul_labels(q.labels)
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label
        + ": "
        + (render_criterion(false_crit, limit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label
        + ": "
        + (render_criterion(true_crit, limit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_prefixes(
    tokenizer: Any, questions: Sequence[Question], *, max_len: int, head_max_len: int
) -> list[QuestionPrefix]:
    """Tokenize every question into its state-independent row prefix.

    Option counts are checked for every question before anything is tokenized.
    Headers and options are then tokenized in one batch each, cut to the UTF-8
    bytes a prefix can use (``QUESTION_BYTES_PER_TOKEN`` per token).

    Raises:
        InvalidInputError: If a question's options do not fit so that every
            marker lands inside ``max_len`` (the reference's "options exceed
            head_max_len" error), or its criteria cannot be rendered.
    """
    for q in questions:
        check_options_fit(q, max_len=max_len, head_max_len=head_max_len)
    if not questions:
        return []
    mask_tok = tokenizer.mask_token
    option_bytes = _OPTION_MAX_TOKENS * QUESTION_BYTES_PER_TOKEN
    header_bytes = max(_MIN_HEADER_TOKENS, head_max_len) * QUESTION_BYTES_PER_TOKEN
    options = [render_options(q, limit=option_bytes) for q in questions]
    headers = [
        f"{q.qtype} question: {clip_text(str(q.instructions), header_bytes).replace(mask_tok, ' ')}" for q in questions
    ]
    option_texts = [" " + clip_text(opt, option_bytes).replace(mask_tok, " ") for opts in options for opt in opts]
    header_ids = tokenizer(headers, add_special_tokens=False)["input_ids"]
    option_ids = iter(tokenizer(option_texts, add_special_tokens=False)["input_ids"])
    return [
        _assemble_prefix(
            tokenizer,
            q,
            head_ids,
            [next(option_ids) for _ in opts],
            max_len=max_len,
            head_max_len=head_max_len,
        )
        for q, head_ids, opts in zip(questions, header_ids, options, strict=True)
    ]


def build_prefix(tokenizer: Any, q: Question, *, max_len: int, head_max_len: int) -> QuestionPrefix:
    """Tokenize one question into its state-independent row prefix (see :func:`build_prefixes`)."""
    return build_prefixes(tokenizer, [q], max_len=max_len, head_max_len=head_max_len)[0]


def _assemble_prefix(
    tokenizer: Any,
    q: Question,
    head_ids: Sequence[int],
    option_ids: Sequence[Sequence[int]],
    *,
    max_len: int,
    head_max_len: int,
) -> QuestionPrefix:
    """Fit a question's header and option tokens into the head budget, as the reference does."""
    opt_ids = [[tokenizer.mask_token_id, *ids[:_OPTION_MAX_TOKENS]] for ids in option_ids]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < _MIN_OPTION_BUDGET:
        per = max(_MIN_OPTION_TOKENS, (head_max_len - _MIN_OPTION_BUDGET) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    ids = [tokenizer.cls_token_id, *head_ids[: max(_MIN_HEADER_TOKENS, opt_budget)], tokenizer.sep_token_id]
    markers: list[int] = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tokenizer.sep_token_id)
    if any(m >= max_len for m in markers):
        raise InvalidInputError(_options_overflow(q, head_max_len))
    return QuestionPrefix(question=q, ids=ids, markers=markers)


def assemble_row(
    prefix: QuestionPrefix,
    state_ids: Sequence[int],
    *,
    sep_token_id: int,
    max_len: int,
    truncate_left: bool,
) -> list[int]:
    """Join a question prefix and a tokenized state into one row, truncating the state to fit."""
    room = max(0, max_len - len(prefix.ids) - 1)
    # not state_ids[-room:]: with no room left, [-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    return [*prefix.ids, *st, sep_token_id][:max_len]


# ---------------------------------------------------------------------------
# Calibration and decoding
# ---------------------------------------------------------------------------


def clamp_temperature(t: Any, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: ``t`` confined to [lo, hi], falling back to 1.0 if it is not a finite number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(hi, max(lo, t))


def temp_bucket(qtype: str, k: int) -> str:
    """Calibration bucket key ``<type>:<2|3-5|6-10|11+>`` for a question with ``k`` options."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


@dataclass(frozen=True, slots=True)
class Calibration:
    """Clamped per-type and per-bucket temperatures from ``rl_agent_config.json``."""

    temperature: tuple[float, float, float]
    temperature_by_options: dict[str, float]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Calibration:
        raw = cfg.get("temperature", [1.0, 1.0, 1.0])
        if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != len(QTYPES):
            raw = [1.0, 1.0, 1.0]
        by_options = cfg.get("temperature_by_options") or {}
        if not isinstance(by_options, Mapping):
            by_options = {}
        temps = tuple(clamp_temperature(t) for t in raw)
        return cls(
            temperature=(temps[0], temps[1], temps[2]),
            temperature_by_options={str(k): clamp_temperature(v) for k, v in by_options.items()},
        )

    def temperature_for(self, qtype: str, k: int) -> float:
        return self.temperature_by_options.get(temp_bucket(qtype, k), self.temperature[QTYPES[qtype]])


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def calibrated_probs(logits: np.ndarray, qtype: str, calibration: Calibration) -> np.ndarray:
    """Temperature-scaled softmax over one question's option logits (float32, like the reference)."""
    k = len(logits)
    z = np.asarray(logits, dtype=np.float32) / calibration.temperature_for(qtype, k)
    p = np.exp(z - z.max())
    return p / p.sum()


def decode_answer(q: Question, logits: np.ndarray, calibration: Calibration) -> dict[str, Any]:
    """Turn one question's marker logits into its typed answer."""
    k = len(logits)
    p = calibrated_probs(logits, q.qtype, calibration)
    confidence = confidence_from_probs(p, k)
    if q.qtype == "choice":
        keys = list(q.criteria.keys())
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {key: float(v) for key, v in zip(keys, p, strict=True)},
            "confidence": confidence,
        }
    if q.qtype == "score":
        return {
            "type": "score",
            "score": float((np.arange(k) * p).sum()),
            "legend": {str(i): c for i, c in enumerate(q.criteria)},
            "probabilities": {str(i): float(v) for i, v in enumerate(p)},
            "confidence": confidence,
        }
    p_true = float(p[1])
    return {
        "type": "noul",
        "noul": p_true,
        "answer": bool(p[1] > p[0]),
        "confidence": max(p_true, 1.0 - p_true),
    }
