"""RFC 8785 JSON Canonicalization Scheme, and digests over canonical content.

Why this file exists
--------------------
Evidence is verified across two repositories and two languages: the run is
recorded here in Python, and the website checks the numbers it publishes in
TypeScript. A digest over *file bytes* would report a mismatch whenever the two
copies differ only in formatting, which they do -- one is formatted with tabs,
the other with two spaces. So the digest has to be over canonical *content*.

The obvious shortcut, "sorted keys and compact separators", does not survive
contact with two languages:

    python  json.dumps(...)   -> {"score":1.0}
    node    JSON.stringify(...) -> {"score":1}

Identical content, different bytes, different SHA-256. This is not a corner
case: a top-ranked similarity score is exactly ``1.0``, so the first thing
anyone binds would fail on data that is perfectly correct.

RFC 8785 removes the ambiguity by specifying number serialization as the
ECMAScript ``Number::toString`` algorithm, sorting object keys by UTF-16 code
unit, and fixing string escaping. ``tests/fixtures/canonical-vector.json``
pins a digest that any independent implementation must reproduce.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any

__all__ = ["canonical_json", "canonical_sha256", "es_number_to_string"]

_MAX_EXACT_INT = 2**53

_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


def es_number_to_string(value: float) -> str:
    """Serialize a number exactly as ECMAScript ``Number::toString`` would.

    RFC 8785 section 3.2.2.3. Python's ``repr`` already yields the shortest
    round-tripping decimal, which is the same digit string ECMAScript picks;
    only the *placement* of the decimal point and the exponent form differ, so
    this reformats those digits according to the ECMAScript rules.
    """
    if isinstance(value, bool):  # bool is a subclass of int; never a JSON number
        raise TypeError("bool is not a JSON number")
    if isinstance(value, int):
        # Exactness, not magnitude. 2**53 and 2**54 are powers of two and survive
        # binary64 unharmed; 2**53 + 1 does not. Rejecting everything at or above
        # 2**53 refused values that canonicalize perfectly well, so ask the only
        # question that matters: does the round trip come back unchanged?
        as_double = float(value)
        if int(as_double) != value:
            raise ValueError(f"integer {value} is not exactly representable as an IEEE 754 double")
        return es_number_to_string(as_double)
    if not math.isfinite(value):
        raise ValueError("NaN and Infinity have no JSON representation")
    if value == 0:
        return "0"  # collapses -0.0, as ECMAScript does

    sign = "-" if value < 0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    digits = list(digit_tuple)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    text = "".join(str(digit) for digit in digits)
    k = len(text)
    n = k + int(exponent)

    if k <= n <= 21:
        return f"{sign}{text}{'0' * (n - k)}"
    if 0 < n <= 21:
        return f"{sign}{text[:n]}.{text[n:]}"
    if -6 < n <= 0:
        return f"{sign}0.{'0' * -n}{text}"
    power = n - 1
    mantissa = text if k == 1 else f"{text[0]}.{text[1:]}"
    return f"{sign}{mantissa}e{'+' if power >= 0 else '-'}{abs(power)}"


def _escape(text: str) -> str:
    out = ['"']
    for character in text:
        code = ord(character)
        escape = _SHORT_ESCAPES.get(code)
        if escape is not None:
            out.append(escape)
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(character)  # every other code point stays literal UTF-8
    out.append('"')
    return "".join(out)


def _sort_key(key: str) -> bytes:
    """Sort by UTF-16 code unit, as RFC 8785 requires.

    Python compares strings by code point, which diverges from UTF-16 order
    above the BMP. Encoding to UTF-16 big-endian and comparing bytes gives the
    order the specification asks for.
    """
    return key.encode("utf-16-be")


def _serialize(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif isinstance(value, bool):
        out.append("true" if value else "false")
    elif isinstance(value, (int, float)):
        out.append(es_number_to_string(value))
    elif isinstance(value, str):
        out.append(_escape(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _serialize(item, out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"object keys must be strings, got {type(key).__name__}")
        out.append("{")
        for index, key in enumerate(sorted(value, key=_sort_key)):
            if index:
                out.append(",")
            out.append(_escape(key))
            out.append(":")
            _serialize(value[key], out)
        out.append("}")
    else:
        raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 encoding of ``value``."""
    out: list[str] = []
    _serialize(value, out)
    return "".join(out).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """SHA-256 over the canonical encoding, so formatting never changes it."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_canonical_sha256(path: Any) -> str:
    """Digest the *content* of a JSON file, ignoring how it is formatted."""
    with open(path, "r", encoding="utf-8") as handle:
        return canonical_sha256(json.load(handle))
