"""The canonicalization must agree with an independent implementation.

``CANONICAL_VECTOR_SHA256`` is not a value this module computes; it is the
digest an independent RFC 8785 implementation in JavaScript produced for
``fixtures/canonical-vector.json``. Any consumer in another language -- the
website's TypeScript checks, for one -- must reproduce it before its digests
can be compared with the ones recorded here. Pinning it makes the two
implementations *proven* to agree rather than assumed to.

The vector deliberately contains the values that break the naive
"sorted keys, compact separators" approach: a whole-number float, a negative
zero, and exponents on both sides of the range where ECMAScript switches
notation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_to_markdown.canonical import canonical_json, canonical_sha256, es_number_to_string

VECTOR_PATH = Path(__file__).parent / "fixtures" / "canonical-vector.json"
CANONICAL_VECTOR_SHA256 = "23bd34e6997123ca5be4feadac3defc3c54bd17412b6343b7cc6f698931700b6"


def test_vector_digest_matches_the_independent_implementation() -> None:
    value = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    assert canonical_sha256(value) == CANONICAL_VECTOR_SHA256


def test_formatting_never_changes_the_digest() -> None:
    value = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    reformatted = json.loads(json.dumps(value, indent=4, ensure_ascii=False, sort_keys=True))
    assert canonical_sha256(reformatted) == canonical_sha256(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The case that makes byte-identical output impossible across languages:
        # Python's json.dumps writes 1.0, JavaScript's JSON.stringify writes 1.
        (1.0, "1"),
        (-1.0, "-1"),
        (0.0, "0"),
        (-0.0, "0"),
        (159, "159"),
        (100.0, "100"),
        (2.5, "2.5"),
        (123.456, "123.456"),
        # Recorded similarity scores from neighbouring examples.
        (0.0023956298828125, "0.0023956298828125"),
        (0.9921875, "0.9921875"),
        (-0.008386863945257427, "-0.008386863945257427"),
        # ECMAScript switches to exponent notation outside 1e-6 .. 1e21.
        (1e-7, "1e-7"),
        (1e-6, "0.000001"),
        (1e16, "10000000000000000"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (1e22, "1e+22"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
    ],
)
def test_numbers_serialize_as_ecmascript_does(value: float, expected: str) -> None:
    assert es_number_to_string(value) == expected


def test_keys_sort_by_utf16_code_unit() -> None:
    assert canonical_json({"ä": 1, "Z": 2, "a": 3}) == b'{"Z":2,"a":3,"\xc3\xa4":1}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused(value: float) -> None:
    with pytest.raises(ValueError):
        es_number_to_string(value)


def test_booleans_are_not_numbers() -> None:
    with pytest.raises(TypeError):
        es_number_to_string(True)
