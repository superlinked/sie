"""The settled charge surfaced in a response's ``usage`` block (#2434).

The gateway publishes ``usage.credits_charged`` + ``usage.rate_book_version``
on every settled response, and keeps the ``x-sie-credits-debited`` header
unchanged for compatibility. The SDK reads the body first — it is the only
source that also names the rate book — and falls back to the header for a
gateway that predates in-body surfacing.
"""

from __future__ import annotations

import pytest
from sie_sdk.client._shared import parse_request_metadata
from sie_sdk.client.async_ import _parse_generate_result_async
from sie_sdk.client.sync import _parse_generate_result

BOOK = "2026-07-22-production-bootstrap-v1"

_GENERATE_PARSERS = (_parse_generate_result, _parse_generate_result_async)


@pytest.mark.parametrize("parse", _GENERATE_PARSERS)
def test_generate_usage_block_carries_the_settled_charge(parse) -> None:
    """The parser rebuilds ``usage`` field-by-field; the charge must survive."""
    result = parse(
        {
            "model": "m",
            "text": "hi",
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
                "credits_charged": 7,
                "rate_book_version": BOOK,
            },
        },
        request=None,
    )

    assert result["usage"]["credits_charged"] == 7
    assert result["usage"]["rate_book_version"] == BOOK
    assert result["usage"]["total_tokens"] == 12


@pytest.mark.parametrize("parse", _GENERATE_PARSERS)
def test_generate_usage_block_stays_charge_free_when_nothing_settled(parse) -> None:
    result = parse(
        {"model": "m", "text": "hi", "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}},
        request=None,
    )

    assert "credits_charged" not in result["usage"]
    assert "rate_book_version" not in result["usage"]


def test_body_is_the_source_of_the_settled_charge() -> None:
    metadata = parse_request_metadata(
        {"x-sie-request-id": "req-1", "x-sie-units-input-tokens": "11", "x-sie-credits-debited": "7"},
        {"model": "m", "usage": {"input_tokens": 11, "credits_charged": 7, "rate_book_version": BOOK}},
    )

    assert metadata is not None
    assert metadata["credits_debited"] == 7
    assert metadata["rate_book_version"] == BOOK
    assert metadata["usage"]["credits_charged"] == 7
    assert metadata["usage"]["rate_book_version"] == BOOK
    # The unit headers still describe the same request.
    assert metadata["usage"]["input_tokens"] == 11


def test_header_still_carries_the_charge_when_the_body_does_not() -> None:
    """A gateway that predates in-body surfacing must keep working."""
    metadata = parse_request_metadata(
        {"x-sie-credits-debited": "7"},
        {"model": "m", "usage": {"prompt_tokens": 4}},
    )

    assert metadata is not None
    assert metadata["credits_debited"] == 7
    assert "rate_book_version" not in metadata
    # `usage` here mirrors the meter HEADERS, which this response has none of;
    # the body's own token counts stay on the response envelope.
    assert "usage" not in metadata


def test_no_settlement_means_no_charge_anywhere() -> None:
    """A billing fault surfaces neither a body charge nor a header one."""
    metadata = parse_request_metadata({"x-sie-request-id": "req-fault"}, {"model": "m"})

    assert metadata is not None
    assert "credits_debited" not in metadata
    assert "rate_book_version" not in metadata


def test_a_zero_charge_is_a_charge() -> None:
    metadata = parse_request_metadata({}, {"usage": {"credits_charged": 0, "rate_book_version": BOOK}})

    assert metadata is not None
    assert metadata["credits_debited"] == 0
    assert metadata["usage"]["credits_charged"] == 0


def test_half_a_charge_is_no_charge() -> None:
    """Neither half is publishable alone, so a partial block falls back."""
    for usage in (
        {"credits_charged": 7},
        {"rate_book_version": BOOK},
        {"credits_charged": -1, "rate_book_version": BOOK},
        {"credits_charged": "7", "rate_book_version": BOOK},
        {"credits_charged": True, "rate_book_version": BOOK},
        {"credits_charged": 7, "rate_book_version": ""},
    ):
        metadata = parse_request_metadata({"x-sie-credits-debited": "3"}, {"usage": usage})
        assert metadata is not None, usage
        assert metadata["credits_debited"] == 3, usage
        assert "rate_book_version" not in metadata, usage


def test_a_body_that_is_not_an_envelope_is_ignored() -> None:
    for body in (None, [], "usage", {"usage": 3}):
        metadata = parse_request_metadata({"x-sie-credits-debited": "5"}, body)
        assert metadata is not None
        assert metadata["credits_debited"] == 5
