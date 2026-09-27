"""Generate responses keep the engine's prefix-cache count in ``usage``."""

from __future__ import annotations

import pytest
from sie_sdk.client.async_ import _parse_generate_result_async
from sie_sdk.client.sync import _parse_generate_result

_GENERATE_PARSERS = (_parse_generate_result, _parse_generate_result_async)


@pytest.mark.parametrize("parse", _GENERATE_PARSERS)
def test_generate_usage_keeps_cached_prompt_tokens(parse) -> None:
    result = parse(
        {
            "model": "m",
            "text": "hi",
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 3,
                "total_tokens": 93,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        },
        request=None,
    )

    assert result["usage"]["prompt_tokens_details"] == {"cached_tokens": 64}


@pytest.mark.parametrize("parse", _GENERATE_PARSERS)
@pytest.mark.parametrize(
    "details",
    [None, {}, {"cached_tokens": -1}, {"cached_tokens": True}, {"cached_tokens": "64"}, [64]],
)
def test_generate_usage_drops_absent_or_malformed_cached_prompt_tokens(parse, details) -> None:
    usage: dict[str, object] = {"prompt_tokens": 90, "completion_tokens": 3, "total_tokens": 93}
    if details is not None:
        usage["prompt_tokens_details"] = details

    result = parse({"model": "m", "text": "hi", "usage": usage}, request=None)

    assert "prompt_tokens_details" not in result["usage"]
