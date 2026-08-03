"""Tests for DuskBedrockClient -- the model-call wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from bedrock_client import DuskBedrockClient, DuskBlockedError


def test_converse_forwards_to_underlying_client():
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "a normal reply"}]}}
    }
    wrapper = DuskBedrockClient(client=mock_client)

    result = wrapper.converse(messages=[{"role": "user", "content": [{"text": "hi"}]}])

    assert result["output"]["message"]["content"][0]["text"] == "a normal reply"
    mock_client.converse.assert_called_once()
    _, kwargs = mock_client.converse.call_args
    assert kwargs["modelId"] == wrapper.model_id
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


def test_dusk_blocked_request_carries_full_payload():
    verdict: dict[str, Any] = {
        "verdict": "BLOCK",
        "score": 0.93,
        "reasons": ["out of baseline", "privileged term introduced"],
    }

    with pytest.raises(DuskBlockedError) as exc_info:
        raise DuskBlockedError(verdict)

    assert exc_info.value.verdict == verdict
    assert "out of baseline" in str(exc_info.value)
