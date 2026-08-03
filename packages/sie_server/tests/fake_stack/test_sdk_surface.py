"""Fake Engine SDK-layer coverage (#1850): SIEClient against the fake stack.

The MODEL_LOADING race tests assert the raw wire semantics; this module
asserts the layer above — `sie_sdk.SIEClient` transparently absorbs the
cold-load 503s (its documented MODEL_LOADING auto-retry) and returns correct,
deterministic results from every fake surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sie_sdk import SIEClient

pytestmark = [pytest.mark.fake_stack, pytest.mark.integration]


@pytest.fixture(scope="module")
def client(fake_server: str) -> Iterator[SIEClient]:
    sie = SIEClient(fake_server, timeout_s=60.0)
    yield sie
    sie.close()


def test_sdk_score_rides_out_cold_load(client: SIEClient) -> None:
    """First touch of sie-fake:small-a hits the 3 s slow-load fault; the SDK
    must absorb the 503 MODEL_LOADING wave and deliver ranked scores.
    """
    result = client.score("sie-fake:small-a", {"text": "q"}, [{"text": "d1"}, {"text": "d2"}])
    scores = result["scores"]
    assert len(scores) == 2
    ranks = [entry["rank"] for entry in scores]
    assert ranks == sorted(ranks)


def test_sdk_encode_deterministic(client: SIEClient) -> None:
    first = client.encode("sie-fake", {"text": "hello"}, output_types=["dense"])
    second = client.encode("sie-fake", {"text": "hello"}, output_types=["dense"])
    assert list(first["dense"]) == list(second["dense"])
    assert len(first["dense"]) == 384


def test_sdk_generate_deterministic(client: SIEClient) -> None:
    first = client.generate("sie-fake", "a prompt", max_new_tokens=16)
    second = client.generate("sie-fake", "a prompt", max_new_tokens=16)
    assert first["text"]
    assert first["text"] == second["text"]
    assert first["finish_reason"] == "length"
    assert first["usage"]["completion_tokens"] == 16
