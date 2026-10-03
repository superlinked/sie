"""Fake Engine regression test (#1850): the MODEL_LOADING race, client-visible.

Boots a real server with the fake bundle (zero downloads) and a slow-load
fault, then races concurrent first-touch requests into the cold model.

Both lazy-load lanes are now NON-BLOCKING (the #1726 asymmetry is closed):

- rerank/score/generate: every racer gets a fast 503 MODEL_LOADING with
  Retry-After; no request rides the load.
- /v1/embeddings: the first touch also returns a fast 503 MODEL_LOADING
  with Retry-After instead of holding the connection through the cold
  load — single-node parity with the gateway's /v1/embeddings. Retrying
  eventually returns 200 once the model is resident.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from .conftest import SLOW_LOAD_S as _SLOW_LOAD_S

pytestmark = [pytest.mark.fake_stack, pytest.mark.integration]

# Raw httpx (not SIEClient) is deliberate here: these tests assert the
# 503/Retry-After wire semantics that the SDK exists to hide (it auto-retries
# MODEL_LOADING). The SDK-layer view of the same server lives in
# test_sdk_surface.py.


def _assert_valid_retry_after(response: httpx.Response) -> None:
    """Every 503 MODEL_LOADING must carry a positive numeric Retry-After hint."""
    retry_after = {k.lower(): v for k, v in response.headers.items()}.get("retry-after")
    assert retry_after is not None, "MODEL_LOADING must carry a Retry-After hint"
    assert float(retry_after) > 0, f"Retry-After must be positive, got {retry_after!r}"


def _rerank_once(base: str) -> httpx.Response:
    return httpx.post(
        f"{base}/v1/rerank",
        json={"model": "sie-fake:small-a", "query": "q", "documents": ["a", "b"], "top_n": 1},
        timeout=30.0,
    )


def test_model_loading_race_nonblocking_lane(fake_server: str) -> None:
    """Eight concurrent cold-load racers on the non-blocking lane: every one
    gets a FAST 503 MODEL_LOADING with a Retry-After hint — none rides the
    in-flight load — and retrying eventually succeeds with ranked results.
    """
    racers = 8
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=racers) as pool:
        responses = list(pool.map(lambda _: _rerank_once(fake_server), range(racers)))
    first_wave_s = time.monotonic() - start

    for response in responses:
        assert response.status_code == 503, response.text
        assert "MODEL_LOADING" in response.text
        assert "retry-after" in {k.lower() for k in response.headers}
    # Non-blocking means the whole racing wave returns well before the load
    # finishes — nobody waited out the slow load on the connection.
    assert first_wave_s < _SLOW_LOAD_S, "non-blocking lane must not ride the cold load"

    deadline = time.monotonic() + _SLOW_LOAD_S * 10
    while True:
        response = _rerank_once(fake_server)
        if response.status_code == 200:
            break
        assert response.status_code == 503, response.text
        if time.monotonic() >= deadline:
            pytest.fail("model never finished loading")
        time.sleep(0.5)
    body = response.json()
    assert len(body["results"]) == 1


def test_model_loading_nonblocking_embeddings_lane(fake_server: str) -> None:
    """The /v1/embeddings first touch is NON-BLOCKING: it returns a FAST 503
    MODEL_LOADING with a Retry-After hint (OpenAI-shaped body on this surface)
    instead of riding the cold load, then a retry loop returns 200 once the
    model is resident — mirroring the rerank/score cold-load lanes.

    This is the regression guard for the usability fix: the previous
    behavior blocked the connection through the whole slow load and returned
    a straight 200, which this test used to assert. See PR #3183.
    """
    start = time.monotonic()
    first = httpx.post(
        f"{fake_server}/v1/embeddings",
        json={"model": "sie-fake", "input": ["hello"]},
        timeout=_SLOW_LOAD_S * 10,
    )
    first_touch_s = time.monotonic() - start

    # Fast, retryable 503 — the request must NOT have ridden the cold load.
    assert first.status_code == 503, first.text
    assert first_touch_s < _SLOW_LOAD_S, "non-blocking lane must not ride the cold load"
    _assert_valid_retry_after(first)
    # Top-level OpenAI-shaped envelope on this surface: {"error": {...}} (#3184).
    error = first.json()["error"]
    assert error["code"] == "MODEL_LOADING", first.text
    assert error["type"] == "server_error"

    # Retrying (as the SDK does transparently) eventually succeeds once loaded.
    deadline = time.monotonic() + _SLOW_LOAD_S * 10
    while True:
        response = httpx.post(
            f"{fake_server}/v1/embeddings",
            json={"model": "sie-fake", "input": ["hello"]},
            timeout=_SLOW_LOAD_S * 10,
        )
        if response.status_code == 200:
            break
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "MODEL_LOADING", response.text
        # Retry-After must be present and valid on EVERY 503, not just the first.
        _assert_valid_retry_after(response)
        if time.monotonic() >= deadline:
            pytest.fail("model never finished loading")
        time.sleep(0.5)
    assert len(response.json()["data"][0]["embedding"]) == 384
