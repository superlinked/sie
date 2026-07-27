"""Fake Engine regression test (#1850): the MODEL_LOADING race, client-visible.

Boots a real server with the fake bundle (zero downloads) and a slow-load
fault, then races concurrent first-touch requests into the cold model.
Characterizes both lazy-load lanes (#1726 asymmetry):

- rerank/score/generate: NON-BLOCKING — every racer gets a fast
  503 MODEL_LOADING with Retry-After; no request rides the load.
- /v1/embeddings: BLOCKING — the first touch holds the connection and
  returns 200 only after the load completes.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from .conftest import SLOW_LOAD_S as _SLOW_LOAD_S

pytestmark = [pytest.mark.fake_stack, pytest.mark.integration]

# Raw httpx (not SIEClient) is deliberate here — AGENTS.md's raw-HTTP
# exception for debugging transport behavior applies: these tests assert the
# 503/Retry-After wire semantics that the SDK exists to hide (it auto-retries
# MODEL_LOADING). The SDK-layer view of the same server lives in
# test_sdk_surface.py.


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


def test_model_loading_blocking_embeddings_lane(fake_server: str) -> None:
    """The /v1/embeddings first touch BLOCKS through the cold load (the
    documented asymmetry): one request, ~slow-load latency, straight 200.
    """
    start = time.monotonic()
    response = httpx.post(
        f"{fake_server}/v1/embeddings",
        json={"model": "sie-fake", "input": ["hello"]},
        timeout=_SLOW_LOAD_S * 10,
    )
    elapsed = time.monotonic() - start
    assert response.status_code == 200, response.text
    assert elapsed >= _SLOW_LOAD_S * 0.9, "blocking lane must ride the cold load"
    assert len(response.json()["data"][0]["embedding"]) == 384
