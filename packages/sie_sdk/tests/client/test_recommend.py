"""Tests for the model recommendation read (``POST /v1/recommend``, #2845).

The SDK's job here is narrow: send the task, return the gateway's answer
unchanged, and map transport failures onto the same typed errors every other
method uses.

"Unchanged" is the load-bearing part. The answer carries ``basis`` and a
per-choice ``evidence_guarded``, and those two fields are the entire reason a
caller can decide whether to trust a pick. An SDK that returned only the model
id — or that quietly dropped a field it did not recognise — would convert a
checkable recommendation into an unfalsifiable one.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import RequestError, ServerError

RANKED: dict[str, Any] = {
    "task": "rerank",
    "label": "Rerank",
    "basis": "ranked",
    "shared_benchmarks": ["mteb__AskUbuntuDupQuestions"],
    "fast": {
        "intent": "fast",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "runtime_id": "Qwen/Qwen3-Reranker-0.6B",
        "profile": "default",
        "alias": "rerank-fast",
        "available": True,
        "quality_ref": "quality-evidence/rerank-fast.json",
        "performance_ref": None,
        "measurement_status": "verified",
        "evidence_guarded": True,
    },
    "best": {
        "intent": "smart",
        "model": "Qwen/Qwen3-Reranker-4B",
        "runtime_id": "Qwen/Qwen3-Reranker-4B",
        "profile": "default",
        "alias": "rerank-best",
        "available": True,
        "quality_ref": "quality-evidence/rerank-best.json",
        "performance_ref": None,
        "measurement_status": "verified",
        "evidence_guarded": False,
    },
}

NO_EVIDENCE: dict[str, Any] = {
    "task": "structured-output",
    "label": "Structured output",
    "basis": "no_evidence",
    "shared_benchmarks": [],
    "fast": {
        "intent": "fast",
        "model": "Qwen/Qwen3.5-4B",
        "runtime_id": "Qwen/Qwen3.5-4B",
        "profile": "default",
        "alias": None,
        "available": True,
        "evidence_guarded": False,
    },
}

UNKNOWN_TASK = {
    "detail": {
        "code": "TASK_NOT_FOUND",
        "message": 'unknown task "embeddings"; this release recommends for: chat, ocr, rerank, search',
    }
}


def _resp(status: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = body
    resp.content = json.dumps(body).encode()
    resp.text = json.dumps(body)
    return resp


def _aio(status: int, body: dict[str, Any]) -> _AioResponse:
    return _AioResponse(status, json.dumps(body).encode(), {"content-type": "application/json"})


class TestSyncRecommend:
    def test_posts_the_task_and_returns_the_answer_verbatim(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, RANKED))
            client = SIEClient("http://localhost:8080")

            answer = client.recommend("rerank")

            assert answer == RANKED
            url = mock_client.return_value.post.call_args[0]
            kwargs = mock_client.return_value.post.call_args[1]
            assert url[0] == "/v1/recommend"
            assert kwargs["json"] == {"task": "rerank"}
            client.close()

    def test_carries_the_fields_that_make_a_pick_checkable(self) -> None:
        """`basis` and `evidence_guarded` must survive the round trip.

        `rerank-best` here is measured but UNGUARDED — the live `search` and
        `redact-pii` shape. A caller that cannot see that flag cannot tell a
        floored number from an unfloored one.
        """
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, RANKED))
            client = SIEClient("http://localhost:8080")

            answer = client.recommend("rerank")

            assert answer["basis"] == "ranked"
            assert answer["shared_benchmarks"] == ["mteb__AskUbuntuDupQuestions"]
            assert answer["best"]["alias"] == "rerank-best"
            assert answer["best"]["evidence_guarded"] is False
            assert answer["fast"]["evidence_guarded"] is True
            client.close()

    def test_a_family_with_no_evidence_says_so_rather_than_omitting_the_basis(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, NO_EVIDENCE))
            client = SIEClient("http://localhost:8080")

            answer = client.recommend("structured-output")

            assert answer["basis"] == "no_evidence"
            assert "best" not in answer
            client.close()

    def test_unknown_task_raises_request_error(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(404, UNKNOWN_TASK))
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError):
                client.recommend("embeddings")
            client.close()

    def test_server_error_maps_to_server_error(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(
                return_value=_resp(503, {"detail": {"code": "UNAVAILABLE", "message": "down"}})
            )
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError):
                client.recommend("rerank")
            client.close()


class TestAsyncRecommend:
    @pytest.mark.asyncio
    async def test_matches_the_sync_twin(self) -> None:
        """The async client must reach the same route with the same body.

        The two drifting is the failure this pins: `recommend` shipped on the
        sync client first, and an async caller silently lacking it is the kind
        of gap that only shows up in someone else's code.
        """
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(200, RANKED))  # type: ignore[method-assign]

            answer = await client.recommend("rerank")

            assert answer == RANKED
            assert client._post.await_args[0][0] == "/v1/recommend"
            assert client._post.await_args[1]["json_data"] == {"task": "rerank"}
            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_task_raises_request_error(self) -> None:
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(404, UNKNOWN_TASK))  # type: ignore[method-assign]

            with pytest.raises(RequestError):
                await client.recommend("embeddings")
            await client.close()
