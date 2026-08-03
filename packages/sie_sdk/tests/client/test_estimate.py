"""Tests for the cost-estimate dry run (``POST /v1/estimate``, #2435).

The SDK's job on this route is narrow and load-bearing:

- send the target request VERBATIM inside ``{endpoint, request}`` — a body the
  SDK reshaped would be a quote for a request the caller never sends;
- return the gateway's projection unchanged, so the customer reads the same
  ceiling the metered path would hold;
- map "the book cannot price this" onto a typed
  :class:`EstimateUnroutableError` rather than a generic 5xx, because that
  verdict is the request's, not the estimator's.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client._shared import build_estimate_envelope
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import EstimateUnroutableError, RequestError, ServerError

QUOTE: dict[str, Any] = {
    "endpoint": "/v1/encode/BAAI/bge-m3",
    "identity": {
        "model": "BAAI/bge-m3",
        "profile": "default",
        "operation": "encode",
        "region": "us",
    },
    "estimated_credits": 261,
    "unit_ceilings": {"input_tokens": 261},
    "applied_rates": [
        {"unit": "input_tokens", "rate_numerator": 1, "rate_denominator": 1},
    ],
    "rate_book_version": "2026-07-26-production-bootstrap-v2",
    "rate_book_sha256": "a" * 64,
    "rounding_rule": "ceil-once-per-terminal-event",
    "estimate_basis": "rate-book 2026-07-26-production-bootstrap-v2 conservative pre-dispatch reservation ceiling",
    "minimum_billed_units": None,
}

UNROUTABLE = {
    "detail": {
        "code": "QUEUE_UNAVAILABLE",
        "message": 'missing rate for model="acme/unpriced", profile="default", operation="encode", region="us"',
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


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


class TestEstimateEnvelope:
    def test_carries_the_target_body_verbatim(self) -> None:
        request = {"items": [{"text": "Hello"}], "params": {"output_types": ["dense"]}}
        envelope = build_estimate_envelope("/v1/encode/BAAI/bge-m3", request)
        assert envelope == {"endpoint": "/v1/encode/BAAI/bge-m3", "request": request}

    def test_detaches_the_caller_mapping(self) -> None:
        """A caller mutating their dict after the call must not change what was priced."""
        request: dict[str, Any] = {"items": [{"text": "Hello"}]}
        envelope = build_estimate_envelope("/v1/encode/m", request)
        request["items"] = []
        assert envelope["request"]["items"] == [{"text": "Hello"}]

    @pytest.mark.parametrize(
        ("endpoint", "body"),
        [
            ("v1/encode/m", {"items": []}),
            ("", {"items": []}),
            ("/v1/encode/m", ["not", "a", "mapping"]),
            ("/v1/encode/m", "not a mapping"),
        ],
    )
    def test_rejects_a_malformed_envelope_client_side(self, endpoint: Any, body: Any) -> None:
        with pytest.raises(ValueError, match="estimate"):
            build_estimate_envelope(endpoint, body)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class TestSyncEstimate:
    def test_posts_the_envelope_and_returns_the_quote(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, QUOTE))
            client = SIEClient("http://localhost:8080")

            quote = client.estimate("/v1/encode/BAAI/bge-m3", {"items": [{"text": "Hello"}]})

            assert quote == QUOTE
            assert quote["estimated_credits"] == 261
            assert quote["unit_ceilings"] == {"input_tokens": 261}
            assert quote["rate_book_version"] == "2026-07-26-production-bootstrap-v2"
            assert quote["applied_rates"][0]["rate_denominator"] == 1

            url, kwargs = mock_client.return_value.post.call_args[0], mock_client.return_value.post.call_args[1]
            assert url[0] == "/v1/estimate"
            assert kwargs["json"] == {
                "endpoint": "/v1/encode/BAAI/bge-m3",
                "request": {"items": [{"text": "Hello"}]},
            }
            client.close()

    def test_compat_body_rides_through_untouched(self) -> None:
        """The model stays in the BODY for an OpenAI-compatible target."""
        body = {
            "model": "Qwen/Qwen3.6-27B",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "n": 3,
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, QUOTE))
            client = SIEClient("http://localhost:8080")
            client.estimate("/v1/chat/completions", body)
            sent = mock_client.return_value.post.call_args[1]["json"]
            assert sent["endpoint"] == "/v1/chat/completions"
            assert sent["request"] == body
            client.close()

    def test_unpriced_identity_raises_the_typed_unroutable_error(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(503, UNROUTABLE))
            client = SIEClient("http://localhost:8080")

            with pytest.raises(EstimateUnroutableError) as excinfo:
                client.estimate("/v1/encode/acme/unpriced", {"items": [{"text": "Hello"}]})

            assert excinfo.value.code == "QUEUE_UNAVAILABLE"
            assert excinfo.value.status_code == 503
            # The planner's own reason survives to the caller — it names the
            # identity, which is the whole point of a fail-closed estimate.
            assert "acme/unpriced" in str(excinfo.value)
            # Existing 5xx handlers keep working.
            assert isinstance(excinfo.value, ServerError)
            client.close()

    def test_a_retryable_billing_capacity_503_stays_a_plain_server_error(self) -> None:
        """Gateway capacity pressure is not an unroutable verdict about the request."""
        capacity = {
            "detail": {
                "code": "BILLING_CAPACITY_UNAVAILABLE",
                "message": "audio billing preflight is at capacity",
            }
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(503, capacity))
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as excinfo:
                client.estimate("/v1/extract/audio/asr", {"items": [{"text": "x"}]})

            assert not isinstance(excinfo.value, EstimateUnroutableError)
            assert excinfo.value.code == "BILLING_CAPACITY_UNAVAILABLE"
            client.close()

    def test_an_unroutable_model_is_a_404_not_a_quote(self) -> None:
        """The gateway checks routability, not just priceability.

        A model this data plane does not serve answers the live 404 even when
        the rate book prices it — a regression turning that into a 200 quote
        would otherwise pass unnoticed.
        """
        not_found = {"detail": {"code": "MODEL_NOT_FOUND", "message": 'Model "acme/absent" not found.'}}
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(404, not_found))
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as excinfo:
                client.estimate("/v1/encode/acme/absent", {"items": [{"text": "Hello"}]})

            assert not isinstance(excinfo.value, EstimateUnroutableError)
            assert excinfo.value.status_code == 404
            assert excinfo.value.code == "MODEL_NOT_FOUND"
            client.close()

    def test_a_malformed_target_body_is_a_request_error(self) -> None:
        invalid = {"detail": {"code": "INVALID_REQUEST", "message": "'items' must be an array"}}
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(400, invalid))
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as excinfo:
                client.estimate("/v1/encode/BAAI/bge-m3", {"items": "nope"})

            assert excinfo.value.status_code == 400
            assert excinfo.value.code == "INVALID_REQUEST"
            client.close()

    def test_a_sealed_rate_card_carries_its_minimum_billed_floor(self) -> None:
        sealed = {
            **QUOTE,
            "endpoint": "/v1/generate/org-42/support-gen",
            "identity": {
                "model": "custom.l4",
                "profile": "default",
                "operation": "sealed",
                "region": "us",
            },
            "unit_ceilings": {"gpu_second": 130},
            "applied_rates": [
                {"unit": "gpu_second", "rate_numerator": 7, "rate_denominator": 1},
            ],
            "estimated_credits": 910,
            "minimum_billed_units": {"gpu_second": 1},
            "estimate_basis": "sealed custom-lane rate card; duration is execution-dependent",
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, sealed))
            client = SIEClient("http://localhost:8080")

            quote = client.estimate(
                "/v1/generate/org-42/support-gen",
                {"prompt": "hi", "max_new_tokens": 32},
            )

            assert quote["minimum_billed_units"] == {"gpu_second": 1}
            assert quote["applied_rates"][0]["unit"] == "gpu_second"
            assert "duration" in quote["estimate_basis"]
            client.close()

    def test_client_side_envelope_validation_never_reaches_the_network(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock()
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ValueError, match="estimate endpoint"):
                client.estimate("v1/encode/m", {"items": []})

            mock_client.return_value.post.assert_not_called()
            client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class TestAsyncEstimate:
    @pytest.mark.asyncio
    async def test_posts_the_envelope_and_returns_the_quote(self) -> None:
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(200, QUOTE))  # type: ignore[method-assign]

            quote = await client.estimate("/v1/encode/BAAI/bge-m3", {"items": [{"text": "Hello"}]})

            assert quote == QUOTE
            assert client._post.await_args[0][0] == "/v1/estimate"
            assert client._post.await_args[1]["json_data"] == {
                "endpoint": "/v1/encode/BAAI/bge-m3",
                "request": {"items": [{"text": "Hello"}]},
            }
            await client.close()

    @pytest.mark.asyncio
    async def test_unpriced_identity_raises_the_typed_unroutable_error(self) -> None:
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(503, UNROUTABLE))  # type: ignore[method-assign]

            with pytest.raises(EstimateUnroutableError) as excinfo:
                await client.estimate("/v1/encode/acme/unpriced", {"items": [{"text": "Hello"}]})

            assert excinfo.value.code == "QUEUE_UNAVAILABLE"
            assert "acme/unpriced" in str(excinfo.value)
            await client.close()

    @pytest.mark.asyncio
    async def test_an_unroutable_model_is_a_404_not_a_quote(self) -> None:
        """The async client duplicates the sync error-dispatch block.

        Without this the two could diverge on the routability contract and
        nothing would notice.
        """
        not_found = {"detail": {"code": "MODEL_NOT_FOUND", "message": 'Model "acme/absent" not found.'}}
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(404, not_found))  # type: ignore[method-assign]

            with pytest.raises(RequestError) as excinfo:
                await client.estimate("/v1/encode/acme/absent", {"items": [{"text": "Hello"}]})

            assert not isinstance(excinfo.value, EstimateUnroutableError)
            assert excinfo.value.status_code == 404
            assert excinfo.value.code == "MODEL_NOT_FOUND"
            await client.close()

    @pytest.mark.asyncio
    async def test_client_side_envelope_validation_never_reaches_the_network(self) -> None:
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock()  # type: ignore[method-assign]

            with pytest.raises(ValueError, match="estimate endpoint"):
                await client.estimate("v1/encode/m", {"items": []})

            client._post.assert_not_awaited()
            await client.close()

    @pytest.mark.asyncio
    async def test_sync_and_async_send_the_same_envelope(self) -> None:
        """The two clients must not drift on WHAT they price."""
        request = {"model": "m", "input": "hello"}
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(return_value=_aio(200, QUOTE))  # type: ignore[method-assign]
            await client.estimate("/v1/embeddings", request)
            async_body = client._post.await_args[1]["json_data"]
            await client.close()

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_resp(200, QUOTE))
            sync_client = SIEClient("http://localhost:8080")
            sync_client.estimate("/v1/embeddings", request)
            sync_body = mock_client.return_value.post.call_args[1]["json"]
            sync_client.close()

        assert async_body == sync_body
