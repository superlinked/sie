# Tests for SDK auto-retry on 503 RESOURCE_EXHAUSTED.
#
# Mirrors the test pattern in ``test_gateway_timeout_retry.py``: stub the
# underlying ``httpx.Client.post`` (sync) or async transport with a
# canned sequence of responses, then assert the retry counter, error class,
# and final result. Backoff timing is decoupled from real wall-clock by
# patching ``time.sleep`` / ``asyncio.sleep`` to no-ops.

from __future__ import annotations

import asyncio  # noqa: F401 — referenced via ``sie_sdk.client.async_.asyncio.sleep`` patch target
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import ResourceExhaustedError


def _resp_oom(retry_after: str = "0.01") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 503
    resp.headers = {"Retry-After": retry_after, "content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": "RESOURCE_EXHAUSTED", "message": "Server resource pressure"}}
    return resp


def _resp_200_encode() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]},
        use_bin_type=True,
    )
    return resp


def _resp_200_extract() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {"model": "m", "items": [{"entities": [], "relations": [], "classifications": [], "objects": []}]},
        use_bin_type=True,
    )
    return resp


# --------------------------------------------------------------------------
# Sync client
# --------------------------------------------------------------------------


class TestSyncOomRetry:
    def test_retry_then_success(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_oom(), _resp_oom(), _resp_200_encode()])
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 3
            client.close()

    def test_exhausted_retries_raises(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            # Always OOM. With max_oom_retries=2, the 3rd 503 raises.
            mock_client.return_value.post = MagicMock(side_effect=[_resp_oom(), _resp_oom(), _resp_oom()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ResourceExhaustedError) as excinfo:
                client.encode("bge-m3", {"text": "hi"}, max_oom_retries=2)

            assert excinfo.value.retries == 2
            assert excinfo.value.code == "RESOURCE_EXHAUSTED"
            assert excinfo.value.status_code == 503
            assert mock_client.return_value.post.call_count == 3
            client.close()

    def test_max_oom_retries_zero_disables_retry(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_oom()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ResourceExhaustedError):
                client.encode("bge-m3", {"text": "hi"}, max_oom_retries=0)

            # No retry — single request only.
            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_extract_path_retries(self) -> None:
        """Score and extract paths share the OOM-retry block; spot-check extract."""
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_oom(), _resp_200_extract()])
            client = SIEClient("http://localhost:8080")

            result = client.extract("gliner", {"text": "hi"}, labels=["x"])
            assert isinstance(result, dict)
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_provision_timeout_bounds_oom_retry_wall_time(self) -> None:
        """``provision_timeout_s`` clamps OOM backoff so callers aren't stranded.

        Regression for the bug where the OOM retry block ignored
        ``elapsed >= timeout`` (unlike the MODEL_LOADING block above it).
        A caller passing a tight ``provision_timeout_s`` could be blocked
        for the full default backoff (5+10+20=35s) regardless.
        """
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            # Stub time.monotonic to advance past the timeout immediately
            # after the first OOM. The retry loop should NOT sleep again.
            base = 1000.0
            ticks = iter([base, base, base + 100.0, base + 100.0, base + 100.0])

            with patch.object(time, "monotonic", side_effect=lambda: next(ticks)):
                mock_client.return_value.post = MagicMock(side_effect=[_resp_oom(), _resp_oom(), _resp_200_encode()])
                client = SIEClient("http://localhost:8080")

                with pytest.raises(ResourceExhaustedError):
                    client.encode(
                        "bge-m3",
                        {"text": "hi"},
                        provision_timeout_s=10.0,  # 100s elapsed > 10s budget
                    )
                # Single round-trip, then the timeout-elapsed check raises
                # on the second iteration without making a second post.
                assert mock_client.return_value.post.call_count == 1
                client.close()

    def test_next_backoff_exceeding_budget_raises_oom_not_provisioning(self) -> None:
        """Sustained OOM near the budget boundary surfaces as
        ``ResourceExhaustedError``, not ``ProvisioningError``.

        Regression: previously the OOM block would clamp ``delay`` to
        ``min(raw_delay, remaining)`` and sleep through the rest of the
        budget; the outer loop's ``remaining <= 0`` branch would then
        raise ``ProvisioningError``, masking the root cause. The new
        guard raises ``ResourceExhaustedError`` when the next backoff
        would exhaust the remaining budget, so callers see "the server
        ran out of capacity" rather than "we timed out provisioning".
        """
        from sie_sdk.client.errors import ProvisioningError

        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            # First OOM at t=0s, then jump to t=8s for the retry decision.
            # provision_timeout_s=10s leaves 2s remaining, but
            # compute_oom_backoff(retry_after=None, attempt=0) = 5s, which
            # exceeds 2s. The new guard must raise ResourceExhaustedError
            # immediately rather than sleep the budget away.
            base = 1000.0
            ticks = iter([base, base, base + 8.0, base + 8.0, base + 8.0])

            # Use a response with NO Retry-After so compute_oom_backoff
            # falls back to its default 5s base delay.
            def resp_oom_no_retry_after() -> MagicMock:
                resp = MagicMock()
                resp.status_code = 503
                resp.headers = {"content-type": "application/json"}
                resp.json.return_value = {
                    "detail": {"code": "RESOURCE_EXHAUSTED", "message": "Server resource pressure"}
                }
                return resp

            with patch.object(time, "monotonic", side_effect=lambda: next(ticks)):
                mock_client.return_value.post = MagicMock(
                    side_effect=[resp_oom_no_retry_after(), resp_oom_no_retry_after()]
                )
                client = SIEClient("http://localhost:8080")

                with pytest.raises(ResourceExhaustedError):
                    client.encode("bge-m3", {"text": "hi"}, provision_timeout_s=10.0)
                # Exactly one post — the second iteration's _handle_oom_retry
                # raises before another post happens.
                assert mock_client.return_value.post.call_count == 1
                # Crucially, NOT ProvisioningError:
                # (pytest.raises above already enforced ResourceExhaustedError,
                # but assert it is NOT a ProvisioningError subclass either)
                client.close()

        # Sanity: ResourceExhaustedError and ProvisioningError must not
        # be the same exception in the type hierarchy.
        assert not issubclass(ResourceExhaustedError, ProvisioningError)

    def test_negative_retry_after_does_not_crash(self) -> None:
        """A malformed ``Retry-After: -5`` must not crash ``time.sleep``.

        Regression: ``compute_oom_backoff`` previously passed the raw
        value through, so a buggy/malicious upstream could drop a negative
        header and abort the client call with ``ValueError`` from
        ``time.sleep``.
        """
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_oom(retry_after="-5"), _resp_200_encode()])
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            # All sleep calls received a non-negative value.
            for call in mock_sleep.call_args_list:
                (delay,) = call.args
                assert delay >= 0, f"time.sleep called with negative {delay}"
            client.close()


# --------------------------------------------------------------------------
# Async client
# --------------------------------------------------------------------------


def _aio_oom() -> object:
    return _AioResponse(
        503,
        json.dumps({"detail": {"code": "RESOURCE_EXHAUSTED", "message": "Server resource pressure"}}).encode(),
        {"Retry-After": "0.01", "content-type": "application/json"},
    )


def _aio_200_encode() -> object:
    return _AioResponse(
        200,
        msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
        {"content-type": "application/msgpack"},
    )


class TestAsyncOomRetry:
    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep"),
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_oom(), _aio_oom(), _aio_200_encode()])

            result = await client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert client._post.await_count == 3
            await client.close()

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep"),
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_oom(), _aio_oom(), _aio_oom()])

            with pytest.raises(ResourceExhaustedError) as excinfo:
                await client.encode("bge-m3", {"text": "hi"}, max_oom_retries=2)

            assert excinfo.value.retries == 2
            assert client._post.await_count == 3
            await client.close()
