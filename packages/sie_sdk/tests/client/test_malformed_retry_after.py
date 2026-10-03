"""Regression tests for BUG 5 — malformed ``Retry-After`` (NaN / inf / negative).

A non-finite or negative ``Retry-After`` header used to be returned verbatim
by :func:`get_retry_after`. Downstream that meant:

* sync ``generate()`` -> ``time.sleep(nan)`` raises ``ValueError`` (crash).
* async ``generate()`` -> ``asyncio.sleep(nan)`` / ``asyncio.sleep(-10)``
  return INSTANTLY, so the retry loop busy-spins for the entire
  provision-timeout budget (DoS amplification).

The source fix makes :func:`get_retry_after` return ``None`` for any
non-finite or negative value (treated as "no usable hint" -> caller falls
back to its default delay).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client._shared import get_retry_after, provisioning_retry_delay
from sie_sdk.client.errors import ModelLoadingError, ProvisioningError


def _resp_with_retry_after(header_value: str) -> MagicMock:
    response = MagicMock()
    response.headers = {"Retry-After": header_value}
    return response


def _aio_resp(status: int, body: dict, headers: dict | None = None) -> object:
    from sie_sdk.client.async_ import _AioResponse

    return _AioResponse(status, json.dumps(body).encode("utf-8"), headers or {"content-type": "application/json"})


def _make_session_post(seq_source: AsyncMock):
    """Build a ``session.post`` that yields a raw aiohttp-like response per call.

    Mirrors the adapter in ``test_generate.py``: the async ``generate`` path
    posts inline via ``_ensure_session().post(...)``, reading ``raw.status``,
    ``await raw.read()`` and ``raw.headers``.
    """

    def _post(*_args: object, **_kwargs: object) -> MagicMock:
        ctx = MagicMock()

        async def _aenter() -> MagicMock:
            aio = await seq_source()
            raw = MagicMock()
            raw.status = aio.status_code
            raw.headers = aio.headers
            raw.read = AsyncMock(return_value=aio.content)
            return raw

        ctx.__aenter__ = AsyncMock(side_effect=_aenter)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    return MagicMock(side_effect=_post)


class TestGetRetryAfterMalformed:
    """(a) ``get_retry_after`` returns ``None`` for non-finite / negative values."""

    @pytest.mark.parametrize("header", ["nan", "NaN", "inf", "-inf", "Infinity", "-10", "-0.0001"])
    def test_returns_none_for_non_finite_or_negative(self, header: str) -> None:
        assert get_retry_after(_resp_with_retry_after(header)) is None

    @pytest.mark.parametrize(("header", "expected"), [("5", 5.0), ("0", 0.0), ("2.5", 2.5)])
    def test_returns_finite_value_for_valid_input(self, header: str, expected: float) -> None:
        assert get_retry_after(_resp_with_retry_after(header)) == expected


class TestGetRetryAfterHttpDate:
    """``get_retry_after`` parses the RFC 7231 HTTP-date form for cross-SDK
    parity with the TS SDK (which already accepts both seconds and HTTP-date).
    """

    def test_future_http_date_returns_positive_delta(self) -> None:
        from datetime import UTC, datetime, timedelta
        from email.utils import format_datetime

        when = datetime.now(UTC) + timedelta(seconds=30)
        header = format_datetime(when, usegmt=True)
        delta = get_retry_after(_resp_with_retry_after(header))
        assert delta is not None
        # ~30s, allowing a little slack for test execution time.
        assert 25.0 <= delta <= 31.0

    def test_past_http_date_clamps_to_zero(self) -> None:
        from datetime import UTC, datetime, timedelta
        from email.utils import format_datetime

        when = datetime.now(UTC) - timedelta(seconds=30)
        header = format_datetime(when, usegmt=True)
        assert get_retry_after(_resp_with_retry_after(header)) == 0.0

    def test_unparseable_value_returns_none(self) -> None:
        assert get_retry_after(_resp_with_retry_after("not-a-date")) is None


class TestProvisioningRetryDelay:
    def test_retry_after_zero_means_immediate_retry(self) -> None:
        response = _resp_with_retry_after("0")
        assert (
            provisioning_retry_delay(
                response,
                gpu="l4",
                wait_for_capacity=True,
                start_time=time.monotonic(),
                timeout=10.0,
            )
            == 0.0
        )


class TestSyncGenerateMalformedRetryAfter:
    """(b) sync ``generate()`` against a perpetual ``503 PROVISIONING Retry-After: nan``
    must NOT raise ``ValueError`` (no ``time.sleep(nan)`` crash) and must
    terminate within the provision budget with a bounded number of HTTP calls.
    """

    def test_perpetual_503_provisioning_nan_does_not_crash_and_is_bounded(self) -> None:

        def _resp_503_provisioning_nan() -> MagicMock:
            response = MagicMock()
            response.status_code = 503
            response.headers = {"Retry-After": "nan", "content-type": "application/json"}
            response.json.return_value = {"error": {"code": "PROVISIONING", "message": "provisioning"}}
            response.content = json.dumps(response.json.return_value).encode("utf-8")
            return response

        sleeps: list[float] = []
        real_sleep = time.sleep  # capture before patching to avoid recursion

        # Real (but fast) sleep so the loop actually consumes the budget; record
        # each delay so we can assert it never sees a NaN/negative value. Before
        # the fix, ``time.sleep(nan)`` would raise ValueError on the first call.
        def _fake_sleep(d: float) -> None:
            sleeps.append(d)
            real_sleep(min(d, 0.01))

        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep", side_effect=_fake_sleep),
        ):
            mock_client.return_value.post.return_value = _resp_503_provisioning_nan()
            client = SIEClient("http://localhost:8080")
            # Must terminate (ProvisioningError) within the budget — NOT ValueError.
            with pytest.raises(ProvisioningError):
                client.generate("m", prompt="hi", max_new_tokens=8, provision_timeout_s=0.2)
            client.close()

        # Every recorded delay is finite and non-negative (no NaN crash, no
        # negative passed to sleep) — proves the malformed hint was discarded
        # and the default-derived delay (capped to remaining budget) was used.
        assert sleeps, "expected at least one retry sleep"
        assert all(math.isfinite(d) and d >= 0 for d in sleeps)
        # Bounded number of HTTP calls (default delay ~5s caps to remaining
        # budget; with a 0.2s budget this is a small handful, not thousands).
        assert mock_client.return_value.post.call_count < 50


class TestAsyncGenerateMalformedRetryAfter:
    """(c) async ``generate()`` against a perpetual ``503 MODEL_LOADING
    Retry-After: -10`` (or ``nan``) must make a BOUNDED number of attempts
    within a small budget (no busy-loop) and raise the expected loading /
    provisioning error.

    Before the fix the malformed header was passed through, so
    ``asyncio.sleep(nan)`` / ``asyncio.sleep(-10)`` returned INSTANTLY and the
    loop busy-spun (~1000 attempts in 0.3s). After the fix the hint is
    discarded and the default 5s delay applies; since that exceeds the tiny
    0.3s budget, ``generate()`` raises the typed ``ModelLoadingError``
    immediately (no sleep, single attempt) instead of sleeping through the
    budget.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("header", ["-10", "nan"])
    async def test_perpetual_503_model_loading_is_bounded(self, header: str) -> None:
        client = SIEAsyncClient("http://localhost:8080")

        def _make_resp() -> object:
            return _aio_resp(
                503,
                {"error": {"code": "MODEL_LOADING", "message": "loading"}},
                {"X-SIE-Error-Code": "MODEL_LOADING", "Retry-After": header, "content-type": "application/json"},
            )

        # Perpetual MODEL_LOADING.
        client._post = AsyncMock(side_effect=lambda *a, **k: _make_resp())  # type: ignore[method-assign]

        slept: list[float] = []
        real_sleep = asyncio.sleep  # capture before patching to avoid recursion

        async def _recording_sleep(d: float) -> None:
            slept.append(d)
            # Run a *real* (tiny) sleep so the monotonic budget actually
            # advances; cap it so the test stays fast.
            await real_sleep(min(d, 0.01))

        with patch.object(client, "_ensure_session") as ensure:
            ensure.return_value.post = _make_session_post(client._post)
            with patch("sie_sdk.client.async_.asyncio.sleep", side_effect=_recording_sleep):
                try:
                    # Tiny budget; if the malformed Retry-After busy-loops, the
                    # call count explodes into the hundreds/thousands.
                    with pytest.raises(ModelLoadingError):
                        await client.generate("m", prompt="hi", max_new_tokens=8, provision_timeout_s=0.3)
                finally:
                    await client.close()

        # No busy-loop: the malformed -10 / nan hint was discarded, so the
        # default 5s delay applies; it exceeds the 0.3s budget, so the SDK
        # raises the typed ModelLoadingError immediately — no sleeps, single
        # attempt. (Before the fix the hint was passed through and asyncio
        # slept instantly, busy-spinning for hundreds of attempts.)
        assert slept == []
        assert all(math.isfinite(d) and d >= 0 for d in slept)
        assert client._post.call_count == 1
