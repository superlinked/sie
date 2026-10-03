"""Outbound HTTP client for an operator-defined upstream.

The client never follows a redirect, ignores ambient proxy variables, verifies
TLS, and reads the credential from its environment variable for each request,
so no long-lived object holds the value.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx

from sie_server.config.upstreams import Upstream
from sie_server.core.upstream_deadline import DeadlineTransport

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class UpstreamRedirectRefusedError(httpx.HTTPError):
    """The upstream answered with a redirect. Credentials never follow one."""


class _EnvBearerAuth(httpx.Auth):
    def __init__(self, upstream: Upstream) -> None:
        self._upstream = upstream

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        api_key = self._upstream.api_key()
        if api_key is not None:
            request.headers["Authorization"] = f"Bearer {api_key}"
        yield request


_REDIRECT_REFUSED = "upstream answered with a redirect, which is refused"


async def _refuse_redirect(response: httpx.Response) -> None:
    if 300 <= response.status_code < 400:
        await response.aclose()
        raise UpstreamRedirectRefusedError(_REDIRECT_REFUSED)


def _refuse_redirect_sync(response: httpx.Response) -> None:
    if 300 <= response.status_code < 400:
        response.close()
        raise UpstreamRedirectRefusedError(_REDIRECT_REFUSED)


def upstream_client(
    upstream: Upstream,
    *,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the async egress client; an explicit transport owns proxy/TLS policy."""
    return httpx.AsyncClient(
        base_url=upstream.base_url,
        auth=_EnvBearerAuth(upstream),
        follow_redirects=False,
        trust_env=False,
        proxy=upstream.proxy_url if transport is None else None,
        verify=True,
        timeout=timeout,
        transport=transport,
        event_hooks={"response": [_refuse_redirect]},
    )


def upstream_sync_client(
    upstream: Upstream,
    *,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build the synchronous egress client; an explicit transport owns proxy/TLS policy."""
    transport = transport if transport is not None else DeadlineTransport(proxy=upstream.proxy_url)
    return httpx.Client(
        base_url=upstream.base_url,
        auth=_EnvBearerAuth(upstream),
        follow_redirects=False,
        trust_env=False,
        verify=True,
        timeout=timeout,
        transport=transport,
        event_hooks={"response": [_refuse_redirect_sync]},
    )
