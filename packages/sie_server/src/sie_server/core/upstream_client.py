"""Outbound HTTP client for an operator-defined upstream.

The client never follows a redirect, ignores ambient proxy variables, verifies
TLS, and reads the credential from its environment variable for each request,
so no long-lived object holds the value.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx

from sie_server.config.upstreams import Upstream

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


async def _refuse_redirect(response: httpx.Response) -> None:
    if 300 <= response.status_code < 400:
        await response.aclose()
        raise UpstreamRedirectRefusedError("upstream answered with a redirect, which is refused")


def upstream_client(
    upstream: Upstream,
    *,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the only client that may call ``upstream``."""
    return httpx.AsyncClient(
        base_url=upstream.base_url,
        auth=_EnvBearerAuth(upstream),
        follow_redirects=False,
        trust_env=False,
        proxy=upstream.proxy_url,
        verify=True,
        timeout=timeout,
        transport=transport,
        event_hooks={"response": [_refuse_redirect]},
    )
