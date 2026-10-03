"""Synchronous HTTP transport with a deadline across individual socket operations."""

from __future__ import annotations

import ssl
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import httpcore
import httpx

DEADLINE_EXTENSION = "sie_deadline"
_DEADLINE: ContextVar[float | None] = ContextVar("upstream_deadline", default=None)
SocketOption = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


def _timeout(timeout: float | None, error: type[Exception]) -> float | None:
    deadline = _DEADLINE.get()
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise error("upstream request exceeded its deadline")
    return remaining if timeout is None else min(timeout, remaining)


@contextmanager
def _request_deadline(deadline: float | None) -> Iterator[None]:
    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


@contextmanager
def _httpx_errors() -> Iterator[None]:
    try:
        yield
    except (
        httpcore.TimeoutException,
        httpcore.NetworkError,
        httpcore.ProxyError,
        httpcore.UnsupportedProtocol,
        httpcore.ProtocolError,
    ) as exc:
        for name in (
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "ConnectError",
            "ReadError",
            "WriteError",
            "ProxyError",
            "UnsupportedProtocol",
            "LocalProtocolError",
            "RemoteProtocolError",
            "TimeoutException",
            "NetworkError",
            "ProtocolError",
        ):
            if isinstance(exc, getattr(httpcore, name)):
                raise getattr(httpx, name)("upstream transport failed") from exc
        raise httpx.TransportError("upstream transport failed") from exc


class _DeadlineStream(httpcore.NetworkStream):
    def __init__(self, stream: httpcore.NetworkStream) -> None:
        self._stream = stream

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._stream.read(max_bytes, timeout=_timeout(timeout, httpcore.ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(buffer, timeout=_timeout(timeout, httpcore.WriteTimeout))

    def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._stream.start_tls(
                ssl_context, server_hostname=server_hostname, timeout=_timeout(timeout, httpcore.ConnectTimeout)
            )
        )

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)

    def close(self) -> None:
        self._stream.close()


class _DeadlineBackend(httpcore.NetworkBackend):
    def __init__(self) -> None:
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._backend.connect_tcp(
                host,
                port,
                timeout=_timeout(timeout, httpcore.ConnectTimeout),
                local_address=local_address,
                socket_options=socket_options,
            )
        )

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Iterable[SocketOption] | None = None
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._backend.connect_unix_socket(
                path,
                timeout=_timeout(timeout, httpcore.ConnectTimeout),
                socket_options=socket_options,
            )
        )

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(_timeout(seconds, httpcore.ConnectTimeout) or 0)


class _ResponseStream(httpx.SyncByteStream):
    def __init__(self, stream: Iterable[bytes], deadline: float | None) -> None:
        self._stream, self._deadline = stream, deadline

    def __iter__(self) -> Iterator[bytes]:
        iterator = iter(self._stream)
        while True:
            with _request_deadline(self._deadline), _httpx_errors():
                try:
                    part = next(iterator)
                except StopIteration:
                    return
            yield part

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class DeadlineTransport(httpx.BaseTransport):
    """Verified TLS, declared proxy, no retries; pool and socket reads share the request deadline.

    Each network operation receives the remaining budget, including reads of
    incomplete headers and chunk framing. OS hostname resolution still follows
    the platform resolver's own timeout.
    """

    def __init__(self, *, proxy: str | None = None) -> None:
        ssl_context = httpx.create_ssl_context(verify=True, trust_env=False)
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_context,
            proxy=(
                httpcore.Proxy(proxy, ssl_context=ssl_context if httpx.URL(proxy).scheme == "https" else None)
                if proxy
                else None
            ),
            network_backend=_DeadlineBackend(),
            retries=0,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.SyncByteStream)
        deadline = request.extensions.get(DEADLINE_EXTENSION)
        extensions = dict(request.extensions)
        with _request_deadline(deadline), _httpx_errors():
            extensions["timeout"] = {
                name: _timeout(value, httpcore.PoolTimeout) for name, value in extensions.get("timeout", {}).items()
            }
            response = self._pool.handle_request(
                httpcore.Request(
                    request.method,
                    httpcore.URL(
                        scheme=request.url.raw_scheme,
                        host=request.url.raw_host,
                        port=request.url.port,
                        target=request.url.raw_path,
                    ),
                    headers=request.headers.raw,
                    content=request.stream,
                    extensions=extensions,
                )
            )
        assert isinstance(response.stream, Iterable)
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_ResponseStream(cast("Iterable[bytes]", response.stream), deadline),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._pool.close()
