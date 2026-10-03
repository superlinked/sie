"""Absolute socket budgets stop header and chunk-framing trickle attacks."""

import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from sie_server.core.upstream_deadline import DEADLINE_EXTENSION, DeadlineTransport


@contextmanager
def trickling_server(prefix):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    stop, finished = threading.Event(), threading.Event()

    def serve():
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(1)
                connection.recv(65536)
                connection.sendall(prefix)
                while not stop.wait(0.015):
                    connection.sendall(b"1")
        except (OSError, TimeoutError):
            pass
        finally:
            finished.set()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        stop.set()
        listener.close()
        worker.join(3)
        assert finished.is_set()


@pytest.mark.parametrize(
    "prefix",
    [
        b"HTTP/1.1 200 OK\r\nX-Incomplete: ",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
    ],
)
def test_deadline_covers_header_and_chunk_framing_reads(prefix):
    with trickling_server(prefix) as url, httpx.Client(transport=DeadlineTransport(), timeout=1) as client:
        request = client.build_request("GET", url)
        started = time.monotonic()
        request.extensions[DEADLINE_EXTENSION] = started + 0.15
        with pytest.raises(httpx.ReadTimeout, match="upstream transport failed"):
            client.send(request)
        assert time.monotonic() - started < 0.7


def test_expired_request_sends_nothing_and_does_not_poison_next_request():
    with trickling_server(b"HTTP/1.1 200 OK\r\nX-Incomplete: ") as url:
        with httpx.Client(transport=DeadlineTransport(), timeout=1) as client:
            request = client.build_request("GET", url)
            request.extensions[DEADLINE_EXTENSION] = time.monotonic() - 1
            with pytest.raises(httpx.PoolTimeout):
                client.send(request)
            request = client.build_request("GET", url)
            request.extensions[DEADLINE_EXTENSION] = time.monotonic() + 0.15
            with pytest.raises(httpx.ReadTimeout):
                client.send(request)
