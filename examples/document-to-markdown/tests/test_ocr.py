from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import msgpack
import pytest
from sie_sdk import RequestError

from document_to_markdown.ocr import convert_image

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN3cAAAAASUVORK5CYII=")


@pytest.mark.parametrize("url", ["http://localhost:8080", "https://chosen.example/gateway"])
def test_original_png_options_and_markdown_through_actual_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(PNG)
    calls: list[httpx.Request] = []
    clients: list[httpx.Client] = []
    markdown = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
    client_type = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            content=msgpack.packb({"items": [{"entities": [], "data": {"markdown": markdown}}]}, use_bin_type=True),
            headers={"Content-Type": "application/msgpack"},
        )

    def create_client(**kwargs: Any) -> httpx.Client:
        client = client_type(transport=httpx.MockTransport(reply), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("sie_sdk.client.sync.httpx.Client", create_client)
    monkeypatch.setenv("SIE_API_KEY", "test-key")
    assert convert_image(image, sie_url=url, model="example/model") == markdown
    assert len(calls) == 1
    assert str(calls[0].url) == f"{url}/v1/extract/example/model"
    assert calls[0].headers["Authorization"] == "Bearer test-key"
    body = msgpack.unpackb(calls[0].content, raw=False)
    assert body["items"][0]["images"][0]["data"] == PNG
    assert body["params"]["options"] == {"max_new_tokens": 4096, "temperature": 0.1, "top_p": 1.0}
    assert clients[0].is_closed


def test_terminal_error_closes_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(PNG)
    clients: list[httpx.Client] = []
    client_type = httpx.Client

    def create_client(**kwargs: Any) -> httpx.Client:
        transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "invalid request"}))
        client = client_type(transport=transport, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("sie_sdk.client.sync.httpx.Client", create_client)
    with pytest.raises(RequestError):
        convert_image(image, sie_url="https://chosen.example")
    assert clients[0].is_closed
