from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse


def _extract_response() -> bytes:
    return msgpack.packb({"model": "whisper", "items": [{"entities": []}]}, use_bin_type=True)


def test_sync_extract_serializes_audio_and_preserves_other_media() -> None:
    response = MagicMock(status_code=200, content=_extract_response())

    with patch("sie_sdk.client.sync.httpx.Client") as http_client:
        http_client.return_value.post.return_value = response
        client = SIEClient("http://localhost:8080")
        client.extract(
            "whisper",
            {
                "audio": {
                    "data": b"RIFF fake wav",
                    "format": "wav",
                    "sample_rate": 16_000,
                },
                "images": [b"\xff\xd8\xff\xe0"],
                "document": b"%PDF-1.4",
            },
        )

        body = http_client.return_value.post.call_args.kwargs["content"]
        item = msgpack.unpackb(body, raw=False)["items"][0]
        assert item["audio"] == {
            "data": b"RIFF fake wav",
            "format": "wav",
            "sample_rate": 16_000,
        }
        assert item["images"] == [{"data": b"\xff\xd8\xff\xe0", "format": "jpeg"}]
        assert item["document"] == {"data": b"%PDF-1.4", "format": None}
        client.close()


@pytest.mark.asyncio
async def test_async_extract_serializes_audio_path(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"ID3 fake mp3")

    client = SIEAsyncClient("http://localhost:8080")
    post = AsyncMock(return_value=_AioResponse(200, _extract_response(), {}))
    with patch.object(client, "_post", post):
        await client.extract("whisper", {"audio": path})

    body = post.call_args.kwargs["data"]
    item = msgpack.unpackb(body, raw=False)["items"][0]
    assert item["audio"] == {
        "data": b"ID3 fake mp3",
        "format": "mp3",
        "sample_rate": None,
    }
    await client.close()
