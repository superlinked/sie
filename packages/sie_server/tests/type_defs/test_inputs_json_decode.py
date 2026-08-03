"""Wire-format decoding tests for media inputs (regression for #1026).

`POST /v1/{encode,score,extract}/{model}` with `Content-Type: application/json`
and a base64-encoded image must base64-decode the inner `data` field into
`bytes`, exactly like the msgpack path. msgspec only base64-decodes a JSON
string when the target field is *typed* `bytes` — so ``Item`` must reference the
typed media input definitions (whose ``data`` is ``bytes``) rather than
``dict[str, Any]``. With ``Any``, the string flows straight into the
preprocessor's ``io.BytesIO(...)`` and raises
``TypeError: a bytes-like object is required, not 'str'``.
"""

import base64
from typing import Any

import msgspec
import pytest
from sie_server.types.inputs import Item, is_image_input
from sie_server.types.requests import EncodeRequest, ExtractRequest, ScoreRequest

# A tiny but real 1x1 PNG so the bytes are non-trivial and round-trip cleanly.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


def test_json_image_data_decodes_to_bytes() -> None:
    """Repro for #1026: JSON image `data` must be base64-decoded to bytes, not str."""
    body = msgspec.json.encode({"items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}]})

    request = msgspec.json.decode(body, type=EncodeRequest)

    img = request.items[0].images[0]
    assert isinstance(img["data"], bytes), "JSON image data must be base64-decoded to bytes"
    assert img["data"] == _PNG_BYTES
    # The guard every preprocessor relies on must accept the decoded payload.
    assert is_image_input(img)


def test_msgpack_image_data_stays_bytes() -> None:
    """The msgpack path already works; lock it in so the fix keeps parity."""
    body = msgspec.msgpack.encode({"items": [{"id": "t", "images": [{"data": _PNG_BYTES, "format": "png"}]}]})

    request = msgspec.msgpack.decode(body, type=EncodeRequest)

    img = request.items[0].images[0]
    assert isinstance(img["data"], bytes)
    assert img["data"] == _PNG_BYTES


@pytest.mark.parametrize("request_type", [EncodeRequest, ExtractRequest])
def test_json_image_data_decodes_to_bytes_all_request_types(request_type: type) -> None:
    """Every items-bearing request type must base64-decode image data on JSON."""
    body = msgspec.json.encode({"items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}]})

    request = msgspec.json.decode(body, type=request_type)

    assert isinstance(request.items[0].images[0]["data"], bytes)


def test_json_score_query_and_items_image_data_decodes_to_bytes() -> None:
    """ScoreRequest carries images on both `query` and `items`."""
    body = msgspec.json.encode(
        {
            "query": {"id": "q", "images": [{"data": _PNG_B64, "format": "png"}]},
            "items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}],
        }
    )

    request = msgspec.json.decode(body, type=ScoreRequest)

    assert isinstance(request.query.images[0]["data"], bytes)
    assert isinstance(request.items[0].images[0]["data"], bytes)


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        pytest.param(msgspec.json.encode, msgspec.json.decode, id="json"),
        pytest.param(msgspec.msgpack.encode, msgspec.msgpack.decode, id="msgpack"),
    ],
)
def test_encode_request_still_decodes_with_runtime_options(encode, decode) -> None:
    """Score-only nested validation must not affect EncodeRequest construction."""
    body = encode(
        {
            "items": [{"text": "hello"}],
            "params": {"options": {"is_query": True}},
        }
    )

    request = decode(body, type=EncodeRequest)

    assert request.params is not None
    assert request.params.options == {"is_query": True}


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        pytest.param(msgspec.json.encode, msgspec.json.decode, id="json"),
        pytest.param(msgspec.msgpack.encode, msgspec.msgpack.decode, id="msgpack"),
    ],
)
@pytest.mark.parametrize("instruction", [None, "", "rank by relevance"])
def test_score_options_instruction_accepts_only_valid_values(encode, decode, instruction: str | None) -> None:
    body = encode(
        {
            "query": {"text": "query"},
            "items": [{"text": "document"}],
            "options": {"instruction": instruction},
        }
    )

    request = decode(body, type=ScoreRequest)

    assert request.options == {"instruction": instruction}


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        pytest.param(msgspec.json.encode, msgspec.json.decode, id="json"),
        pytest.param(msgspec.msgpack.encode, msgspec.msgpack.decode, id="msgpack"),
    ],
)
@pytest.mark.parametrize("instruction", [False, 7, 1.5, [], {}])
def test_score_options_instruction_rejects_non_string_values(encode, decode, instruction: object) -> None:
    body = encode(
        {
            "query": {"text": "query"},
            "items": [{"text": "document"}],
            "instruction": "valid top-level value does not bypass nested validation",
            "options": {"instruction": instruction},
        }
    )

    with pytest.raises(msgspec.ValidationError, match=r"\$\.options\.instruction"):
        decode(body, type=ScoreRequest)


def test_msgpack_score_options_instruction_rejects_binary_value() -> None:
    body = msgspec.msgpack.encode(
        {
            "query": {"text": "query"},
            "items": [{"text": "document"}],
            "options": {"instruction": b"binary-is-not-a-string"},
        }
    )

    with pytest.raises(msgspec.ValidationError, match=r"\$\.options\.instruction"):
        msgspec.msgpack.decode(body, type=ScoreRequest)


def test_json_audio_video_document_data_decodes_to_bytes() -> None:
    """The sibling media fields share the same latent bug; lock the contract."""
    body = msgspec.json.encode(
        {
            "items": [
                {
                    "id": "t",
                    "audio": {"data": _PNG_B64, "format": "wav", "sample_rate": 16000},
                    "video": {"data": _PNG_B64, "format": "mp4"},
                    "document": {"data": _PNG_B64, "format": "pdf"},
                }
            ]
        }
    )

    item: Item = msgspec.json.decode(body, type=EncodeRequest).items[0]

    assert isinstance(item.audio["data"], bytes)
    assert isinstance(item.video["data"], bytes)
    assert isinstance(item.document["data"], bytes)


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_audio_rejects_unknown_fields(codec: Any) -> None:
    data = _PNG_B64 if codec is msgspec.json else _PNG_BYTES
    body = codec.encode(
        {"items": [{"audio": {"data": data, "format": "wav", "sample_rate": 16_000, "surprise": True}}]}
    )

    with pytest.raises(msgspec.ValidationError, match="unknown field `surprise`"):
        codec.decode(body, type=ExtractRequest)


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_audio_requires_binary_data(codec: Any) -> None:
    body = codec.encode({"items": [{"audio": {"format": "wav", "sample_rate": 16_000}}]})

    with pytest.raises(msgspec.ValidationError, match="Object missing required field `data`"):
        codec.decode(body, type=ExtractRequest)
