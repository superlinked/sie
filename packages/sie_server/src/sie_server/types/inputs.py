"""Input types for SIE Server API (wire format).

These types define the structure of items received over the wire after msgpack
deserialization. The SDK converts flexible Python types (PIL.Image, numpy arrays,
file paths) to these wire format types before transport.

Mapping-shaped media uses TypedDict where permissive compatibility is required;
audio uses a strict msgspec struct because it crosses a bounded binary boundary.
"""

from typing import Any, Literal, TypedDict, TypeGuard, cast, overload

import msgspec


class ImageInput(TypedDict, total=False):
    """Image input for multimodal models (wire format).

    On the wire, images are sent as bytes with format hint.

    Attributes:
        data: Image data as bytes.
        format: Image format hint: 'jpeg', 'png', etc. Inferred if not provided.
    """

    data: bytes
    format: str | None


class AudioInput(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Audio input for audio models (wire format).

    On the wire, audio is sent as bytes with format and sample rate metadata.

    Attributes:
        data: Audio data as bytes.
        format: Audio format: 'wav', 'mp3', etc.
        sample_rate: Sample rate in Hz.
    """

    data: bytes
    format: str | None = None
    sample_rate: int | None = None

    @overload
    def __getitem__(self, key: Literal["data"]) -> bytes:
        pass

    @overload
    def __getitem__(self, key: Literal["format"]) -> str | None:
        pass

    @overload
    def __getitem__(self, key: Literal["sample_rate"]) -> int | None:
        pass

    def __getitem__(self, key: str) -> object:
        if key not in {"data", "format", "sample_rate"}:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        if key not in {"data", "format", "sample_rate"}:
            return default
        return getattr(self, key)


class VideoInput(TypedDict, total=False):
    """Video input for video models (wire format).

    On the wire, video is sent as bytes with format hint.

    Attributes:
        data: Video data as bytes.
        format: Video format: 'mp4', 'webm', etc.
    """

    data: bytes
    format: str | None


class DocumentInput(TypedDict, total=False):
    """Document input for composite-document extractors (wire format).

    On the wire, documents are sent as bytes with a format hint
    (e.g., 'pdf', 'docx', 'html'). The hint is advisory — adapters may
    still sniff the bytes when format is missing or unrecognized.

    Attributes:
        data: Document bytes (raw file content).
        format: Document format hint: 'pdf', 'docx', 'html', etc.
    """

    data: bytes
    format: str | None


class Item(msgspec.Struct):
    """A single item to encode, score, or extract from.

    All fields are optional. Models accept text-only, image-only, or multimodal
    items depending on their capabilities.
    """

    id: str | None = None
    text: str | None = None
    # These reference typed media inputs (data: bytes) rather than dict[str, Any]
    # so msgspec base64-decodes the `data` field on the JSON path, matching the
    # msgpack path. See issue #1026.
    images: list[ImageInput] | None = None
    audio: AudioInput | None = None
    video: VideoInput | None = None
    document: DocumentInput | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# Type Guards
# =============================================================================


def is_image_input(obj: Any) -> TypeGuard[ImageInput]:
    """Check if obj is a valid ImageInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_audio_input(obj: Any) -> TypeGuard[AudioInput]:
    """Check if obj is a valid AudioInput struct or compatible mapping.

    Args:
        obj: Object to validate.

    Returns:
        True if obj carries a 'data' field containing bytes.
    """
    if isinstance(obj, AudioInput):
        return isinstance(obj.data, bytes)
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_video_input(obj: Any) -> TypeGuard[VideoInput]:
    """Check if obj is a valid VideoInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_document_input(obj: Any) -> TypeGuard[DocumentInput]:
    """Check if obj is a valid DocumentInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_item(obj: Any) -> TypeGuard[Item | dict[str, Any]]:
    """Check if obj is a valid Item or Item-like dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is an Item Struct or a dict.
    """
    return isinstance(obj, (dict, Item))


# =============================================================================
# Validated media access
# =============================================================================


class InvalidInputError(ValueError):
    """A caller-controlled inference input violates an adapter contract.

    This typed boundary distinguishes invalid requests from internal inference
    failures on both the direct HTTP and queue/sidecar paths.
    """


class InvalidMediaError(InvalidInputError):
    """A media input's ``data`` field is missing or not bytes.

    Subclasses :class:`InvalidInputError` so both request paths surface it as
    structured ``INVALID_INPUT`` (HTTP 400) rather than a generic 500:

    - HTTP / in-process: the endpoints' ``except ValueError`` routes it through
      ``InferenceErrorHandler.handle_value_error`` (see ``api/encode.py``).
    - Queue / sidecar: ``queue_executor._inference_exception_outcome`` maps it
      to ``ErrorCode.INVALID_INPUT`` before the sidecar publishes the result.

    Without this, an un-decoded base64 ``str`` slipping past the wire boundary
    raised ``TypeError: a bytes-like object is required, not 'str'`` deep inside
    a preprocessor/adapter — a generic 500, and one trigger for a malformed
    tensor reaching a CUDA kernel. See issue #1026.
    """


def media_bytes(media: object, *, kind: str = "media") -> bytes:
    """Return the validated ``data`` bytes from a media input mapping.

    Every image/video/document consumer relies on the wire contract that
    ``data`` is raw ``bytes``. msgspec only enforces that on a typed ``bytes``
    field, and only on decode paths that run through it — the queue path builds
    ``Item`` from a plain dict and bypasses that check. This is the single
    enforcement point all consumers funnel through, turning any contract
    violation into a clean :class:`InvalidMediaError` at the point of use.

    Args:
        media: The media input mapping (e.g. an :class:`ImageInput`).
        kind: Human label used in the error message ("image", "document", ...).

    Returns:
        The ``data`` as ``bytes`` (``bytearray``/``memoryview`` are coerced).

    Raises:
        InvalidMediaError: If ``media`` is not a mapping, lacks ``data``, or
            ``data`` is not a bytes-like object.
    """
    if isinstance(media, AudioInput):
        data = media.data
        if isinstance(data, bytes):
            return data
        raise InvalidMediaError(
            f"{kind} data must be bytes, got {type(data).__name__} "
            "(base64 JSON strings must be decoded to bytes before inference)"
        )
    if not isinstance(media, dict):
        raise InvalidMediaError(f"{kind} input must be a mapping with a 'data' field")
    mapping = cast("dict[str, Any]", media)
    if "data" not in mapping:
        raise InvalidMediaError(f"{kind} input must be a mapping with a 'data' field")
    data = mapping["data"]
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise InvalidMediaError(
        f"{kind} data must be bytes, got {type(data).__name__} "
        "(base64 JSON strings must be decoded to bytes before inference)"
    )


# =============================================================================
# Validated item decode
# =============================================================================


def decode_item(raw: dict[str, Any]) -> Item:
    """Validate a wire-format item mapping into a typed :class:`Item`.

    The HTTP ingress path decodes request bodies straight into typed msgspec
    structs (``msgspec.json/msgpack.decode(body, type=...)``), so an item whose
    fields have the wrong type is rejected at the seam with ``INVALID_INPUT``.
    The queue / IPC path receives each item as a plain ``dict``
    (``msgpack.unpackb``) and historically built ``Item(**kwargs)``, which
    performs *no* type validation — a malformed item slipped through to fail
    deep inside a preprocessor/adapter (see issue #1026).

    This is the single validating decode for dict-sourced items: it runs the
    same ``Item`` contract through :func:`msgspec.convert`, so the queue path
    rejects type violations exactly where the HTTP path does and base64-decodes
    ``str`` media ``data`` to ``bytes`` the same way the JSON path does. The
    Permissive ``*Input`` TypedDicts are ``total=False``, so missing/empty
    media ``data`` is still enforced at point of use by :func:`media_bytes`;
    the strict audio struct rejects unknown fields during conversion.

    Args:
        raw: The wire-format item mapping. The SDK ships ``content`` as an
            alias for ``text``; it is remapped when ``text`` is absent.

    Returns:
        A validated :class:`Item`.

    Raises:
        msgspec.ValidationError: If a field is present with the wrong type.
            Both ingress paths surface this as ``INVALID_INPUT`` (HTTP 400).
    """
    if "content" in raw and "text" not in raw:
        raw = {**raw, "text": raw["content"]}
    return msgspec.convert(raw, type=Item)
