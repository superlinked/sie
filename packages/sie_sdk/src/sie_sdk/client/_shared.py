from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any, Protocol

import msgpack
import numpy as np


class _HttpResponse(Protocol):
    """Structural type for HTTP responses (httpx.Response or _AioResponse)."""

    status_code: int
    content: bytes
    headers: Any

    @property
    def text(self) -> str: ...
    def json(self) -> Any: ...


from sie_sdk.types import (
    Classification,
    DetectedObject,
    EncodeResult,
    EntityResult,
    ExtractResult,
    Relation,
    ScoreEntry,
    ScoreResult,
    SparseResult,
)

from .errors import RequestError, ServerError

# Content types
MSGPACK_CONTENT_TYPE = "application/msgpack"
JSON_CONTENT_TYPE = "application/json"

# HTTP status code thresholds
HTTP_ACCEPTED = 202
HTTP_CLIENT_ERROR = 400
HTTP_SERVER_ERROR = 500

# Default provisioning settings
DEFAULT_PROVISION_TIMEOUT_S = 900.0  # 15 minutes
DEFAULT_RETRY_DELAY_S = 5.0  # Retry every 5 seconds if no Retry-After header

# Pool settings
DEFAULT_LEASE_RENEWAL_INTERVAL_S = 60.0  # Renew lease every 60s (lease is 1200s)

# LoRA loading retry settings
LORA_LOADING_MAX_RETRIES = 10  # Max retries for LoRA loading (usually completes in 1-2s)
LORA_LOADING_DEFAULT_DELAY_S = 1.0  # Default retry delay if no Retry-After header
LORA_LOADING_ERROR_CODE = "LORA_LOADING"  # Error code from server

# Model loading retry settings
MODEL_LOADING_MAX_RETRIES = 60  # Max retries (60 * 5s = 5 min, matches provision timeout)
MODEL_LOADING_DEFAULT_DELAY_S = 5.0  # Default retry delay (model loads take longer than LoRA)
MODEL_LOADING_ERROR_CODE = "MODEL_LOADING"  # Error code from server

# Version negotiation headers
SDK_VERSION_HEADER = "X-SIE-SDK-Version"
SERVER_VERSION_HEADER = "X-SIE-Server-Version"


def get_sdk_version() -> str:
    try:
        return pkg_version("sie-sdk")
    except PackageNotFoundError:
        return "unknown"


def check_version_skew(sdk_version: str, server_version: str) -> str | None:
    try:
        sdk_parts = sdk_version.split(".")
        server_parts = server_version.split(".")
        if len(sdk_parts) < 2 or len(server_parts) < 2:
            return None

        sdk_major, sdk_minor = int(sdk_parts[0]), int(sdk_parts[1])
        server_major, server_minor = int(server_parts[0]), int(server_parts[1])

        if sdk_major != server_major:
            return (
                f"SDK version {sdk_version} has different major version than server {server_version}. Please upgrade."
            )

        if abs(sdk_minor - server_minor) > 1:
            return (
                f"SDK version {sdk_version} is more than one minor version "
                f"{'behind' if sdk_minor < server_minor else 'ahead of'} "
                f"server {server_version}. Consider upgrading."
            )
    except (ValueError, IndexError):
        pass
    return None


def parse_gpu_param(gpu: str) -> tuple[str | None, str]:
    """Parse GPU parameter to extract pool name and GPU type.

    Args:
        gpu: GPU string, either "pool_name/gpu_type" or just "gpu_type".

    Returns:
        Tuple of (pool_name, gpu_type). pool_name is None if not specified.

    Examples:
        >>> parse_gpu_param("eval-bench/l4")
        ("eval-bench", "l4")
        >>> parse_gpu_param("l4")
        (None, "l4")
    """
    if "/" in gpu:
        parts = gpu.split("/", 1)
        return parts[0], parts[1]
    return None, gpu


def get_retry_after(response: _HttpResponse) -> float | None:
    """Extract Retry-After header value from response.

    Args:
        response: HTTP response that may contain Retry-After header.

    Returns:
        Retry delay in seconds, or None if header not present.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            return None
    return None


def get_error_code(response: _HttpResponse) -> str | None:
    """Extract error code from response body.

    Args:
        response: HTTP response to parse.

    Returns:
        Error code string, or None if not found.
    """
    try:
        if MSGPACK_CONTENT_TYPE in response.headers.get("content-type", ""):
            data = msgpack.unpackb(response.content, raw=False)
        else:
            data = response.json()

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                return error.get("code")
            return None  # error is a string, no code
        if "detail" in data:
            detail = data["detail"]
            if isinstance(detail, dict):
                return detail.get("code")
    except (ValueError, KeyError, TypeError):
        pass
    return None


def handle_error(response: _HttpResponse) -> None:
    """Handle error response from server.

    Raises:
        RequestError: For 4xx responses.
        ServerError: For 5xx responses.
    """
    code = None
    message = f"HTTP {response.status_code}"

    try:
        # Try msgpack first
        if MSGPACK_CONTENT_TYPE in response.headers.get("content-type", ""):
            data = msgpack.unpackb(response.content, raw=False)
        else:
            data = response.json()

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message", message)
            else:
                # error is a string, use it as the message
                message = str(error)
        elif "detail" in data:
            detail = data["detail"]
            if isinstance(detail, dict):
                code = detail.get("code")
                message = detail.get("message", str(detail))
            else:
                message = str(detail)
    except (ValueError, KeyError, TypeError):
        # Fall back to raw text
        message = response.text or message

    if response.status_code >= HTTP_SERVER_ERROR:
        raise ServerError(message, code=code, status_code=response.status_code)
    raise RequestError(message, code=code, status_code=response.status_code)


def parse_encode_results(items: list[dict[str, Any]]) -> list[EncodeResult]:
    """Parse encode response items into EncodeResult TypedDicts.

    Extracts numpy arrays from the wire format. Arrays are expected to be
    numpy arrays from msgpack-numpy deserialization.
    """
    results: list[EncodeResult] = []

    for item in items:
        result: EncodeResult = {}

        # Copy id if present
        if "id" in item:
            result["id"] = item["id"]

        # Extract dense embedding (may be None if not requested)
        if "dense" in item and item["dense"] is not None:
            values = item["dense"]["values"]
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["dense"] = values

        # Extract sparse embedding (may be None if not requested)
        if "sparse" in item and item["sparse"] is not None:
            sparse = item["sparse"]
            indices = sparse["indices"]
            values = sparse["values"]
            assert isinstance(indices, np.ndarray), "Expected numpy array from msgpack-numpy"
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["sparse"] = SparseResult(indices=indices, values=values)

        # Extract multivector embedding (may be None if not requested)
        if "multivector" in item and item["multivector"] is not None:
            values = item["multivector"]["values"]
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["multivector"] = values

        results.append(result)

    return results


def parse_score_result(data: dict[str, Any]) -> ScoreResult:
    """Parse score response into ScoreResult TypedDict."""
    result: ScoreResult = {
        "model": data["model"],
        "scores": [
            ScoreEntry(
                item_id=s["item_id"],
                score=s["score"],
                rank=s["rank"],
            )
            for s in data["scores"]
        ],
    }
    if data.get("query_id") is not None:
        result["query_id"] = data["query_id"]
    return result


def parse_extract_results(items: list[dict[str, Any]]) -> list[ExtractResult]:
    """Parse extract response items into ExtractResult TypedDicts."""
    results: list[ExtractResult] = []

    for item in items:
        result: ExtractResult = {
            "entities": [
                EntityResult(
                    text=e["text"],
                    label=e["label"],
                    score=e["score"],
                    start=e.get("start"),
                    end=e.get("end"),
                    bbox=e.get("bbox"),
                )
                for e in item.get("entities", [])
            ],
            "relations": [
                Relation(
                    head=r["head"],
                    tail=r["tail"],
                    relation=r["relation"],
                    score=r["score"],
                )
                for r in item.get("relations", [])
            ],
            "classifications": [
                Classification(label=c["label"], score=c["score"]) for c in item.get("classifications", [])
            ],
            "objects": [
                DetectedObject(label=o["label"], score=o["score"], bbox=o["bbox"]) for o in item.get("objects", [])
            ],
        }

        # Copy optional fields
        if item.get("id") is not None:
            result["id"] = item["id"]
        if item.get("data"):
            result["data"] = item["data"]

        results.append(result)

    return results
