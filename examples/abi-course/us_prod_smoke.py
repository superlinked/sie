# ruff: noqa: INP001
"""Make one inexpensive, deterministic embeddings request to managed SIE."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, TextIO

API_KEY_ENV = "SIE_API_KEY"
BASE_URL_ENV = "SIE_BASE_URL"
MODEL_ENV = "SIE_MODEL"
DEFAULT_MODEL = "BAAI/bge-m3"
SMOKE_INPUT = "ABI course managed SIE smoke test."
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 30.0
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 599

Transport = Callable[[urllib.request.Request, float], Any]


class SmokeError(Exception):
    """Expected smoke-test failure with a stable, secret-free error code."""

    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "error": self.code,
            "message": self.message,
        }
        if self.status is not None:
            result["status"] = self.status
        return result


@dataclass(frozen=True)
class SmokeConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SmokeConfig:
        api_key = env.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise SmokeError(
                "CONFIG_MISSING",
                f"Set {API_KEY_ENV} in the environment; do not put it in this file.",
            )

        base_url = env.get(BASE_URL_ENV, "").strip()
        if not base_url:
            raise SmokeError(
                "CONFIG_MISSING",
                f"Set {BASE_URL_ENV} to the managed API HTTPS origin.",
            )

        model = env.get(MODEL_ENV, DEFAULT_MODEL).strip()
        if not model:
            raise SmokeError("CONFIG_INVALID", f"{MODEL_ENV} cannot be empty.")

        return cls(api_key=api_key, base_url=_validate_base_url(base_url), model=model)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the bearer credential on the explicitly configured origin."""

    def redirect_request(
        self,
        _req: urllib.request.Request,
        _fp: Any,
        _code: int,
        _msg: str,
        _headers: Any,
        _newurl: str,
    ) -> None:
        return None


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeError(
            "CONFIG_INVALID",
            f"{BASE_URL_ENV} must be a bare HTTPS origin, for example https://api.superlinked.com.",
        )
    return value.rstrip("/")


def _default_transport(request: urllib.request.Request, timeout: float) -> Any:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _request(config: SmokeConfig) -> urllib.request.Request:
    body = json.dumps(
        {
            "model": config.model,
            "input": SMOKE_INPUT,
            "encoding_format": "float",
        },
        separators=(",", ":"),
    ).encode()
    return urllib.request.Request(  # noqa: S310 - origin is HTTPS-only validated.
        f"{config.base_url}/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )


def _fail_http(status: int) -> NoReturn:
    failures = {
        401: (
            "AUTH_INVALID",
            "The API key is missing, invalid, or no longer active.",
        ),
        402: (
            "CREDITS_EXHAUSTED",
            "The account does not have enough credits for this request.",
        ),
        403: (
            "ACCESS_FORBIDDEN",
            "The API key is not permitted to use this model or endpoint.",
        ),
        404: (
            "MODEL_UNAVAILABLE",
            "The configured model is not available on this deployment.",
        ),
        429: (
            "RATE_LIMITED",
            "The service rate limit was reached; retry later.",
        ),
    }
    if status in failures:
        code, message = failures[status]
    elif SERVER_ERROR_MIN <= status <= SERVER_ERROR_MAX:
        code, message = (
            "SERVICE_ERROR",
            "The managed service returned a server error; retry later.",
        )
    else:
        code, message = (
            "REQUEST_REJECTED",
            "The managed service rejected the smoke request.",
        )
    raise SmokeError(code, message, status=status)


def _read_json(response: Any) -> dict[str, Any]:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SmokeError("INVALID_RESPONSE", "The response exceeded the size limit.")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(
            "INVALID_RESPONSE",
            "The service did not return valid JSON.",
        ) from error
    if not isinstance(payload, dict):
        raise SmokeError(
            "INVALID_RESPONSE",
            "The service returned an unexpected JSON value.",
        )
    return payload


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str) and _finite_number(item)}


def _safe_headers(headers: Any) -> dict[str, str]:
    allowed_prefixes = (
        "x-sie-units-",
        "x-sie-credit-",
        "x-sie-credits-",
        "x-sie-cost-",
    )
    result: dict[str, str] = {}
    for key, value in headers.items():
        normalized = key.lower()
        if normalized.startswith(allowed_prefixes):
            result[normalized] = str(value)
    return result


def _parse_success(
    payload: dict[str, Any],
    headers: Any,
    client_latency_ms: float,
    requested_model: str,
) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise SmokeError(
            "INVALID_RESPONSE",
            "The service returned no embedding vector.",
        )
    embedding = data[0].get("embedding")
    if not isinstance(embedding, list) or not embedding or not all(_finite_number(value) for value in embedding):
        raise SmokeError(
            "INVALID_RESPONSE",
            "The service returned an invalid embedding vector.",
        )

    server_time = headers.get("X-Inference-Time")
    result: dict[str, Any] = {
        "ok": True,
        "model": payload.get("model") or requested_model,
        "dimensions": len(embedding),
        "preview": [round(float(value), 6) for value in embedding[:5]],
        "l2_norm": round(
            math.sqrt(sum(float(value) ** 2 for value in embedding)),
            6,
        ),
        "request_id": headers.get("X-SIE-Request-Id"),
        "client_latency_ms": round(client_latency_ms, 1),
        "usage": _safe_usage(payload.get("usage")),
    }
    if server_time is not None:
        result["server_inference_time"] = server_time
    metering = _safe_headers(headers)
    if metering:
        result["metering"] = metering
    return result


def run_smoke(
    config: SmokeConfig,
    *,
    transport: Transport = _default_transport,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    request = _request(config)
    started = clock()
    try:
        with transport(request, TIMEOUT_SECONDS) as response:
            payload = _read_json(response)
            elapsed_ms = (clock() - started) * 1000
            return _parse_success(
                payload,
                response.headers,
                elapsed_ms,
                config.model,
            )
    except urllib.error.HTTPError as error:
        _fail_http(error.code)
    except (urllib.error.URLError, TimeoutError) as error:
        raise SmokeError(
            "NETWORK_ERROR",
            "Could not reach the configured managed API origin.",
        ) from error


def main(
    *,
    env: Mapping[str, str] = os.environ,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    transport: Transport = _default_transport,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    try:
        config = SmokeConfig.from_env(env)
        result = run_smoke(config, transport=transport, clock=clock)
    except SmokeError as error:
        print(json.dumps(error.as_dict(), sort_keys=True), file=stderr)
        return 2
    print(json.dumps(result, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
