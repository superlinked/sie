# ruff: noqa: PT009, PT027

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any, Self

MODULE_PATH = Path(__file__).parents[1] / "us_prod_smoke.py"
SPEC = importlib.util.spec_from_file_location("us_prod_smoke", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
smoke: ModuleType = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = json.dumps(payload).encode()
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.body


class RecordingTransport:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request: urllib.request.Request | None = None
        self.timeout: float | None = None

    def __call__(
        self,
        request: urllib.request.Request,
        timeout: float,
    ) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        return self.response


class SmokeTest(unittest.TestCase):
    def test_missing_configuration_is_typed(self) -> None:
        stderr = io.StringIO()

        exit_code = smoke.main(env={}, stderr=stderr)

        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stderr.getvalue())["error"], "CONFIG_MISSING")

    def test_base_url_must_be_bare_https_origin(self) -> None:
        invalid_urls = (
            "http://api.example.com",
            "https://user:password@api.example.com",
            "https://api.example.com/v1",
            "https://api.example.com?key=value",
        )

        for url in invalid_urls:
            with self.subTest(url=url), self.assertRaises(smoke.SmokeError) as caught:
                smoke.SmokeConfig.from_env(
                    {
                        "SIE_API_KEY": "fixture-value",
                        "SIE_BASE_URL": url,
                    }
                )
            self.assertEqual(caught.exception.code, "CONFIG_INVALID")

    def test_success_uses_default_model_and_prints_safe_summary(self) -> None:
        transport = RecordingTransport(
            FakeResponse(
                {
                    "data": [{"embedding": [0.25, -0.5, 0.75]}],
                    "model": "BAAI/bge-m3",
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
                {
                    "X-SIE-Request-Id": "request-123",
                    "X-Inference-Time": "0.012",
                    "X-SIE-Units-Input-Tokens": "7",
                    "Authorization": "must-not-print",
                },
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = smoke.main(
            env={
                "SIE_API_KEY": "secret-fixture-value",
                "SIE_BASE_URL": "https://api.example.com",
            },
            stdout=stdout,
            stderr=stderr,
            transport=transport,
            clock=iter((10.0, 10.025)).__next__,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["model"], "BAAI/bge-m3")
        self.assertEqual(result["dimensions"], 3)
        self.assertEqual(result["request_id"], "request-123")
        self.assertEqual(result["client_latency_ms"], 25.0)
        self.assertEqual(result["metering"]["x-sie-units-input-tokens"], "7")
        self.assertNotIn("secret-fixture-value", stdout.getvalue())
        self.assertNotIn("must-not-print", stdout.getvalue())

        assert transport.request is not None
        self.assertEqual(
            transport.request.full_url,
            "https://api.example.com/v1/embeddings",
        )
        body = json.loads(transport.request.data)
        self.assertEqual(body["model"], "BAAI/bge-m3")
        self.assertEqual(
            transport.request.get_header("Authorization"),
            "Bearer secret-fixture-value",
        )

    def test_model_can_be_overridden(self) -> None:
        config = smoke.SmokeConfig.from_env(
            {
                "SIE_API_KEY": "fixture-value",
                "SIE_BASE_URL": "https://api.example.com/",
                "SIE_MODEL": "model/from-catalog",
            }
        )

        self.assertEqual(config.model, "model/from-catalog")
        self.assertEqual(config.base_url, "https://api.example.com")

    def test_common_http_failures_are_typed_without_reading_error_body(self) -> None:
        cases = {
            401: "AUTH_INVALID",
            402: "CREDITS_EXHAUSTED",
            403: "ACCESS_FORBIDDEN",
            500: "SERVICE_ERROR",
            503: "SERVICE_ERROR",
        }
        for status, expected_code in cases.items():
            secret = f"secret-body-{status}"
            error = urllib.error.HTTPError(
                "https://api.example.com/v1/embeddings",
                status,
                secret,
                {},
                io.BytesIO(secret.encode()),
            )

            def failing_transport(
                _request: urllib.request.Request,
                _timeout: float,
                *,
                error: urllib.error.HTTPError = error,
            ) -> FakeResponse:
                raise error

            stderr = io.StringIO()
            with self.subTest(status=status):
                exit_code = smoke.main(
                    env={
                        "SIE_API_KEY": "fixture-value",
                        "SIE_BASE_URL": "https://api.example.com",
                    },
                    stderr=stderr,
                    transport=failing_transport,
                )
                output = stderr.getvalue()
                self.assertEqual(exit_code, 2)
                self.assertEqual(json.loads(output)["error"], expected_code)
                self.assertNotIn(secret, output)

    def test_invalid_embedding_shape_is_rejected(self) -> None:
        stderr = io.StringIO()

        exit_code = smoke.main(
            env={
                "SIE_API_KEY": "fixture-value",
                "SIE_BASE_URL": "https://api.example.com",
            },
            stderr=stderr,
            transport=RecordingTransport(FakeResponse({"data": []})),
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stderr.getvalue())["error"], "INVALID_RESPONSE")


if __name__ == "__main__":
    unittest.main()
