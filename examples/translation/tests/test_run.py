"""Offline checks: the SDK sends the recorded request, and failures print no detail."""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from unittest import mock

import httpx
from sie_sdk import ServerError, SIEClient

import run

BASE = "http://localhost:8080"
RECORDED_REQUEST = run.RECORDED["recorded"]["request"]
RECORDED_RESPONSE = run.RECORDED["recorded"]["response"]
SOURCE = run.RECORDED["source"]
# The SIE response also carries these; none of them is printed.
EXTRA_FIELDS = {
    "attempt_id": "offline-attempt",
    "ttft_ms": 12.5,
    "tpot_ms": 3.0,
}


def respond_with(body: dict, calls: list[httpx.Request], status: int = 200):
    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=body)

    return respond


def mock_client(respond, api_key: str = "") -> SIEClient:
    http_client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(respond))
    return SIEClient(BASE, api_key=api_key, http_client=http_client)


def run_main(argv: list[str], respond=None, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """Run main() with a mocked transport; return (exit code, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()

    def factory(**_: object) -> SIEClient:
        if respond is None:
            raise AssertionError("main() built a client when it should send nothing")
        return mock_client(respond)

    with (
        mock.patch.dict(os.environ, env or {}, clear=True),
        mock.patch.object(run, "SIEClient", factory),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            code = run.main(argv)
        except SystemExit as exc:  # argparse errors
            code = int(exc.code)
    return code, stdout.getvalue(), stderr.getvalue()


class RequestTests(unittest.TestCase):
    def test_prompt_and_body_are_the_recorded_ones(self) -> None:
        self.assertEqual(run.PATH, RECORDED_REQUEST["path"])
        self.assertEqual(run.request_body(SOURCE["text"], SOURCE["translate_into"]), RECORDED_REQUEST["body"])
        self.assertEqual(run.build_prompt(f"  {SOURCE['text']}\n", "Japanese"), RECORDED_REQUEST["body"]["prompt"])

    def test_the_output_cap_grows_with_the_input_like_the_page(self) -> None:
        self.assertEqual(run.output_cap(SOURCE["text"]), 256)
        # The page's Hy-MT2 check: up to 27 characters get 64, up to 66 get 128,
        # up to 144 get 256, up to 297 get 512, and longer text gets 1024.
        limits = [27, 66, 144, 297]
        for length in range(1, 1201):
            index = next((i for i, limit in enumerate(limits) if length <= limit), len(limits))
            self.assertEqual(run.output_cap("x" * length), run.OUTPUT_TOKEN_BUCKETS[index], length)
        self.assertEqual(run.output_cap("x" * 5000), 1024)
        self.assertEqual(run.output_cap(f"  {'x' * 10}  "), 64)
        # Code points, not UTF-16 units or bytes.
        self.assertEqual(run.output_cap("😀" * 144), 256)
        self.assertEqual(run.output_cap("😀" * 145), 512)

    def test_sdk_sends_one_post_with_the_recorded_body(self) -> None:
        calls: list[httpx.Request] = []
        with mock_client(respond_with({**RECORDED_RESPONSE, **EXTRA_FIELDS}, calls)) as client:
            output = run.translate(client, SOURCE["text"], SOURCE["translate_into"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].method, "POST")
        self.assertEqual(calls[0].url.path, RECORDED_REQUEST["path"])
        # Parsed, not byte for byte: the SDK escapes non-ASCII characters.
        self.assertEqual(json.loads(calls[0].content), RECORDED_REQUEST["body"])
        self.assertNotIn("authorization", calls[0].headers)
        self.assertEqual(output["text"], RECORDED_RESPONSE["text"])
        self.assertEqual(output["finish_reason"], "stop")
        self.assertEqual(output["usage"], {"prompt_tokens": 50, "completion_tokens": 51})

    def test_the_key_is_sent_as_a_bearer_token(self) -> None:
        calls: list[httpx.Request] = []
        with mock_client(respond_with(RECORDED_RESPONSE, calls), api_key="offline-test-key") as client:
            run.translate(client, SOURCE["text"], SOURCE["translate_into"])
        self.assertEqual(calls[0].headers["authorization"], "Bearer offline-test-key")

    def test_a_gateway_timeout_is_not_sent_again(self) -> None:
        calls: list[httpx.Request] = []
        with mock_client(respond_with({"detail": "Gateway timeout"}, calls, 504)) as client:
            with self.assertRaises(ServerError):
                run.translate(client, SOURCE["text"], SOURCE["translate_into"])
        self.assertEqual(len(calls), 1)

    def test_the_read_timeout_covers_the_model_limit_not_the_capacity_wait(self) -> None:
        calls: list[httpx.Request] = []
        with mock_client(respond_with(RECORDED_RESPONSE, calls)) as client:
            run.translate(client, SOURCE["text"], SOURCE["translate_into"])
        self.assertEqual(calls[0].extensions["timeout"]["read"], 600)


class MainTests(unittest.TestCase):
    def test_show_prints_the_request_and_sends_nothing(self) -> None:
        code, out, _ = run_main(["--show"])
        self.assertEqual(code, 0)
        shown = json.loads(out)
        self.assertEqual(shown["url"], "https://api.superlinked.com" + RECORDED_REQUEST["path"])
        self.assertEqual(shown["body"], RECORDED_REQUEST["body"])

    def test_show_uses_sie_base_url(self) -> None:
        code, out, _ = run_main(["--show"], env={"SIE_BASE_URL": BASE + "/"})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["url"], BASE + RECORDED_REQUEST["path"])

    def test_no_key_and_no_base_url_stops_before_any_call(self) -> None:
        code, out, err = run_main([])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("SIE_API_KEY", err)

    def test_live_run_prints_the_translation_and_compares_it_with_the_recording(self) -> None:
        calls: list[httpx.Request] = []
        code, out, _ = run_main([], respond_with({**RECORDED_RESPONSE, **EXTRA_FIELDS}, calls), {"SIE_BASE_URL": BASE})
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        printed = json.loads(out)
        self.assertEqual(printed["text"], RECORDED_RESPONSE["text"])
        self.assertIs(printed["matches_recorded"], True)
        self.assertEqual(printed["recorded_on"], "2026-10-25")
        for field in EXTRA_FIELDS:
            self.assertNotIn(field, out)

    def test_a_different_output_is_reported_not_hidden(self) -> None:
        reply = {**RECORDED_RESPONSE, "text": "Android 13 以降、別の訳です。"}
        code, out, _ = run_main([], respond_with(reply, []), {"SIE_BASE_URL": BASE})
        self.assertEqual(code, 0)
        self.assertIs(json.loads(out)["matches_recorded"], False)

    def test_custom_text_is_not_compared(self) -> None:
        calls: list[httpx.Request] = []
        reply = {**RECORDED_RESPONSE, "text": "Bonjour."}
        code, out, _ = run_main(
            ["--text", "Hello.", "--into", "French"], respond_with(reply, calls), {"SIE_BASE_URL": BASE}
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(calls[0].content)["prompt"], run.build_prompt("Hello.", "French"))
        self.assertEqual(json.loads(calls[0].content)["max_new_tokens"], 64)
        self.assertNotIn("matches_recorded", json.loads(out))

    def test_long_text_gets_a_larger_output_cap(self) -> None:
        calls: list[httpx.Request] = []
        long_text = " ".join([SOURCE["text"]] * 6)
        code, _, _ = run_main(["--text", long_text], respond_with(RECORDED_RESPONSE, calls), {"SIE_BASE_URL": BASE})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(calls[0].content)["max_new_tokens"], 1024)

    def test_a_cut_off_translation_is_flagged(self) -> None:
        reply = {
            **RECORDED_RESPONSE,
            "finish_reason": "length",
            "usage": {"prompt_tokens": 50, "completion_tokens": 256},
        }
        code, out, err = run_main([], respond_with(reply, []), {"SIE_BASE_URL": BASE})
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["finish_reason"], "length")
        self.assertIn("cut off at the 256-token output cap", err)

    def test_a_failed_call_prints_its_kind_and_status_only(self) -> None:
        echoed = "offline-test-key echoed by a server"
        code, out, err = run_main(
            [], respond_with({"detail": {"code": "UNAUTHORIZED", "message": echoed}}, [], 401), {"SIE_BASE_URL": BASE}
        )
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("HTTP 401", err)
        self.assertNotIn(echoed, err)


if __name__ == "__main__":
    unittest.main()
