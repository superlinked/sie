import contextlib
import io
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from sie_sdk import SIEClient

from run_paraphrase import parse_args
from sie_chat import chat_text


class ReaderDemoTests(unittest.TestCase):
    def setUp(self):
        self.client = SIEClient("https://example.invalid", api_key="test-credential")

    def tearDown(self):
        self.client.close()

    def response(self, status, body):
        return httpx.Response(status, json=body,
                              request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))

    def test_sdk_serializes_payload_and_returns_completed_text(self):
        payload = {"model": "test-model", "messages": [{"role": "user", "content": "Translate."}],
                   "max_tokens": 1024, "temperature": 0}
        response = self.response(200, {
            "choices": [{"message": {"content": " Translated. "}, "finish_reason": "stop"}]
        })
        with patch.object(self.client._client, "post", return_value=response) as post:
            self.assertEqual(chat_text(self.client, payload), "Translated.")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "/v1/chat/completions")
        body = json.loads(post.call_args.kwargs["content"])
        for key, value in payload.items():
            self.assertEqual(body[key], value)

    def test_failed_requests_are_not_retried_or_logged_with_secrets(self):
        payload = {"model": "test-model", "messages": []}
        for status in [401, 402, 429, 503, 504]:
            with self.subTest(status=status):
                response = self.response(status, {"detail": "test-credential"})
                with patch.object(self.client._client, "post", return_value=response) as post:
                    with self.assertRaises(RuntimeError) as exc:
                        chat_text(self.client, payload)
                post.assert_called_once()
                self.assertNotIn("test-credential", str(exc.exception))
        with patch.object(self.client._client, "post", side_effect=httpx.ReadTimeout("test-credential")) as post:
            with self.assertRaises(RuntimeError) as exc:
                chat_text(self.client, payload)
        post.assert_called_once()
        self.assertNotIn("test-credential", str(exc.exception))

    def test_incomplete_empty_and_malformed_outputs_are_rejected(self):
        bodies = [
            {"choices": [{"message": {"content": "cut off"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
            {"choices": []},
        ]
        for body in bodies:
            with patch.object(self.client._client, "post", return_value=self.response(200, body)):
                with self.assertRaises(RuntimeError):
                    chat_text(self.client, {"model": "test-model", "messages": []})

    def test_cli_refuses_overwriting_saved_evidence(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--output", str(Path(__file__).with_name("samples.json"))])
            with self.assertRaises(SystemExit):
                parse_args(["--limit", "0"])

    def test_fresh_table_matches_all_168_rows(self):
        root = Path(__file__).parent / "fresh-translation-2026-09-13"
        payload = json.loads((root / "results.json").read_text())
        records = payload["records"]
        self.assertEqual(len(records), 168)
        self.assertEqual(len({(r["id"], r["arm"]) for r in records}), 168)
        self.assertEqual(len({r["prompt_index"] for r in records}), 21)
        for row in records:
            n, g = row["scored_pairs"], row["green_hits"]
            self.assertAlmostEqual(row["z"], (g - 0.25 * n) / math.sqrt(n * 0.25 * 0.75))
        self.assertEqual(len(payload["pairs"]), 84)
        for bias, expected in [(0, (0, 0)), (2, (19, 8)), (3, (21, 15)), (4, (21, 15))]:
            pairs = [p for p in payload["pairs"] if p["bias"] == bias]
            self.assertEqual(len(pairs), 21)
            self.assertEqual(tuple(sum(p[side]["z"] > 3 for p in pairs)
                                   for side in ["before", "after"]), expected)


if __name__ == "__main__":
    unittest.main()
