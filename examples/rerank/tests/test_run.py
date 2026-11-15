"""Offline SDK transport checks; synthetic fixtures are not model measurements."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx
    import msgpack
    import sie_sdk
except ImportError:
    SDK_AVAILABLE = False
else:
    SDK_AVAILABLE = True

ROOT = Path(__file__).resolve().parents[1]
score_spec = importlib.util.spec_from_file_location("rerank_trial_score", ROOT / "score.py")
assert score_spec and score_spec.loader
scorer = importlib.util.module_from_spec(score_spec)
score_spec.loader.exec_module(scorer)
run_spec = importlib.util.spec_from_file_location("rerank_trial_run", ROOT / "run.py")
assert run_spec and run_spec.loader
runner = importlib.util.module_from_spec(run_spec)
with patch.dict(sys.modules, {"score": scorer}):
    run_spec.loader.exec_module(runner)


@unittest.skipUnless(SDK_AVAILABLE, "Install the repository SDK dependencies for offline wire tests")
class URLTrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "inputs").mkdir()
        self.cases = [
            {
                "id": f"case-{letter}",
                "query": f"Question {letter}: preserve Ö and /OI",
                "candidates": [
                    {"id": f"document-{letter}-{i}", "text": f"Full text {letter}/{i}: €184.50 · ZŁ"} for i in range(20)
                ],
            }
            for letter in ["a", "b"]
        ]
        (self.directory / "inputs/cases_test.json").write_text(json.dumps({"cases": self.cases}))
        self.output = self.directory / "trial.json"
        self.requests = []
        self.clients = []
        self.model_revision = scorer.MODEL_REVISION
        self.bad_candidate_set = False
        self.execution_revision = "a" * 64
        self.next_execution_revision = self.execution_revision
        self.invalid_score = False
        self.invalid_score_value = None
        self.missing_score = False

    def transport(self, request):
        self.requests.append(request)
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": runner.MODEL, "revision": self.model_revision}]})
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, "/v1/score/Qwen/Qwen3-Reranker-4B")
        body = msgpack.unpackb(request.content, raw=False)
        items = body["items"][:-1] if self.bad_candidate_set else body["items"]
        reply = {
            "model": runner.MODEL,
            "scores": [{"item_id": item["id"], "score": 1.0 - i / 20, "rank": i} for i, item in enumerate(items)],
        }
        if self.missing_score:
            del reply["scores"][0]["score"]
        elif self.invalid_score:
            reply["scores"][0]["score"] = self.invalid_score_value
        headers = {"Content-Type": "application/msgpack"}
        if self.execution_revision is not None:
            headers["X-SIE-Model-Revision"] = self.execution_revision
        self.execution_revision = self.next_execution_revision
        return httpx.Response(
            200,
            content=msgpack.packb(reply, use_bin_type=True),
            headers=headers,
        )

    def invoke(self, arguments=(), environment=None):
        assert SDK_AVAILABLE
        actual_client = sie_sdk.SIEClient

        def client(base_url, **kwargs):
            transport = httpx.Client(base_url=base_url, transport=httpx.MockTransport(self.transport), trust_env=False)
            result = actual_client(base_url, http_client=transport, **kwargs)
            self.clients.append(transport)
            return result

        variables = {"SIE_API_KEY": "", "SIE_BASE_URL": "", "SIE_URL": "", **(environment or {})}
        argv = ["run.py", "--data", str(self.directory), "--out", str(self.output), *arguments]
        with (
            patch.dict(os.environ, variables),
            patch.object(sys, "argv", argv),
            patch.object(sie_sdk, "SIEClient", side_effect=client),
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            runner.main()
        self.assertTrue(all(client.is_closed for client in self.clients))
        return printed.getvalue()

    def score_requests(self):
        return [request for request in self.requests if request.method == "POST"]

    def test_selected_arm_preserves_full_inputs_order_and_optional_auth(self):
        for key in ["", "synthetic-test-key"]:
            with self.subTest(auth=bool(key)):
                self.requests.clear()
                self.invoke(
                    ["--record", "--limit", "2", "--arm", "proposed"],
                    {"SIE_URL": "https://fixture.invalid", "SIE_API_KEY": key},
                )
                posts = self.score_requests()
                self.assertEqual(len(posts), 2)
                for case, request in zip(self.cases, posts, strict=True):
                    body = msgpack.unpackb(request.content, raw=False)
                    self.assertEqual(body["query"]["text"], case["query"])
                    self.assertEqual(body["items"], case["candidates"])
                    self.assertEqual(body["instruction"], runner.RULES["proposed"])
                    self.assertEqual(request.headers.get("Authorization"), f"Bearer {key}" if key else None)
                saved = json.loads(self.output.read_text())
                self.assertEqual([row["case"] for row in saved], [case["id"] for case in self.cases])
                self.assertEqual([row["arm"] for row in saved], ["proposed", "proposed"])
                self.assertTrue(all(row["model_revision"] == scorer.MODEL_REVISION for row in saved))
                self.assertTrue(all(row["execution_revision"] == "a" * 64 for row in saved))
                self.output.unlink()

    def test_paired_default_and_url_precedence(self):
        for args, env, expected in [
            (
                ["--base-url", "https://explicit.invalid"],
                {"SIE_URL": "https://new.invalid", "SIE_BASE_URL": "https://legacy.invalid"},
                "explicit.invalid",
            ),
            ([], {"SIE_URL": "https://new.invalid", "SIE_BASE_URL": "https://legacy.invalid"}, "new.invalid"),
            ([], {"SIE_BASE_URL": "https://legacy.invalid"}, "legacy.invalid"),
        ]:
            with self.subTest(host=expected):
                self.requests.clear()
                self.invoke(["--record", *args], env)
                posts = self.score_requests()
                self.assertEqual(len(posts), 2)
                self.assertTrue(all(request.url.host == expected for request in self.requests))
                self.assertEqual(
                    [msgpack.unpackb(request.content, raw=False)["instruction"] for request in posts],
                    list(runner.RULES.values()),
                )
                self.output.unlink()

    def test_selected_preview_and_overwrite_refusal_send_no_requests(self):
        preview = self.invoke(["--show", "case-a", "--arm", "proposed"])
        self.assertEqual(json.loads(preview), {"proposed": runner.envelope(self.cases[0], "proposed")})
        self.assertFalse(self.requests)
        self.output.write_text("preserve previous trial")
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.invoke(["--record"])
        self.assertEqual(self.output.read_text(), "preserve previous trial")
        self.assertFalse(self.requests)

    def test_revision_gate_and_bad_response_close_client_without_trial(self):
        self.model_revision = "wrong-checkpoint"
        with self.assertRaisesRegex(SystemExit, "recorded model revision"):
            self.invoke(["--record"])
        self.assertFalse(self.score_requests())
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertFalse(self.output.exists())
        self.model_revision = scorer.MODEL_REVISION
        self.bad_candidate_set = True
        with self.assertRaisesRegex(SystemExit, "candidate set differs"):
            self.invoke(["--record", "--arm", "proposed"])
        self.assertEqual(len(self.score_requests()), 1)
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertFalse(self.output.exists())

    def test_invalid_scores_close_client_without_trial(self):
        for value in [None, "not a number", True, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(score=repr(value)):
                self.requests.clear()
                self.invalid_score = True
                self.invalid_score_value = value
                with self.assertRaisesRegex(SystemExit, "scores are invalid"):
                    self.invoke(["--record", "--arm", "proposed"])
                self.assertEqual(len(self.score_requests()), 1)
                self.assertTrue(all(client.is_closed for client in self.clients))
                self.assertFalse(self.output.exists())
        self.missing_score = True
        with self.assertRaisesRegex(SystemExit, "scores are invalid"):
            self.invoke(["--record", "--arm", "proposed"])
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertFalse(self.output.exists())

    def test_execution_revision_drift_closes_client_without_trial(self):
        for next_revision in ["b" * 64, None]:
            with self.subTest(next_revision=next_revision):
                self.requests.clear()
                self.execution_revision = "a" * 64
                self.next_execution_revision = next_revision
                with self.assertRaisesRegex(SystemExit, "execution revision changed"):
                    self.invoke(["--record", "--limit", "2", "--arm", "proposed"])
                self.assertEqual(len(self.score_requests()), 2)
                self.assertTrue(all(client.is_closed for client in self.clients))
                self.assertFalse(self.output.exists())

    def test_missing_execution_revision_is_recorded_separately_from_weights(self):
        self.execution_revision = None
        self.next_execution_revision = None
        self.invoke(["--record", "--limit", "2", "--arm", "proposed"])
        saved = json.loads(self.output.read_text())
        self.assertTrue(all(row["model_revision"] == scorer.MODEL_REVISION for row in saved))
        self.assertTrue(all(row["execution_revision"] is None for row in saved))


if __name__ == "__main__":
    unittest.main()
