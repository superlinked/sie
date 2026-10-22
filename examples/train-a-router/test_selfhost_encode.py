"""Exercise the real pinned SDK against an in-memory fake HTTP server."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from embedding import recipes, selfhost, wire
from test_public_adapters import source_fixture


class Tokenizer:
    def __init__(self, n=12):
        self.n = n

    def __call__(self, texts, *, truncation):
        assert truncation is False
        return {"input_ids": [[1] * self.n for _ in texts]}


class FakeServer:
    def __init__(self, *, wrong_revision=False, fail=False):
        self.requests = []
        self.wrong_revision = wrong_revision
        self.fail = fail

    def __call__(self, request):
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"name": recipes.MODEL, "revision": "0" * 40 if self.wrong_revision else recipes.REVISION,
                                            "dims": {"dense": 2560}, "loaded": True})
        if self.fail:
            return httpx.Response(503, json={"error": {"code": "MODEL_LOADING", "message": "synthetic loading"}})
        body = wire.unpackb(request.content, numeric_arrays=True)
        vectors = [{"id": row["id"], "dense": {"dims": 2560, "values": np.asarray([3.0, 4.0] + [0.0] * 2558, dtype=np.float32)}}
                   for row in body["items"]]
        content = wire.packb({"model": recipes.MODEL, "items": vectors, "usage": {"input_tokens": 12}}, use_bin_type=True)
        return httpx.Response(200, content=content, headers={"content-type": "application/msgpack",
                              "x-sie-request-id": "synthetic-request", "x-sie-model-revision": "a" * 64})

    def factory(self, base_url, capture):
        return selfhost.make_client(base_url, capture, transport=httpx.MockTransport(self))


class SelfHostTests(unittest.TestCase):
    def run_fixture(self, directory, server, tokenizer=None):
        root = Path(directory)
        source = source_fixture(root)
        plan = selfhost.prepare(source, "http://127.0.0.1:8080", 60)
        binding = {"model": recipes.MODEL, "hf_revision": recipes.REVISION,
                   "files_sha256": {"synthetic-tokenizer.json": "b" * 64}}
        result = selfhost.run(source, root / "encoded", plan, tokenizer or Tokenizer(), binding,
                              client_factory=server.factory)
        return source, root / "encoded", result

    def test_sdk_vectors_enter_existing_learner_and_cloud_key_never_leaves(self):
        server = FakeServer()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SIE_API_KEY": "unused-cloud-key",
                                                                                "SIE_SELF_HOSTED_API_KEY": ""}):
            source, output, result = self.run_fixture(directory, server)
            self.assertEqual(result["status"], "complete")
            self.assertIsNone(result["encoder_cost_usd"])
            self.assertTrue(all("authorization" not in request.headers for request in server.requests))
            posts = [request for request in server.requests if request.method == "POST"]
            self.assertEqual(len(posts), 1)
            body = wire.unpackb(posts[0].content, numeric_arrays=True)
            self.assertEqual(body["params"], {"output_types": ["dense"], "instruction": recipes.INSTRUCTION,
                                               "output_dtype": "float32", "options": {"is_query": True}})
            run = Path(directory) / "run"
            plan = recipes.prepare(source, Path(__file__).with_name("pair.json"), run)
            for role in recipes.ROLES:
                rows = recipes.role_rows(run, plan, role)
                matrix, _ = recipes.features(plan, rows, output / "features", role)
                self.assertEqual(matrix.shape, (1, 2560))
                self.assertAlmostEqual(float(matrix[0, 0]), 0.6, places=6)
            fees = json.loads((output / "unpriced-encoder.json").read_bytes())
            self.assertTrue(all(row["projected_singleton_usd"] is None for row in fees["rows"]))
            self.assertFalse((output / "fee-projection.json").exists())

    def test_wrong_revision_stops_before_any_encode_call(self):
        server = FakeServer(wrong_revision=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ready pinned model"):
                self.run_fixture(directory, server)
            self.assertFalse(any(request.method == "POST" for request in server.requests))

    def test_failed_physical_attempt_is_preserved_and_not_retried(self):
        server = FakeServer(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Exception):
                self.run_fixture(directory, server)
            self.assertEqual(sum(request.method == "POST" for request in server.requests), 1)
            output = Path(directory) / "encoded"
            self.assertTrue((output / "batches/000000/intent.json").is_file())
            self.assertTrue((output / "batches/000000/response.bin").is_file())
            terminal = json.loads((output / "terminal.json").read_bytes())
            self.assertEqual(terminal["status"], "incomplete")
            self.assertEqual(terminal["unresolved_batch"], "batches/000000")

    def test_context_overflow_precedes_even_catalog_request(self):
        server = FakeServer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "context overflow"):
                self.run_fixture(directory, server, Tokenizer(32769))
            self.assertEqual(server.requests, [])


if __name__ == "__main__":
    unittest.main()
