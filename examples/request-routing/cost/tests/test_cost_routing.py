"""Offline tests: the knn20 scoring and routing rule on a tiny synthetic bank, and evidence verification.

The banks, manifests and files here are synthetic fixtures, not study data.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import msgpack
import msgpack_numpy
import numpy as np
from sie_sdk import SIEClient

import fetch
import run
import score
import study


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    return v / np.linalg.norm(v)


class KnnScoreTests(unittest.TestCase):
    def test_weights_the_k_most_similar_labels_by_cosine(self) -> None:
        similarities = np.array([0.9, 0.1, 0.5, 0.7])
        y = np.array([1.0, 0.0, 0.0, 0.5])
        # The two nearest are 0.9 (y=1) and 0.7 (y=0.5).
        self.assertAlmostEqual(study.knn_score(similarities, y, k=2), (0.9 * 1.0 + 0.7 * 0.5) / (0.9 + 0.7))

    def test_negative_cosines_carry_no_weight(self) -> None:
        similarities = np.array([0.4, -0.8])
        y = np.array([1.0, 0.0])
        self.assertEqual(study.knn_score(similarities, y, k=2), 1.0)

    def test_falls_back_to_the_plain_mean_when_no_neighbour_is_positive(self) -> None:
        similarities = np.array([-0.1, -0.2, 0.0])
        y = np.array([1.0, 0.0, 0.5])
        self.assertAlmostEqual(study.knn_score(similarities, y, k=2), 0.75)

    def test_label_is_one_unless_the_strong_answer_is_better(self) -> None:
        q_cheap = np.array([1.0, 0.0, 1.0, 0.25])
        q_strong = np.array([1.0, 1.0, 0.0, 0.75])
        np.testing.assert_array_equal(study.labels(q_cheap, q_strong), [1.0, 0.0, 1.0, 0.5])


class RoutingRuleTests(unittest.TestCase):
    def test_score_at_the_threshold_goes_cheap(self) -> None:
        self.assertTrue(study.routes_cheap(0.95, 0.95))
        self.assertFalse(study.routes_cheap(0.9499, 0.95))

    def test_null_threshold_never_goes_cheap(self) -> None:
        self.assertFalse(study.routes_cheap(1.0, None))

    def tiny_bank(self, pair: str = "openai", tau: float = 0.75) -> study.Bank:
        vectors = np.array([unit(1, 0, 0), unit(1, 0.2, 0), unit(0, 1, 0), unit(0, 0, 1)])
        return study.Bank(pair, ["a", "b", "c", "d"], vectors, np.array([1.0, 1.0, 0.0, 0.0]), tau)

    def test_bank_routes_a_request_near_cheap_wins_to_the_cheap_model(self) -> None:
        decision = self.tiny_bank().route(np.array([3.0, 0.1, 0.0]))
        self.assertEqual(decision["route"], "cheap")
        self.assertEqual(decision["model"], "gpt-6-luna")
        self.assertGreaterEqual(decision["score"], decision["tau"])

    def test_bank_routes_a_request_near_strong_wins_to_the_strong_model(self) -> None:
        decision = self.tiny_bank(pair="anthropic").route(np.array([0.0, 2.0, 0.1]))
        self.assertEqual(decision["route"], "strong")
        self.assertEqual(decision["model"], "claude-sonnet-5-5")

    def test_bank_can_leave_out_its_own_request(self) -> None:
        bank = self.tiny_bank()
        with_self = bank.score(unit(0, 0, 1))
        without_self = bank.score(unit(0, 0, 1), exclude="d")
        self.assertEqual(with_self, 0.0)
        # Without "d", the only positive neighbours left are orthogonal, so every weight is zero: plain mean.
        self.assertAlmostEqual(without_self, 2 / 3)

    def test_rejects_a_zero_or_non_finite_vector(self) -> None:
        for vector in (np.zeros(3), np.array([1.0, np.nan, 0.0])):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.tiny_bank().score(vector)


class StudyRuleTests(unittest.TestCase):
    def outcomes(self) -> study.Outcomes:
        n = 4
        return study.Outcomes(
            q_cheap=np.array([1.0, 1.0, 0.0, 1.0]),
            q_strong=np.array([1.0, 1.0, 1.0, 1.0]),
            usd_cheap=np.full(n, 0.001),
            usd_strong=np.full(n, 0.01),
            usd_router=np.zeros(n),
            score=np.zeros(n),
            route=["strong"] * n,
        )

    def test_leave_one_out_scores_skip_the_request_itself(self) -> None:
        vectors = np.array([unit(1, 0), unit(1, 0.01), unit(0, 1)])
        y = np.array([1.0, 0.0, 0.5])
        scores = score.leave_one_out_scores(vectors, y, np.array([True, True, False]), k=1)
        np.testing.assert_array_equal(scores, [0.0, 1.0, 0.0])

    def test_threshold_is_the_cheapest_that_keeps_98_percent(self) -> None:
        o = self.outcomes()
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        mask = np.ones(4, bool)
        # Sending the 0.7 request cheap costs a quarter of quality, so the threshold stops above it.
        self.assertEqual(score.choose_tau(o, scores, np.zeros(4), mask), 0.8)

    def test_threshold_is_infinite_when_nothing_can_go_cheap(self) -> None:
        o = self.outcomes()
        scores = np.array([0.1, 0.1, 0.9, 0.1])
        self.assertEqual(score.choose_tau(o, scores, np.zeros(4), np.ones(4, bool)), np.inf)

    def test_summary_includes_the_router_cost(self) -> None:
        o = self.outcomes()
        quality, cost, cheap = score.routed(o, np.array([0.9, 0.8, 0.7, 0.6]), np.full(4, 0.0001), 0.8)
        row = score.summarize(o, quality, cost, cheap, np.ones(4, bool))
        self.assertAlmostEqual(row["cost_per_1k"], 1000 * (2 * 0.001 + 2 * 0.01 + 4 * 0.0001) / 4)
        self.assertAlmostEqual(row["cheap_share"], 0.5)
        self.assertAlmostEqual(row["quality_vs_strong"], 1.0)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def encode(self, model: str, items: dict) -> dict:
        self.calls.append((model, items))
        return {"dense": np.array([0.5, 0.5], dtype=np.float32)}


class EmbedTests(unittest.TestCase):
    def test_embeds_one_text_item_with_bge_m3(self) -> None:
        client = FakeEncoder()
        vector = run.embed(client, "Solve for x")
        self.assertEqual(client.calls, [("BAAI/bge-m3", {"text": "Solve for x"})])
        self.assertEqual(vector.dtype, np.float64)

    def test_default_client_sends_one_encode_request_to_sie_serverless(self) -> None:
        calls: list[httpx.Request] = []
        dense = np.array([0.6, 0.8, 0.0], dtype=np.float32)

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            body = {"model": "BAAI/bge-m3", "items": [{"dense": {"dims": 3, "dtype": "float32", "values": dense}}]}
            return httpx.Response(
                200,
                headers={"Content-Type": "application/msgpack"},
                content=msgpack.packb(body, default=msgpack_numpy.encode, use_bin_type=True),
            )

        transport = httpx.Client(base_url="https://api.superlinked.com", transport=httpx.MockTransport(respond))
        with (
            mock.patch.dict(os.environ, {"SIE_API_KEY": "test-key"}, clear=True),
            SIEClient(http_client=transport) as c,
        ):
            vector = run.embed(c, "Solve for x")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.host, "api.superlinked.com")
        self.assertEqual(calls[0].url.path, "/v1/encode/BAAI/bge-m3")
        self.assertEqual(calls[0].headers["authorization"], "Bearer test-key")
        self.assertEqual(msgpack.unpackb(calls[0].content, raw=False)["items"], [{"text": "Solve for x"}])
        np.testing.assert_array_equal(vector, dense.astype(np.float64))


class EvidenceTests(unittest.TestCase):
    def fixture(self) -> tuple[dict[str, bytes], bytes]:
        files = {name: f"synthetic {name}\n".encode() for name in study.FILES}
        entries = {name: {"bytes": len(body), "sha256": study.sha256(body)} for name, body in files.items()}
        entries["unused.json"] = {"bytes": 1, "sha256": "0" * 64}
        manifest = json.dumps({"schema": "test", "files": entries}).encode()
        return files, manifest

    def serve(self, files: dict[str, bytes], manifest: bytes):
        by_url = {fetch.url(name): body for name, body in files.items()} | {fetch.url(study.MANIFEST): manifest}
        requested: list[str] = []

        def download(address: str, max_bytes: int) -> bytes:
            requested.append(address)
            return by_url[address]

        return download, requested

    def test_pinned_revision_and_manifest_digest(self) -> None:
        self.assertEqual(study.REVISION, "95d34b775b61bf1d584cada099bfed053710fc2b")
        self.assertEqual(study.MANIFEST_SHA256, "82f870f6a64916aa4e217eea03df8f17d3355f06f2c6e2718064b795c40de713")
        self.assertTrue(fetch.url("results.json").startswith(f"https://huggingface.co/datasets/{study.DATASET}/"))
        self.assertIn(f"/resolve/{study.REVISION}/{study.FOLDER}/results.json", fetch.url("results.json"))

    def test_manifest_must_match_its_pinned_digest(self) -> None:
        _, manifest = self.fixture()
        with self.assertRaises(study.EvidenceError):
            study.manifest_entries(manifest)
        with mock.patch.object(study, "MANIFEST_SHA256", study.sha256(manifest)):
            self.assertEqual(set(study.manifest_entries(manifest)), set(study.FILES))

    def test_manifest_must_list_every_file_used(self) -> None:
        manifest = json.dumps({"files": {"results.json": {"bytes": 1, "sha256": "0" * 64}}}).encode()
        with (
            mock.patch.object(study, "MANIFEST_SHA256", study.sha256(manifest)),
            self.assertRaises(study.EvidenceError),
        ):
            study.manifest_entries(manifest)

    def test_file_must_match_its_size_and_digest(self) -> None:
        body = b"recorded"
        entry = (study.sha256(body), len(body))
        study.check_file("results.json", body, entry)
        for altered in (b"recorded!", b"Recorded"):
            with self.subTest(altered=altered), self.assertRaises(study.EvidenceError):
                study.check_file("results.json", altered, entry)

    def test_fetch_downloads_verifies_and_never_overwrites(self) -> None:
        files, manifest = self.fixture()
        download, requested = self.serve(files, manifest)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(study, "MANIFEST_SHA256", study.sha256(manifest)):
            destination = Path(tmp) / "evidence"
            self.assertTrue(fetch.fetch(destination, download))
            self.assertEqual(len(requested), len(study.FILES) + 1)
            self.assertEqual((destination / "records/routers/bge.jsonl.gz").read_bytes(), files[study.FILES[-1]])
            study.verify_evidence(destination)
            self.assertFalse(fetch.fetch(destination, download))
            self.assertEqual(len(requested), len(study.FILES) + 1)

            (destination / "results.json").write_bytes(b"edited")
            with self.assertRaises(study.EvidenceError):
                study.verify_evidence(destination)
            with self.assertRaises(study.EvidenceError):
                fetch.fetch(destination, download)
            self.assertEqual((destination / "results.json").read_bytes(), b"edited")

    def test_fetch_leaves_nothing_behind_when_a_file_does_not_verify(self) -> None:
        files, manifest = self.fixture()
        tampered = files | {"items-openai.jsonl": b"tampered\n"}
        download, _ = self.serve(tampered, manifest)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(study, "MANIFEST_SHA256", study.sha256(manifest)):
            destination = Path(tmp) / "evidence"
            with self.assertRaises(study.EvidenceError):
                fetch.fetch(destination, download)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_fetch_rejects_a_manifest_that_is_not_the_pinned_one(self) -> None:
        files, manifest = self.fixture()
        download, requested = self.serve(files, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(study.EvidenceError):
                fetch.fetch(Path(tmp) / "evidence", download)
            self.assertEqual(requested, [fetch.url(study.MANIFEST)])

    def test_scoring_refuses_unverified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(study.EvidenceError):
            study.load_study(Path(tmp))


if __name__ == "__main__":
    unittest.main()
