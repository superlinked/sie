import contextlib
import io
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run
import score
from run import (
    DEFAULT_ENDPOINT,
    Loaders,
    build_parser,
    document_text,
    load_quality_data,
    maxsim_scores,
    plan_quality,
    run_quality,
    sparse_scores,
    top_k_predictions,
)
from score import K_VALUES, compare, metric_rows, parse_task_key, retrieval_metrics, task_metrics

REFERENCE_ID = "example-reference-quality-01"
REVISION = "d56462d0e63a25450459c4f213e49ffdb866f7f9"
CORPUS = {
    "d1": {"title": "Alpha", "text": "alpha body"},
    "d2": {"title": "", "text": "beta body"},
    "d3": {"text": " gamma body "},
    "d4": {"title": "", "text": "delta body"},
    "d5": {"title": "", "text": "epsilon body"},
}
QUERIES = {"q1": "first query", "q2": "second query", "q3": "third query", "q4": "unjudged query"}
QRELS = {"q1": {"d2": 1}, "q2": {"d3": 1}, "q3": {"d2": 2, "d4": 1}, "q9": {"d1": 1}, "q4": {}}
TOP_RANKED = {"q1": ["d2", "d1"], "q2": ["d3", "d5"], "q3": ["d4", "d2"]}
SPARSE_VECTORS = {
    "Alpha alpha body": ([0], [1.0]),
    "beta body": ([2, 5], [1.0, 3.0]),
    "gamma body": ([7], [1.0]),
    "delta body": ([0, 2], [0.5, 0.5]),
    "epsilon body": ([], []),
    "first query": ([0, 2], [1.0, 2.0]),
    "second query": ([7], [2.0]),
    "third query": ([5], [1.0]),
}
MULTIVECTORS = {
    "Alpha alpha body": [[1.0, 0.0]],
    "beta body": [[0.0, 1.0], [1.0, 1.0]],
    "gamma body": [[3.0, 4.0]],
    "delta body": [[0.0, 0.0], [1.0, 0.0]],
    "epsilon body": [[-1.0, 0.0]],
    "first query": [[1.0, 0.0], [0.0, 1.0]],
    "second query": [[0.0, 1.0]],
    "third query": [[1.0, 0.0]],
}
RESULT_KEYS = {
    "kind",
    "schema_version",
    "reference_id",
    "comparable",
    "comparable_reasons",
    "evaluation_mode",
    "model",
    "profile",
    "operation",
    "model_version",
    "execution_revisions",
    "ingress",
    "task_key",
    "dataset",
    "payload",
    "client",
    "scorer",
    "run",
    "validity",
    "metrics",
    "comparison",
    "comparison_basis",
}


class FakeClient:
    def __init__(self, *, retries=(), revisions=("rev-a",), fail_at=None):
        self.calls = []
        self.last_retry_count = 0
        self.last_model_revision = None
        self._retries = list(retries)
        self._revisions = list(revisions)
        self._fail_at = fail_at

    def _observe(self):
        index = len(self.calls) - 1
        self.last_retry_count = self._retries[index] if index < len(self._retries) else 0
        if index == self._fail_at:
            raise RuntimeError("boom")
        self.last_model_revision = self._revisions[index % len(self._revisions)]

    def encode(self, model, items, **kwargs):
        self.calls.append(("encode", model, items, kwargs))
        self._observe()
        return [self._encode_item(item, kwargs["output_types"]) for item in items]

    def _encode_item(self, item, output_types):
        if output_types == ["sparse"]:
            indices, values = SPARSE_VECTORS[item["text"]]
            sparse = {"indices": np.asarray(indices, dtype=np.int32), "values": np.asarray(values, dtype=np.float32)}
            return {"id": item["id"], "sparse": sparse}
        return {"id": item["id"], "multivector": np.asarray(MULTIVECTORS[item["text"]], dtype=np.float16)}

    def score(self, model, query, items, **kwargs):
        self.calls.append(("score", model, (query, items), kwargs))
        self._observe()
        entries = [
            {"item_id": item["id"], "score": float(len(items) - position), "rank": position}
            for position, item in enumerate(items)
        ]
        return {"model": model, "query_id": "", "scores": entries}


class ShortClient(FakeClient):
    def encode(self, model, items, **kwargs):
        return super().encode(model, items, **kwargs)[:-1]


class NoneScoreClient(FakeClient):
    def score(self, model, query, items, **kwargs):
        result = super().score(model, query, items, **kwargs)
        if len(self.calls) > 1:
            result["scores"] = [{"item_id": items[0]["id"], "score": None, "rank": 0}]
        return result


def tiny_loaders():
    return Loaders(
        corpus=lambda name, revision: {doc_id: dict(doc) for doc_id, doc in CORPUS.items()},
        queries=lambda name, revision: dict(QUERIES),
        qrels=lambda name, revision: {query_id: dict(docs) for query_id, docs in QRELS.items()},
        top_ranked=lambda name, revision: {query_id: list(docs) for query_id, docs in TOP_RANKED.items()},
    )


def reference_file(directory, **overrides):
    reference = {
        "kind": "published-benchmark-reference",
        "schema_version": 1,
        "reference_id": REFERENCE_ID,
        "evaluation_mode": "quality",
        "model": "example/sparse-encoder",
        "model_version": "0" * 40,
        "profile": "default",
        "operation": "encode",
        "task_key": "mteb/SciFact",
        "dataset": {"name": "mteb/SciFact", "revision": REVISION, "subset": "test", "parameters": []},
        "payload": {"items_per_request": 2, "output_contract": "sparse", "parameters": []},
        "timing": {
            "warmup_s": 0.0,
            "settle_s": 0.0,
            "drain_s": 0.0,
            "request_timeout_s": 300.0,
            "target_readiness_timeout_s": None,
            "max_in_flight_per_process": 1,
            "scheduling_lag_tolerance_ms": 20.0,
            "retry": {"backoff": "none", "max_attempts": 1, "record_attempts": True},
        },
        "client": {"worker_count": 1, "processes_per_worker": 1},
        "scorer": {"version": "2.7.5", "implementation_digest": "sha256:" + "0" * 64},
        "metrics": [
            {"name": "main_score", "unit": "ratio", "aggregation": "task_native", "value": 0.5},
            {"name": "ndcg_at_10", "unit": "ratio", "aggregation": "task_native", "value": 0.5},
            {"name": "mrr_at_1", "unit": "ratio", "aggregation": "task_native", "value": 1.0},
        ],
        "accepted_at": "2026-09-05T05:18:08Z",
        "measurement_started_at": None,
    }
    reference.update(overrides)
    path = Path(directory) / f"{REFERENCE_ID}.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    return path


def parse(*argv):
    return build_parser().parse_args(["quality", *argv])


def clock():
    return datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.reference = reference_file(self.temporary.name)

    def test_reference_fixes_model_output_and_items_per_request(self):
        for flags in (["--model", "x"], ["--output", "sparse"], ["--items-per-request", "4"]):
            with self.assertRaises(ValueError):
                plan_quality(parse("--reference", str(self.reference), *flags))

    def test_reference_task_must_match_or_be_a_smoke_key(self):
        with self.assertRaises(ValueError):
            plan_quality(parse("--reference", str(self.reference), "--task", "mteb/NFCorpus"))
        with self.assertRaises(ValueError):
            plan_quality(parse("--reference", str(self.reference), "--task", "mteb/SciFact?split=train"))
        same = plan_quality(parse("--reference", str(self.reference), "--task", "mteb/SciFact"))
        self.assertEqual(same.comparable_reasons, [])
        smoke = plan_quality(parse("--reference", str(self.reference), "--task", "mteb/SciFact?limit=2"))
        self.assertEqual(smoke.comparable_reasons, ["smoke key mteb/SciFact?limit=2"])
        self.assertEqual(smoke.task_params, {"limit": "2"})

    def test_smoke_limit_must_be_a_positive_integer(self):
        for limit in ("abc", "0", ""):
            with self.assertRaises(ValueError):
                plan_quality(parse("--reference", str(self.reference), "--task", f"mteb/SciFact?limit={limit}"))
            with self.assertRaises(ValueError):
                plan_quality(
                    parse(
                        "--model",
                        "m",
                        "--task",
                        f"mteb/SciFact?limit={limit}",
                        "--output",
                        "sparse",
                        "--items-per-request",
                        "2",
                    )
                )

    def test_reference_values_are_taken_from_the_file(self):
        plan = plan_quality(parse("--reference", str(self.reference)))
        self.assertEqual(plan.reference_id, REFERENCE_ID)
        self.assertEqual(
            (plan.model, plan.profile, plan.route), ("example/sparse-encoder", "default", "example/sparse-encoder")
        )
        self.assertEqual((plan.output_contract, plan.items_per_request, plan.request_timeout_s), ("sparse", 2, 300.0))
        self.assertEqual((plan.model_version, plan.revision, plan.endpoint), ("0" * 40, REVISION, DEFAULT_ENDPOINT))
        self.assertEqual(plan.comparable_reasons, [])

    def test_endpoint_override_is_a_comparable_reason(self):
        plan = plan_quality(parse("--reference", str(self.reference), "--endpoint", "http://localhost:8080"))
        self.assertEqual(plan.comparable_reasons, [f"endpoint http://localhost:8080 != {DEFAULT_ENDPOINT}"])

    def test_endpoint_is_normalised_before_it_is_compared(self):
        for endpoint in ("https://api.superlinked.com/", "HTTPS://API.superlinked.com"):
            with self.subTest(endpoint=endpoint):
                plan = plan_quality(parse("--reference", str(self.reference), "--endpoint", endpoint))
                self.assertEqual(plan.comparable_reasons, [])
                self.assertEqual(plan.endpoint, DEFAULT_ENDPOINT)
        local = plan_quality(parse("--reference", str(self.reference), "--endpoint", "http://localhost:8080/"))
        self.assertEqual(local.comparable_reasons, [f"endpoint http://localhost:8080/ != {DEFAULT_ENDPOINT}"])
        self.assertEqual(local.endpoint, "http://localhost:8080")
        flags = plan_quality(
            parse("--model", "m", "--task", "mteb/AskUbuntuDupQuestions", "--endpoint", "HTTP://LocalHost:8080/sie/")
        )
        self.assertEqual(flags.endpoint, "http://localhost:8080/sie")
        with self.assertRaises(ValueError):
            plan_quality(parse("--reference", str(self.reference), "--endpoint", "localhost:8080"))

    def test_default_ports_are_normalised_away(self):
        same = plan_quality(parse("--reference", str(self.reference), "--endpoint", "https://api.superlinked.com:443/"))
        self.assertEqual((same.endpoint, same.comparable_reasons), (DEFAULT_ENDPOINT, []))
        plain = plan_quality(parse("--reference", str(self.reference), "--endpoint", "http://localhost:80/"))
        self.assertEqual(plain.endpoint, "http://localhost")
        local = plan_quality(parse("--reference", str(self.reference), "--endpoint", "http://localhost:8080"))
        self.assertEqual(local.endpoint, "http://localhost:8080")

    def test_an_endpoint_without_a_host_or_with_a_bad_port_is_rejected(self):
        for endpoint in ("http://:8080", "http://localhost:99999"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                plan_quality(parse("--reference", str(self.reference), "--endpoint", endpoint))

    def test_reference_kind_and_mode_are_checked(self):
        other = reference_file(self.temporary.name, evaluation_mode="isolated_performance")
        with self.assertRaises(ValueError):
            plan_quality(parse("--reference", str(other)))

    def test_reference_id_must_be_a_plain_file_name(self):
        for reference_id in ("", ".", "..", "../escape", "nested/id", "/abs"):
            with self.subTest(reference_id=reference_id):
                other = reference_file(self.temporary.name, reference_id=reference_id)
                with self.assertRaises(ValueError):
                    plan_quality(parse("--reference", str(other)))

    def test_flags_without_reference(self):
        with self.assertRaises(ValueError):
            plan_quality(parse("--model", "BAAI/bge-m3:sparse"))
        with self.assertRaises(ValueError):
            plan_quality(parse("--model", "BAAI/bge-m3:sparse", "--task", "mteb/NFCorpus"))
        with self.assertRaises(ValueError):
            plan_quality(parse("--model", "x", "--task", "mteb/AskUbuntuDupQuestions", "--output", "sparse"))
        with self.assertRaises(ValueError):
            plan_quality(parse("--model", "x", "--task", "mteb/AskUbuntuDupQuestions", "--items-per-request", "20"))
        encode = plan_quality(
            parse(
                "--model",
                "BAAI/bge-m3:sparse",
                "--task",
                "mteb/NFCorpus",
                "--output",
                "sparse",
                "--items-per-request",
                "128",
            )
        )
        self.assertEqual((encode.model, encode.profile, encode.route), ("BAAI/bge-m3", "sparse", "BAAI/bge-m3:sparse"))
        self.assertEqual(
            (encode.operation, encode.output_contract, encode.items_per_request), ("encode", "sparse", 128)
        )
        self.assertEqual(encode.comparable_reasons, ["no reference"])
        self.assertIsNone(encode.reference_id)
        self.assertEqual(encode.revision, score.HF_DATASETS["mteb/NFCorpus"][1])
        rerank = plan_quality(
            parse("--model", "Qwen/Qwen3-Reranker-0.6B", "--task", "mteb/AskUbuntuDupQuestions?limit=5")
        )
        self.assertEqual(
            (rerank.operation, rerank.output_contract, rerank.items_per_request), ("score", "scores", None)
        )
        self.assertEqual(rerank.comparable_reasons, ["no reference", "smoke key mteb/AskUbuntuDupQuestions?limit=5"])


class DataTest(unittest.TestCase):
    def test_only_queries_with_a_qrels_entry_are_kept_in_dataset_order(self):
        plan = plan_quality(
            parse("--model", "m", "--task", "mteb/SciFact", "--output", "sparse", "--items-per-request", "2")
        )
        data = load_quality_data(plan, tiny_loaders())
        self.assertEqual(list(data.queries), ["q1", "q2", "q3"])
        self.assertEqual(set(data.qrels), {"q1", "q2", "q3"})
        self.assertIsNone(data.top_ranked)

    def test_smoke_key_keeps_the_first_queries_and_the_whole_corpus(self):
        plan = plan_quality(
            parse("--model", "m", "--task", "mteb/SciFact?limit=2", "--output", "sparse", "--items-per-request", "2")
        )
        data = load_quality_data(plan, tiny_loaders())
        self.assertEqual(list(data.queries), ["q1", "q2"])
        self.assertEqual(list(data.qrels), ["q1", "q2"])
        self.assertEqual(len(data.corpus), 5)

    def test_document_text_joins_title_and_text_like_the_published_method(self):
        self.assertEqual(document_text({"title": "Alpha", "text": "alpha body"}), "Alpha alpha body")
        self.assertEqual(document_text({"title": "", "text": " beta body "}), "beta body")
        self.assertEqual(document_text({"text": " gamma body "}), "gamma body")


class RankingTest(unittest.TestCase):
    def test_sparse_dot_uses_a_shared_column_count(self):
        queries = [{"sparse": {"indices": [0, 2], "values": [1.0, 2.0]}}]
        documents = [
            {"sparse": {"indices": [0], "values": [1.0]}},
            {"sparse": {"indices": [2, 5], "values": [1.0, 3.0]}},
            {"sparse": {"indices": [7], "values": [1.0]}},
        ]
        scores = sparse_scores(queries, documents)
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(scores.tolist(), [[1.0, 2.0, 0.0]])

    def test_maxsim_sums_the_best_token_match_over_normalised_rows(self):
        queries = [np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)]
        documents = [
            np.asarray([[2.0, 0.0]], dtype=np.float16),
            np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float16),
            np.asarray([[3.0, 4.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        ]
        scores = maxsim_scores(queries, documents)
        self.assertEqual(scores.shape, (1, 4))
        expected = [1.0, 1.0 + 1.0 / np.sqrt(2.0), 1.4, 1.0]
        for value, target in zip(scores[0].tolist(), expected, strict=True):
            self.assertAlmostEqual(value, target, places=5)
        with self.assertRaises(ValueError):
            maxsim_scores(queries, [np.zeros((0, 2), dtype=np.float32)])

    def test_maxsim_chunks_documents_and_groups_queries(self):
        rng = np.random.default_rng(7)
        queries = [rng.standard_normal((rng.integers(1, 9), 4)) for _ in range(9)]
        documents = [rng.standard_normal((rng.integers(1, 7), 4)) for _ in range(23)]
        with patch("run.DOCUMENT_CHUNK", 5), patch("run.QUERY_TOKEN_ROWS", 6):
            chunked = maxsim_scores(queries, documents)
        plain = maxsim_scores(queries, documents)
        for row, query in zip(plain, queries, strict=True):
            unit_query = query / np.linalg.norm(query, axis=1, keepdims=True)
            for value, document in zip(row, documents, strict=True):
                unit_document = document / np.linalg.norm(document, axis=1, keepdims=True)
                self.assertAlmostEqual(value, float((unit_query @ unit_document.T).max(axis=1).sum()), places=4)
        np.testing.assert_allclose(chunked, plain, rtol=1e-5, atol=1e-6)

    def test_top_k_keeps_score_order_and_breaks_ties_by_corpus_order(self):
        scores = np.asarray([[1.0, 2.0, 0.0, 1.0]], dtype=np.float32)
        predictions = top_k_predictions(scores, ["q"], ["a", "b", "c", "d"], k=3)
        self.assertEqual(list(predictions["q"].items()), [("b", 2.0), ("a", 1.0), ("d", 1.0)])


class EncodeRunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "results"
        self.reference = reference_file(self.temporary.name)

    def run_reference(self, client, *flags):
        plan = plan_quality(parse("--reference", str(self.reference), *flags))
        data = load_quality_data(plan, tiny_loaders())
        return run_quality(plan, data, client, results_dir=self.results, now=clock)

    def test_documents_then_queries_in_reference_batches(self):
        client = FakeClient()
        record, result_path = self.run_reference(client)
        measured = client.calls[1:]
        self.assertEqual([len(items) for _, _, items, _ in measured], [2, 2, 1, 2, 1])
        self.assertEqual([kwargs["is_query"] for _, _, _, kwargs in measured], [False, False, False, True, True])
        for _, model, _, kwargs in measured:
            self.assertEqual(model, "example/sparse-encoder")
            self.assertEqual(kwargs["output_types"], ["sparse"])
            self.assertFalse(kwargs["wait_for_capacity"])
            self.assertEqual(kwargs["provision_timeout_s"], 300.0)
            self.assertEqual(kwargs["max_oom_retries"], 0)
            self.assertNotIn("instruction", kwargs)
        sent = [item for _, _, items, _ in measured[:3] for item in items]
        self.assertEqual([item["id"] for item in sent], ["d1", "d2", "d3", "d4", "d5"])
        self.assertEqual(sent[0]["text"], "Alpha alpha body")
        self.assertEqual(sent[2]["text"], "gamma body")
        self.assertEqual(result_path, self.results / f"{REFERENCE_ID}-20261102T120000Z.json")
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), record)

    def test_predictions_are_top_k_sparse_dot_products(self):
        record, _ = self.run_reference(FakeClient())
        predictions_path = Path(record["run"]["predictions_path"])
        self.assertEqual(predictions_path, self.results / f"{REFERENCE_ID}-20261102T120000Z.predictions.json")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        self.assertEqual(
            list(predictions["q1"].items()), [("d2", 2.0), ("d4", 1.5), ("d1", 1.0), ("d3", 0.0), ("d5", 0.0)]
        )
        self.assertEqual(next(iter(predictions["q2"])), "d3")
        self.assertEqual(next(iter(predictions["q3"])), "d2")
        self.assertEqual(set(predictions), {"q1", "q2", "q3"})

    def test_result_record_schema_and_comparison(self):
        record, _ = self.run_reference(FakeClient())
        self.assertEqual(set(record), RESULT_KEYS)
        self.assertEqual(record["kind"], "published-benchmark-reproduction")
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["reference_id"], REFERENCE_ID)
        self.assertTrue(record["comparable"])
        self.assertEqual(record["comparable_reasons"], [])
        self.assertEqual(record["comparison_basis"], "same parameters")
        self.assertEqual(record["evaluation_mode"], "quality")
        self.assertEqual(
            (record["model"], record["profile"], record["operation"]), ("example/sparse-encoder", "default", "encode")
        )
        self.assertEqual(record["model_version"], "0" * 40)
        self.assertEqual(record["execution_revisions"], ["rev-a"])
        self.assertEqual(record["ingress"], {"url": DEFAULT_ENDPOINT})
        self.assertEqual(record["task_key"], "mteb/SciFact")
        self.assertEqual(
            record["dataset"], {"name": "mteb/SciFact", "revision": REVISION, "subset": "test", "parameters": []}
        )
        self.assertEqual(record["payload"], {"items_per_request": 2, "output_contract": "sparse", "parameters": []})
        self.assertEqual(record["client"]["worker_count"], 1)
        self.assertEqual(record["client"]["processes_per_worker"], 1)
        self.assertTrue(record["client"]["description"].startswith("single process, "))
        self.assertEqual(record["scorer"]["id"], "pytrec-eval")
        self.assertFalse(record["scorer"]["reference_scorer_matched"])
        self.assertEqual(record["run"]["started_at"], "2026-11-02T12:00:00Z")
        self.assertEqual(record["run"]["endpoint"], DEFAULT_ENDPOINT)
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (5, 0))
        self.assertEqual(record["validity"], {"valid": True, "reasons": [], "sdk_retries": 0})
        names = {row["name"] for row in record["metrics"]}
        self.assertIn("main_score", names)
        self.assertIn("accuracy", names)
        self.assertEqual(sum(name.startswith("ndcg_at_") for name in names), len(K_VALUES))
        self.assertEqual({row["aggregation"] for row in record["metrics"]}, {"task_native"})
        self.assertEqual([row["name"] for row in record["comparison"]], ["main_score", "mrr_at_1", "ndcg_at_10"])
        main = next(row for row in record["comparison"] if row["name"] == "main_score")
        self.assertEqual(main["reference_value"], 0.5)
        self.assertAlmostEqual(main["delta"], main["value"] - 0.5, places=5)

    def test_smoke_key_is_not_comparable(self):
        record, _ = self.run_reference(FakeClient(), "--task", "mteb/SciFact?limit=2")
        self.assertFalse(record["comparable"])
        self.assertEqual(record["comparable_reasons"], ["smoke key mteb/SciFact?limit=2"])
        self.assertEqual(
            record["comparison_basis"], "same method, different parameters: smoke key mteb/SciFact?limit=2"
        )
        self.assertEqual(record["dataset"]["parameters"], [{"name": "limit", "value": "2"}])
        self.assertEqual(record["task_key"], "mteb/SciFact?limit=2")
        self.assertEqual(record["run"]["request_count"], 4)

    def test_sdk_retry_makes_the_run_invalid(self):
        record, _ = self.run_reference(FakeClient(retries=(0, 0, 1, 0, 2)))
        self.assertEqual(record["validity"], {"valid": False, "reasons": ["sdk retries 3"], "sdk_retries": 3})
        self.assertTrue(record["comparable"])

    def test_execution_revision_change_is_collected_and_not_comparable(self):
        record, _ = self.run_reference(FakeClient(revisions=("rev-b", "rev-a", "rev-b")))
        self.assertEqual(record["execution_revisions"], ["rev-a", "rev-b"])
        self.assertFalse(record["comparable"])
        self.assertEqual(record["comparable_reasons"], ["execution revision changed"])
        self.assertTrue(record["validity"]["valid"])

    def test_failed_request_stops_the_run_and_is_recorded(self):
        record, result_path = self.run_reference(FakeClient(fail_at=3))
        self.assertTrue(result_path.exists())
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (3, 0 + 1))
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertEqual(record["metrics"], [])
        self.assertEqual(record["comparison"], [])
        self.assertFalse(record["validity"]["valid"])
        self.assertEqual(record["validity"]["reasons"], ["request failed: RuntimeError: boom"])

    def test_retries_before_a_failed_request_are_recorded(self):
        record, _ = self.run_reference(FakeClient(retries=(0, 0, 1, 2), fail_at=3))
        self.assertEqual(record["validity"]["sdk_retries"], 3)
        self.assertEqual(record["validity"]["reasons"], ["sdk retries 3", "request failed: RuntimeError: boom"])

    def test_readiness_request_is_the_first_call_and_is_not_counted(self):
        client = FakeClient()
        record, _ = self.run_reference(client)
        kind, model, items, kwargs = client.calls[0]
        self.assertEqual((kind, model), ("encode", "example/sparse-encoder"))
        self.assertEqual([item["id"] for item in items], ["d1", "d2"])
        self.assertEqual(items, client.calls[1][2])
        self.assertTrue(kwargs["wait_for_capacity"])
        self.assertEqual(kwargs["provision_timeout_s"], 300.0)
        self.assertEqual(
            (kwargs["max_oom_retries"], kwargs["is_query"], kwargs["output_types"]), (0, False, ["sparse"])
        )
        self.assertEqual(len(client.calls), 6)
        self.assertEqual(record["run"]["request_count"], 5)

    def test_readiness_failure_skips_the_measured_phase(self):
        client = FakeClient(fail_at=0)
        record, result_path = self.run_reference(client)
        self.assertTrue(result_path.exists())
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(record["validity"]["valid"])
        self.assertEqual(record["validity"]["reasons"], ["readiness request failed: RuntimeError: boom"])
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (0, 0))
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertEqual((record["metrics"], record["comparison"]), ([], []))

    def test_retries_on_the_readiness_request_are_not_counted(self):
        record, _ = self.run_reference(FakeClient(retries=(4,)))
        self.assertEqual(record["validity"], {"valid": True, "reasons": [], "sdk_retries": 0})

    def test_a_malformed_response_is_recorded_as_a_failed_run(self):
        record, result_path = self.run_reference(ShortClient())
        self.assertTrue(result_path.exists())
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (1, 0))
        self.assertFalse(record["validity"]["valid"])
        (reason,) = record["validity"]["reasons"]
        self.assertTrue(reason.startswith("request failed: ValueError: "), reason)

    def test_multivector_run_without_reference(self):
        client = FakeClient(revisions=("r2", "r1"))
        plan = plan_quality(
            parse(
                "--model", "m/colbert", "--task", "mteb/SciFact", "--output", "multivector", "--items-per-request", "3"
            )
        )
        record, _ = run_quality(
            plan, load_quality_data(plan, tiny_loaders()), client, results_dir=self.results, now=clock
        )
        self.assertEqual([len(items) for _, _, items, _ in client.calls[1:]], [3, 2, 3])
        self.assertEqual({kwargs["output_types"][0] for _, _, _, kwargs in client.calls}, {"multivector"})
        predictions = json.loads(Path(record["run"]["predictions_path"]).read_text(encoding="utf-8"))
        self.assertEqual(list(predictions["q1"]), ["d2", "d3", "d1", "d4", "d5"])
        self.assertIsNone(record["reference_id"])
        self.assertIsNone(record["model_version"])
        self.assertEqual(record["comparable_reasons"], ["no reference", "execution revision changed"])
        self.assertEqual(record["execution_revisions"], ["r1", "r2"])
        self.assertEqual(record["comparison"], [])
        self.assertIsNone(record["comparison_basis"])
        self.assertTrue(Path(record["run"]["predictions_path"]).name.startswith("m-colbert-mteb-SciFact-"))


class RerankRunTest(unittest.TestCase):
    def test_one_score_call_per_query_with_its_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient()
            plan = plan_quality(parse("--model", "Qwen/Qwen3-Reranker-0.6B", "--task", "mteb/AskUbuntuDupQuestions"))
            data = load_quality_data(plan, tiny_loaders())
            record, _ = run_quality(plan, data, client, results_dir=Path(temporary) / "results", now=clock)
            predictions = json.loads(Path(record["run"]["predictions_path"]).read_text(encoding="utf-8"))
        self.assertEqual([kind for kind, _, _, _ in client.calls], ["score", "score", "score", "score"])
        self.assertEqual(client.calls[0][2], client.calls[1][2])
        self.assertTrue(client.calls[0][3]["wait_for_capacity"])
        measured = client.calls[1:]
        queries = [query for _, _, (query, _), _ in measured]
        self.assertEqual(queries, [{"text": "first query"}, {"text": "second query"}, {"text": "third query"}])
        items = [items for _, _, (_, items), _ in measured]
        self.assertEqual(
            [[item["id"] for item in batch] for batch in items], [["d2", "d1"], ["d3", "d5"], ["d4", "d2"]]
        )
        self.assertEqual(items[0][1]["text"], "Alpha alpha body")
        self.assertEqual(record["run"]["request_count"], 3)
        for _, _, _, kwargs in measured:
            self.assertFalse(kwargs["wait_for_capacity"])
            self.assertEqual(kwargs["provision_timeout_s"], 300.0)
            self.assertEqual(kwargs["max_oom_retries"], 0)
        self.assertEqual(
            predictions, {"q1": {"d2": 2.0, "d1": 1.0}, "q2": {"d3": 2.0, "d5": 1.0}, "q3": {"d4": 2.0, "d2": 1.0}}
        )
        self.assertEqual(record["payload"], {"items_per_request": 2, "output_contract": "scores", "parameters": []})
        self.assertEqual(record["operation"], "score")
        main = next(row for row in record["metrics"] if row["name"] == "main_score")
        map_1000 = next(row for row in record["metrics"] if row["name"] == "map_at_1000")
        self.assertEqual(main["value"], map_1000["value"])


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.reference = reference_file(self.temporary.name)

    def command(self, fake, *flags, environ=None):
        def measured(plan, data, client):
            return run_quality(plan, data, fake, results_dir=Path(self.temporary.name) / "results", now=clock)

        def loader(plan):
            return load_quality_data(plan, tiny_loaders())

        stdout, stderr = io.StringIO(), io.StringIO()
        code = None
        with (
            patch.dict("os.environ", environ or {}, clear=True),
            patch.object(run, "load_quality_data", side_effect=loader) as self.loader,
            patch.object(run, "run_quality", side_effect=measured),
            patch.object(run, "SIEClient") as client,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                run.main(["quality", "--reference", str(self.reference), *flags])
            except SystemExit as caught:
                code = caught.code
        return code, client, stderr.getvalue()

    def test_a_reference_run_without_a_key_fails_before_any_call(self):
        code, client, stderr = self.command(FakeClient())
        self.assertEqual(code, 2)
        client.assert_not_called()
        self.assertIn(f"SIE_API_KEY must be set in the environment for {DEFAULT_ENDPOINT}", stderr)

    def test_a_local_endpoint_runs_without_a_key(self):
        code, client, stderr = self.command(FakeClient(), "--endpoint", "http://localhost:8080")
        self.assertIsNone(code)
        client.assert_called_once_with("http://localhost:8080", api_key="", timeout_s=300.0)
        self.assertEqual(stderr, "")

    def test_a_key_is_passed_as_is(self):
        code, client, _ = self.command(FakeClient(), environ={"SIE_API_KEY": "test-key"})
        self.assertIsNone(code)
        client.assert_called_once_with(DEFAULT_ENDPOINT, api_key="test-key", timeout_s=300.0)

    def test_a_cloud_key_is_not_sent_to_another_endpoint(self):
        code, client, stderr = self.command(
            FakeClient(), "--endpoint", "http://localhost:8080", environ={"SIE_API_KEY": "cloud-key"}
        )
        self.assertIsNone(code)
        client.assert_called_once_with("http://localhost:8080", api_key="", timeout_s=300.0)
        self.assertIn(
            "note: SIE_API_KEY is scoped to https://api.superlinked.com and was not sent to http://localhost:8080",
            stderr,
        )

    def test_a_key_scoped_by_base_url_is_sent_to_that_endpoint(self):
        environ = {"SIE_BASE_URL": "http://localhost:8080/", "SIE_API_KEY": "local-token"}
        code, client, stderr = self.command(FakeClient(), "--endpoint", "http://localhost:8080", environ=environ)
        self.assertIsNone(code)
        client.assert_called_once_with("http://localhost:8080", api_key="local-token", timeout_s=300.0)
        self.assertEqual(stderr, "")

    def test_a_key_scope_is_an_origin_so_a_path_on_the_endpoint_still_matches(self):
        environ = {"SIE_BASE_URL": "http://localhost:8080", "SIE_API_KEY": "local-token"}
        code, client, stderr = self.command(FakeClient(), "--endpoint", "http://localhost:8080/sie/", environ=environ)
        self.assertIsNone(code)
        client.assert_called_once_with("http://localhost:8080/sie", api_key="local-token", timeout_s=300.0)
        self.assertEqual(stderr, "")

    def test_a_key_scoped_elsewhere_fails_on_the_default_endpoint(self):
        environ = {"SIE_BASE_URL": "http://localhost:8080", "SIE_API_KEY": "local-token"}
        code, client, stderr = self.command(FakeClient(), environ=environ)
        self.assertEqual(code, 2)
        client.assert_not_called()
        self.assertIn(f"SIE_API_KEY must be set in the environment for {DEFAULT_ENDPOINT}", stderr)

    def test_a_failed_run_prints_its_reasons_and_exits_1(self):
        for fake, reason in (
            (FakeClient(fail_at=0), "readiness request failed: RuntimeError: boom"),
            (FakeClient(fail_at=1), "request failed: RuntimeError: boom"),
            (ShortClient(), "request failed: ValueError: encode returned a different number of results"),
        ):
            with self.subTest(reason=reason):
                code, _, stderr = self.command(fake, "--endpoint", "http://localhost:8080")
                self.assertEqual(code, 1)
                self.assertTrue(stderr.startswith(f"invalid: {reason}"), stderr)
                self.assertNotIn("Traceback", stderr)

    def test_an_invalid_run_with_retries_prints_the_reason_and_exits_0(self):
        code, _, stderr = self.command(FakeClient(retries=(0, 2)), "--endpoint", "http://localhost:8080")
        self.assertIsNone(code)
        self.assertEqual(stderr, "invalid: sdk retries 2\n")

    def test_a_missing_reference_key_is_named(self):
        broken = json.loads(self.reference.read_text(encoding="utf-8"))
        del broken["task_key"]
        self.reference.write_text(json.dumps(broken), encoding="utf-8")
        code, client, stderr = self.command(FakeClient(), "--endpoint", "http://localhost:8080")
        self.assertEqual(code, 2)
        client.assert_not_called()
        self.assertIn(f"reference {self.reference} is missing key 'task_key'", stderr)

    def test_an_endpoint_without_a_scheme_is_a_usage_error(self):
        code, client, stderr = self.command(FakeClient(), "--endpoint", "localhost:8080")
        self.assertEqual(code, 2)
        client.assert_not_called()
        self.assertIn("endpoint must be an http(s) URL", stderr)

    def test_an_endpoint_without_a_host_or_with_a_bad_port_is_a_usage_error(self):
        for endpoint in ("http://:8080", "http://localhost:99999"):
            with self.subTest(endpoint=endpoint):
                code, client, stderr = self.command(FakeClient(), "--endpoint", endpoint)
                self.assertEqual(code, 2)
                self.loader.assert_not_called()
                client.assert_not_called()
                self.assertIn("endpoint must be an http(s) URL", stderr)

    def test_a_score_without_a_value_is_recorded_as_a_failed_run(self):
        self.reference = reference_file(
            self.temporary.name,
            model="Qwen/Qwen3-Reranker-0.6B",
            operation="score",
            task_key="mteb/AskUbuntuDupQuestions",
            dataset={"name": "mteb/AskUbuntuDupQuestions", "revision": REVISION, "subset": "test", "parameters": []},
            payload={"items_per_request": 2, "output_contract": "scores", "parameters": []},
        )
        code, _, stderr = self.command(NoneScoreClient(), "--endpoint", "http://localhost:8080")
        self.assertEqual(code, 1)
        (result_path,) = (Path(self.temporary.name) / "results").glob("*-????????T??????Z.json")
        record = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertIsNone(record["run"]["predictions_path"])
        (reason,) = record["validity"]["reasons"]
        self.assertTrue(reason.startswith("request failed: TypeError"), reason)
        self.assertTrue(stderr.startswith("invalid: request failed: TypeError"), stderr)


class ScoringTest(unittest.TestCase):
    def test_two_query_hand_computed_case(self):
        qrels = {"q1": {"a": 1}, "q2": {"c": 1}}
        predictions = {"q1": {"a": 1.0, "b": 0.5}, "q2": {"d": 1.0, "c": 0.5}}
        metrics = retrieval_metrics(predictions, qrels)
        expected = {
            "ndcg_at_1": 0.5,
            "ndcg_at_3": round((1.0 + 1.0 / np.log2(3.0)) / 2.0, 5),
            "map_at_1": 0.5,
            "map_at_3": 0.75,
            "recall_at_1": 0.5,
            "recall_at_3": 1.0,
            "precision_at_1": 0.5,
            "precision_at_3": 0.33333,
            "mrr_at_1": 0.5,
            "mrr_at_3": 0.75,
            "accuracy": 0.5,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value, places=5, msg=name)
        for prefix in ("ndcg", "map", "recall", "precision", "mrr"):
            for k in K_VALUES:
                self.assertIn(f"{prefix}_at_{k}", metrics)
        self.assertEqual(metrics["recall_at_1000"], 1.0)
        self.assertEqual(len(metrics), 5 * len(K_VALUES) + 1)

    def test_mrr_is_the_unrounded_mean_reciprocal_rank(self):
        qrels = {"q1": {"a": 1}, "q2": {"a": 1}, "q3": {"a": 1}}
        predictions = {"q1": {"a": 1.0}, "q2": {"b": 1.0}, "q3": {"b": 1.0}}
        metrics = retrieval_metrics(predictions, qrels)
        self.assertEqual(metrics["mrr_at_1"], 1.0 / 3.0)
        self.assertEqual(metrics["precision_at_1"], 0.33333)

    def test_graded_zero_is_not_relevant_and_queries_must_match(self):
        metrics = retrieval_metrics({"q": {"a": 1.0, "b": 0.5}}, {"q": {"a": 0, "b": 1}})
        self.assertEqual((metrics["mrr_at_3"], metrics["recall_at_1"], metrics["ndcg_at_1"]), (0.5, 0.0, 0.0))
        with self.assertRaises(ValueError):
            retrieval_metrics({"q": {"a": 1.0}}, {"q": {"a": 1}, "r": {"a": 1}})
        with self.assertRaises(ValueError):
            retrieval_metrics({"q": {"a": 1.0}, "r": {"a": 1.0}}, {"q": {"a": 1}})

    def test_task_metrics_copy_the_main_score(self):
        metrics = task_metrics({"q": {"a": 1.0}}, {"q": {"a": 1}}, "mteb/AskUbuntuDupQuestions")
        self.assertEqual(metrics["main_score"], metrics["map_at_1000"])
        metrics = task_metrics({"q": {"a": 1.0}}, {"q": {"a": 1}}, "mteb/NFCorpus")
        self.assertEqual(metrics["main_score"], metrics["ndcg_at_10"])

    def test_metric_rows_and_compare(self):
        rows = metric_rows({"ndcg_at_10": 0.5, "main_score": 0.5})
        self.assertEqual(rows[0], {"name": "main_score", "unit": "ratio", "aggregation": "task_native", "value": 0.5})
        reference = [
            {"name": "main_score", "unit": "ratio", "aggregation": "task_native", "value": 0.33056},
            {"name": "ndcg_at_10", "unit": "ratio", "aggregation": "p50", "value": 0.1},
        ]
        comparison = compare(rows, reference)
        self.assertEqual(
            comparison,
            [
                {
                    "name": "main_score",
                    "aggregation": "task_native",
                    "reference_value": 0.33056,
                    "value": 0.5,
                    "delta": 0.16944,
                }
            ],
        )
        lines = score.format_rows(rows, comparison)
        self.assertEqual(lines[0], "main_score           0.50000  reference 0.33056  delta +0.16944")
        self.assertEqual(lines[1], "ndcg_at_10           0.50000")

    def test_parse_task_key(self):
        self.assertEqual(parse_task_key("mteb/SciFact"), ("mteb/SciFact", {}))
        self.assertEqual(parse_task_key("mteb/SciFact?limit=50"), ("mteb/SciFact", {"limit": "50"}))
        with self.assertRaises(ValueError):
            parse_task_key("mteb/Unknown")


class ScoreCliTest(unittest.TestCase):
    def test_rescore_reproduces_the_result_and_writes_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = reference_file(temporary)
            plan = plan_quality(parse("--reference", str(reference), "--task", "mteb/SciFact?limit=2"))
            data = load_quality_data(plan, tiny_loaders())
            record, result_path = run_quality(
                plan, data, FakeClient(), results_dir=Path(temporary) / "results", now=clock
            )
            full_qrels = tiny_loaders().qrels("mteb/SciFact", REVISION)
            with patch("score.load_qrels", return_value=full_qrels) as loader:
                again, rows, comparison = score.rescore(result_path)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    score.main([str(result_path), "--write"])
            loader.assert_called_with("mteb/SciFact", REVISION, "test")
            self.assertEqual(again, record)
            self.assertEqual(rows, record["metrics"])
            self.assertEqual(comparison, record["comparison"])
            self.assertEqual(output.getvalue().splitlines(), score.format_rows(rows, comparison))
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), record)

    def test_rescore_refuses_a_result_whose_run_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = reference_file(temporary)
            plan = plan_quality(parse("--reference", str(reference)))
            data = load_quality_data(plan, tiny_loaders())
            _, result_path = run_quality(
                plan, data, FakeClient(fail_at=0), results_dir=Path(temporary) / "results", now=clock
            )
            with self.assertRaises(ValueError):
                score.rescore(result_path)
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                score.main([str(result_path)])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
