import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import httpx
from sie_sdk import ProvisioningError, ServerError, SIEConnectionError

import run
import score
from run import (
    DEFAULT_ENDPOINT,
    Dispatcher,
    Loaders,
    PerformanceData,
    build_parser,
    load_level_name,
    load_performance_data,
    performance_text,
    performance_validity,
    plan_performance,
    request_payload,
    request_rows,
    request_status,
    run_performance,
    schedule_offsets,
    select_ids,
)
from score import (
    PERFORMANCE_METRICS,
    nearest_rank,
    parse_task_key,
    performance_metric_rows,
    performance_metrics,
)

REFERENCE_ID = "example-reference-performance-01"
REVISION = "d56462d0e63a25450459c4f213e49ffdb866f7f9"
CORPUS_KEY = "mteb/SciFact?split=test&role=corpus&limit=4&filter=nonempty"
SCORE_KEY = "mteb/AskUbuntuDupQuestions?split=test&limit=2&candidates-k=2"
CORPUS = {
    "10": {"title": "Ten", "text": "ten body"},
    "9": {"title": "", "text": "nine body"},
    "100": {"title": "Hundred", "text": "   "},
    "2": {"text": "two body"},
    "33": {"title": "T", "text": "thirty-three body"},
}
SELECTED = ["10", "2", "33", "9"]
QUERIES = {"q2": "second query", "q1": "first query", "q10": "tenth query"}
TOP_RANKED = {"q1": ["10", "9", "2"], "q2": ["2", "33"], "q10": ["9"]}
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
    "load_level_name",
    "schedule",
    "timing",
    "run",
    "validity",
    "metrics",
    "comparison",
    "comparison_basis",
}


def timeout_error():
    error = SIEConnectionError("Request timed out")
    error.__cause__ = httpx.ReadTimeout("read timed out")
    return error


class FakeClient:
    def __init__(self, *, delay_s=0.005, failures=None, retries=None, revisions=("rev-a",), ready=True):
        self.calls = []
        self.delay_s = delay_s
        self.failures = dict(failures or {})
        self.retries = dict(retries or {})
        self.revisions = list(revisions)
        self.ready = ready
        self._lock = threading.Lock()
        self._state = threading.local()

    @property
    def last_retry_count(self):
        return getattr(self._state, "retry_count", 0)

    @property
    def last_model_revision(self):
        return getattr(self._state, "revision", None)

    def _observe(self, call, key, kwargs):
        with self._lock:
            self.calls.append(call)
            index = len(self.calls) - 1
        if kwargs.get("wait_for_capacity") and not self.ready:
            raise ProvisioningError("no capacity")
        time.sleep(self.delay_s)
        self._state.retry_count = self.retries.get(key, 0)
        self._state.revision = None
        failure = self.failures.get(key)
        if failure is not None:
            raise failure
        self._state.revision = self.revisions[index % len(self.revisions)]

    def encode(self, model, items, **kwargs):
        self._observe(("encode", model, items, kwargs), items[0]["id"], kwargs)
        return [{"id": item["id"], "sparse": {"indices": [0], "values": [1.0]}} for item in items]

    def score(self, model, query, items, **kwargs):
        self._observe(("score", model, (query, items), kwargs), query["text"], kwargs)
        return {"model": model, "query_id": "", "scores": [{"item_id": item["id"], "score": 1.0} for item in items]}


def tiny_loaders():
    return Loaders(
        corpus=lambda name, revision: {doc_id: dict(doc) for doc_id, doc in CORPUS.items()},
        queries=lambda name, revision: dict(QUERIES),
        qrels=lambda name, revision: {},
        top_ranked=lambda name, revision: {query_id: list(docs) for query_id, docs in TOP_RANKED.items()},
    )


def reference_file(directory, **overrides):
    reference = {
        "kind": "published-benchmark-reference",
        "schema_version": 1,
        "reference_id": REFERENCE_ID,
        "evaluation_mode": "isolated_performance",
        "model": "example/sparse-encoder",
        "model_version": "0" * 40,
        "profile": "default",
        "operation": "encode",
        "task_key": CORPUS_KEY,
        "dataset": {
            "name": "mteb/SciFact",
            "revision": REVISION,
            "subset": "test",
            "parameters": [
                {"name": "limit", "value": 4},
                {"name": "role", "value": "corpus"},
                {"name": "filter", "value": "nonempty"},
            ],
        },
        "payload": {
            "items_per_request": 1,
            "output_contract": "sparse",
            "parameters": [{"name": "is_query", "value": False}],
        },
        "load_level_name": "fixed-50-rps",
        "schedule": {
            "arrival": "deterministic",
            "levels": [{"name": "fixed-50-rps", "rate": 50.0, "duration_s": 0.4}],
            "seed": 0,
            "unit": "requests_per_second",
        },
        "timing": {
            "warmup_s": 0.0,
            "settle_s": 0.0,
            "drain_s": 0.0,
            "request_timeout_s": 5.0,
            "target_readiness_timeout_s": 30.0,
            "max_in_flight_per_process": 4,
            "scheduling_lag_tolerance_ms": 20.0,
            "retry": {"backoff": "none", "max_attempts": 1, "record_attempts": True},
        },
        "client": {"worker_count": 2, "processes_per_worker": 2},
        "scorer": None,
        "metrics": [
            {
                "name": "offered_requests_per_second",
                "unit": "requests_per_second",
                "aggregation": "campaign_global",
                "value": 50.0,
            },
            {"name": "success_rate", "unit": "ratio", "aggregation": "campaign_global", "value": 1.0},
            {"name": "latency_p50_ms", "unit": "milliseconds", "aggregation": "p50", "value": 10.0},
        ],
        "accepted_at": "2026-09-05T05:18:08Z",
        "measurement_started_at": "2026-09-05T04:47:35Z",
    }
    reference.update(overrides)
    path = Path(directory) / f"{REFERENCE_ID}.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    return path


def parse(*argv):
    return build_parser().parse_args(["performance", *argv])


def clock():
    return datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)


def request_row(sequence, lag_ms, latency_ms, status="success", retries=0):
    offset_ns = sequence * 1_000_000_000
    sent_ns = offset_ns + int(lag_ms * 1_000_000)
    return {
        "sequence": sequence,
        "level_name": "fixed-1-rps",
        "offset_s": float(sequence),
        "actual_send_offset_ns": sent_ns,
        "completed_offset_ns": sent_ns + int(latency_ms * 1_000_000),
        "scheduling_lag_ms": lag_ms,
        "status": status,
        "attempt": 1,
        "client_retry_count": retries,
        "in_flight_at_send": 0,
    }


LEVEL_ROWS = [
    request_row(0, 1.0, 100.0),
    request_row(1, 2.0, 200.0),
    request_row(2, 0.0, 300.0),
    request_row(3, 3.0, 30000.0, status="timeout"),
    request_row(4, 0.5, 400.0),
    request_row(5, 1.0, 500.0, retries=1),
    request_row(6, 25.0, 600.0),
    request_row(7, 0.0, 10.0, status="error:RuntimeError"),
    request_row(8, 0.0, 2600.0),
    request_row(9, 4.0, 700.0),
]


class ScheduleTest(unittest.TestCase):
    def test_offsets_are_exact_fractions_below_the_duration(self):
        for rate, count in ((0.25, 15), (1.0, 60), (10.0, 600)):
            with self.subTest(rate=rate):
                offsets = schedule_offsets(rate, 60.0)
                self.assertEqual(len(offsets), count)
                self.assertEqual(offsets, [Fraction(index) / Fraction(str(rate)) for index in range(count)])
                self.assertLess(offsets[-1], 60)
        self.assertEqual(schedule_offsets(50.0, 0.4), [Fraction(index, 50) for index in range(20)])
        self.assertEqual(schedule_offsets(3.0, 1.0), [Fraction(0), Fraction(1, 3), Fraction(2, 3)])

    def test_level_name_writes_the_decimal_point_as_a_hyphen(self):
        self.assertEqual(load_level_name(0.25), "fixed-0-25-rps")
        self.assertEqual(load_level_name(1.0), "fixed-1-rps")
        self.assertEqual(load_level_name(10.0), "fixed-10-rps")
        self.assertEqual(load_level_name(2.5), "fixed-2-5-rps")


class RowAssignmentTest(unittest.TestCase):
    def test_encode_rows_wrap_modulo_the_dataset_length(self):
        self.assertEqual([request_rows(sequence, 1, 4) for sequence in range(6)], [[0], [1], [2], [3], [0], [1]])
        self.assertEqual(request_rows(0, 3, 5), [0, 1, 2])
        self.assertEqual(request_rows(1, 3, 5), [3, 4, 0])
        self.assertEqual(request_rows(2, 3, 5), [1, 2, 3])

    def test_payloads_follow_the_selected_list(self):
        encode = plan_performance(
            parse("--model", "m", "--task", CORPUS_KEY, "--output", "sparse", "--items-per-request", "3", "--rate", "1")
        )
        data = load_performance_data(encode, tiny_loaders())
        self.assertEqual([item["id"] for item in request_payload(encode, data, 1)], ["9", "10", "2"])
        rerank = plan_performance(parse("--model", "m", "--task", SCORE_KEY, "--rate", "1"))
        data = load_performance_data(rerank, tiny_loaders())
        self.assertEqual(request_payload(rerank, data, 0), request_payload(rerank, data, 2))
        self.assertEqual(request_payload(rerank, data, 3)[0], {"text": "tenth query"})
        with self.assertRaises(ValueError):
            request_payload(rerank, PerformanceData([], []), 0)


class SelectionTest(unittest.TestCase):
    def test_nonempty_filter_then_string_sorted_ids_then_first_n(self):
        texts = {doc_id: doc["text"] for doc_id, doc in CORPUS.items()}
        self.assertEqual(select_ids(texts, {"filter": "nonempty", "limit": "2"}), ["10", "2"])
        self.assertEqual(select_ids(texts, {"filter": "nonempty"}), SELECTED)
        self.assertEqual(select_ids(texts, {"limit": "3"}), ["10", "100", "2"])

    def test_corpus_role_sends_title_newline_text(self):
        plan = plan_performance(
            parse("--model", "m", "--task", CORPUS_KEY, "--output", "sparse", "--items-per-request", "1", "--rate", "1")
        )
        data = load_performance_data(plan, tiny_loaders())
        self.assertEqual(data.selected_ids, SELECTED)
        self.assertEqual(
            data.payloads,
            [
                {"id": "10", "text": "Ten\nten body"},
                {"id": "2", "text": "two body"},
                {"id": "33", "text": "T\nthirty-three body"},
                {"id": "9", "text": "nine body"},
            ],
        )
        self.assertEqual(performance_text({"title": "", "text": " beta "}), " beta ")

    def test_queries_role_by_default(self):
        plan = plan_performance(
            parse(
                "--model",
                "m",
                "--task",
                "mteb/SciFact?limit=2",
                "--output",
                "sparse",
                "--items-per-request",
                "1",
                "--rate",
                "1",
            )
        )
        data = load_performance_data(plan, tiny_loaders())
        self.assertEqual(data.selected_ids, ["q1", "q10"])
        self.assertEqual(data.payloads, [{"id": "q1", "text": "first query"}, {"id": "q10", "text": "tenth query"}])

    def test_score_task_keeps_each_query_candidates_in_source_order(self):
        plan = plan_performance(parse("--model", "m", "--task", SCORE_KEY, "--rate", "1"))
        data = load_performance_data(plan, tiny_loaders())
        self.assertEqual(data.selected_ids, ["q1", "q10"])
        self.assertEqual(
            data.payloads,
            [
                ({"text": "first query"}, [{"id": "10", "text": "Ten\nten body"}, {"id": "9", "text": "nine body"}]),
                ({"text": "tenth query"}, [{"id": "9", "text": "nine body"}]),
            ],
        )
        whole = plan_performance(parse("--model", "m", "--task", "mteb/AskUbuntuDupQuestions", "--rate", "1"))
        data = load_performance_data(whole, tiny_loaders())
        self.assertEqual(data.selected_ids, ["q1", "q10", "q2"])
        self.assertEqual([doc["id"] for doc in data.payloads[0][1]], ["10", "9", "2"])

    def test_parse_task_key_keeps_every_performance_parameter(self):
        self.assertEqual(
            parse_task_key("mteb/NFCorpus?split=test&role=corpus&limit=256&filter=nonempty"),
            ("mteb/NFCorpus", {"split": "test", "role": "corpus", "limit": "256", "filter": "nonempty"}),
        )
        self.assertEqual(
            parse_task_key("mteb/AskUbuntuDupQuestions?split=test&limit=64&candidates-k=20"),
            ("mteb/AskUbuntuDupQuestions", {"split": "test", "limit": "64", "candidates-k": "20"}),
        )


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.reference = reference_file(self.temporary.name)

    def test_reference_values_are_taken_from_the_file(self):
        plan = plan_performance(parse("--reference", str(self.reference)))
        self.assertEqual(
            (plan.reference_id, plan.model, plan.profile, plan.route),
            (REFERENCE_ID, "example/sparse-encoder", "default", "example/sparse-encoder"),
        )
        self.assertEqual((plan.operation, plan.output_contract, plan.items_per_request), ("encode", "sparse", 1))
        self.assertEqual((plan.rate, plan.duration_s, plan.warmup_s), (50.0, 0.4, 0.0))
        self.assertEqual((plan.settle_s, plan.drain_s, plan.request_timeout_s), (0.0, 0.0, 5.0))
        self.assertEqual((plan.target_readiness_timeout_s, plan.max_in_flight, plan.lag_tolerance_ms), (30.0, 4, 20.0))
        self.assertEqual((plan.task_key, plan.task_name), (CORPUS_KEY, "mteb/SciFact"))
        self.assertEqual(plan.task_params, {"split": "test", "role": "corpus", "limit": "4", "filter": "nonempty"})
        self.assertEqual((plan.revision, plan.model_version, plan.endpoint), (REVISION, "0" * 40, DEFAULT_ENDPOINT))
        self.assertEqual(plan.comparable_reasons, ["client processes 1 != 4"])
        self.assertEqual(plan.level_name, "fixed-50-rps")

    def test_overrides_are_comparable_reasons(self):
        plan = plan_performance(parse("--reference", str(self.reference), "--rate", "100"))
        self.assertEqual(plan.comparable_reasons, ["rate 100 != 50", "client processes 1 != 4"])
        self.assertEqual((plan.rate, plan.level_name), (100.0, "fixed-100-rps"))
        plan = plan_performance(
            parse("--reference", str(self.reference), "--rate", "50", "--duration", "1", "--warmup", "2")
        )
        self.assertEqual(plan.comparable_reasons, ["duration 1 != 0.4", "warmup 2 != 0", "client processes 1 != 4"])
        plan = plan_performance(parse("--reference", str(self.reference), "--endpoint", "http://localhost:8080"))
        self.assertEqual(
            plan.comparable_reasons,
            [f"endpoint http://localhost:8080 != {DEFAULT_ENDPOINT}", "client processes 1 != 4"],
        )

    def test_reference_fixes_model_output_items_and_task(self):
        for flags in (
            ["--model", "x"],
            ["--output", "sparse"],
            ["--items-per-request", "4"],
            ["--task", "mteb/SciFact"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                plan_performance(parse("--reference", str(self.reference), *flags))
        same = plan_performance(parse("--reference", str(self.reference), "--task", CORPUS_KEY))
        self.assertEqual(same.comparable_reasons, ["client processes 1 != 4"])

    def test_reference_kind_mode_and_id_are_checked(self):
        quality = reference_file(self.temporary.name, evaluation_mode="quality")
        with self.assertRaises(ValueError):
            plan_performance(parse("--reference", str(quality)))
        escaped = reference_file(self.temporary.name, reference_id="../escape")
        with self.assertRaises(ValueError):
            plan_performance(parse("--reference", str(escaped)))

    def test_timing_values_must_be_sane(self):
        for flags in (["--rate", "0"], ["--rate", "-1"], ["--duration", "0"], ["--warmup", "-1"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                plan_performance(parse("--reference", str(self.reference), *flags))

    def test_flags_without_reference(self):
        with self.assertRaises(ValueError):
            plan_performance(
                parse("--model", "m", "--task", CORPUS_KEY, "--output", "sparse", "--items-per-request", "1")
            )
        with self.assertRaises(ValueError):
            plan_performance(parse("--model", "m", "--task", CORPUS_KEY, "--rate", "1"))
        with self.assertRaises(ValueError):
            plan_performance(parse("--model", "m", "--task", SCORE_KEY, "--output", "sparse", "--rate", "1"))
        with self.assertRaises(ValueError):
            plan_performance(parse("--model", "m", "--task", SCORE_KEY, "--items-per-request", "20", "--rate", "1"))
        encode = plan_performance(
            parse(
                "--model",
                "BAAI/bge-m3:sparse",
                "--task",
                CORPUS_KEY,
                "--output",
                "sparse",
                "--items-per-request",
                "2",
                "--rate",
                "10",
            )
        )
        self.assertEqual((encode.model, encode.profile, encode.route), ("BAAI/bge-m3", "sparse", "BAAI/bge-m3:sparse"))
        self.assertEqual((encode.operation, encode.output_contract, encode.items_per_request), ("encode", "sparse", 2))
        self.assertEqual((encode.rate, encode.duration_s, encode.warmup_s), (10.0, 60.0, 30.0))
        self.assertEqual((encode.settle_s, encode.drain_s, encode.target_readiness_timeout_s), (10.0, 10.0, 300.0))
        self.assertEqual((encode.request_timeout_s, encode.max_in_flight, encode.lag_tolerance_ms), (30.0, 64, 20.0))
        self.assertEqual(encode.comparable_reasons, ["no reference"])
        self.assertIsNone(encode.reference_id)
        self.assertEqual(encode.revision, score.HF_DATASETS["mteb/SciFact"][1])
        rerank = plan_performance(
            parse("--model", "Qwen/Qwen3-Reranker-0.6B", "--task", SCORE_KEY, "--rate", "0.25", "--warmup", "0")
        )
        self.assertEqual(
            (rerank.operation, rerank.output_contract, rerank.items_per_request), ("score", "scores", None)
        )
        self.assertEqual((rerank.request_timeout_s, rerank.max_in_flight, rerank.warmup_s), (300.0, 128, 0.0))
        self.assertEqual(rerank.level_name, "fixed-0-25-rps")

    def test_key_parameters_are_checked(self):
        for key in (
            "mteb/SciFact?split=train&role=corpus",
            "mteb/SciFact?role=documents",
            "mteb/SciFact?filter=all",
            "mteb/SciFact?limit=0",
            "mteb/AskUbuntuDupQuestions?candidates-k=0",
            "mteb/AskUbuntuDupQuestions?role=corpus",
        ):
            flags = [] if key.startswith("mteb/AskUbuntu") else ["--output", "sparse", "--items-per-request", "1"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_performance(parse("--model", "m", "--task", key, "--rate", "1", *flags))


class AggregationTest(unittest.TestCase):
    def test_every_metric_of_a_hand_built_level(self):
        metrics = performance_metrics(LEVEL_ROWS, 1.0, 10.0)
        self.assertEqual(
            metrics,
            {
                "offered_requests_per_second": 1.0,
                "achieved_requests_per_second": 0.7,
                "eventual_achieved_requests_per_second": 0.8,
                "in_window_success_rate": 0.7,
                "in_window_retry_free_success_rate": 0.6,
                "success_rate": 0.8,
                "retry_free_success_rate": 0.7,
                "scheduling_lag_p99_ms": 25.0,
                "latency_p50_ms": 400.0,
                "latency_p95_ms": 2600.0,
                "latency_p99_ms": 2600.0,
            },
        )
        rows = performance_metric_rows(metrics)
        self.assertEqual([row["name"] for row in rows], list(PERFORMANCE_METRICS))
        self.assertEqual(
            rows[1],
            {
                "name": "achieved_requests_per_second",
                "unit": "requests_per_second",
                "aggregation": "common_window",
                "value": 0.7,
            },
        )
        self.assertEqual(rows[-1]["aggregation"], "p99")
        lines = score.format_rows(rows, [])
        self.assertEqual(lines[2], "eventual_achieved_requests_per_second 0.80000")

    def test_nearest_rank_percentiles(self):
        fifteen = list(range(1, 16))
        self.assertEqual((nearest_rank(fifteen, 50), nearest_rank(fifteen, 95), nearest_rank(fifteen, 99)), (8, 15, 15))
        hundred = list(range(100, 0, -1))
        self.assertEqual(
            (nearest_rank(hundred, 50), nearest_rank(hundred, 95), nearest_rank(hundred, 99)), (50, 95, 99)
        )
        self.assertEqual(nearest_rank([7.0], 99), 7.0)
        with self.assertRaises(ValueError):
            nearest_rank([], 50)

    def test_no_success_leaves_the_latency_metrics_out(self):
        rows = [request_row(sequence, 0.0, 100.0, status="timeout") for sequence in range(3)]
        metrics = performance_metrics(rows, 1.0, 3.0)
        self.assertEqual(metrics["success_rate"], 0.0)
        self.assertEqual(metrics["achieved_requests_per_second"], 0.0)
        self.assertNotIn("latency_p50_ms", metrics)
        self.assertEqual(len(performance_metric_rows(metrics)), 8)
        with self.assertRaises(ValueError):
            performance_metrics([], 1.0, 3.0)

    def test_request_status_classification(self):
        self.assertEqual(request_status(timeout_error()), "timeout")
        self.assertEqual(request_status(httpx.ConnectTimeout("slow")), "timeout")
        self.assertEqual(request_status(SIEConnectionError("refused")), "error:SIEConnectionError")
        self.assertEqual(request_status(ServerError("boom", status_code=503)), "error:ServerError")
        self.assertEqual(request_status(KeyError("scores")), "error:KeyError")


class ValidityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.plan = plan_performance(parse("--reference", str(reference_file(self.temporary.name))))

    def dispatcher(self):
        return Dispatcher(self.plan, PerformanceData([], []), None)

    def test_clean_run_is_valid(self):
        validity = performance_validity(self.plan, self.dispatcher(), {"scheduling_lag_p99_ms": 20.0})
        self.assertEqual(
            validity,
            {"valid": True, "reasons": [], "lag_max_ms": 0.0, "in_flight_peak": 0, "timeouts": 0, "sdk_retries": 0},
        )

    def test_each_rule_names_its_reason(self):
        dispatcher = self.dispatcher()
        dispatcher.log.sdk_retries = 2
        self.assertEqual(performance_validity(self.plan, dispatcher, {})["reasons"], ["sdk retries 2"])
        validity = performance_validity(self.plan, self.dispatcher(), {"scheduling_lag_p99_ms": 20.5})
        self.assertEqual(validity["reasons"], ["scheduling lag p99 20.500 ms > 20 ms"])
        dispatcher = self.dispatcher()
        dispatcher.in_flight_peak = 5
        self.assertEqual(performance_validity(self.plan, dispatcher, {})["reasons"], ["in-flight peak 5 > 4"])
        dispatcher = self.dispatcher()
        dispatcher.readiness_failure = "ProvisioningError: no capacity"
        self.assertEqual(
            performance_validity(self.plan, dispatcher, {})["reasons"],
            ["readiness request failed: ProvisioningError: no capacity"],
        )
        dispatcher = self.dispatcher()
        dispatcher.log.execution_revisions = ["rev-a", "rev-b"]
        self.assertEqual(performance_validity(self.plan, dispatcher, {})["reasons"], ["execution revision changed"])

    def test_counts_come_from_the_rows(self):
        dispatcher = self.dispatcher()
        dispatcher.rows = [request_row(0, 1.5, 10.0), request_row(1, 0.0, 10.0, status="timeout")]
        validity = performance_validity(self.plan, dispatcher, {"scheduling_lag_p99_ms": 1.5})
        self.assertEqual((validity["valid"], validity["lag_max_ms"], validity["timeouts"]), (True, 1.5, 1))


class RunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "results"
        self.reference = reference_file(self.temporary.name)

    def run_reference(self, client, *flags, reference=None):
        plan = plan_performance(parse("--reference", str(reference or self.reference), *flags))
        data = load_performance_data(plan, tiny_loaders())
        return run_performance(plan, data, client, results_dir=self.results, now=clock)

    def test_timed_fixed_rate_run(self):
        client = FakeClient()
        record, result_path = self.run_reference(client)
        self.assertEqual(result_path, self.results / f"{REFERENCE_ID}-20261102T120000Z.json")
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), record)
        readiness, *scheduled = client.calls
        self.assertEqual(len(scheduled), 20)
        self.assertEqual(readiness[0], "encode")
        self.assertEqual([item["id"] for item in readiness[2]], ["10"])
        self.assertTrue(readiness[3]["wait_for_capacity"])
        self.assertEqual(readiness[3]["provision_timeout_s"], 30.0)
        for kind, model, items, kwargs in scheduled:
            self.assertEqual((kind, model, len(items)), ("encode", "example/sparse-encoder", 1))
            self.assertEqual(kwargs["output_types"], ["sparse"])
            self.assertFalse(kwargs["is_query"])
            self.assertFalse(kwargs["wait_for_capacity"])
            self.assertEqual(kwargs["provision_timeout_s"], 5.0)
            self.assertEqual(kwargs["max_oom_retries"], 0)
            self.assertNotIn("instruction", kwargs)
        self.assertEqual(
            Counter(items[0]["id"] for _, _, items, _ in scheduled), Counter({"10": 5, "2": 5, "33": 5, "9": 5})
        )
        self.assertEqual(
            {items[0]["text"] for _, _, items, _ in scheduled if items[0]["id"] == "10"}, {"Ten\nten body"}
        )
        requests_path = Path(record["run"]["requests_path"])
        self.assertEqual(requests_path, self.results / f"{REFERENCE_ID}-20261102T120000Z.requests.jsonl")
        rows = score.read_jsonl(requests_path)
        self.assertEqual([row["sequence"] for row in rows], list(range(20)))
        for row in rows:
            self.assertEqual(row["offset_s"], row["sequence"] / 50)
            self.assertEqual(row["level_name"], "fixed-50-rps")
            self.assertEqual(row["status"], "success")
            self.assertEqual((row["attempt"], row["client_retry_count"]), (1, 0))
            self.assertLessEqual(row["sequence"] * 20_000_000, row["actual_send_offset_ns"])
            self.assertLessEqual(row["actual_send_offset_ns"], row["completed_offset_ns"])
            self.assertGreaterEqual(row["scheduling_lag_ms"], 0.0)
            self.assertGreaterEqual(row["in_flight_at_send"], 0)
            self.assertEqual(
                list(row),
                [
                    "sequence",
                    "level_name",
                    "offset_s",
                    "actual_send_offset_ns",
                    "completed_offset_ns",
                    "scheduling_lag_ms",
                    "status",
                    "attempt",
                    "client_retry_count",
                    "in_flight_at_send",
                ],
            )
        self.assertEqual(set(record), RESULT_KEYS)
        self.assertEqual((record["kind"], record["schema_version"]), ("published-benchmark-reproduction", 1))
        self.assertEqual(record["evaluation_mode"], "isolated_performance")
        self.assertFalse(record["comparable"])
        self.assertEqual(record["comparable_reasons"], ["client processes 1 != 4"])
        self.assertEqual(record["comparison_basis"], "same method, different parameters: client processes 1 != 4")
        self.assertEqual(record["execution_revisions"], ["rev-a"])
        self.assertEqual(record["dataset"]["selected_ids"], SELECTED)
        self.assertEqual(
            record["dataset"]["parameters"],
            [
                {"name": "split", "value": "test"},
                {"name": "role", "value": "corpus"},
                {"name": "limit", "value": "4"},
                {"name": "filter", "value": "nonempty"},
            ],
        )
        self.assertEqual(
            record["payload"],
            {"items_per_request": 1, "output_contract": "sparse", "parameters": [{"name": "is_query", "value": False}]},
        )
        self.assertEqual(record["load_level_name"], "fixed-50-rps")
        self.assertEqual(
            record["schedule"],
            {"arrival": "deterministic", "levels": [{"name": "fixed-50-rps", "rate": 50.0, "duration_s": 0.4}]},
        )
        self.assertEqual(record["timing"]["max_in_flight_per_process"], 4)
        self.assertEqual(record["timing"]["retry"], {"backoff": "none", "max_attempts": 1, "record_attempts": True})
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (20, 0))
        self.assertEqual(record["run"]["started_at"], "2026-11-02T12:00:00Z")
        self.assertEqual(
            list(record["validity"]), ["valid", "reasons", "lag_max_ms", "in_flight_peak", "timeouts", "sdk_retries"]
        )
        self.assertEqual((record["validity"]["timeouts"], record["validity"]["sdk_retries"]), (0, 0))
        self.assertGreaterEqual(record["validity"]["in_flight_peak"], 1)
        self.assertLessEqual(record["validity"]["in_flight_peak"], 4)
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual(set(metrics), set(PERFORMANCE_METRICS))
        self.assertEqual(metrics["offered_requests_per_second"], 50.0)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["eventual_achieved_requests_per_second"], 50.0)
        self.assertGreaterEqual(metrics["latency_p50_ms"], 5.0)
        self.assertEqual(
            [row["name"] for row in record["comparison"]],
            ["offered_requests_per_second", "success_rate", "latency_p50_ms"],
        )
        self.assertEqual(record["comparison"][0]["delta"], 0.0)
        self.assertNotIn("scorer", record)

    def test_requests_file_re_aggregates_to_the_same_metrics(self):
        record, result_path = self.run_reference(FakeClient())
        again, rows, comparison = score.rescore(result_path)
        self.assertEqual(again, record)
        self.assertEqual(rows, record["metrics"])
        self.assertEqual(comparison, record["comparison"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            score.main([str(result_path), "--write"])
        self.assertEqual(output.getvalue().splitlines(), score.format_rows(rows, comparison))
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), record)

    def test_timeouts_and_errors_lower_the_success_rates(self):
        failures = {"2": timeout_error(), "33": ServerError("boom", status_code=503), "9": RuntimeError("malformed")}
        record, result_path = self.run_reference(FakeClient(failures=failures))
        rows = score.read_jsonl(Path(record["run"]["requests_path"]))
        expected = {1: "timeout", 2: "error:ServerError", 3: "error:RuntimeError"}
        self.assertEqual(
            [row["status"] for row in rows], [expected.get(row["sequence"] % 4, "success") for row in rows]
        )
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (20, 15))
        self.assertEqual(record["validity"]["timeouts"], 5)
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual((metrics["success_rate"], metrics["eventual_achieved_requests_per_second"]), (0.25, 12.5))
        self.assertEqual([r for r in record["validity"]["reasons"] if not r.startswith("scheduling lag")], [])
        self.assertEqual(score.rescore(result_path)[1], record["metrics"])

    def test_sdk_retries_invalidate_and_lower_the_retry_free_rates(self):
        record, _ = self.run_reference(FakeClient(retries={"9": 1}))
        rows = score.read_jsonl(Path(record["run"]["requests_path"]))
        self.assertEqual(
            [row["client_retry_count"] for row in rows], [1 if row["sequence"] % 4 == 3 else 0 for row in rows]
        )
        self.assertEqual(record["validity"]["sdk_retries"], 5)
        self.assertIn("sdk retries 5", record["validity"]["reasons"])
        self.assertFalse(record["validity"]["valid"])
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual((metrics["success_rate"], metrics["retry_free_success_rate"]), (1.0, 0.75))
        self.assertLessEqual(metrics["in_window_retry_free_success_rate"], 0.75)

    def test_execution_revision_change_is_not_comparable_and_not_valid(self):
        record, _ = self.run_reference(FakeClient(revisions=("rev-a", "rev-b")))
        self.assertEqual(set(record["execution_revisions"]), {"rev-a", "rev-b"})
        self.assertEqual(record["comparable_reasons"], ["client processes 1 != 4", "execution revision changed"])
        self.assertIn("execution revision changed", record["validity"]["reasons"])
        self.assertFalse(record["validity"]["valid"])

    def test_readiness_failure_writes_the_result_and_sends_nothing_else(self):
        client = FakeClient(ready=False)
        record, result_path = self.run_reference(client)
        self.assertTrue(result_path.exists())
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(record["run"]["requests_path"])
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (0, 0))
        self.assertEqual((record["metrics"], record["comparison"]), ([], []))
        self.assertEqual(
            record["validity"],
            {
                "valid": False,
                "reasons": ["readiness request failed: ProvisioningError: no capacity"],
                "lag_max_ms": 0.0,
                "in_flight_peak": 0,
                "timeouts": 0,
                "sdk_retries": 0,
            },
        )
        with self.assertRaises(ValueError):
            score.rescore(result_path)
        client = FakeClient(failures={"10": RuntimeError("malformed")})
        record, _ = self.run_reference(client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(record["validity"]["reasons"], ["readiness request failed: RuntimeError: malformed"])

    def test_warmup_is_a_closed_loop_on_row_zero_that_is_not_recorded(self):
        client = FakeClient()
        record, _ = self.run_reference(client, "--warmup", "0.05")
        self.assertEqual(record["comparable_reasons"], ["warmup 0.05 != 0", "client processes 1 != 4"])
        self.assertEqual(record["timing"]["warmup_s"], 0.05)
        warm = client.calls[1 : len(client.calls) - 20]
        self.assertGreaterEqual(len(warm), 2)
        self.assertEqual({items[0]["id"] for _, _, items, _ in warm}, {"10"})
        self.assertEqual({kwargs["wait_for_capacity"] for _, _, _, kwargs in warm}, {False})
        self.assertEqual(record["run"]["request_count"], 20)

    def test_the_pool_bounds_in_flight_requests_and_the_lag_gate_binds(self):
        reference = Path(self.temporary.name) / "bounded.json"
        payload = json.loads(self.reference.read_text(encoding="utf-8"))
        payload["schedule"]["levels"] = [{"name": "fixed-100-rps", "rate": 100.0, "duration_s": 0.1}]
        payload["timing"]["max_in_flight_per_process"] = 2
        reference.write_text(json.dumps(payload), encoding="utf-8")
        record, _ = self.run_reference(FakeClient(delay_s=0.05), reference=reference)
        rows = score.read_jsonl(Path(record["run"]["requests_path"]))
        self.assertEqual(len(rows), 10)
        self.assertEqual(record["validity"]["in_flight_peak"], 2)
        self.assertLessEqual(max(row["in_flight_at_send"] for row in rows), 1)
        self.assertFalse(record["validity"]["valid"])
        self.assertTrue(record["validity"]["reasons"][0].startswith("scheduling lag p99 "))
        self.assertTrue(record["validity"]["reasons"][0].endswith(" ms > 20 ms"))
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertGreater(metrics["scheduling_lag_p99_ms"], 20.0)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertLess(metrics["achieved_requests_per_second"], metrics["eventual_achieved_requests_per_second"])

    def test_score_run_sends_one_query_with_its_candidates(self):
        payload = json.loads(self.reference.read_text(encoding="utf-8"))
        payload.update(
            {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "operation": "score",
                "task_key": SCORE_KEY,
                "dataset": {
                    "name": "mteb/AskUbuntuDupQuestions",
                    "revision": REVISION,
                    "subset": "test",
                    "parameters": [],
                },
                "payload": {"items_per_request": 2, "output_contract": "scores", "parameters": []},
                "client": {"worker_count": 1, "processes_per_worker": 2},
            }
        )
        reference = Path(self.temporary.name) / "score.json"
        reference.write_text(json.dumps(payload), encoding="utf-8")
        client = FakeClient()
        record, _ = self.run_reference(client, reference=reference)
        readiness, *scheduled = client.calls
        self.assertEqual(
            readiness[2],
            ({"text": "first query"}, [{"id": "10", "text": "Ten\nten body"}, {"id": "9", "text": "nine body"}]),
        )
        self.assertEqual(
            Counter(query["text"] for _, _, (query, _), _ in scheduled), Counter({"first query": 10, "tenth query": 10})
        )
        for kind, model, _, kwargs in scheduled:
            self.assertEqual((kind, model), ("score", "Qwen/Qwen3-Reranker-0.6B"))
            self.assertFalse(kwargs["wait_for_capacity"])
            self.assertEqual((kwargs["provision_timeout_s"], kwargs["max_oom_retries"]), (5.0, 0))
            self.assertNotIn("output_types", kwargs)
        self.assertEqual(record["operation"], "score")
        self.assertEqual(record["payload"], {"items_per_request": 2, "output_contract": "scores", "parameters": []})
        self.assertEqual(record["dataset"]["selected_ids"], ["q1", "q10"])
        self.assertEqual(record["comparable_reasons"], ["client processes 1 != 2"])


class CommandTest(unittest.TestCase):
    def test_a_reference_run_without_a_key_fails_before_any_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = reference_file(temporary)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch.object(run, "load_performance_data") as loader,
                patch.object(run, "SIEClient") as client,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                run.main(["performance", "--reference", str(reference)])
        self.assertEqual(caught.exception.code, 2)
        loader.assert_not_called()
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
