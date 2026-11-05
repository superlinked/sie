import contextlib
import io
import json
import math
import shlex
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_performance
from datasets import Dataset
from test_quality import FakeClient, clock, parse, reference_file, tiny_loaders

import run
import score
from run import (
    DEFAULT_ENDPOINT,
    Loaders,
    load_performance_data,
    load_quality_data,
    plan_performance,
    plan_quality,
    run_performance,
    run_quality,
)
from score import BASELINE_METRICS, compute_verdict, format_verdict, recompute_verdict, task_metrics

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
LOCAL = "http://localhost:8080"
FLAGS = (
    "--model",
    "example/sparse-encoder",
    "--task",
    "mteb/SciFact",
    "--output",
    "sparse",
    "--items-per-request",
    "2",
)
DENSE_FLAGS = ("--model", "example/dense", "--task", "mteb/SciFact", "--output", "dense", "--items-per-request", "2")
PROLOGUE = "uv run python run.py quality"
VERDICT_FIELDS = {
    "verdict",
    "reason",
    "metric",
    "current",
    "baseline",
    "delta_pct",
    "threshold_low",
    "threshold_high",
    "margin",
}
ADDED_DATASETS = {
    "mteb/FiQA2018": ("mteb/fiqa", "27a168819829fe9bcd655c2df245fb19452e8e06"),
    "mteb/SCIDOCS": ("mteb/scidocs", "f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88"),
    "mteb/CQADupstackPhysicsRetrieval": ("mteb/cqadupstack-physics", "79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4"),
    "mteb/CosQA": ("CoIR-Retrieval/cosqa", "bc5efb7e9d437246ce393ed19d772e08e4a79535"),
    "mteb/StackOverflowQA": ("CoIR-Retrieval/stackoverflow-qa", "db8f169f3894c14a00251061f957b2063eef2bd5"),
    "mteb/LegalBenchConsumerContractsQA": (
        "mteb/legalbench_consumer_contracts_qa",
        "b23590301ec94e8087e2850b21d43d4956b1cca9",
    ),
}
DENSE_VECTORS = {
    "Alpha alpha body": [10.0, 10.0],
    "beta body": [1.0, 0.0],
    "gamma body": [0.0, 5.0],
    "delta body": [3.0, 4.0],
    "epsilon body": [-1.0, 0.0],
    "first query": [1.0, 0.0],
    "second query": [0.0, 1.0],
    "third query": [1.0, 1.0],
}


class DenseClient(FakeClient):
    def _encode_item(self, item, output_types):
        if output_types == ["dense"]:
            return {"id": item["id"], "dense": np.asarray(DENSE_VECTORS[item["text"]], dtype=np.float32)}
        return super()._encode_item(item, output_types)


def supplied(metric="ndcg_at_10", value=0.32, source="model card"):
    return {"kind": "supplied", "metric": metric, "value": value, "source": source}


def judged_qrels():
    return {query_id: dict(tiny_loaders().qrels("", "")[query_id]) for query_id in ("q1", "q2", "q3")}


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_flags(self, client, *flags):
        plan = plan_quality(parse(*flags))
        data = load_quality_data(plan, tiny_loaders())
        return run_quality(plan, data, client, results_dir=self.directory / "results", now=clock)

    def command(self, fake, *argv):
        def measured(plan, data, client):
            return run_quality(plan, data, fake, results_dir=self.directory / "results", now=clock)

        def loader(plan):
            return load_quality_data(plan, tiny_loaders())

        stdout, stderr = io.StringIO(), io.StringIO()
        code = None
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(run, "load_quality_data", side_effect=loader),
            patch.object(run, "run_quality", side_effect=measured),
            patch.object(run, "SIEClient"),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                run.main(["quality", *argv])
            except SystemExit as caught:
                code = caught.code
        return code, stdout.getvalue(), stderr.getvalue()

    def refused(self, *argv):
        stderr = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(run, "load_quality_data") as loader,
            patch.object(run, "load_performance_data") as performance_loader,
            patch.object(run, "SIEClient") as client,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            run.main(list(argv))
        self.assertEqual(caught.exception.code, 2)
        loader.assert_not_called()
        performance_loader.assert_not_called()
        client.assert_not_called()
        return stderr.getvalue()


class BandTest(unittest.TestCase):
    baseline = supplied(value=0.32)
    low = 0.32 * 0.98
    high = 0.32 / 0.98

    def verdict(self, current, **kwargs):
        return compute_verdict({"ndcg_at_10": current}, self.baseline, limited=False, **kwargs)

    def assert_verdict(self, verdict, label, reason):
        self.assertEqual((verdict["verdict"], verdict["reason"]), (label, reason))

    def test_exactly_at_the_low_bound_passes(self):
        self.assert_verdict(self.verdict(self.low), "PASS", "within_margin")

    def test_just_below_the_low_bound_fails(self):
        self.assert_verdict(self.verdict(math.nextafter(self.low, 0.0)), "FAIL (regression)", "below_margin")

    def test_exactly_at_the_high_bound_passes(self):
        self.assert_verdict(self.verdict(self.high), "PASS", "within_margin")

    def test_just_above_the_high_bound_warns(self):
        self.assert_verdict(
            self.verdict(math.nextafter(self.high, 1.0)), "WARN (suspicious-improvement)", "above_margin"
        )

    def test_zero_is_no_output_ahead_of_every_other_case(self):
        for baseline, limited in ((self.baseline, False), (self.baseline, True), (None, False), (None, True)):
            with self.subTest(baseline=baseline, limited=limited):
                metrics = {"ndcg_at_10": 0.0, "main_score": 0.0}
                verdict = compute_verdict(metrics, baseline, limited=limited)
                self.assert_verdict(verdict, "FAIL (no-output)", "current_zero")
                self.assertEqual(verdict["current"], 0.0)

    def test_no_predictions_is_no_output_ahead_of_every_other_case(self):
        for baseline, limited in ((self.baseline, False), (self.baseline, True), (None, False), (None, True)):
            with self.subTest(baseline=baseline, limited=limited):
                verdict = compute_verdict({}, baseline, limited=limited)
                self.assert_verdict(verdict, "FAIL (no-output)", "no_predictions")
                self.assertIsNone(verdict["current"])

    def test_a_limit_key_is_a_measurement_even_with_a_baseline_out_of_band(self):
        verdict = compute_verdict({"ndcg_at_10": 0.01}, self.baseline, limited=True)
        self.assert_verdict(verdict, "MEASUREMENT", "limited_run")
        self.assertEqual(verdict["current"], 0.01)
        for field in ("baseline", "delta_pct", "threshold_low", "threshold_high"):
            self.assertIsNone(verdict[field], field)

    def test_no_baseline_is_a_measurement_of_the_main_score(self):
        verdict = compute_verdict({"main_score": 0.4, "ndcg_at_10": 0.9}, None, limited=False)
        self.assert_verdict(verdict, "MEASUREMENT", "no_baseline")
        self.assertEqual((verdict["metric"], verdict["current"]), ("main_score", 0.4))
        for field in ("baseline", "delta_pct", "threshold_low", "threshold_high"):
            self.assertIsNone(verdict[field], field)

    def test_the_named_metric_is_compared_and_not_the_main_score(self):
        verdict = compute_verdict({"main_score": 0.9, "ndcg_at_10": 0.32}, self.baseline, limited=False)
        self.assert_verdict(verdict, "PASS", "within_margin")
        self.assertEqual((verdict["metric"], verdict["current"]), ("ndcg_at_10", 0.32))

    def test_field_names_and_arithmetic(self):
        verdict = self.verdict(0.3)
        self.assertEqual(set(verdict), VERDICT_FIELDS)
        self.assertEqual(verdict["baseline"], 0.32)
        self.assertEqual(verdict["margin"], 0.98)
        self.assertEqual(verdict["threshold_low"], 0.32 * 0.98)
        self.assertEqual(verdict["threshold_high"], 0.32 / 0.98)
        self.assertEqual(verdict["delta_pct"], (0.3 - 0.32) / 0.32 * 100.0)


class VerdictLinesTest(unittest.TestCase):
    def lines(self, metrics, baseline=None, *, limited=False):
        return format_verdict(compute_verdict(metrics, baseline, limited=limited), baseline, "uv run python run.py")

    def test_a_pass_is_one_line_with_the_delta_and_the_band(self):
        baseline = supplied(value=0.3245, source="the paper")
        self.assertEqual(
            self.lines({"ndcg_at_10": 0.3199}, baseline),
            ["verdict: PASS ndcg_at_10 0.31990 against 0.32450 from the paper (-1.42%, band 0.31801 to 0.33112)"],
        )

    def test_a_warning_is_one_line(self):
        self.assertEqual(
            self.lines({"ndcg_at_10": 0.4}, supplied()),
            [
                (
                    "verdict: WARN (suspicious-improvement) ndcg_at_10 0.40000 against 0.32000 from model card"
                    " (+25.00%, band 0.31360 to 0.32653)"
                )
            ],
        )

    def test_measurements_say_why_there_is_no_comparison(self):
        self.assertEqual(self.lines({"main_score": 0.4}), ["verdict: MEASUREMENT no baseline"])
        self.assertEqual(
            self.lines({"ndcg_at_10": 0.4}, supplied(), limited=True),
            ["verdict: MEASUREMENT limited run, not compared with a full-dataset number"],
        )

    def test_a_regression_adds_expected_got_and_reproduce(self):
        self.assertEqual(
            self.lines({"ndcg_at_10": 0.25}, supplied()),
            [
                (
                    "verdict: FAIL (regression) ndcg_at_10 0.25000 against 0.32000 from model card"
                    " (-21.88%, band 0.31360 to 0.32653)"
                ),
                "expected: ndcg_at_10 from 0.31360 to 0.32653, 0.32000 from model card",
                "got: ndcg_at_10 0.25000",
                "reproduce: uv run python run.py",
            ],
        )

    def test_no_output_adds_expected_got_and_reproduce(self):
        self.assertEqual(
            self.lines({}, supplied()),
            [
                "verdict: FAIL (no-output) no predictions",
                "expected: ndcg_at_10 from 0.31360 to 0.32653, 0.32000 from model card",
                "got: no predictions",
                "reproduce: uv run python run.py",
            ],
        )
        self.assertEqual(
            self.lines({"main_score": 0.0}),
            [
                "verdict: FAIL (no-output) main_score is exactly 0",
                "expected: main_score above 0",
                "got: main_score 0.00000",
                "reproduce: uv run python run.py",
            ],
        )


class BaselineFlagsTest(TemporaryCase):
    def test_bad_forms_are_refused_before_any_request(self):
        for baseline, message in (
            (("--baseline", "ndcg_at_10=0.3"), "go together"),
            (("--baseline-source", "model card"), "go together"),
            (("--baseline", "ndcg_at_10=0.3", "--baseline-source", "  "), "go together"),
            (("--baseline", "bogus=0.3", "--baseline-source", "s"), "names 'bogus'"),
            (("--baseline", "ndcg=0.3", "--baseline-source", "s"), "names 'ndcg'"),
            (("--baseline", "ndcg_at_7=0.3", "--baseline-source", "s"), "names 'ndcg_at_7'"),
            (("--baseline", "NDCG_at_10=0.3", "--baseline-source", "s"), "names 'NDCG_at_10'"),
            (("--baseline", "ndcg_at_10", "--baseline-source", "s"), "must be <metric>=<value>"),
            (("--baseline", "ndcg_at_10=", "--baseline-source", "s"), "must be <metric>=<value>"),
            (("--baseline", "ndcg_at_10=high", "--baseline-source", "s"), "must be <metric>=<value>"),
            (("--baseline", "ndcg_at_10=0", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=-0.3", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=1.01", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=nan", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=inf", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=-inf", "--baseline-source", "s"), "above 0 and at most 1"),
            (("--baseline", "ndcg_at_10=0.3", "--baseline-source", "a\nverdict: PASS"), "one line"),
        ):
            with self.subTest(baseline=baseline):
                stderr = self.refused("quality", *FLAGS, "--endpoint", LOCAL, *baseline)
                self.assertIn(message, stderr)

    def test_baseline_flags_are_refused_beside_a_reference(self):
        reference = reference_file(self.directory)
        for flags, name in (
            (("--baseline", "ndcg_at_10=0.3", "--baseline-source", "s"), "--baseline"),
            (("--baseline", "ndcg_at_10=0.3"), "--baseline"),
            (("--baseline-source", "s"), "--baseline-source"),
        ):
            with self.subTest(flags=flags):
                stderr = self.refused("quality", "--reference", str(reference), "--endpoint", LOCAL, *flags)
                self.assertIn(f"{name} cannot be combined with --reference", stderr)

    def test_performance_has_no_baseline_flags(self):
        for flag in ("--baseline", "--baseline-source"):
            with self.subTest(flag=flag):
                self.refused("performance", "--model", "m", "--task", "mteb/SciFact", "--rate", "1", flag, "x")

    def test_the_accepted_metrics_are_the_ones_a_quality_run_writes(self):
        written = task_metrics({"q": {"a": 1.0}}, {"q": {"a": 1}}, "mteb/SciFact")
        self.assertEqual(BASELINE_METRICS, set(written))
        for metric in sorted(BASELINE_METRICS):
            with self.subTest(metric=metric):
                plan = plan_quality(parse(*FLAGS, "--baseline", f"{metric}=0.5", "--baseline-source", "s"))
                self.assertEqual(plan.baseline["metric"], metric)

    def test_a_value_of_one_is_accepted_and_the_source_is_recorded(self):
        record, _ = self.run_flags(FakeClient(), *FLAGS, "--baseline", "recall_at_1=1", "--baseline-source", "a paper")
        self.assertEqual(
            record["baseline"], {"kind": "supplied", "metric": "recall_at_1", "value": 1.0, "source": "a paper"}
        )

    def test_a_score_task_takes_a_baseline(self):
        record, _ = self.run_flags(
            FakeClient(),
            "--model",
            "m",
            "--task",
            "mteb/AskUbuntuDupQuestions",
            "--baseline",
            "map_at_1000=1",
            "--baseline-source",
            "s",
        )
        self.assertEqual(record["verdict"]["verdict"], "PASS")
        self.assertEqual(record["verdict"]["metric"], "map_at_1000")


class RecordTest(TemporaryCase):
    def test_a_run_without_a_baseline_records_a_measurement(self):
        record, _ = self.run_flags(FakeClient(), *FLAGS)
        self.assertIsNone(record["baseline"])
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(set(record["verdict"]), VERDICT_FIELDS)
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["reason"]), ("MEASUREMENT", "no_baseline"))

    def test_a_limit_key_records_a_measurement_and_keeps_the_baseline(self):
        record, _ = self.run_flags(
            FakeClient(),
            *FLAGS[:2],
            "--task",
            "mteb/SciFact?limit=2",
            *FLAGS[4:],
            "--baseline",
            "recall_at_1=0.1",
            "--baseline-source",
            "s",
        )
        self.assertEqual(record["baseline"]["value"], 0.1)
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["reason"]), ("MEASUREMENT", "limited_run"))

    def test_a_run_that_wrote_no_predictions_records_no_output(self):
        record, _ = self.run_flags(
            FakeClient(fail_at=1), *FLAGS, "--baseline", "recall_at_1=1", "--baseline-source", "s"
        )
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertEqual(
            (record["verdict"]["verdict"], record["verdict"]["reason"]), ("FAIL (no-output)", "no_predictions")
        )

    def test_nothing_in_the_verdict_changes_validity_or_comparable(self):
        record, _ = self.run_flags(FakeClient(), *FLAGS, "--baseline", "recall_at_1=0.9", "--baseline-source", "s")
        self.assertEqual(record["verdict"]["verdict"], "FAIL (regression)")
        self.assertTrue(record["validity"]["valid"])
        self.assertEqual(record["comparable_reasons"], ["no reference"])


class ReferenceBaselineTest(TemporaryCase):
    def test_every_committed_quality_reference_gets_its_published_main_score(self):
        paths = sorted((EXAMPLE_DIR / "references").glob("*-quality-*.json"))
        self.assertEqual(len(paths), 6)
        kinds = set()
        for path in paths:
            with self.subTest(path.name):
                reference = json.loads(path.read_text(encoding="utf-8"))
                main_score = next(row["value"] for row in reference["metrics"] if row["name"] == "main_score")
                plan = plan_quality(parse("--reference", str(path)))
                self.assertEqual(
                    plan.baseline,
                    {
                        "kind": "published",
                        "metric": "main_score",
                        "value": main_score,
                        "source": reference["reference_id"],
                    },
                )
                kinds.add(reference["task_key"])
                if reference["task_key"] == "mteb/AskUbuntuDupQuestions":
                    map_1000 = next(row["value"] for row in reference["metrics"] if row["name"] == "map_at_1000")
                    self.assertEqual(main_score, map_1000)
        self.assertIn("mteb/AskUbuntuDupQuestions", kinds)

    def test_a_reference_without_a_main_score_is_refused(self):
        reference = reference_file(
            self.directory,
            metrics=[{"name": "ndcg_at_10", "unit": "ratio", "aggregation": "task_native", "value": 0.5}],
        )
        stderr = self.refused("quality", "--reference", str(reference), "--endpoint", LOCAL)
        self.assertIn("no main_score metric", stderr)

    def test_a_reference_run_is_compared_with_the_records_main_score(self):
        reference = reference_file(self.directory)
        record, _ = self.run_flags(FakeClient(), "--reference", str(reference))
        self.assertEqual(
            record["baseline"],
            {"kind": "published", "metric": "main_score", "value": 0.5, "source": "example-reference-quality-01"},
        )
        self.assertEqual(record["verdict"]["metric"], "main_score")
        self.assertEqual(record["verdict"]["baseline"], 0.5)
        self.assertEqual(record["verdict"]["verdict"], "WARN (suspicious-improvement)")

    def test_an_askubuntu_reference_run_is_compared_on_map_and_not_on_ndcg(self):
        reference = reference_file(
            self.directory,
            model="Qwen/Qwen3-Reranker-0.6B",
            operation="score",
            task_key="mteb/AskUbuntuDupQuestions",
            dataset={"name": "mteb/AskUbuntuDupQuestions", "revision": "0" * 40, "subset": "test", "parameters": []},
            payload={"items_per_request": 2, "output_contract": "scores", "parameters": []},
            metrics=[{"name": "main_score", "unit": "ratio", "aggregation": "task_native", "value": 1.0}],
        )
        record, _ = self.run_flags(FakeClient(), "--reference", str(reference))
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual(metrics["main_score"], metrics["map_at_1000"])
        self.assertLess(metrics["ndcg_at_10"], 0.98)
        self.assertEqual(record["baseline"]["kind"], "published")
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["current"]), ("PASS", 1.0))

    def test_another_endpoint_keeps_comparable_false_and_still_gets_a_verdict(self):
        reference = reference_file(self.directory)
        record, _ = self.run_flags(FakeClient(), "--reference", str(reference), "--endpoint", LOCAL)
        self.assertFalse(record["comparable"])
        self.assertEqual(record["comparable_reasons"], [f"endpoint {LOCAL} != {DEFAULT_ENDPOINT}"])
        self.assertEqual(record["verdict"]["verdict"], "WARN (suspicious-improvement)")

    def test_a_smoke_key_keeps_the_published_baseline_and_is_a_measurement(self):
        reference = reference_file(self.directory)
        record, _ = self.run_flags(FakeClient(), "--reference", str(reference), "--task", "mteb/SciFact?limit=2")
        self.assertEqual(record["baseline"]["kind"], "published")
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["reason"]), ("MEASUREMENT", "limited_run"))


class InstructionTest(TemporaryCase):
    def test_refused_beside_a_reference(self):
        reference = reference_file(self.directory)
        stderr = self.refused("quality", "--reference", str(reference), "--endpoint", LOCAL, "--instruction", "x")
        self.assertIn("--instruction cannot be combined with --reference", stderr)

    def test_refused_on_the_score_task(self):
        stderr = self.refused(
            "quality", "--model", "m", "--task", "mteb/AskUbuntuDupQuestions", "--endpoint", LOCAL, "--instruction", "x"
        )
        self.assertIn("--instruction is not accepted for the score task", stderr)

    def test_refused_when_empty(self):
        for text in ("", "  "):
            with self.subTest(text=text):
                stderr = self.refused("quality", *FLAGS, "--endpoint", LOCAL, "--instruction", text)
                self.assertIn("--instruction must not be empty", stderr)

    def test_performance_has_no_such_flag(self):
        self.refused("performance", "--model", "m", "--task", "mteb/SciFact", "--rate", "1", "--instruction", "x")

    def test_sent_with_queries_only_and_recorded(self):
        client = FakeClient()
        record, _ = self.run_flags(client, *FLAGS, "--instruction", "Find the passage")
        readiness, *measured = client.calls
        self.assertFalse(readiness[3]["is_query"])
        self.assertNotIn("instruction", readiness[3])
        queries = [kwargs for _, _, _, kwargs in measured if kwargs["is_query"]]
        documents = [kwargs for _, _, _, kwargs in measured if not kwargs["is_query"]]
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(documents), 3)
        for kwargs in queries:
            self.assertEqual(kwargs["instruction"], "Find the passage")
        for kwargs in documents:
            self.assertNotIn("instruction", kwargs)
        self.assertEqual(record["payload"]["parameters"], [{"name": "instruction", "value": "Find the passage"}])

    def test_absent_unless_given(self):
        client = FakeClient()
        record, _ = self.run_flags(client, *FLAGS)
        for _, _, _, kwargs in client.calls:
            self.assertNotIn("instruction", kwargs)
        self.assertEqual(record["payload"]["parameters"], [])


class DenseTest(TemporaryCase):
    def test_ranked_by_cosine_where_the_raw_dot_product_gives_another_order(self):
        record, _ = self.run_flags(DenseClient(), *DENSE_FLAGS)
        predictions = json.loads(Path(record["run"]["predictions_path"]).read_text(encoding="utf-8"))
        cosine_order = ["d2", "d1", "d4", "d3", "d5"]
        self.assertEqual(list(predictions["q1"]), cosine_order)
        self.assertAlmostEqual(predictions["q1"]["d2"], 1.0, places=6)
        self.assertAlmostEqual(predictions["q1"]["d1"], 1.0 / math.sqrt(2.0), places=6)
        self.assertAlmostEqual(predictions["q1"]["d4"], 0.6, places=6)
        self.assertAlmostEqual(predictions["q1"]["d5"], -1.0, places=6)
        documents = {
            doc_id: np.asarray(DENSE_VECTORS[text], dtype=np.float32)
            for doc_id, text in (
                ("d1", "Alpha alpha body"),
                ("d2", "beta body"),
                ("d3", "gamma body"),
                ("d4", "delta body"),
                ("d5", "epsilon body"),
            )
        }
        query = np.asarray(DENSE_VECTORS["first query"], dtype=np.float32)
        dot_order = sorted(documents, key=lambda doc_id: float(query @ documents[doc_id]), reverse=True)
        self.assertEqual(dot_order, ["d1", "d4", "d2", "d3", "d5"])
        self.assertNotEqual(dot_order, cosine_order)

    def test_a_row_below_the_norm_floor_is_divided_by_the_floor(self):
        self.assertAlmostEqual(float(run.dense_scores([[1.0, 0.0]], [[1e-13, 0.0]])[0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(run.dense_scores([[1e-13, 0.0]], [[1.0, 0.0]])[0, 0]), 0.1, places=5)

    def test_a_row_at_or_above_the_norm_floor_scores_its_cosine_and_a_zero_row_scores_zero(self):
        self.assertAlmostEqual(float(run.dense_scores([[1.0, 0.0]], [[3e-12, 4e-12]])[0, 0]), 0.6, places=5)
        self.assertEqual(float(run.dense_scores([[1.0, 0.0]], [[0.0, 0.0]])[0, 0]), 0.0)

    def test_the_norm_floor_is_on_the_dense_path_only(self):
        self.assertAlmostEqual(float(run.maxsim_scores([[[1.0, 0.0]]], [[[1e-13, 0.0]]])[0, 0]), 1.0, places=5)

    def test_a_quality_run_refuses_a_muvera_profile(self):
        stderr = self.refused(
            "quality",
            "--model",
            "lightonai/GTE-ModernColBERT-v1:muvera",
            "--task",
            "mteb/SciFact",
            "--output",
            "dense",
            "--items-per-request",
            "2",
            "--endpoint",
            LOCAL,
        )
        self.assertIn("muvera", stderr)
        self.assertIn("cosine", stderr)

    def test_the_muvera_refusal_is_for_a_dense_quality_run_with_that_profile_only(self):
        task = ("--task", "mteb/SciFact", "--items-per-request", "2")
        other_profile = plan_quality(parse("--model", "m:other", "--output", "dense", *task))
        self.assertEqual((other_profile.profile, other_profile.output_contract), ("other", "dense"))
        other_output = plan_quality(parse("--model", "m:muvera", "--output", "sparse", *task))
        self.assertEqual((other_output.profile, other_output.output_contract), ("muvera", "sparse"))
        performance = plan_performance(
            test_performance.parse(
                "--model",
                "m:muvera",
                "--task",
                "mteb/SciFact?split=test&role=corpus&limit=4&filter=nonempty",
                "--output",
                "dense",
                "--items-per-request",
                "1",
                "--rate",
                "1",
            )
        )
        self.assertEqual((performance.profile, performance.output_contract), ("muvera", "dense"))

    def test_a_result_carries_the_dense_contract_and_every_request_asks_for_it(self):
        client = DenseClient()
        record, _ = self.run_flags(client, *DENSE_FLAGS)
        self.assertEqual(record["payload"]["output_contract"], "dense")
        self.assertEqual(len(client.calls), 6)
        for _, _, _, kwargs in client.calls:
            self.assertEqual(kwargs["output_types"], ["dense"])

    def test_a_response_of_the_wrong_shape_is_a_failed_run(self):
        record, _ = self.run_flags(FakeClient(), *DENSE_FLAGS)
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertEqual(record["verdict"]["reason"], "no_predictions")

    def test_vectors_of_different_lengths_are_a_failed_run(self):
        class Uneven(DenseClient):
            def _encode_item(self, item, output_types):
                result = super()._encode_item(item, output_types)
                if item["text"] == "gamma body":
                    result["dense"] = np.zeros(3, dtype=np.float32)
                return result

        record, _ = self.run_flags(Uneven(), *DENSE_FLAGS)
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertFalse(record["validity"]["valid"])

    def test_the_score_task_still_refuses_an_output(self):
        stderr = self.refused(
            "quality", "--model", "m", "--task", "mteb/AskUbuntuDupQuestions", "--output", "dense", "--endpoint", LOCAL
        )
        self.assertIn("not accepted for the score task", stderr)
        stderr = self.refused(
            "performance",
            "--model",
            "m",
            "--task",
            "mteb/AskUbuntuDupQuestions",
            "--rate",
            "1",
            "--output",
            "dense",
            "--endpoint",
            LOCAL,
        )
        self.assertIn("not accepted for the score task", stderr)

    def test_a_performance_run_sends_dense_documents_and_has_no_verdict(self):
        key = "mteb/SciFact?split=test&role=corpus&limit=4&filter=nonempty"
        plan = plan_performance(
            test_performance.parse(
                "--model", "m", "--task", key, "--output", "dense", "--items-per-request", "1", "--rate", "20"
            )
        )
        plan = replace(plan, duration_s=0.1, warmup_s=0.0, settle_s=0.0, drain_s=0.0)
        client = test_performance.FakeClient()
        data = load_performance_data(plan, test_performance.tiny_loaders())
        record, _ = run_performance(plan, data, client, results_dir=self.directory / "results", now=clock)
        self.assertEqual(record["payload"]["output_contract"], "dense")
        self.assertEqual(len(client.calls), 3)
        for _, _, _, kwargs in client.calls:
            self.assertEqual(kwargs["output_types"], ["dense"])
            self.assertFalse(kwargs["is_query"])
            self.assertNotIn("instruction", kwargs)
        self.assertNotIn("baseline", record)
        self.assertNotIn("verdict", record)


class CommandLineTest(TemporaryCase):
    def test_a_reference_run(self):
        reference = reference_file(self.directory)
        record, _ = self.run_flags(FakeClient(), "--reference", str(reference))
        self.assertEqual(
            record["run"]["command"], f"{PROLOGUE} --reference references/example-reference-quality-01.json"
        )

    def test_a_smoke_key_and_another_endpoint_are_kept(self):
        reference = reference_file(self.directory)
        record, _ = self.run_flags(
            FakeClient(), "--reference", str(reference), "--task", "mteb/SciFact?limit=2", "--endpoint", f"{LOCAL}/"
        )
        self.assertEqual(
            record["run"]["command"],
            f"{PROLOGUE} --reference references/example-reference-quality-01.json"
            f" --task 'mteb/SciFact?limit=2' --endpoint {LOCAL}",
        )

    def test_the_default_endpoint_is_left_out(self):
        for endpoint in (DEFAULT_ENDPOINT, f"{DEFAULT_ENDPOINT}/"):
            with self.subTest(endpoint=endpoint):
                record, _ = self.run_flags(FakeClient(), *FLAGS, "--endpoint", endpoint)
                self.assertNotIn("--endpoint", record["run"]["command"])

    def test_the_command_rebuilds_the_same_quality_plan(self):
        for flags in (
            FLAGS,
            (*DENSE_FLAGS, "--endpoint", f"{LOCAL}/sie"),
            (
                *DENSE_FLAGS,
                "--baseline",
                "ndcg_at_10=0.1234567890123",
                "--baseline-source",
                "the model's card",
                "--instruction",
                "Find the passage",
                "--endpoint",
                LOCAL,
            ),
            (
                *DENSE_FLAGS,
                "--baseline",
                "ndcg_at_10=0.3",
                "--baseline-source=-a dash first",
                "--instruction=--a dash first",
            ),
            (
                "--model",
                "BAAI/bge-m3:sparse",
                "--task",
                "mteb/SciFact",
                "--output",
                "sparse",
                "--items-per-request",
                "1",
            ),
            ("--model", "m", "--task", "mteb/AskUbuntuDupQuestions"),
        ):
            with self.subTest(flags=flags):
                plan = plan_quality(parse(*flags))
                command = run.quality_command_line(plan)
                self.assertEqual(shlex.split(command)[:5], shlex.split(PROLOGUE))
                self.assertEqual(plan_quality(parse(*shlex.split(command)[5:])), plan)

    def test_the_command_rebuilds_the_same_performance_plan(self):
        key = "mteb/SciFact?split=test&role=corpus&limit=4&filter=nonempty"
        for flags in (
            ("--model", "m", "--task", key, "--output", "dense", "--items-per-request", "1", "--rate", "0.25"),
            (
                "--model",
                "m",
                "--task",
                key,
                "--output",
                "sparse",
                "--items-per-request",
                "1",
                "--rate",
                "1",
                "--warmup",
                "0",
            ),
            ("--model", "m", "--task", "mteb/AskUbuntuDupQuestions?limit=2", "--rate", "2", "--endpoint", LOCAL),
        ):
            with self.subTest(flags=flags):
                plan = plan_performance(test_performance.parse(*flags))
                command = run.performance_command_line(plan)
                self.assertEqual(plan_performance(test_performance.parse(*shlex.split(command)[5:])), plan)

    def test_a_performance_reference_command_repeats_the_same_plan(self):
        reference = test_performance.reference_file(self.directory)
        plan = plan_performance(test_performance.parse("--reference", str(reference), "--rate", "2"))
        command = run.performance_command_line(plan)
        self.assertIn("--reference references/example-reference-performance-01.json", command)
        argv = shlex.split(command)[5:]
        argv[1] = str(reference)
        self.assertEqual(plan_performance(test_performance.parse(*argv)), plan)

    def test_url_credentials_reach_neither_a_quality_nor_a_performance_result(self):
        endpoint = "http://someone:s3cret@localhost:8080/sie?token=t0ken#fr4gment"
        reference = reference_file(self.directory)
        quality, _ = self.run_flags(FakeClient(), "--reference", str(reference), "--endpoint", endpoint)
        flags, _ = self.run_flags(FakeClient(), *FLAGS, "--endpoint", endpoint)

        performance_reference = test_performance.reference_file(self.directory)
        plan = plan_performance(
            test_performance.parse("--reference", str(performance_reference), "--endpoint", endpoint)
        )
        data = load_performance_data(plan, test_performance.tiny_loaders())
        performance, _ = run_performance(
            plan, data, test_performance.FakeClient(), results_dir=self.directory / "results", now=clock
        )
        for name, record in (("quality", quality), ("flags", flags), ("performance", performance)):
            with self.subTest(name):
                text = json.dumps(record)
                for secret in ("someone", "s3cret", "t0ken", "fr4gment"):
                    self.assertNotIn(secret, text)
                self.assertTrue(record["run"]["command"].endswith("--endpoint http://localhost:8080/sie"))
                self.assertEqual(record["run"]["endpoint"], "http://localhost:8080/sie")
                self.assertIn(f"endpoint http://localhost:8080/sie != {DEFAULT_ENDPOINT}", record["comparable_reasons"])


class ExitStatusTest(TemporaryCase):
    def test_a_regression_prints_its_lines_on_stdout_and_exits_0(self):
        code, stdout, stderr = self.command(
            FakeClient(),
            *FLAGS,
            "--baseline",
            "recall_at_1=0.9",
            "--baseline-source",
            "model card",
            "--endpoint",
            LOCAL,
        )
        self.assertIsNone(code)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout.splitlines()[-4:],
            [
                (
                    "verdict: FAIL (regression) recall_at_1 0.83333 against 0.90000 from model card"
                    " (-7.41%, band 0.88200 to 0.91837)"
                ),
                "expected: recall_at_1 from 0.88200 to 0.91837, 0.90000 from model card",
                "got: recall_at_1 0.83333",
                (
                    f"reproduce: {PROLOGUE} --model example/sparse-encoder --task mteb/SciFact --output sparse"
                    " --items-per-request 2 --baseline recall_at_1=0.9 '--baseline-source=model card'"
                    f" --endpoint {LOCAL}"
                ),
            ],
        )

    def test_a_warning_and_a_pass_exit_0(self):
        for value, label in (("0.5", "WARN (suspicious-improvement)"), ("0.83333", "PASS")):
            with self.subTest(label=label):
                code, stdout, _ = self.command(
                    FakeClient(),
                    *FLAGS,
                    "--baseline",
                    f"recall_at_1={value}",
                    "--baseline-source",
                    "s",
                    "--endpoint",
                    LOCAL,
                )
                self.assertIsNone(code)
                self.assertTrue(stdout.splitlines()[-1].startswith(f"verdict: {label} recall_at_1"))

    def test_the_verdict_comes_after_the_metrics(self):
        _, stdout, _ = self.command(FakeClient(), *FLAGS, "--endpoint", LOCAL)
        lines = stdout.splitlines()
        self.assertEqual(lines[-1], "verdict: MEASUREMENT no baseline")
        self.assertEqual(sum(line.startswith("verdict:") for line in lines), 1)
        self.assertTrue(lines[-2].startswith("recall_at_5"))

    def test_no_predictions_exits_1_and_prints_the_verdict(self):
        for fake in (FakeClient(fail_at=0), FakeClient(fail_at=1)):
            with self.subTest(fail_at=fake._fail_at):
                code, stdout, stderr = self.command(
                    fake, *FLAGS, "--baseline", "recall_at_1=0.9", "--baseline-source", "s", "--endpoint", LOCAL
                )
                self.assertEqual(code, 1)
                self.assertEqual(
                    stdout.splitlines()[-4:-1],
                    [
                        "verdict: FAIL (no-output) no predictions",
                        "expected: recall_at_1 from 0.88200 to 0.91837, 0.90000 from s",
                        "got: no predictions",
                    ],
                )
                self.assertTrue(stdout.splitlines()[-1].startswith("reproduce: "))
                self.assertTrue(stderr.startswith("invalid: "))


class ScoreVerdictTest(TemporaryCase):
    def rescored(self, result_path, *flags):
        stdout = io.StringIO()
        with patch("score.load_qrels", return_value=judged_qrels()), contextlib.redirect_stdout(stdout):
            score.main([str(result_path), *flags])
        return stdout.getvalue().splitlines()

    def stored(self, *flags):
        record, result_path = self.run_flags(FakeClient(), *FLAGS, *flags)
        return record, result_path

    def worsen_predictions(self, record):
        path = Path(record["run"]["predictions_path"])
        predictions = json.loads(path.read_text(encoding="utf-8"))
        predictions["q1"] = {doc_id: float(index) for index, doc_id in enumerate(predictions["q1"])}
        score.write_json(path, predictions)

    def test_the_stored_verdict_is_recomputed_the_same(self):
        for value, label in (
            ("0.83333", "PASS"),
            ("0.9", "FAIL (regression)"),
            ("0.5", "WARN (suspicious-improvement)"),
        ):
            with self.subTest(label=label):
                record, result_path = self.stored("--baseline", f"recall_at_1={value}", "--baseline-source", "a paper")
                self.assertEqual(record["verdict"]["verdict"], label)
                with patch("score.load_qrels", return_value=judged_qrels()):
                    _, rows, _ = score.rescore(result_path)
                self.assertEqual(recompute_verdict(record, rows), record["verdict"])
                self.assertEqual(
                    self.rescored(result_path)[-len(format_verdict(record["verdict"], record["baseline"], "x")) :],
                    format_verdict(record["verdict"], record["baseline"], record["run"]["command"]),
                )

    def test_the_verdict_follows_the_predictions_and_not_the_stored_fields(self):
        record, result_path = self.stored("--baseline", "recall_at_1=0.83333", "--baseline-source", "s")
        self.assertEqual(record["verdict"]["verdict"], "PASS")
        self.worsen_predictions(record)
        lines = self.rescored(result_path)
        self.assertTrue(any(line.startswith("verdict: FAIL (regression)") for line in lines), lines)
        self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), record)

    def test_write_updates_the_verdict(self):
        record, result_path = self.stored("--baseline", "recall_at_1=0.83333", "--baseline-source", "s")
        self.worsen_predictions(record)
        self.rescored(result_path, "--write")
        written = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(written["verdict"]["verdict"], "FAIL (regression)")
        self.assertEqual(
            written["verdict"]["current"], next(r["value"] for r in written["metrics"] if r["name"] == "recall_at_1")
        )

    def test_a_result_with_a_null_baseline_prints_a_measurement(self):
        _, result_path = self.stored()
        self.assertEqual(self.rescored(result_path)[-1], "verdict: MEASUREMENT no baseline")

    def test_a_result_with_no_baseline_field_prints_no_verdict(self):
        record, result_path = self.stored("--baseline", "recall_at_1=0.9", "--baseline-source", "s")
        older = {key: value for key, value in record.items() if key not in {"baseline", "verdict"}}
        score.write_json(result_path, older)
        lines = self.rescored(result_path, "--write")
        self.assertTrue(lines)
        self.assertFalse([line for line in lines if line.split(":")[0] in {"verdict", "expected", "got", "reproduce"}])
        written = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertNotIn("verdict", written)
        self.assertNotIn("baseline", written)

    def test_a_committed_performance_result_prints_no_verdict(self):
        paths = [
            path
            for path in sorted((EXAMPLE_DIR / "validation").glob("*.json"))
            if json.loads(path.read_text(encoding="utf-8"))["evaluation_mode"] == score.PERFORMANCE_MODE
        ]
        self.assertTrue(paths)
        with contextlib.chdir(self.directory):
            for path in paths:
                with self.subTest(path.name):
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        score.main([str(path)])
                    self.assertNotIn("verdict:", stdout.getvalue())


class AddedDatasetsTest(TemporaryCase):
    def test_the_tables_name_the_revisions_and_the_main_score(self):
        for task_name, entry in ADDED_DATASETS.items():
            with self.subTest(task_name):
                self.assertEqual(score.HF_DATASETS[task_name], entry)
                self.assertEqual(score.MAIN_SCORE[task_name], "ndcg_at_10")
                self.assertEqual(score.ROLE_SPLITS[task_name], {"corpus": "corpus", "queries": "queries"})
                self.assertNotIn(task_name, run.SCORE_TASKS)

    def test_quality_and_performance_keys_are_accepted_as_encode_tasks(self):
        for task_name, (_, revision) in ADDED_DATASETS.items():
            with self.subTest(task_name):
                plan = plan_quality(
                    parse("--model", "m", "--task", task_name, "--output", "dense", "--items-per-request", "2")
                )
                self.assertEqual((plan.operation, plan.task_name, plan.revision), ("encode", task_name, revision))
                performance = plan_performance(
                    test_performance.parse(
                        "--model",
                        "m",
                        "--task",
                        f"{task_name}?split=test&role=corpus&limit=256&filter=nonempty",
                        "--output",
                        "dense",
                        "--items-per-request",
                        "1",
                        "--rate",
                        "1",
                    )
                )
                self.assertEqual(
                    (performance.operation, performance.task_name, performance.revision),
                    ("encode", task_name, revision),
                )

    def test_each_role_loads_from_its_pinned_config_and_split(self):
        rows = {
            "corpus": Dataset.from_dict({"_id": ["d1"], "title": ["A"], "text": ["body"], "partition": ["x"]}),
            "queries": Dataset.from_dict({"_id": ["q1"], "text": ["query"], "language": ["en"]}),
            "default": Dataset.from_dict({"query-id": ["q1"], "corpus-id": ["d1"], "score": [1.0]}),
        }
        for task_name, (dataset, revision) in ADDED_DATASETS.items():
            with (
                self.subTest(task_name),
                patch("score.load_dataset", side_effect=lambda _, config, **__: rows[config]) as load,
            ):
                self.assertEqual(score.load_corpus(task_name, revision), {"d1": {"title": "A", "text": "body"}})
                self.assertEqual(score.load_queries(task_name, revision), {"q1": "query"})
                self.assertEqual(score.load_qrels(task_name, revision), {"q1": {"d1": 1}})
                self.assertEqual(
                    [(call.args, call.kwargs) for call in load.call_args_list],
                    [
                        ((dataset, "corpus"), {"split": "corpus", "revision": revision}),
                        ((dataset, "queries"), {"split": "queries", "revision": revision}),
                        ((dataset, "default"), {"split": "test", "revision": revision}),
                    ],
                )


class QualityTaskKeyTest(TemporaryCase):
    ENCODE = ("--output", "sparse", "--items-per-request", "2")

    def test_a_parameter_other_than_limit_is_refused_before_any_request(self):
        baseline = ("--baseline", "ndcg_at_10=0.3", "--baseline-source", "model card")
        for key, shape, parameter in (
            ("mteb/SciFact?split=train", self.ENCODE, "split"),
            ("mteb/SciFact?split=test", self.ENCODE, "split"),
            ("mteb/SciFact?role=corpus", self.ENCODE, "role"),
            ("mteb/SciFact?Limit=5", self.ENCODE, "Limit"),
            ("mteb/SciFact?limit=5&filter=nonempty", self.ENCODE, "filter"),
            ("mteb/AskUbuntuDupQuestions?candidates-k=5", (), "candidates-k"),
        ):
            with self.subTest(key):
                stderr = self.refused("quality", "--model", "m", "--task", key, *shape, *baseline)
                self.assertIn(f"takes only limit, got {parameter}", stderr)

    def test_a_plain_key_and_a_limit_key_are_accepted(self):
        for key, params in (("mteb/SciFact", {}), ("mteb/SciFact?limit=5", {"limit": "5"})):
            with self.subTest(key):
                plan = plan_quality(parse("--model", "m", "--task", key, *self.ENCODE))
                self.assertEqual(plan.task_params, params)


class IdenticalIdsTest(TemporaryCase):
    def test_only_fiqa_drops_a_document_with_the_query_id(self):
        predictions = {"7": {"7": 9.0, "a": 1.0}}
        qrels = {"7": {"a": 1}}
        self.assertEqual(score.IGNORE_IDENTICAL_IDS, {"mteb/FiQA2018"})
        fiqa = task_metrics(predictions, qrels, "mteb/FiQA2018")
        self.assertEqual((fiqa["recall_at_1"], fiqa["mrr_at_1"], fiqa["main_score"]), (1.0, 1.0, 1.0))
        for task_name in set(score.HF_DATASETS) - {"mteb/FiQA2018"}:
            with self.subTest(task_name):
                other = task_metrics(predictions, qrels, task_name)
                self.assertEqual((other["recall_at_1"], other["mrr_at_1"]), (0.0, 0.0))
                self.assertLess(other["ndcg_at_10"], 1.0)

    def test_the_input_is_left_as_it_was(self):
        predictions = {"7": {"7": 9.0, "a": 1.0}}
        task_metrics(predictions, {"7": {"a": 1}}, "mteb/FiQA2018")
        self.assertEqual(predictions, {"7": {"7": 9.0, "a": 1.0}})

    def test_the_run_and_score_py_agree_and_the_predictions_file_keeps_the_document(self):
        corpus = {"q1": {"title": "", "text": "beta body"}, "d4": {"title": "", "text": "delta body"}}
        loaders = Loaders(
            corpus=lambda name, revision: corpus,
            queries=lambda name, revision: {"q1": "first query"},
            qrels=lambda name, revision: {"q1": {"d4": 1}},
            top_ranked=lambda name, revision: {},
        )
        plan = plan_quality(
            parse("--model", "m", "--task", "mteb/FiQA2018", "--output", "sparse", "--items-per-request", "2")
        )
        data = load_quality_data(plan, loaders)
        record, result_path = run_quality(plan, data, FakeClient(), results_dir=self.directory / "results", now=clock)
        predictions = json.loads(Path(record["run"]["predictions_path"]).read_text(encoding="utf-8"))
        self.assertEqual(list(predictions["q1"]), ["q1", "d4"])
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual(metrics["recall_at_1"], 1.0)
        with patch("score.load_qrels", return_value={"q1": {"d4": 1}}):
            _, rows, _ = score.rescore(result_path)
        self.assertEqual(rows, record["metrics"])


if __name__ == "__main__":
    unittest.main()
