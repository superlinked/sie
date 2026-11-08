import contextlib
import io
import json
import shlex
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import test_performance
from datasets import Dataset
from test_quality import FakeClient, clock, parse, reference_file, tiny_loaders
from test_verdict import DENSE_FLAGS, DENSE_VECTORS, DenseClient, TemporaryCase, judged_qrels

import run
import score
from run import (
    DEFAULT_ENDPOINT,
    Loaders,
    candidate_batches,
    load_performance_data,
    load_quality_data,
    plan_performance,
    plan_quality,
    run_performance,
    run_quality,
)
from score import eval_split, parse_task_key, recompute_verdict

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
LOCAL = "http://localhost:8080"
NEW_TASKS = {
    "mteb/MMarcoReranking": (
        "mteb/MMarcoReranking",
        "91c0f9d5cbd4cdc7ecd7039c5c2e57740cb62b00",
        "dev",
        "score",
        "map_at_1000",
    ),
    "mteb/T2Reranking": ("mteb/T2Reranking", "a34fe6bc0dff185af1228e49a0f6fb1de1565627", "dev", "score", "map_at_1000"),
    "mteb/CMedQAv1-reranking": (
        "mteb/CMedQAv1-reranking",
        "f77f0345c9b6ac0a2447a3a71483d5b27393cb35",
        "test",
        "score",
        "map_at_1000",
    ),
    "mteb/CMedQAv2-reranking": (
        "mteb/CMedQAv2-reranking",
        "01399b6956a4c6bf9edc5b6b90e176aeeb399686",
        "test",
        "score",
        "map_at_1000",
    ),
    "mteb/NanoFiQA2018Retrieval": (
        "zeta-alpha-ai/NanoFiQA2018",
        "4163ba032953d5044a7a6244261413f609c14342",
        "train",
        "encode",
        "ndcg_at_10",
    ),
}
RERANK_TASKS = [task for task, entry in NEW_TASKS.items() if entry[3] == "score"]
TWO_STAGE = "mteb/SciFact?candidates_model=org/encoder&candidates_k=2"
RERANKER = ("--model", "org/reranker")
ENCODE_SHAPE = ("--output", "dense", "--items-per-request", "2")


def corpus_of(count):
    return {f"d{index}": {"title": "", "text": f"doc {index}"} for index in range(count)}


def many_candidates(count):
    return Loaders(
        corpus=lambda name, revision: corpus_of(count),
        queries=lambda name, revision: {"q1": "only query"},
        qrels=lambda name, revision: {"q1": {"d0": 1}},
        top_ranked=lambda name, revision: {"q1": [f"d{index}" for index in range(count)]},
    )


class GeneratedClient(FakeClient):
    """Dense vectors from the text; documents with a low number sit closest to the query."""

    def _encode_item(self, item, output_types):
        number = 0.0 if item["text"] == "only query" else float(item["text"].split()[-1])
        return {"id": item["id"], "dense": np.asarray([1.0, number], dtype=np.float32)}


class ReversingClient(DenseClient):
    """A reranker that scores the retriever's last candidate highest, so its order is visible."""

    def score(self, model, query, items, **kwargs):
        result = super().score(model, query, items, **kwargs)
        result["scores"] = [
            {"item_id": item["id"], "score": float(position + 1), "rank": len(items) - position}
            for position, item in enumerate(items)
        ]
        return result


class ZeroClient(FakeClient):
    def __init__(self, zero=("document", "query"), **kwargs):
        super().__init__(**kwargs)
        self.zero = zero

    def encode(self, model, items, **kwargs):
        self.is_query = kwargs["is_query"]
        return super().encode(model, items, **kwargs)

    def _encode_item(self, item, output_types):
        role = "query" if self.is_query else "document"
        vector = [0.0, 0.0] if role in self.zero else DENSE_VECTORS[item["text"]]
        return {"id": item["id"], "dense": np.asarray(vector, dtype=np.float32)}


class TaskTablesTest(unittest.TestCase):
    def test_the_tables_name_each_new_task(self):
        for task, (dataset, revision, split, operation, main) in NEW_TASKS.items():
            with self.subTest(task):
                self.assertEqual(score.HF_DATASETS[task], (dataset, revision))
                self.assertEqual(score.MAIN_SCORE[task], main)
                self.assertEqual(eval_split(task), split)
                self.assertEqual(task in run.SCORE_TASKS, operation == "score")

    def test_every_table_covers_the_same_tasks_and_older_tasks_stay_on_test(self):
        self.assertEqual(set(score.HF_DATASETS), set(score.MAIN_SCORE))
        self.assertEqual(set(score.HF_DATASETS), set(score.ROLE_SPLITS))
        for task in set(score.HF_DATASETS) - set(NEW_TASKS):
            self.assertEqual(eval_split(task), "test")

    def test_each_new_task_plans_with_its_operation_and_revision(self):
        for task, (_, revision, _, operation, _) in NEW_TASKS.items():
            with self.subTest(task):
                shape = ENCODE_SHAPE if operation == "encode" else ()
                plan = plan_quality(parse("--model", "m", "--task", task, *shape))
                self.assertEqual((plan.operation, plan.task_name, plan.revision), (operation, task, revision))

    def test_the_rerank_tasks_refuse_an_output(self):
        for task in RERANK_TASKS:
            with self.subTest(task), self.assertRaises(ValueError):
                plan_quality(parse("--model", "m", "--task", task, *ENCODE_SHAPE))


class LoaderTest(unittest.TestCase):
    RERANK_ROWS: ClassVar[dict] = {
        "corpus": {"_id": ["d1"], "text": ["body"], "title": ["A"]},
        "queries": {"_id": ["q1"], "text": ["query"]},
        "default": {"query-id": ["q1"], "corpus-id": ["d1"], "score": [2]},
        "top_ranked": {"query-id": ["q1"], "corpus-ids": [["d1"]]},
    }

    def test_a_rerank_task_reads_every_role_from_its_own_split(self):
        rows = {config: Dataset.from_dict(columns) for config, columns in self.RERANK_ROWS.items()}
        for task in RERANK_TASKS:
            dataset, revision, split, _, _ = NEW_TASKS[task]
            with (
                self.subTest(task),
                patch("score.load_dataset", side_effect=lambda _, config, **__: rows[config]) as load,
            ):
                self.assertEqual(score.load_corpus(task, revision), {"d1": {"title": "A", "text": "body"}})
                self.assertEqual(score.load_queries(task, revision), {"q1": "query"})
                self.assertEqual(score.load_qrels(task, revision), {"q1": {"d1": 2}})
                self.assertEqual(score.load_top_ranked(task, revision), {"q1": ["d1"]})
                self.assertEqual(
                    [(call.args, call.kwargs) for call in load.call_args_list],
                    [
                        ((dataset, config), {"split": split, "revision": revision})
                        for config in ("corpus", "queries", "default", "top_ranked")
                    ],
                )

    def test_nano_has_no_title_its_qrels_config_and_grade_one(self):
        task = "mteb/NanoFiQA2018Retrieval"
        dataset, revision, split, _, _ = NEW_TASKS[task]
        rows = {
            "corpus": Dataset.from_dict({"_id": ["d1", "d2"], "text": ["one", "two"]}),
            "queries": Dataset.from_dict({"_id": ["q1"], "text": ["query"]}),
            "qrels": Dataset.from_dict({"query-id": ["q1", "q1"], "corpus-id": ["d1", "d2"]}),
        }
        with patch("score.load_dataset", side_effect=lambda _, config, **__: rows[config]) as load:
            self.assertEqual(
                score.load_corpus(task, revision),
                {"d1": {"title": "", "text": "one"}, "d2": {"title": "", "text": "two"}},
            )
            self.assertEqual(score.load_queries(task, revision), {"q1": "query"})
            self.assertEqual(score.load_qrels(task, revision), {"q1": {"d1": 1, "d2": 1}})
        self.assertEqual(
            [(call.args, call.kwargs) for call in load.call_args_list],
            [((dataset, config), {"split": split, "revision": revision}) for config in ("corpus", "queries", "qrels")],
        )

    def test_a_split_named_by_a_result_still_wins(self):
        rows = Dataset.from_dict(self.RERANK_ROWS["default"])
        with patch("score.load_dataset", return_value=rows) as load:
            score.load_qrels("mteb/T2Reranking", "r", "other")
        self.assertEqual(load.call_args.kwargs["split"], "other")


class EvalSplitRecordTest(TemporaryCase):
    def test_a_quality_result_names_the_task_split_and_score_py_reads_it(self):
        for task, split in (("mteb/T2Reranking", "dev"), ("mteb/CMedQAv1-reranking", "test")):
            with self.subTest(task):
                record, result_path = self.run_flags(FakeClient(), "--model", "m", "--task", task)
                self.assertEqual(record["dataset"]["subset"], split)
                self.assertEqual(record["verdict"]["metric"], "main_score")
                with patch("score.load_qrels", return_value=judged_qrels()) as loader:
                    _, rows, _ = score.rescore(result_path)
                loader.assert_called_once_with(task, NEW_TASKS[task][1], split)
                self.assertEqual(rows, record["metrics"])

    def test_nano_runs_as_an_encode_task_on_its_split(self):
        record, _ = self.run_flags(DenseClient(), "--model", "m", "--task", "mteb/NanoFiQA2018Retrieval", *ENCODE_SHAPE)
        self.assertEqual(record["dataset"]["subset"], "train")
        self.assertEqual(record["operation"], "encode")
        self.assertTrue(record["validity"]["valid"])

    def perf(self, key, *flags):
        return plan_performance(test_performance.parse("--model", "m", "--task", key, "--rate", "1", *flags))

    def test_the_performance_path_takes_the_task_split_and_records_it(self):
        plan = self.perf("mteb/T2Reranking?split=dev&limit=2&candidates-k=2", "--duration", "1", "--warmup", "0")
        plan = replace(plan, duration_s=0.1, settle_s=0.0, drain_s=0.0, rate=20.0)
        data = load_performance_data(plan, test_performance.tiny_loaders())
        record, _ = run_performance(
            plan, data, test_performance.FakeClient(), results_dir=self.directory / "results", now=clock
        )
        self.assertEqual(record["dataset"]["subset"], "dev")

    def test_the_performance_path_refuses_another_split(self):
        for key, split in (("mteb/T2Reranking?split=test", "dev"), ("mteb/SciFact?split=dev", "test")):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"only split={split} is available"):
                self.perf(
                    key, *(() if key.startswith("mteb/T2") else ("--output", "dense", "--items-per-request", "1"))
                )

    def test_a_performance_score_request_stays_within_the_item_cap(self):
        with self.assertRaisesRegex(ValueError, "candidates-k must be at most 100"):
            self.perf("mteb/MMarcoReranking?split=dev&limit=1&candidates-k=101")
        plan = self.perf("mteb/MMarcoReranking?split=dev&limit=1")
        data = load_performance_data(plan, many_candidates(250))
        self.assertEqual(len(data.payloads[0][1]), 100)
        plan = self.perf("mteb/MMarcoReranking?split=dev&limit=1&candidates-k=100")
        self.assertEqual(len(load_performance_data(plan, many_candidates(250)).payloads[0][1]), 100)


class AliasTest(TemporaryCase):
    ALIASES: ClassVar[dict] = {
        "mteb/CMedQAv1Reranking": "mteb/CMedQAv1-reranking",
        "mteb/CMedQAv2Reranking": "mteb/CMedQAv2-reranking",
    }

    def test_parse_normalises_and_keeps_the_parameters(self):
        for alias, canonical in self.ALIASES.items():
            with self.subTest(alias):
                self.assertEqual(parse_task_key(f"{alias}?limit=3"), (canonical, {"limit": "3"}))

    def test_the_users_key_stays_in_task_key_and_the_dataset_is_the_canonical_one(self):
        for alias, canonical in self.ALIASES.items():
            with self.subTest(alias):
                record, result_path = self.run_flags(FakeClient(), "--model", "m", "--task", alias)
                self.assertEqual(record["task_key"], alias)
                self.assertEqual(record["dataset"]["name"], canonical)
                self.assertEqual(record["dataset"]["revision"], score.HF_DATASETS[canonical][1])
                self.assertIn(alias, record["run"]["command"])
                with patch("score.load_qrels", return_value=judged_qrels()) as loader:
                    _, rows, _ = score.rescore(result_path)
                loader.assert_called_once_with(canonical, score.HF_DATASETS[canonical][1], "test")
                self.assertEqual(rows, record["metrics"])

    def test_the_command_rebuilds_the_same_plan(self):
        plan = plan_quality(parse("--model", "m", "--task", "mteb/CMedQAv1Reranking"))
        self.assertEqual(plan_quality(parse(*shlex.split(run.quality_command_line(plan))[5:])), plan)

    def test_a_performance_key_takes_an_alias_too(self):
        plan = plan_performance(
            test_performance.parse("--model", "m", "--task", "mteb/CMedQAv2Reranking?limit=2", "--rate", "1")
        )
        self.assertEqual((plan.task_key, plan.task_name), ("mteb/CMedQAv2Reranking?limit=2", "mteb/CMedQAv2-reranking"))


class CandidateSplitTest(unittest.TestCase):
    def run_rerank(self, client, count, task="mteb/MMarcoReranking"):
        plan = plan_quality(parse("--model", "m", "--task", task))
        data = load_quality_data(plan, many_candidates(count))
        with tempfile.TemporaryDirectory() as temporary:
            record, _ = run_quality(plan, data, client, results_dir=Path(temporary) / "results", now=clock)
            path = record["run"]["predictions_path"]
            predictions = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
        return record, predictions

    def test_the_batches_keep_order_and_the_cap(self):
        ids = [str(index) for index in range(250)]
        batches = candidate_batches(ids)
        self.assertEqual([len(batch) for batch in batches], [100, 100, 50])
        self.assertEqual([item for batch in batches for item in batch], ids)
        self.assertEqual([len(batch) for batch in candidate_batches(ids[:100])], [100])
        self.assertEqual(candidate_batches([]), [])

    def test_a_query_is_split_and_its_scores_are_merged(self):
        client = FakeClient()
        record, predictions = self.run_rerank(client, 250)
        readiness, *measured = client.calls
        self.assertEqual(len(readiness[2][1]), 100)
        self.assertTrue(readiness[3]["wait_for_capacity"])
        self.assertEqual([len(call[2][1]) for call in measured], [100, 100, 50])
        self.assertEqual({call[2][0]["text"] for call in measured}, {"only query"})
        self.assertEqual(len(predictions["q1"]), 250)
        self.assertEqual(record["run"]["request_count"], 3)
        self.assertEqual(record["payload"]["items_per_request"], 100)

    def test_twenty_candidates_stay_one_request(self):
        client = FakeClient()
        record, predictions = self.run_rerank(client, 20, "mteb/AskUbuntuDupQuestions")
        self.assertEqual([len(call[2][1]) for call in client.calls], [20, 20])
        self.assertEqual(record["payload"]["items_per_request"], 20)
        self.assertEqual(len(predictions["q1"]), 20)

    def test_exactly_one_hundred_is_one_request(self):
        client = FakeClient()
        self.run_rerank(client, 100)
        self.assertEqual([len(call[2][1]) for call in client.calls], [100, 100])

    def test_a_failure_in_a_later_batch_is_a_failed_run(self):
        record, predictions = self.run_rerank(FakeClient(fail_at=2), 250)
        self.assertIsNone(predictions)
        self.assertEqual(record["validity"]["reasons"], ["request failed: RuntimeError: boom"])
        self.assertEqual((record["run"]["request_count"], record["run"]["failure_count"]), (2, 1))

    def test_a_batch_that_answers_with_other_ids_is_a_failed_run(self):
        class OtherIds(FakeClient):
            def score(self, model, query, items, **kwargs):
                result = super().score(model, query, items, **kwargs)
                if len(self.calls) == 3:
                    result["scores"][0]["item_id"] = "d0"
                return result

        record, predictions = self.run_rerank(OtherIds(), 250)
        self.assertIsNone(predictions)
        self.assertIn("differ from the candidates", record["validity"]["reasons"][0])


class TwoStagePlanTest(TemporaryCase):
    def test_a_key_plans_a_score_run_with_the_encoder_and_k(self):
        plan = plan_quality(parse(*RERANKER, "--task", TWO_STAGE))
        self.assertEqual((plan.operation, plan.output_contract, plan.items_per_request), ("score", "scores", None))
        self.assertEqual((plan.candidates_model, plan.candidates_k), ("org/encoder", 2))
        self.assertEqual((plan.route, plan.task_name), ("org/reranker", "mteb/SciFact"))
        self.assertEqual(plan.comparable_reasons, ["no reference"])
        self.assertIsNone(plan.baseline)

    def test_limit_goes_with_it_and_keeps_its_meaning(self):
        plan = plan_quality(parse(*RERANKER, "--task", f"{TWO_STAGE}&limit=2"))
        self.assertEqual(plan.task_params["limit"], "2")
        self.assertEqual(plan.comparable_reasons, ["no reference", f"smoke key {TWO_STAGE}&limit=2"])

    def test_every_encode_task_takes_it(self):
        for task in set(score.HF_DATASETS) - run.SCORE_TASKS:
            with self.subTest(task):
                key = f"{task}?candidates_model=org/encoder&candidates_k=5"
                self.assertEqual(plan_quality(parse(*RERANKER, "--task", key)).candidates_k, 5)

    def test_the_command_rebuilds_the_same_plan(self):
        for flags in (
            ("--task", TWO_STAGE),
            ("--task", f"{TWO_STAGE}&limit=2", "--endpoint", LOCAL),
            ("--task", TWO_STAGE, "--baseline", "ndcg_at_10=0.3", "--baseline-source", "the page"),
        ):
            with self.subTest(flags=flags):
                plan = plan_quality(parse(*RERANKER, *flags))
                self.assertEqual(plan_quality(parse(*shlex.split(run.quality_command_line(plan))[5:])), plan)

    def test_bad_forms_exit_2_before_any_request(self):
        for key, message in (
            ("mteb/SciFact?candidates_model=org/encoder", "go together"),
            ("mteb/SciFact?candidates_k=5", "go together"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=0", "candidates_k must be a positive integer"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=-1", "candidates_k must be a positive integer"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=abc", "candidates_k must be a positive integer"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=", "candidates_k must be a positive integer"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=1.5", "candidates_k must be a positive integer"),
            ("mteb/SciFact?candidates_model=&candidates_k=5", "plain <org>/<name>"),
            ("mteb/SciFact?candidates_model=noslash&candidates_k=5", "plain <org>/<name>"),
            ("mteb/SciFact?candidates_model=org/encoder:muvera&candidates_k=5", "plain <org>/<name>"),
            ("mteb/SciFact?candidates_model=org/a%20b&candidates_k=5", "plain <org>/<name>"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=5&candidates_k=6", "repeats a parameter"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=5&limit=0", "limit must be a positive integer"),
            ("mteb/SciFact?candidates_model=org/encoder&candidates_k=5&split=test", "takes only limit, got split"),
            (
                "mteb/AskUbuntuDupQuestions?candidates_model=org/encoder&candidates_k=5",
                "mteb/AskUbuntuDupQuestions is a score task",
            ),
            ("mteb/T2Reranking?candidates_model=org/encoder&candidates_k=5", "mteb/T2Reranking is a score task"),
        ):
            with self.subTest(key):
                self.assertIn(message, self.refused("quality", *RERANKER, "--task", key, "--endpoint", LOCAL))

    def test_the_fixed_flags_are_refused(self):
        for flags, message in (
            (("--output", "dense"), "--output and --items-per-request are not accepted"),
            (("--items-per-request", "2"), "--output and --items-per-request are not accepted"),
            (("--instruction", "x"), "--instruction is not accepted"),
        ):
            with self.subTest(flags=flags):
                stderr = self.refused("quality", *RERANKER, "--task", TWO_STAGE, "--endpoint", LOCAL, *flags)
                self.assertIn(message, stderr)

    def test_a_reference_run_and_a_performance_run_refuse_the_key(self):
        reference = reference_file(self.directory)
        self.assertIn("--task must equal", self.refused("quality", "--reference", str(reference), "--task", TWO_STAGE))
        stderr = self.refused(
            "performance",
            *RERANKER,
            "--task",
            TWO_STAGE,
            "--rate",
            "1",
            "--output",
            "dense",
            "--items-per-request",
            "1",
            "--endpoint",
            LOCAL,
        )
        self.assertIn("got candidates_k, candidates_model", stderr)


class TwoStageRunTest(TemporaryCase):
    def run_key(self, client, key=TWO_STAGE, *flags):
        return self.run_flags(client, *RERANKER, "--task", key, *flags)

    def predictions(self, record):
        return json.loads(Path(record["run"]["predictions_path"]).read_text(encoding="utf-8"))

    def test_readiness_for_both_models_then_encode_then_score(self):
        client = DenseClient()
        self.run_key(client)
        encoder_ready, reranker_ready, *measured = client.calls
        self.assertEqual((encoder_ready[0], encoder_ready[1]), ("encode", "org/encoder"))
        self.assertTrue(encoder_ready[3]["wait_for_capacity"])
        self.assertEqual(encoder_ready[3]["output_types"], ["dense"])
        self.assertFalse(encoder_ready[3]["is_query"])
        self.assertEqual((reranker_ready[0], reranker_ready[1]), ("score", "org/reranker"))
        self.assertTrue(reranker_ready[3]["wait_for_capacity"])
        self.assertEqual(reranker_ready[2][0], {"text": "first query"})
        self.assertEqual([item["id"] for item in reranker_ready[2][1]], ["d1", "d2"])
        self.assertEqual([call[0] for call in measured], ["encode", "encode", "score", "score", "score"])
        documents, queries = measured[:2]
        self.assertEqual([item["id"] for item in documents[2]], ["d1", "d2", "d3", "d4", "d5"])
        self.assertEqual(documents[2][0]["text"], "Alpha alpha body")
        self.assertFalse(documents[3]["is_query"])
        self.assertEqual([item["id"] for item in queries[2]], ["q1", "q2", "q3"])
        self.assertTrue(queries[3]["is_query"])
        for call in (documents, queries):
            self.assertEqual(call[1], "org/encoder")
            self.assertEqual(call[3]["output_types"], ["dense"])
            self.assertNotIn("instruction", call[3])
            self.assertFalse(call[3]["wait_for_capacity"])
            self.assertEqual(call[3]["max_oom_retries"], 0)

    def test_only_the_top_k_by_cosine_reach_the_reranker_and_its_scores_rank_them(self):
        client = ReversingClient()
        record, _ = self.run_key(client)
        sent = [(call[2][0]["text"], [item["id"] for item in call[2][1]]) for call in client.calls[-3:]]
        self.assertEqual(
            sent,
            [("first query", ["d2", "d1"]), ("second query", ["d3", "d4"]), ("third query", ["d1", "d4"])],
        )
        self.assertEqual({call[1] for call in client.calls[-3:]}, {"org/reranker"})
        self.assertEqual(
            self.predictions(record),
            {"q1": {"d2": 1.0, "d1": 2.0}, "q2": {"d3": 1.0, "d4": 2.0}, "q3": {"d1": 1.0, "d4": 2.0}},
        )
        metrics = {row["name"]: row["value"] for row in record["metrics"]}
        self.assertEqual(metrics["mrr_at_1"], 1.0 / 3.0)
        self.assertEqual(metrics["main_score"], metrics["ndcg_at_10"])

    def test_ties_keep_corpus_order(self):
        client = DenseClient()
        self.run_key(client, "mteb/SciFact?candidates_model=org/encoder&candidates_k=4")
        self.assertEqual([item["id"] for item in client.calls[-1][2][1]], ["d1", "d4", "d2", "d3"])

    def test_the_encoder_batches_are_64_and_a_long_candidate_list_is_split(self):
        client = GeneratedClient()
        plan = plan_quality(parse(*RERANKER, "--task", "mteb/SciFact?candidates_model=org/encoder&candidates_k=150"))
        record, _ = run_quality(
            plan,
            load_quality_data(plan, many_candidates(250)),
            client,
            results_dir=self.directory / "results",
            now=clock,
        )
        self.assertEqual(len(client.calls[0][2]), 64)
        self.assertEqual(len(client.calls[1][2][1]), 100)
        encodes = [len(call[2]) for call in client.calls[2:] if call[0] == "encode"]
        self.assertEqual(encodes, [64, 64, 64, 58, 1])
        self.assertEqual([len(call[2][1]) for call in client.calls[2:] if call[0] == "score"], [100, 50])
        self.assertEqual(len(self.predictions(record)["q1"]), 150)
        self.assertEqual(record["payload"]["items_per_request"], 100)
        self.assertEqual(record["run"]["request_count"], 7)

    def test_a_k_larger_than_the_corpus_sends_the_corpus(self):
        record, _ = self.run_key(DenseClient(), "mteb/SciFact?candidates_model=org/encoder&candidates_k=999")
        self.assertEqual(len(self.predictions(record)["q1"]), 5)
        self.assertEqual(record["payload"]["items_per_request"], 5)

    def test_the_record_names_the_encoder_and_k_and_reads_measurement(self):
        record, _ = self.run_key(DenseClient())
        self.assertEqual(
            record["payload"],
            {
                "items_per_request": 2,
                "output_contract": "scores",
                "parameters": [
                    {"name": "candidates_model", "value": "org/encoder"},
                    {"name": "candidates_k", "value": 2},
                ],
            },
        )
        self.assertEqual(record["operation"], "score")
        self.assertEqual(record["task_key"], TWO_STAGE)
        self.assertEqual((record["verdict"]["verdict"], record["verdict"]["reason"]), ("MEASUREMENT", "no_baseline"))
        self.assertIsNone(record["baseline"])
        self.assertNotIn("--output", record["run"]["command"])

    def test_a_baseline_is_compared(self):
        record, _ = self.run_key(DenseClient(), TWO_STAGE, "--baseline", "recall_at_1=0.9", "--baseline-source", "page")
        self.assertEqual(record["baseline"]["kind"], "supplied")
        self.assertEqual(record["verdict"]["metric"], "recall_at_1")
        self.assertIn(record["verdict"]["reason"], {"below_margin", "within_margin", "above_margin"})

    def test_a_limit_scores_the_first_queries_and_is_a_measurement_even_with_a_baseline(self):
        client = DenseClient()
        record, _ = self.run_key(
            client, f"{TWO_STAGE}&limit=1", "--baseline", "recall_at_1=0.9", "--baseline-source", "page"
        )
        self.assertEqual(list(self.predictions(record)), ["q1"])
        self.assertEqual([item["id"] for item in client.calls[3][2]], ["q1"])
        self.assertEqual(record["verdict"]["reason"], "limited_run")

    def test_two_models_are_not_a_revision_change(self):
        client = DenseClient(revisions=("encoder-rev",) * 4 + ("reranker-rev",) * 3)
        record, _ = self.run_key(client)
        self.assertEqual(record["execution_revisions"], ["encoder-rev", "reranker-rev"])
        self.assertEqual(record["comparable_reasons"], ["no reference"])

    def test_a_change_inside_one_model_is_a_revision_change(self):
        for revisions in (
            ("rev", "rev", "encoder-1", "encoder-2", "reranker", "reranker", "reranker"),
            ("rev", "rev", "encoder", "encoder", "reranker-1", "reranker-2", "reranker-2"),
        ):
            with self.subTest(revisions=revisions):
                record, _ = self.run_key(DenseClient(revisions=revisions))
                self.assertEqual(record["comparable_reasons"], ["no reference", "execution revision changed"])

    def test_a_failure_at_each_stage_is_a_failed_run(self):
        for fail_at, reason, request_count in (
            (0, "readiness request failed: RuntimeError: boom", 0),
            (1, "readiness request failed: RuntimeError: boom", 0),
            (2, "request failed: RuntimeError: boom", 1),
            (3, "request failed: RuntimeError: boom", 2),
            (4, "request failed: RuntimeError: boom", 3),
        ):
            with self.subTest(fail_at=fail_at):
                record, _ = self.run_key(DenseClient(fail_at=fail_at))
                self.assertIsNone(record["run"]["predictions_path"])
                self.assertEqual(record["validity"]["reasons"], [reason])
                self.assertEqual(record["run"]["request_count"], request_count)
                self.assertEqual(record["verdict"]["reason"], "no_predictions")

    def test_an_sdk_retry_in_either_stage_invalidates_the_run(self):
        for retries in ((0, 0, 1), (0, 0, 0, 0, 1)):
            with self.subTest(retries=retries):
                record, _ = self.run_key(DenseClient(retries=retries))
                self.assertFalse(record["validity"]["valid"])
                self.assertEqual(record["validity"]["reasons"], ["sdk retries 1"])

    def test_score_py_recomputes_from_the_predictions_file(self):
        for key, flags in (
            (TWO_STAGE, ()),
            (TWO_STAGE, ("--baseline", "recall_at_1=0.9", "--baseline-source", "page")),
            (f"{TWO_STAGE}&limit=2", ()),
        ):
            with self.subTest(key=key, flags=flags):
                record, result_path = self.run_key(DenseClient(), key, *flags)
                with patch("score.load_qrels", return_value=judged_qrels()) as loader:
                    _, rows, _ = score.rescore(result_path)
                loader.assert_called_once_with("mteb/SciFact", score.HF_DATASETS["mteb/SciFact"][1], "test")
                self.assertEqual(rows, record["metrics"])
                self.assertEqual(recompute_verdict(record, rows), record["verdict"])

    def test_an_encoder_that_returns_only_zeros_is_a_failed_run(self):
        record, _ = self.run_key(ZeroClient())
        self.assertIsNone(record["run"]["predictions_path"])
        self.assertIn("all zeros", record["validity"]["reasons"][0])


class PerformanceKeyTest(TemporaryCase):
    def test_an_unknown_parameter_exits_2_before_any_request(self):
        shape = ("--output", "dense", "--items-per-request", "1")
        for key, parameter in (
            ("mteb/SciFact?Limit=5", "Limit"),
            ("mteb/SciFact?limit=5&foo=1", "foo"),
            ("mteb/SciFact?candidates_k=5", "candidates_k"),
        ):
            with self.subTest(key):
                stderr = self.refused(
                    "performance", "--model", "m", "--task", key, "--rate", "1", *shape, "--endpoint", LOCAL
                )
                self.assertIn(f"got {parameter}", stderr)
        stderr = self.refused(
            "performance",
            "--model",
            "m",
            "--task",
            "mteb/AskUbuntuDupQuestions?Limit=5",
            "--rate",
            "1",
            "--endpoint",
            LOCAL,
        )
        self.assertIn("got Limit", stderr)

    def test_a_reference_with_an_unknown_parameter_exits_2(self):
        reference = test_performance.reference_file(self.directory, task_key="mteb/SciFact?Limit=5")
        self.assertIn("got Limit", self.refused("performance", "--reference", str(reference), "--endpoint", LOCAL))

    def test_every_committed_performance_reference_still_plans(self):
        paths = sorted((EXAMPLE_DIR / "references").glob("*-perf-*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path.name):
                plan_performance(test_performance.parse("--reference", str(path)))


class CredentialsInOutputTest(TemporaryCase):
    SECRET = "s3cret"

    def command_stderr(self, environ, *argv):
        stderr = io.StringIO()
        with (
            patch.dict("os.environ", environ, clear=True),
            patch.object(run, "load_quality_data", side_effect=lambda plan: load_quality_data(plan, tiny_loaders())),
            patch.object(run, "load_performance_data"),
            patch.object(run, "run_quality", side_effect=SystemExit(0)),
            patch.object(run, "run_performance", side_effect=SystemExit(0)),
            patch.object(run, "SIEClient"),
            patch.object(run, "http_fetch", side_effect=ConnectionError),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit),
        ):
            run.main(list(argv))
        return stderr.getvalue()

    def note(self, base_url):
        return self.command_stderr(
            {"SIE_API_KEY": "key", "SIE_BASE_URL": base_url},
            "quality",
            *DENSE_FLAGS,
            "--baseline",
            "ndcg_at_10=0.3",
            "--baseline-source",
            "s",
            "--endpoint",
            LOCAL,
        )

    def test_the_note_drops_credentials_from_the_base_url(self):
        stderr = self.note(f"https://user:{self.SECRET}@example.test:8443/base?token={self.SECRET}")
        self.assertNotIn(self.SECRET, stderr)
        self.assertNotIn("user", stderr)
        self.assertIn("scoped to https://example.test:8443/base and was not sent", stderr)

    def test_an_unparseable_base_url_is_not_echoed(self):
        for base_url in (
            f"http://user:{self.SECRET}@[bad",
            f"http://user:{self.SECRET}@[bad]:80",
            f"http://user:{self.SECRET}@host:port",
            f"user:{self.SECRET}@host:8080",
            f"ftp://user:{self.SECRET}@host",
            f"{self.SECRET}",
        ):
            with self.subTest(base_url=base_url):
                stderr = self.note(base_url)
                self.assertNotIn(self.SECRET, stderr)
                self.assertIn("an SIE_BASE_URL that is not an http(s) URL", stderr)

    def test_an_odd_endpoint_is_not_echoed(self):
        for endpoint in (
            f"http://user:{self.SECRET}@[bad",
            f"http://user:{self.SECRET}@[bad]:80",
            f"http://user:{self.SECRET}@[{self.SECRET}]",
            f"http://user:{self.SECRET}@host:port",
            f"user:{self.SECRET}@host:8080",
            f"ftp://user:{self.SECRET}@host",
            f"http://user:{self.SECRET}@:8080",
            f"http://user:{self.SECRET}@host:0",
        ):
            for command in (
                ("quality", *DENSE_FLAGS),
                ("performance", "--model", "m", "--task", "mteb/AskUbuntuDupQuestions", "--rate", "1"),
            ):
                with self.subTest(endpoint=endpoint, command=command[0]):
                    stderr = self.refused(*command, f"--endpoint={endpoint}")
                    self.assertNotIn(self.SECRET, stderr)
                    self.assertIn("endpoint must be an http(s) URL", stderr)

    def test_the_default_endpoint_error_is_unchanged(self):
        stderr = self.command_stderr(
            {}, "quality", *DENSE_FLAGS, "--baseline", "ndcg_at_10=0.3", "--baseline-source", "s"
        )
        self.assertIn(f"SIE_API_KEY must be set in the environment for {DEFAULT_ENDPOINT}", stderr)


class ZeroVectorTest(TemporaryCase):
    def test_all_zero_vectors_are_a_failed_run_on_either_side(self):
        for zero, role in ((("document", "query"), "document"), (("document",), "document"), (("query",), "query")):
            with self.subTest(zero=zero):
                record, _ = self.run_flags(ZeroClient(zero), *DENSE_FLAGS)
                self.assertFalse(record["validity"]["valid"])
                self.assertEqual(
                    record["validity"]["reasons"],
                    [f"request failed: ValueError: every returned dense {role} vector is all zeros"],
                )
                self.assertIsNone(record["run"]["predictions_path"])
                self.assertEqual(record["metrics"], [])
                self.assertEqual(
                    (record["verdict"]["verdict"], record["verdict"]["reason"]), ("FAIL (no-output)", "no_predictions")
                )

    def test_a_few_zero_vectors_are_left_alone(self):
        class OneZero(DenseClient):
            def _encode_item(self, item, output_types):
                result = super()._encode_item(item, output_types)
                if item["text"] == "gamma body":
                    result["dense"] = np.zeros(2, dtype=np.float32)
                return result

        record, _ = self.run_flags(OneZero(), *DENSE_FLAGS)
        self.assertTrue(record["validity"]["valid"])
        self.assertIsNotNone(record["run"]["predictions_path"])

    def test_the_command_exits_1_and_names_it(self):
        code, stdout, stderr = self.command(
            ZeroClient(), *DENSE_FLAGS, "--baseline", "ndcg_at_10=0.3", "--baseline-source", "s", "--endpoint", LOCAL
        )
        self.assertEqual(code, 1)
        self.assertIn("verdict: FAIL (no-output) no predictions", stdout)
        self.assertIn("invalid: request failed: ValueError: every returned dense document vector is all zeros", stderr)


class ProjectDescriptionTest(unittest.TestCase):
    def test_the_description_says_it_also_tests_a_new_model(self):
        project = tomllib.loads((EXAMPLE_DIR / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertIn("test a new model", project["description"])


if __name__ == "__main__":
    unittest.main()
