"""Tiny mocked lifecycle checks; no API calls, downloads, training, or weights."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import collect
import collect_answers as answers
import grade_v2 as grading
import openai_population as acquisition
from reader_io import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def source_fixture(root):
    source = root / "source"
    meta, files = [], {}
    for role in ("train", "development", "calibration", "final"):
        item = {"id": role + "-q", "request": "Select the orange ball.",
                "response_contract": {"type": "exact_string", "max_length": 2048}}
        meta.append({"id": item["id"], "role": role, "retained": True,
                     "task": "dyck_languages", "family": role + "-family",
                     "request_sha256": hashlib.sha256(item["request"].encode()).hexdigest()})
        path = source / f"question-only/{role}-inputs.json"
        write(path, [item])
        files[f"question-only/{role}-inputs.json"] = sha(path)
    write(source / "meta.json", meta)
    files["meta.json"] = sha(source / "meta.json")
    write(source / "source-plan.json", {"files_sha256": files})
    # A reference file exists, but neither preparation nor acquisition may open it.
    (source / "gold.json").write_text("DO NOT READ THIS FILE")
    return source


def response(item, arm, *, valid=True):
    body = collect.body_for(item, arm)
    usage = {"input_tokens": 3, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 0}}
    return {"id": item["id"], "arm": arm, "model": body["model"], "resolved_model": body["model"],
            "body_sha256": acquisition.digest(json.dumps(body, sort_keys=True).encode()),
            "status": "complete", "terminal_response": True, "execution_unknown": False,
            "created_unix": 0, "elapsed_seconds": 0.1, "http_status": 200, "service_tier": "default",
            "provider_status": "completed", "usage": usage, "cost": collect.openai_cost(usage, body),
            "answer": "orange ball" if valid else None, "valid_answer": valid,
            "messages": [{"type": "output_text", "text": '{"answer":"orange ball"}' if valid else "invalid JSON"}]}


class AnswerAdapterTests(unittest.TestCase):
    def test_question_only_plan_excludes_calibration_final_and_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            items, plan = answers.prepare(source, Path(__file__).with_name("pair.json"), "learn", "1", 20)
            self.assertEqual(list(items), ["train", "development"])
            self.assertEqual(plan["roles"], ["train", "development"])
            self.assertFalse(plan["references_loaded_by_collector"])
            self.assertEqual(collect.body_for(items["train"][0], "sol")["max_output_tokens"], 16384)

    def test_resume_never_repeats_a_terminal_bad_format_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            items, plan = answers.prepare(source, Path(__file__).with_name("pair.json"), "calibration", "1", 20)
            calls = []

            def call(item, arm):
                calls.append((item["id"], arm))
                return response(item, arm, valid=False)

            output = root / "records"
            first = answers.execute(items, plan, output, call=call)
            original = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*-response.json")}
            second = answers.execute(items, plan, output, call=lambda *_: self.fail("Paid terminal response repeated"))
            self.assertEqual(len(calls), 2)
            self.assertEqual(first["physical_attempts"], second["physical_attempts"])
            self.assertEqual(original, {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*-response.json")})
            self.assertGreater(float(second["known_list_usd"]), 0)
            with self.assertRaisesRegex(ValueError, "resumed collection plan"):
                answers.execute(items, {**plan, "budget_usd": "2"}, output, call=call)

    def test_final_is_blocked_before_dispatch_without_exact_frozen_population(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "raw frozen policy bundle"):
                answers.prepare(source, Path(__file__).with_name("pair.json"), "final", "1", 20)
            plan_path = Path(directory) / "wrong-freeze.json"
            write(plan_path, {"format": "public-retrospective-envelope"})
            with self.assertRaises(ValueError):
                answers.prepare(source, Path(__file__).with_name("pair.json"), "final", "1", 20, plan_path)

    def test_changed_source_or_pair_cannot_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            path = source / "question-only/train-inputs.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                answers.prepare(source, Path(__file__).with_name("pair.json"), "learn", "1", 20)

    def test_new_receipts_grade_and_enter_existing_learner_without_rebinding(self):
        from embedding import recipes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            metadata = json.loads((source / "meta.json").read_bytes())
            write(source / "gold.json", [{"id": row["id"], "target": "orange ball"} for row in metadata])
            source_plan = json.loads((source / "source-plan.json").read_bytes())
            source_plan["files_sha256"]["gold.json"] = sha(source / "gold.json")
            write(source / "source-plan.json", source_plan)
            pair = Path(__file__).with_name("pair.json")
            items, plan = answers.prepare(source, pair, "learn", "1", 20)
            records = root / "records"
            answers.execute(items, plan, records, call=response)
            official = root / "official.py"
            official.write_text("# Synthetic official adapter; no upstream execution.\n")
            fixture = SimpleNamespace(preprocess_sample=lambda value: value.strip(),
                                      preprocess_reference=lambda value: value.strip(),
                                      evaluate_correctness=lambda answer, target: answer.strip() == target.strip())
            with patch.object(grading, "load_official", return_value=fixture):
                index = grading.grade(source, records, official, pair, root / "grades")
            run = root / "run"
            learner_plan = recipes.prepare(source, pair, run)
            for role in ("train", "development"):
                rows = recipes.role_rows(run, learner_plan, role)
                y, costs, document = recipes.grades(learner_plan, rows, root / "grades" / f"grades-{role}.json", role)
                self.assertEqual(y.tolist(), [[1, 1]])
                self.assertTrue((costs > 0).all())
                self.assertEqual(document["acquisition_protocol_sha256"], sha(records / "plan.json"))
                self.assertEqual(document["grader_sha256"], sha(Path(grading.__file__)))
            self.assertEqual(set(index["roles"]), {"train", "development"})


class NativeBaseAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Direct native scripts resolve their preserved sibling imports this way too.
        cls.native_path = str(Path(__file__).parent / "native")
        sys.path.insert(0, cls.native_path)
        spec = importlib.util.spec_from_file_location("public_base_adapter", Path(cls.native_path) / "base.py")
        cls.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.base)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.native_path)

    def test_cpu_score_uses_only_requested_input_role_and_same_cost_rule(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "base.json", {"synthetic": True})
            plan = {"source_plan_sha256": "a" * 64, "pair_sha256": "b" * 64,
                    "pair": {"models": {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"}},
                    "native": {"max_len": 8192, "head_max_len": 256}}
            rows = [{"id": "c", "role": "calibration", "request": "calibration question"},
                    {"id": "f", "role": "final", "request": "final question"}]
            stats = {"unit": "USD", "arm_order": ["cheap", "strong"], "class_order": [0, 1],
                     "means": [[1, 2], [10, 20]], "counts": [[1, 1], [1, 1]]}
            with patch.object(self.base, "checked_base", return_value=(plan, rows, {"conditional_cost_usd": stats})), \
                 patch.object(self.base.native, "checkpoint_probabilities", return_value=np.array([[0.25, 0.75]])) as scorer:
                result = self.base.score(root, root, "calibration", root / "scores.json", "cpu")
                self.assertEqual(scorer.call_args.kwargs["device"], "cpu")
                self.assertEqual(scorer.call_args.args[1], [rows[0]])
                self.assertEqual(result["ids"], ["c"])
                self.assertEqual(result["methods"]["base"]["expected_usd"], [[1.25, 17.5]])
                self.assertFalse(result["outcomes_accessed"])
            with self.assertRaisesRegex(ValueError, "role/device"):
                self.base.score(root, root, "train", root / "bad.json", "cpu")


if __name__ == "__main__":
    unittest.main()
