"""Small runtime-contract checks without model downloads, CUDA or real outcomes."""

import contextlib
import copy
import importlib
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

NATIVE = Path(__file__).parent / "native"
sys.path.insert(0, str(NATIVE))
protocol = importlib.import_module("fast_protocol")
runtime = importlib.import_module("fast_runtime")
fast = importlib.import_module("fast_train")
sys.path.pop(0)


class BatchTests(unittest.TestCase):
    def test_membership_and_partial_tail_weighting(self):
        for count in (1, 2, 4, 5, 13, 16, 18, 28, 31, 32, 67):
            items = [{"id": i, "ids": [0] * (19 - i % 17)} for i in range(count)]
            original = copy.deepcopy(items)
            windows = list(runtime.grouped_windows(items))
            self.assertEqual(items, original)
            for index, window in enumerate(windows):
                expected = original[index * 16:(index + 1) * 16]
                self.assertEqual(sorted(x["id"] for x in window), [x["id"] for x in expected])
                if len(expected) % 4:
                    self.assertEqual(window, expected)
                else:
                    self.assertEqual([len(x["ids"]) for x in window], sorted(len(x["ids"]) for x in expected))

    def test_option_rng_remains_assigned_to_original_examples_across_epochs(self):
        config = types.SimpleNamespace(seed=89, shuffle_options=("choice",))
        items = [{"id": i, "length": 2 + (i * 7) % 23} for i in range(35)]

        def draw(item, rng, kinds):
            options = ["incorrect", "correct"]
            rng.shuffle(options)
            return options

        def encode(tokenizer, item, maximum, head, options, parallel):
            return {"id": item["id"], "ids": [0] * item["length"], "options": options}

        lt = types.SimpleNamespace(encode_item=encode, draw_option_order=draw)
        expected_rng, actual_rng = random.Random(89), random.Random(89)
        for epoch in range(3):
            order = list(items)
            random.Random(89 + epoch).shuffle(order)
            expected = {x["id"]: draw(x, expected_rng, config.shuffle_options) for x in order}
            encoded = runtime.encoded_epoch(lt, None, items, config, epoch, actual_rng, 8192, 256)
            grouped = [x for w in runtime.grouped_windows(encoded) for x in w]
            self.assertEqual({x["id"]: x["options"] for x in grouped}, expected)
            self.assertEqual(actual_rng.getstate(), expected_rng.getstate())


class ProvenanceTests(unittest.TestCase):
    def test_reference_source_change_fails_closed(self):
        self.assertEqual(protocol.checked_reference(), protocol.REFERENCE_FILES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in protocol.REFERENCE_FILES:
                (root / name).write_bytes((NATIVE / name).read_bytes())
            with (root / "bbeh_native.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# modified\n")
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                protocol.checked_reference(root)

    def test_gpu_dependency_drift_fails_closed(self):
        with patch.object(protocol.importlib.metadata, "version", side_effect=lambda name: protocol.VERSIONS[name]):
            self.assertEqual(protocol.checked_versions(), protocol.VERSIONS)
        with patch.object(protocol.importlib.metadata, "version", return_value="different"), \
             self.assertRaisesRegex(ValueError, "pinned GPU"):
            protocol.checked_versions()

    def test_wrapper_binds_actual_runtime_and_preserves_reference_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            core = fast.reference.core
            core.write(run / "plan.json", {"source": "unchanged"})
            original_plan = (run / "plan.json").read_bytes()
            plan = {"native": {"gradient_checkpointing": True}}
            recipe = protocol.manifest(core.sha(run / "plan.json"), {"synthetic": True})
            core.write(run / "fast-recipe.json", recipe)
            recorder = fast.Recorder(run / "fast-runtime")

            def original_train(root, training, development, checkpoint, *, commit):
                self.assertEqual((training, development), ("train.json", "development.json"))
                out = root / "native"
                out.mkdir()
                core.write(out / "training-start.json", {"config": plan["native"]})
                core.write(out / "selection-config.json", {"gradient_checkpointing": True})
                commit()
                self.assertFalse(core.read(out / "training-start.json")["config"]["gradient_checkpointing"])
                self.assertFalse(core.read(out / "selection-config.json")["gradient_checkpointing"])
                core.write(out / "trained.json", {"plan_sha256": core.sha(run / "plan.json"),
                           "training_start_sha256": core.sha(out / "training-start.json")})
                commit()

            with patch.object(fast, "prepare", return_value=(run, plan, recipe, recorder)), \
                 patch.object(fast, "fast_execution", return_value=contextlib.nullcontext()), \
                 patch.object(fast.reference, "train", side_effect=original_train):
                trained = fast.train(run, "train.json", "development.json", "checkpoint")
            out = run / "native"
            self.assertEqual(trained["format"], "bbeh-paired-native-fast-v1")
            self.assertEqual(trained["fast_recipe_sha256"], core.sha(out / "fast-recipe.json"))
            self.assertEqual(trained["fast_runtime_sha256"], core.sha(out / "fast-runtime.json"))
            self.assertEqual(trained["training_start_sha256"], core.sha(out / "training-start.json"))
            self.assertEqual(trained["reference_training_start_sha256"], core.sha(out / "training-start.reference.json"))
            self.assertTrue(core.read(out / "training-start.reference.json")["config"]["gradient_checkpointing"])
            self.assertEqual((run / "plan.json").read_bytes(), original_plan)


class HookTests(unittest.TestCase):
    def test_scoring_guard_and_hooks_restored_after_exception(self):
        lt = types.ModuleType("laya.train")
        original_train = lambda *a, **k: "original"
        original_records = lambda *a, **k: [1, 2]
        lt.train_model, lt.calibration_records = original_train, original_records
        laya = types.ModuleType("laya")
        laya.train = lt
        cuda = types.SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                     max_memory_allocated=lambda: 17, max_memory_reserved=lambda: 23)
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        original_select = fast.reference.select_epoch
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict(sys.modules, {"laya": laya, "laya.train": lt, "torch": torch}), \
             patch.object(runtime, "require_sdpa_fp32") as checked, \
             patch.object(runtime, "train_model", return_value=[0.5]) as trained:
            recorder = fast.Recorder(Path(temporary) / "events")
            with self.assertRaisesRegex(RuntimeError, "deliberate"), fast.fast_execution(recorder):
                self.assertEqual(lt.train_model("model"), [0.5])
                self.assertEqual(trained.call_args.kwargs["phase"], "selection")
                self.assertEqual(lt.calibration_records("model"), [1, 2])
                checked.assert_called_once_with("model")
                self.assertEqual(recorder.events[-1]["attention"], "sdpa")
                self.assertFalse(recorder.events[-1]["autocast"])
                raise RuntimeError("deliberate")
            self.assertIs(lt.train_model, original_train)
            self.assertIs(lt.calibration_records, original_records)
            self.assertIs(fast.reference.select_epoch, original_select)


if __name__ == "__main__":
    unittest.main()
