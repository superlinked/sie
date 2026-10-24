"""Offline Cloud price-label compatibility and unchanged settlement checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embedding import cloud_reader, collect_features, encode
from embedding import recipes as core
from test_public_adapters import source_fixture


def fixture(root, *, label="ceil-once-per-terminal-event-micro", mutate=None):
    counts = {"first": 166, "second": 167}
    intent = {
        "ids": list(counts),
        "quote": {
            "applied_rates": [copy.deepcopy(collect_features.RATE)],
            "rounding_rule": label,
            "minimum_billed_units": None,
            "rate_book_version": "test-v40",
            "rate_book_sha256": "a" * 64,
        },
    }
    receipt = {
        "ids": list(counts),
        "credits_decimal": "2.000000",
        "usage": {"input_tokens": 333, "rate_book_version": "test-v40"},
    }
    if mutate:
        mutate(intent, receipt)
    directory = root / "000000"
    core.write(directory / "intent.json", intent)
    receipt["intent_sha256"] = core.sha(directory / "intent.json")
    core.write(directory / "receipt.json", receipt)
    core.write(root / "collection.json", {
        "status": "complete", "unknown_charge_batches": [],
        "batch_receipts": [{"path": "000000/receipt.json", "sha256": core.sha(directory / "receipt.json")}],
    })
    return counts


class CloudReaderTests(unittest.TestCase):
    def test_both_observed_labels_preserve_per_event_rounding_and_raw_receipts(self):
        for label in sorted(cloud_reader.ROUNDING_LABELS):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                counts = fixture(root, label=label)
                before = {p.name: p.read_bytes() for p in (root / "000000").iterdir()}
                book = cloud_reader.check_tokens(root, counts)
                self.assertEqual(book, {"rate_book_version": "test-v40", "rate_book_sha256": "a" * 64})
                self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "000000").iterdir()})
                self.assertEqual(collect_features.credits(sum(counts.values())), 2)
                self.assertEqual(sum(collect_features.credits(n) for n in counts.values()), 3)

    def test_changed_rate_unit_minimum_or_label_is_rejected(self):
        changes = {
            "rate": lambda i, r: i["quote"]["applied_rates"][0].update(rate_numerator=4),
            "unit": lambda i, r: i["quote"]["applied_rates"][0].update(unit="output_tokens"),
            "minimum": lambda i, r: i["quote"].update(minimum_billed_units=1),
            "label": lambda i, r: i["quote"].update(rounding_rule="floor-per-request"),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                counts = fixture(root, mutate=change)
                with self.assertRaisesRegex(ValueError, "pricing basis"):
                    cloud_reader.check_tokens(root, counts)

    def test_fractional_or_per_item_charge_and_wrong_tokens_are_rejected(self):
        changes = {
            "fractional": lambda i, r: r.update(credits_decimal="1.998000"),
            "per_item": lambda i, r: r.update(credits_decimal="3.000000"),
            "tokens": lambda i, r: r["usage"].update(input_tokens=334),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                counts = fixture(root, mutate=change)
                with self.assertRaisesRegex(ValueError, "token/fee projection"):
                    cloud_reader.check_tokens(root, counts)

    def test_changed_receipt_or_intent_is_rejected(self):
        for name, message in (("receipt.json", "Batch binding"), ("intent.json", "Intent changed")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                counts = fixture(root)
                path = root / "000000" / name
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    cloud_reader.check_tokens(root, counts)

    def test_mismatched_ids_or_book_and_unknown_settlement_are_rejected(self):
        changes = {
            "ids": (lambda i, r: r.update(ids=["first"]), "Batch IDs"),
            "book": (lambda i, r: r["usage"].update(rate_book_version="other"), "Usage/quote"),
        }
        for name, (change, message) in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                counts = fixture(root, mutate=change)
                with self.assertRaisesRegex(ValueError, message):
                    cloud_reader.check_tokens(root, counts)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = fixture(root)
            collection = core.read(root / "collection.json")
            collection["unknown_charge_batches"] = [{"id": "unresolved"}]
            (root / "collection.json").write_text(json.dumps(collection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Complete settled"):
                cloud_reader.check_tokens(root, counts)

    def test_reader_wraps_only_one_collection_and_restores_checker_on_failure(self):
        original = collect_features.check_tokens
        for failure in (False, True):
            def run(*args, _failure=failure, **kwargs):
                self.assertIs(collect_features.check_tokens, cloud_reader.check_tokens)
                self.assertEqual(args, ("source", "output"))
                self.assertEqual(kwargs, {"marker": 1})
                if _failure:
                    raise RuntimeError("fixture failure")
                return {"status": "complete"}

            with self.subTest(failure=failure), patch.object(collect_features, "run", side_effect=run):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        cloud_reader.run("source", "output", marker=1)
                else:
                    self.assertEqual(cloud_reader.run("source", "output", marker=1), {"status": "complete"})
                self.assertIs(collect_features.check_tokens, original)

    def test_reader_plan_binds_helper_and_preserves_source_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(Path(directory))
            with patch.object(collect_features, "singleton_rows", return_value=[]):
                plan = encode.prepare(source, "1", 60)
            self.assertEqual(plan["source_sha256"]["cloud_reader"], core.sha(cloud_reader.__file__))
            path = source / "question-only/train-inputs.json"
            rows = json.loads(path.read_bytes())
            rows[0]["request"] = "Changed input"
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Input hash changed"):
                encode.prepare(source, "1", 60)


if __name__ == "__main__":
    unittest.main()
