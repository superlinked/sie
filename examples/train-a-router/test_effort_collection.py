"""No-provider tests for the prospective Luna-low collection lifecycle."""

import copy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

import collect
from effort import collect_low as low


def item(number=0):
    return {"id": f"bbeh:synthetic:{number:03d}", "request": "Which symbol follows A? Return B.",
            "response_contract": {"type": "exact_string", "max_length": 2048}}


def plan(items):
    return {"format": "bbeh-luna-low-acquisition-v1", "protocol": low.PROTOCOL,
            "n": len(items), "ordered_ids": [row["id"] for row in items],
            "inputs_canonical_sha256": low.digest(items),
            "request_body_sha256": {row["id"]: low.digest(low.body_for(row)) for row in items},
            "source_files_sha256": low.source_hashes(), "budget_usd": "20.00", "deadline_seconds": 5400}


def success(row, *, answer="B"):
    body = low.body_for(row)
    usage = {"input_tokens": 100, "output_tokens": 10,
             "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 5},
             "output_tokens_details": {"reasoning_tokens": 4}, "total_tokens": 110}
    return {"id": row["id"], "arm": "luna", "model": low.MODEL, "reasoning_effort": "low",
            "body_sha256": low.digest(body), "created_unix": time.time(), "elapsed_seconds": 0.01,
            "terminal_response": True, "execution_unknown": False, "request_not_sent": False,
            "http_status": 200, "resolved_model": low.MODEL, "service_tier": "default",
            "resolved_reasoning_effort": "low", "provider_status": "completed", "status": "complete",
            "answer": answer, "valid_answer": answer is not None,
            "messages": [{"type": "output_text", "text": json.dumps({"answer": answer})}],
            "usage": usage, "cost": collect.openai_cost(usage, body)}


def transient(row, status=503):
    return {"id": row["id"], "arm": "luna", "model": low.MODEL, "reasoning_effort": "low",
            "body_sha256": low.digest(low.body_for(row)), "created_unix": time.time() - 100,
            "elapsed_seconds": 0.01, "terminal_response": False, "execution_unknown": False,
            "request_not_sent": False, "http_status": status, "retry_after_seconds": 0,
            "status": "provider_error", "error_code": "overloaded", "error_type": "server_error"}


class CollectionTests(unittest.TestCase):
    def test_request_changes_only_luna_effort_and_never_original_adapter(self):
        settings = copy.deepcopy(collect.OPENAI_SETTINGS)
        row = item()
        medium, actual = collect.body_for(row, "luna"), low.body_for(row)
        self.assertEqual(actual, {**medium, "reasoning": {"effort": "low"}})
        self.assertEqual(actual["model"], "gpt-6-luna")
        self.assertEqual(actual["max_output_tokens"], 16384)
        self.assertEqual(collect.OPENAI_SETTINGS, settings)
        with self.assertRaises(ValueError):
            low.body_for({**row, "reference": "B"})

    def test_actual_http_adapter_preserves_readable_usage_and_hides_provider_extras(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"model": low.MODEL, "service_tier": "default", "status": "completed",
                "reasoning": {"effort": "low"}, "id": "provider-response-id",
                "output": [{"type": "reasoning", "encrypted_content": "synthetic-opaque"},
                           {"type": "message", "content": [{"type": "output_text", "text": '{"answer":"B"}'}]}],
                "usage": success(item())["usage"]})

        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-key"}):
            row = low.acquire(item(), lambda: httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(requests, [low.body_for(item())])
        self.assertEqual(row["answer"], "B")
        self.assertEqual(row["cost"], collect.openai_cost(row["usage"], requests[0]))
        self.assertGreaterEqual(row["elapsed_seconds"], 0)
        for forbidden in ("synthetic-key", "provider-response-id", "synthetic-opaque", "Authorization"):
            self.assertNotIn(forbidden, json.dumps(row))

    def test_successful_resume_has_no_new_attempt_or_response_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, root = [item(0), item(1)], Path(temporary)
            checks = []
            result = low.run(rows, root, plan(rows), commit=lambda: checks.append(True), call=success, emit=False)
            self.assertEqual(result["status"], "complete")
            before = {str(path): path.read_bytes() for path in root.glob("attempts/*/*")}
            with patch.object(low, "acquire", side_effect=AssertionError("Must not call")):
                result = low.run(rows, root, plan(rows), call=lambda row: self.fail("Duplicate call"), emit=False)
            self.assertEqual(result["physical_attempts"], 2)
            self.assertEqual(before, {str(path): path.read_bytes() for path in root.glob("attempts/*/*")})
            self.assertTrue(checks)

    def test_no_retry_for_invalid_truncated_or_incomplete_terminal_receipt(self):
        for variant in ("invalid", "truncated", "missing-usage"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                calls = []

                def call(row):
                    calls.append(row["id"])
                    value = success(row, answer=None)
                    if variant == "truncated":
                        value.update(provider_status="incomplete", incomplete_reason="max_output_tokens")
                    if variant == "missing-usage":
                        value.pop("cost")
                        value.pop("usage")
                        value.update(status="receipt_incomplete", execution_unknown=True)
                    return value

                low.run([item()], temporary, plan([item()]), call=call, emit=False)
                low.run([item()], temporary, plan([item()]), call=call, emit=False)
                entry = low.read(Path(temporary) / "selection.json")["entries"][0]
                self.assertEqual(calls, [item()["id"]])
                self.assertEqual(entry["selected_attempt"], 1)

    def test_transient_retries_are_bounded_and_charge_every_unknown_attempt(self):
        for succeed in (True, False):
            with self.subTest(succeed=succeed), tempfile.TemporaryDirectory() as temporary:
                calls = []

                def call(row):
                    calls.append(row["id"])
                    return success(row) if succeed and len(calls) == 3 else transient(row)

                result = low.run([item()], temporary, plan([item()]), call=call, emit=False)
                self.assertEqual(len(calls), 3)
                reserve = collect.reservation(low.body_for(item()), "luna")
                self.assertEqual(Decimal(result["unknown_reservation_usd"]), reserve * (2 if succeed else 3))
                self.assertEqual(result["status"], "complete" if succeed else "incomplete")
                low.run([item()], temporary, plan([item()]), call=call, emit=False)
                self.assertEqual(len(calls), 3)

    def test_uncertain_response_and_missing_receipt_are_not_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            row, root = item(), Path(temporary)
            request_plan = plan([row])
            low.original.dump(root / "plan.json", request_plan)
            path = root / "attempts" / hashlib.sha256(row["id"].encode()).hexdigest() / "001-intent.json"
            reserve = collect.reservation(low.body_for(row), "luna")
            low.original.dump(path, {"id": row["id"], "arm": "luna", "model": low.MODEL, "reasoning_effort": "low",
                "attempt": 1, "body_sha256": low.digest(low.body_for(row)), "plan_sha256": low.sha(root / "plan.json"),
                "reserved_usd": str(reserve), "created_unix": time.time() - 10})
            result = low.run([row], root, request_plan, call=lambda value: self.fail("Unknown repeated"), emit=False)
            self.assertEqual(result["status_counts"], {"uncertain": 1})
            self.assertEqual(Decimal(result["unknown_reservation_usd"]), reserve)
            self.assertIsNone(low.read(root / "selection.json")["entries"][0]["attempts"][0]["record_path"])
            self.assertFalse(path.with_name("001-response.json").exists())
        with tempfile.TemporaryDirectory() as temporary:
            calls = []

            def timeout(row):
                calls.append(row["id"])
                return {**transient(row), "http_status": None, "error_type": "ReadTimeout",
                        "status": "failed_or_unknown", "execution_unknown": True}

            low.run([item()], temporary, plan([item()]), call=timeout, emit=False)
            low.run([item()], temporary, plan([item()]), call=timeout, emit=False)
            self.assertEqual(len(calls), 1)

    def test_identity_and_fatal_quota_stop_and_preserve_first_terminal(self):
        for variant in ("model", "tier", "effort", "quota"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                calls = []

                def call(row):
                    calls.append(row["id"])
                    value = success(row)
                    if variant == "model":
                        value["resolved_model"] = "gpt-6.1-sol"
                    elif variant == "tier":
                        value["service_tier"] = "flex"
                    elif variant == "effort":
                        value["resolved_reasoning_effort"] = "medium"
                    else:
                        value = {**transient(row, 429), "error_code": "credit_balance_exhausted"}
                    return value

                result = low.run([item()], temporary, plan([item()]), call=call, emit=False)
                self.assertEqual(result["stop"], "provider_auth_quota_or_identity_error")
                self.assertEqual(result["status"], "incomplete")
                entry = low.read(Path(temporary) / "selection.json")["entries"][0]
                self.assertEqual(entry["selected_attempt"], None if variant == "quota" else 1)
                low.run([item()], temporary, plan([item()]), call=call, emit=False)
                self.assertEqual(len(calls), 1)

    def test_expired_deadline_and_insufficient_budget_dispatch_nothing(self):
        for variant in ("deadline", "budget"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root, request_plan = Path(temporary), plan([item()])
                if variant == "budget":
                    request_plan["budget_usd"] = "0.00001"
                low.original.dump(root / "plan.json", request_plan)
                if variant == "deadline":
                    low.original.dump(root / "execution.json", {"plan_sha256": low.sha(root / "plan.json"),
                        "started_unix": time.time() - 6000, "deadline_seconds": 5400})
                result = low.run([item()], root, request_plan, call=lambda row: self.fail("Unauthorized call"), emit=False)
                self.assertEqual(result["physical_attempts"], 0)
                self.assertEqual(result["stop"], "dispatch_deadline" if variant == "deadline" else "budget_bound")

    def test_all_declared_source_bindings_and_undeclared_responses_are_checked(self):
        for name in low.source_hashes():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                request_plan = plan([item()])
                request_plan["source_files_sha256"][name] = "0" * 64
                with self.assertRaisesRegex(ValueError, "source changed"):
                    low.run([item()], temporary, request_plan, call=lambda row: self.fail("Called"), emit=False)
        with tempfile.TemporaryDirectory() as temporary:
            root, request_plan = Path(temporary), plan([item()])
            low.original.dump(root / "plan.json", request_plan)
            low.original.dump(root / "attempts/unknown/001-response.json", {"undeclared": True})
            with self.assertRaisesRegex(ValueError, "Unaccounted"):
                low.run([item()], root, request_plan, call=lambda row: self.fail("Called"), emit=False)

    def test_rolling_concurrency_never_exceeds_sixteen_and_intents_precede_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, active, maximum, lock = Path(temporary), 0, 0, threading.Lock()
            committed = set()

            def commit():
                committed.update(path.relative_to(root).as_posix() for path in root.glob("attempts/*/*-intent.json"))

            def call(row):
                nonlocal active, maximum
                name = "attempts/" + hashlib.sha256(row["id"].encode()).hexdigest() + "/001-intent.json"
                self.assertIn(name, committed)
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.01)
                with lock:
                    active -= 1
                return success(row)

            items = [item(number) for number in range(33)]
            result = low.run(items, root, plan(items), commit=commit, call=call, emit=False)
            self.assertEqual(result["complete_receipts"], 33)
            self.assertGreater(maximum, 1)
            self.assertLessEqual(maximum, 16)


if __name__ == "__main__":
    unittest.main()
