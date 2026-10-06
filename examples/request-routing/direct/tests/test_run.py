"""Offline fixtures exercise SDK serialization, validation and single-send behavior."""

import copy
import unittest

import httpx
import msgpack
from sie_sdk import SIEError

from run import LABELS, MODELS, TEXT, RetryDisabled, SingleSendClient, route, select_handler

BASE = "http://localhost:8000"


def fixture_scores() -> list[dict]:
    return [{"label": label, "score": 0.9 if label == "edit calendar events" else 0.01} for label in LABELS]


class RoutingTests(unittest.TestCase):
    def test_sdk_sends_complete_native_contract_and_caller_selection_is_separate(self) -> None:
        calls = []
        scores = fixture_scores()

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                headers={"Content-Type": "application/msgpack"},
                content=msgpack.packb({"items": [{"entities": [], "classifications": scores}]}, use_bin_type=True),
            )

        transport = httpx.Client(base_url=BASE, transport=httpx.MockTransport(respond))
        with SingleSendClient(BASE, api_key="", http_client=transport) as client:
            result = route(client, MODELS[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].method, "POST")
        self.assertEqual(calls[0].url.path, f"/v1/extract/{MODELS[0]}")
        self.assertEqual(
            msgpack.unpackb(calls[0].content, raw=False),
            {"items": [{"text": TEXT}], "params": {"labels": list(LABELS), "options": {"overflow_policy": "error"}}},
        )
        self.assertNotIn("authorization", calls[0].headers)
        self.assertEqual(result["native"]["classifications"], scores)
        self.assertNotIn("selected_handler", result["native"])
        self.assertEqual(result["caller"], {"selected_handler": "edit calendar events"})

    def test_selection_is_independent_of_native_list_order(self) -> None:
        scores = list(reversed(fixture_scores()))
        original = copy.deepcopy(scores)
        self.assertEqual(select_handler({"classifications": scores}, LABELS), "edit calendar events")
        self.assertEqual(scores, original)

    def test_incomplete_duplicate_and_unknown_handler_scores_are_rejected(self) -> None:
        duplicate = fixture_scores()
        duplicate[0]["label"] = duplicate[1]["label"]
        unknown = fixture_scores()
        unknown[0]["label"] = "delete every account"
        for scores in (None, [], fixture_scores()[:-1], duplicate, unknown, [None] * len(LABELS)):
            with self.subTest(scores=scores), self.assertRaises((TypeError, ValueError)):
                select_handler({"classifications": scores}, LABELS)

    def test_nonfinite_boolean_unbounded_and_tied_scores_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), True, -0.1, 1.1, "0.9"):
            scores = fixture_scores()
            scores[0]["score"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_handler({"classifications": scores}, LABELS)
        tied = fixture_scores()
        tied[0]["score"] = 0.9
        with self.assertRaises(ValueError):
            select_handler({"classifications": tied}, LABELS)

    def test_capacity_failures_do_not_send_a_second_inference_request(self) -> None:
        for status, code in (
            (503, "MODEL_LOADING"),
            (503, "RESOURCE_EXHAUSTED"),
            (503, "PROVISIONING"),
            (429, "RATE_LIMIT"),
        ):
            calls = []

            def respond(request: httpx.Request, status=status, code=code, calls=calls) -> httpx.Response:
                calls.append(request)
                return httpx.Response(
                    status,
                    headers={"Retry-After": "1"},
                    json={"detail": {"code": code, "message": "Offline retry fixture"}},
                )

            transport = httpx.Client(base_url=BASE, transport=httpx.MockTransport(respond))
            with self.subTest(code=code), SingleSendClient(BASE, api_key="", http_client=transport) as client:
                with self.assertRaises((SIEError, RetryDisabled)):
                    route(client, MODELS[0])
                self.assertEqual(len(calls), 1)

    def test_a_transport_failure_does_not_send_again(self) -> None:
        calls = []

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadError("Offline transport fixture", request=request)

        transport = httpx.Client(base_url=BASE, transport=httpx.MockTransport(respond))
        with SingleSendClient(BASE, api_key="", http_client=transport) as client, self.assertRaises(SIEError):
            route(client, MODELS[0])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
