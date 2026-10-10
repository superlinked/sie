from __future__ import annotations

import contextlib
import copy
import gzip
import io
import json
import multiprocessing
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

import httpx

import fetch
import run


def source_cases() -> list[dict]:
    request = {
        "messages": [{"role": "system", "content": "Choose a route."}, {"role": "user", "content": "A test record."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"route": {"type": "string", "enum": ["K1", "K2"]}},
                    "required": ["route"],
                    "additionalProperties": False,
                },
            },
        },
    }
    return [
        {
            "case_id": f"case-{i}",
            "episode_id": f"episode-{i // 4}",
            "family": f"family-{i // 8}",
            "control": i % 4 == 0,
            "request": copy.deepcopy(request),
            "request_sha256": run.sha256(run.canonical(request)),
        }
        for i in range(32)
    ]


def settings(**updates) -> dict:
    return {
        "url": "https://example.test/base",
        "model": "public/model",
        "timeout_s": 10,
        "max_completion_tokens": 32768,
        "thinking": "on",
        **updates,
    }


def reply(content='{"route":"K2"}', finish="stop") -> dict:
    return {
        "model": "public/model",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 256,
            "total_tokens": 306,
            "completion_tokens_details": {"reasoning_tokens": 240},
        },
    }


def stalled_worker(case: dict, folder: Path, chosen: dict) -> None:
    run.save_new(folder / "sent.json", b"{}")
    time.sleep(30)


def empty_worker(case: dict, folder: Path, chosen: dict) -> None:
    return


class SourceAndFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cases = source_cases()
        self.inputs = self.root / "inputs.json"
        self.set_source(self.cases)

    def tearDown(self) -> None:
        self.patch.stop()
        self.temporary.cleanup()

    def set_source(self, cases: list[dict]) -> None:
        raw = run.canonical(cases)
        self.inputs.write_bytes(raw)
        if hasattr(self, "patch"):
            self.patch.stop()
        self.patch = mock.patch.dict(
            run.SOURCE["files"], {"inputs.json": {"bytes": len(raw), "sha256": run.sha256(raw)}}
        )
        self.patch.start()

    def frame(self, execute, **overrides) -> list[dict]:
        with contextlib.redirect_stdout(io.StringIO()):
            return run.run_frame(
                self.inputs,
                self.root / "out",
                url="https://example.test/base",
                model="public/model",
                execute=execute,
                **overrides,
            )

    def test_modified_source_rejected_before_dispatch_or_output_creation(self) -> None:
        self.inputs.write_bytes(self.inputs.read_bytes() + b" ")
        execute = mock.Mock()
        with self.assertRaises(ValueError):
            self.frame(execute)
        execute.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_duplicate_id_rejected_even_when_file_pin_matches(self) -> None:
        self.cases[1]["case_id"] = self.cases[0]["case_id"]
        self.set_source(self.cases)
        with self.assertRaises(ValueError):
            self.frame(mock.Mock())

    def test_changed_request_rejected_even_when_file_pin_matches(self) -> None:
        self.cases[1]["request"]["messages"][1]["content"] = "Changed wire"
        self.set_source(self.cases)
        with self.assertRaises(ValueError):
            self.frame(mock.Mock())

    def test_shortened_frame_rejected(self) -> None:
        self.set_source(self.cases[:31])
        with self.assertRaises(ValueError):
            self.frame(mock.Mock())

    def test_existing_output_is_preserved_without_dispatch(self) -> None:
        out = self.root / "out"
        out.mkdir()
        (out / "previous").write_text("preserve")
        execute = mock.Mock()
        with self.assertRaises(FileExistsError):
            self.frame(execute)
        execute.assert_not_called()
        self.assertEqual((out / "previous").read_text(), "preserve")

    def test_wrong_but_valid_first_case_does_not_select_by_gold(self) -> None:
        execute = mock.Mock(
            return_value={
                "status": "ok",
                "finish_reason": "stop",
                "output": {"route": "K2"},
                "physical_sends": 1,
                "halt": False,
            }
        )
        rows = self.frame(execute)
        self.assertEqual(execute.call_count, 32)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["output"] == {"route": "K2"} for row in rows))
        self.assertEqual([row["case_id"] for row in rows], [case["case_id"] for case in self.cases])
        self.assertEqual(len(list((self.root / "out/cases").glob("*/intent.json"))), 32)

    def test_first_structural_failure_retains_all_32_and_untouched_tail(self) -> None:
        execute = mock.Mock(return_value={"status": "failed", "physical_sends": 1, "halt": False})
        rows = self.frame(execute)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertTrue(all(row["status"] == "known_unattempted" for row in rows[1:]))
        self.assertEqual(len(json.loads((self.root / "out/responses.json").read_bytes())), 32)

    def test_later_complete_failure_continues_then_success(self) -> None:
        calls = []

        def execute(case, folder, chosen):
            calls.append(case["case_id"])
            return {
                "status": "failed" if len(calls) == 2 else "ok",
                "physical_sends": 1,
                "halt": False,
                "finish_reason": "stop",
                "output": None if len(calls) == 2 else {"route": "K1"},
            }

        rows = self.frame(execute)
        self.assertEqual(len(calls), 32)
        self.assertEqual([row["status"] for row in rows[:3]], ["ok", "failed", "ok"])

    def test_unknown_stops_with_full_unattempted_tail(self) -> None:
        execute = mock.Mock(
            side_effect=[
                {"status": "ok", "halt": False},
                {"status": "attempt_status_unknown", "physical_sends": 1, "halt": True},
            ]
        )
        rows = self.frame(execute)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(rows[1]["status"], "attempt_status_unknown")
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["status"] == "known_unattempted" for row in rows[2:]))

    def test_persistence_failure_dispatches_no_later_case(self) -> None:
        original = run.save_json
        execute = mock.Mock(return_value={"status": "ok", "physical_sends": 1, "halt": False})
        calls = 0

        def save(path, value):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("test persistence failure")
            original(path, value)

        with mock.patch.object(run, "save_json", side_effect=save), self.assertRaises(OSError):
            self.frame(execute)
        self.assertEqual(execute.call_count, 1)

    def test_invalid_urls_and_timeout_rejected_before_dispatch(self) -> None:
        for url in [
            "file:///tmp/data",
            "https://user:password@example.test",
            "https://example.test/?key=x",
            "https://example.test/#fragment",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                run.run_frame(self.inputs, self.root / "out", url=url, model="m", execute=mock.Mock())
        for seconds in [float("nan"), float("inf"), 0, -1]:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.frame(mock.Mock(), timeout_s=seconds)


class TransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.case = source_cases()[0]
        self.chosen = settings()
        self.expected = run.request_body(self.case, self.chosen)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def worker(self, handler, **chosen) -> dict:
        run.case_worker(self.case, self.folder, settings(**chosen), httpx.MockTransport(handler))
        return json.loads((self.folder / "terminal.json").read_bytes())

    def test_locked_sdk_preserves_full_source_base_path_settings_and_usage(self) -> None:
        requests = []

        def handler(request):
            requests.append(request)
            self.assertEqual(request.url.path, "/base/v1/chat/completions")
            self.assertEqual(json.loads(request.read()), self.expected)
            self.assertEqual(request.headers["accept-encoding"], "gzip, identity")
            self.assertTrue((self.folder / "sent.json").exists())
            return httpx.Response(
                200, stream=httpx.ByteStream(run.canonical(reply())), headers={"content-type": "application/json"}
            )

        terminal = self.worker(handler)
        self.assertEqual(len(requests), 1)
        self.assertEqual(terminal["status"], "ok")
        self.assertEqual(terminal["usage"]["completion_tokens_details"]["reasoning_tokens"], 240)
        self.assertEqual(json.loads((self.folder / "sdk.json").read_bytes())["usage"], reply()["usage"])
        self.assertEqual(terminal["wire"][0]["method"], "POST")
        self.assertEqual(json.loads((self.folder / terminal["wire"][0]["file"]).read_bytes()), reply())

    def test_capacity_rejection_never_retries(self) -> None:
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(429, json={"error": {"message": "test capacity rejection"}})

        terminal = self.worker(handler)
        self.assertEqual(len(requests), 1)
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertEqual(terminal["status"], "failed")
        self.assertTrue(terminal["halt"])

    def test_compressed_reply_is_decoded_once_and_preserves_evidence(self) -> None:
        body = run.canonical(reply())

        def handler(request):
            encoded = gzip.compress(body)
            return httpx.Response(
                200,
                stream=httpx.ByteStream(encoded),
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                    "content-length": str(len(encoded)),
                },
            )

        terminal = self.worker(handler)
        self.assertEqual(terminal["status"], "ok")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertEqual((self.folder / terminal["wire"][0]["file"]).read_bytes(), body)
        self.assertEqual(terminal["wire"][0]["body_representation"], "decoded")
        self.assertEqual(terminal["wire"][0]["content_encoding"], "gzip")
        self.assertEqual(json.loads((self.folder / "sdk.json").read_bytes()), reply())

    def test_gzip_expansion_is_bounded_before_allocation_and_stops(self) -> None:
        bound = 4096
        encoded = gzip.compress(b"x" * (128 * 1024))
        produced = []
        limits = []
        original = zlib.decompressobj

        class ObservedDecoder:
            def __init__(self, *args, **kwargs):
                self.decoder = original(*args, **kwargs)

            def decompress(self, data, max_length=0):
                limits.append(max_length)
                output = self.decoder.decompress(data, max_length)
                produced.append(len(output))
                return output

            def __getattr__(self, name):
                return getattr(self.decoder, name)

        self.assertLess(len(encoded), bound)
        with (
            mock.patch.object(run, "MAX_RESPONSE_BYTES", bound),
            mock.patch.object(zlib, "decompressobj", ObservedDecoder),
        ):
            terminal = self.worker(
                lambda request: httpx.Response(
                    200, stream=httpx.ByteStream(encoded), headers={"content-encoding": "gzip"}
                )
            )
        self.assertEqual(terminal["status"], "attempt_status_unknown")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertTrue(terminal["halt"])
        self.assertTrue(limits)
        self.assertTrue(all(0 < limit <= bound + 1 for limit in limits))
        self.assertLessEqual(max(produced), bound + 1)
        self.assertEqual(terminal["wire"], [])

    def test_truncated_gzip_reply_stays_unresolved(self) -> None:
        encoded = gzip.compress(run.canonical(reply()))[:-8]
        terminal = self.worker(
            lambda request: httpx.Response(200, stream=httpx.ByteStream(encoded), headers={"content-encoding": "gzip"})
        )
        self.assertEqual(terminal["status"], "attempt_status_unknown")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertTrue(terminal["halt"])

    def test_metadata_get_cannot_qualify_unknown_post_and_second_post_is_refused(self) -> None:
        requests = []

        def handler(request):
            requests.append(request.method)
            if request.method == "POST":
                raise httpx.ReadTimeout("test unresolved send", request=request)
            return httpx.Response(200, json={"metadata": "not a model reply"})

        transport = run.OnePost(self.folder, self.chosen["url"], self.expected, httpx.MockTransport(handler))
        transport.handle_request(httpx.Request("GET", "https://example.test/base/v1/models"))
        post = httpx.Request("POST", "https://example.test/base/v1/chat/completions", json=self.expected)
        with self.assertRaises(httpx.ReadTimeout):
            transport.handle_request(post)
        self.assertIsNone(transport.post_status)
        self.assertEqual(transport.sends, 1)
        with self.assertRaises(RuntimeError):
            transport.handle_request(post)
        self.assertEqual(requests, ["GET", "POST"])
        transport.close()

    def test_sdk_completion_get_preserves_original_post_and_complete_reply(self) -> None:
        methods = []

        def handler(request):
            methods.append(request.method)
            if request.method == "POST":
                return httpx.Response(303, headers={"Location": "/base/result?__modal_attempt_token=test"})
            return httpx.Response(200, json=reply())

        terminal = self.worker(handler)
        self.assertEqual(methods, ["POST", "GET"])
        self.assertEqual(terminal["post_status"], 303)
        self.assertEqual(terminal["status"], "ok")
        self.assertEqual([row["method"] for row in terminal["wire"]], ["POST", "GET"])

    def test_completion_get_error_keeps_consumed_unknown(self) -> None:
        def handler(request):
            if request.method == "POST":
                return httpx.Response(303, headers={"Location": "/base/result?__modal_attempt_token=test"})
            raise httpx.ReadTimeout("test continuation lost", request=request)

        terminal = self.worker(handler)
        self.assertEqual(terminal["status"], "attempt_status_unknown")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertTrue(terminal["halt"])

    def test_complete_model_output_failures_are_known_without_retries(self) -> None:
        for content, finish in [
            ("", "stop"),
            ('{"route":"K1"}', "length"),
            ('{"route":"other"}', "stop"),
            ('{"route":"K1","route":"K2"}', "stop"),
            ('{"other":"K1"}', "stop"),
        ]:
            with self.subTest(content=content, finish=finish), tempfile.TemporaryDirectory() as directory:
                run.case_worker(
                    self.case,
                    Path(directory),
                    self.chosen,
                    httpx.MockTransport(
                        lambda request, content=content, finish=finish: httpx.Response(200, json=reply(content, finish))
                    ),
                )
                terminal = json.loads((Path(directory) / "terminal.json").read_bytes())
                self.assertEqual(terminal["status"], "failed")
                self.assertFalse(terminal["halt"])
                self.assertEqual(terminal["finish_reason"], finish)
                self.assertTrue((Path(directory) / "sdk.json").exists())

    def test_oversized_response_is_unresolved_and_stops(self) -> None:
        with mock.patch.object(run, "MAX_RESPONSE_BYTES", 8):
            terminal = self.worker(lambda request: httpx.Response(200, content=b"x" * 9))
        self.assertEqual(terminal["status"], "attempt_status_unknown")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertTrue(terminal["halt"])

    def test_modified_actual_post_is_rejected_before_network(self) -> None:
        inner = httpx.MockTransport(mock.Mock())
        transport = run.OnePost(self.folder, self.chosen["url"], self.expected, inner)
        with self.assertRaises(ValueError):
            transport.handle_request(httpx.Request("POST", "https://example.test/base/v1/chat/completions", json={}))
        self.assertEqual(transport.sends, 0)
        self.assertFalse((self.folder / "sent.json").exists())
        transport.close()

    def test_server_template_setting_omits_caller_thinking_override(self) -> None:
        captured = []

        def handler(request):
            captured.append(json.loads(request.read()))
            return httpx.Response(200, json=reply())

        terminal = self.worker(handler, thinking="server")
        self.assertEqual(terminal["status"], "ok")
        self.assertNotIn("chat_template_kwargs", captured[0])

    def test_wall_guard_reaps_stalled_consumed_child(self) -> None:
        context = multiprocessing.get_context("fork")
        before = {process.pid for process in multiprocessing.active_children()}
        terminal = run.run_case(self.case, self.folder, settings(timeout_s=0.2), worker=stalled_worker, context=context)
        self.assertEqual(terminal["status"], "attempt_status_unknown")
        self.assertEqual(terminal["physical_sends"], 1)
        self.assertTrue(terminal["halt"])
        self.assertLess(terminal["case_wall_s"], 5)
        self.assertEqual({process.pid for process in multiprocessing.active_children()}, before)

    def test_spawned_never_sent_child_is_known_unattempted(self) -> None:
        terminal = run.run_case(self.case, self.folder, self.chosen, worker=empty_worker)
        self.assertEqual(terminal["status"], "known_unattempted")
        self.assertEqual(terminal["physical_sends"], 0)

    def test_process_startup_consumes_the_case_deadline(self) -> None:
        process = mock.Mock(pid=123)
        process.is_alive.return_value = False
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(run.time, "monotonic", side_effect=[100.0, 100.6, 100.6]):
            terminal = run.run_case(self.case, self.folder, settings(timeout_s=1), context=context)
        self.assertAlmostEqual(process.join.call_args_list[0].args[0], 0.4)
        self.assertEqual(process.join.call_args_list[-1].args, ())
        process.start.assert_called_once()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        self.assertEqual(terminal["status"], "known_unattempted")


class FetchTests(unittest.TestCase):
    def test_existing_valid_pin_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.txt").write_bytes(b"source")
            with (
                mock.patch.dict(fetch.SOURCE, {"files": {"source.txt": {"bytes": 6, "sha256": run.sha256(b"source")}}}),
                mock.patch("urllib.request.urlopen") as request,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                fetch.fetch(root)
            request.assert_not_called()

    def test_existing_corrupt_pin_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.txt").write_bytes(b"corrupt")
            with (
                mock.patch.dict(fetch.SOURCE, {"files": {"source.txt": {"bytes": 6, "sha256": run.sha256(b"source")}}}),
                mock.patch("urllib.request.urlopen") as request,
                self.assertRaises(ValueError),
            ):
                fetch.fetch(root)
            request.assert_not_called()
            self.assertEqual((root / "source.txt").read_bytes(), b"corrupt")

    def test_traversal_and_escaping_symlink_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            (root / "outside").symlink_to(Path(directory))
            for name in ["../source.txt", "/source.txt", "outside/source.txt"]:
                with (
                    self.subTest(name=name),
                    mock.patch.dict(fetch.SOURCE, {"files": {name: {"bytes": 0, "sha256": "x"}}}),
                    mock.patch("urllib.request.urlopen") as request,
                    self.assertRaises(ValueError),
                ):
                    fetch.fetch(root)
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
