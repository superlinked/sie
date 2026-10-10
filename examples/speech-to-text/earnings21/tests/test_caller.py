"""Offline controls: pinned metadata and tiny PCM16 fixtures, with external sockets forbidden."""

from __future__ import annotations

import base64
import copy
import gzip
import io
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import wave
import zlib
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import httpx
import msgpack

import common
import fetch
import prepare
import run
import transport
from common import DEFAULT_MODEL, OPTIONS, SOURCE, PersistenceFailure, canonical, sha256


def forbid_socket(*_args, **_kwargs):
    raise AssertionError("External socket access is forbidden in these offline controls")


socket.socket.connect = forbid_socket
socket.create_connection = forbid_socket
socket.getaddrinfo = forbid_socket


def wav_bytes(samples: int = 8) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * samples)
    return buffer.getvalue()


def fixture_frame(counts: list[int]) -> tuple[list[dict], list[dict]]:
    calls = []
    raw = wav_bytes()
    for index, count in enumerate(counts):
        cid = str(1000 + index)
        calls.append(
            {
                "id": cid,
                "mp3": {"file": f"{cid}.mp3"},
                "decoded": {"samples": count * 8},
                "reference": {"text_sha256": sha256(b"original reference")},
                "sie_pieces": [
                    {
                        "start_sample": k * 8,
                        "end_sample": (k + 1) * 8,
                        "start_s": k * 8 / 16000,
                        "end_s": (k + 1) * 8 / 16000,
                        "wav_bytes": len(raw),
                        "wav_sha256": sha256(raw),
                    }
                    for k in range(count)
                ],
            }
        )
    pieces = common.frame_from_calls(
        calls, [call["id"] for call in calls], call_count=len(counts), piece_count=sum(counts)
    )
    return calls, pieces


def terminal_payload(text: str = "fixture text") -> dict:
    return {
        "model": "server-reported/model",
        "usage": {"audio_ms": 1},
        "items": [{"data": {"text": text, "output_tokens": 8192, "finish_reason": "length", "cap_state": "capped"}}],
    }


def terminal_response(text: str = "fixture text") -> httpx.Response:
    return httpx.Response(
        200,
        content=msgpack.packb(terminal_payload(text), use_bin_type=True),
        headers={"content-type": "application/msgpack", "x-sie-model-revision": "reported-revision"},
    )


class BrokenStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"partial response"
        raise httpx.ReadError("https://secret.invalid/?key=secret-api-token")


class SlowStream(httpx.SyncByteStream):
    def __iter__(self):
        while True:
            yield b"part"
            time.sleep(0.05)


class HangingClose(httpx.MockTransport):
    def close(self) -> None:
        time.sleep(60)


def fixture_worker(piece, folder, settings, halt):
    index = piece["piece_index"]
    common.save_new(folder / "start.json", canonical({"at": time.monotonic(), "pid": os.getpid()}))
    time.sleep((5 - index % 5) * 0.005)
    common.save_new(
        folder / "terminal.json",
        canonical({"status": "ok", "text": str(index), "halt": False, "end": time.monotonic()}),
    )


def unknown_worker(piece, folder, settings, halt):
    common.save_new(folder / "intent.json", b"{}")
    if piece["piece_index"] == 0:
        halt.set()
        common.save_new(folder / "terminal.json", canonical({"status": "UNKNOWN", "text": "", "halt": True}))
    else:
        time.sleep(0.03)
        common.save_new(folder / "terminal.json", canonical({"status": "ok", "text": "kept", "halt": False}))


def sleeping_worker(piece, folder, settings, halt):
    time.sleep(60)


def partial_worker(piece, folder, settings, halt):
    run.piece_worker(
        piece, folder, settings, halt, httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowStream()))
    )


def close_worker(piece, folder, settings, halt):
    run.piece_worker(piece, folder, settings, halt, HangingClose(lambda _: terminal_response()))


def stock_fixture_worker(piece, folder, settings, halt):
    run.piece_worker(piece, folder, settings, halt, httpx.MockTransport(lambda _: terminal_response()))


def descendant_worker(piece, folder, settings, halt):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    common.save_new(folder / "descendant.json", canonical({"pid": process.pid}))
    time.sleep(60)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pieces_dir = self.root / "audio"
        self.pieces_dir.mkdir()
        self.calls, self.pieces = fixture_frame([5])
        for piece in self.pieces:
            common.piece_path(self.pieces_dir, piece).write_bytes(wav_bytes())
        self.settings = {
            "url": "https://fixture.example.test:9443/edge/base",
            "model": DEFAULT_MODEL,
            "api_key": "secret-api-token",
            "pieces_dir": str(self.pieces_dir),
            "case_seconds": 3.0,
            "whole_seconds": 10.0,
            "concurrency": 8,
        }

    def worker(self, handler, *, inner=None):
        folder = self.root / f"direct-{len(list(self.root.glob('direct-*')))}"
        folder.mkdir()
        settings = {
            **self.settings,
            "case_start_monotonic": time.monotonic(),
            "case_deadline_at": time.monotonic() + 3,
        }
        halt = multiprocessing.get_context("fork").Event()
        run.piece_worker(self.pieces[0], folder, settings, halt, inner or httpx.MockTransport(handler))
        return json.loads((folder / "terminal.json").read_bytes()), folder, halt

    def execute(self, worker=fixture_worker, **settings):
        output = self.root / "output"
        before = {child.pid for child in multiprocessing.active_children()}
        rows = run.execute(
            self.calls,
            self.pieces,
            output,
            {**self.settings, **settings},
            worker=worker,
            context=multiprocessing.get_context("fork"),
        )
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)
        return rows, output

    def test_real_pinned_frame_and_hf_url_bindings(self):
        fixtures = Path(__file__).parent / "fixtures"
        common.verify_file(fixtures / "calls.json", SOURCE["files"]["calls.json"])
        common.verify_file(fixtures / "all_ids.txt", SOURCE["files"]["all_ids.txt"])
        calls = json.loads((fixtures / "calls.json").read_bytes())
        ids = (fixtures / "all_ids.txt").read_text().splitlines()
        self.assertEqual(len(common.frame_from_calls(calls, ids)), 219)
        self.assertEqual(len(calls), 44)
        self.assertEqual(
            SOURCE["raw_base"],
            f"https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/{SOURCE['revision']}",
        )
        self.assertTrue(
            SOURCE["decoder_notes_url"].startswith(
                "https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/fba647ac"
            )
        )
        for mutation in (calls[:-1], calls[::-1], calls + calls[:1]):
            with self.assertRaises(ValueError):
                common.frame_from_calls(mutation, ids)
        bad = copy.deepcopy(calls)
        bad[-1]["sie_pieces"].pop()
        with self.assertRaises(ValueError):
            common.frame_from_calls(bad, ids)

    def test_explicit_local_setup_enforces_bytes_hash_and_reuse(self):
        source = self.root / "source"
        source.write_bytes(b"fixture")
        pin = {"bytes": 7, "sha256": sha256(b"fixture")}
        destination = self.root / "fetched"
        fetch.fetch_file("https://never-called.invalid", destination, pin, source)
        self.assertEqual(destination.read_bytes(), b"fixture")
        source.write_bytes(b"changed-longer")
        fetch.fetch_file("https://never-called.invalid", destination, pin, source)
        destination.unlink()
        with self.assertRaises(ValueError):
            fetch.fetch_file("https://never-called.invalid", destination, pin, source)
        self.assertFalse(destination.exists())
        source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            fetch.fetch_file("https://never-called.invalid", destination, pin, source)
        self.assertFalse(destination.exists())

    def test_source_reference_hash_and_manifest_checks(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        bodies = {
            "calls.json": canonical(self.calls),
            "all_ids.txt": b"1000\n",
            "references.jsonl": canonical({"id": "1000", "text": "original reference"}) + b"\n",
        }
        files = {name: {"bytes": len(raw), "sha256": sha256(raw)} for name, raw in bodies.items()}
        manifest = canonical({"files": [{"path": name, **pin} for name, pin in files.items()]})
        bodies["manifest.json"] = manifest
        files["manifest.json"] = {"bytes": len(manifest), "sha256": sha256(manifest)}
        for name, raw in bodies.items():
            (evidence / name).write_bytes(raw)
        source = {"files": files, "calls": 1, "pieces": 5}
        self.assertEqual(common.load_frame(evidence, source=source)[1], self.pieces)
        (evidence / "references.jsonl").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            common.load_frame(evidence, source=source)
        (evidence / "references.jsonl").write_bytes(bodies["references.jsonl"])
        bad = copy.deepcopy(self.calls)
        bad[0]["reference"]["text_sha256"] = "0" * 64
        changed = canonical(bad)
        (evidence / "calls.json").write_bytes(changed)
        files["calls.json"] = {"bytes": len(changed), "sha256": sha256(changed)}
        manifest = canonical(
            {"files": [{"path": name, **pin} for name, pin in files.items() if name != "manifest.json"]}
        )
        (evidence / "manifest.json").write_bytes(manifest)
        files["manifest.json"] = {"bytes": len(manifest), "sha256": sha256(manifest)}
        with self.assertRaises(ValueError):
            common.load_frame(evidence, source=source)

    def test_paths_sizes_hashes_and_post_qualification_mutation(self):
        common.qualify_pieces(self.pieces_dir, self.pieces)
        path = common.piece_path(self.pieces_dir, self.pieces[-1])
        original = path.read_bytes()
        for changed in (original[:-1], original[:-2] + b"xx"):
            path.write_bytes(changed)
            with self.assertRaises(ValueError):
                common.qualify_pieces(self.pieces_dir, self.pieces)
        path.unlink()
        with self.assertRaises(ValueError):
            common.qualify_pieces(self.pieces_dir, self.pieces)
        path.write_bytes(original)
        self.pieces[0]["wav_sha256"] = sha256(b"changed")
        row, _, halt = self.worker(lambda _: self.fail("Mutated piece reached egress"))
        self.assertEqual(row["status"], "known_unattempted")
        self.assertTrue(halt.is_set())

    def test_actual_stock_sdk_request_model_url_and_result(self):
        self.assertEqual(version("sie-sdk"), "0.10.0")
        seen = []
        self.settings["model"] = "another/model"

        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), "https://fixture.example.test:9443/edge/base/v1/extract/another/model")
            self.assertEqual(
                msgpack.unpackb(request.read(), raw=False),
                {
                    "items": [{"audio": {"data": wav_bytes(), "format": "wav", "sample_rate": None}}],
                    "params": {"options": OPTIONS},
                },
            )
            self.assertEqual(request.headers["authorization"], "Bearer secret-api-token")
            return terminal_response()

        row, folder, halt = self.worker(handler)
        self.assertEqual(len(seen), 1)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["reported_revision"], "reported-revision")
        self.assertEqual(row["reported_model"], "server-reported/model")
        self.assertEqual(row["cap_state"], "capped")
        self.assertFalse(halt.is_set())
        self.assertEqual(json.loads((folder / "sdk-result.json").read_bytes())["data"]["text"], "fixture text")

    def test_absent_api_key_does_not_read_ambient_sdk_key(self):
        self.settings["api_key"] = ""
        with patch.dict(os.environ, {"SIE_API_KEY": "unintended-ambient-key"}):
            row, _, _ = self.worker(
                lambda request: (
                    terminal_response() if "authorization" not in request.headers else self.fail("Ambient auth leaked")
                )
            )
        self.assertEqual(row["status"], "ok")

    def test_complete_failure_fences_every_retry_and_preserves_first_body(self):
        fixtures = [
            (503, "MODEL_LOADING"),
            (503, "RESOURCE_EXHAUSTED"),
            (503, "QUEUE_FULL"),
            (429, "RATE_LIMIT"),
            (504, "TIMEOUT"),
            (400, "UNSUPPORTED_MODEL"),
            (302, "REDIRECT"),
        ]
        for status, code in fixtures:
            with self.subTest(status=status, code=code):
                seen = []

                def handler(request, seen=seen, status=status, code=code):
                    seen.append(request)
                    return httpx.Response(
                        status,
                        json={"detail": {"code": code, "message": "fixture failure"}},
                        headers={"Retry-After": "0"},
                    )

                with patch("sie_sdk.client.sync.time.sleep", return_value=None):
                    row, folder, halt = self.worker(handler)
                self.assertEqual(len(seen), 1)
                self.assertEqual(row["status"], "failed")
                self.assertFalse(halt.is_set())
                evidence = json.loads((folder / "wire-000.json").read_bytes())
                self.assertTrue(evidence["complete"])
                self.assertEqual(evidence["status"], status)
                self.assertEqual(evidence["payload"]["detail"]["code"], code)
                if code in {"MODEL_LOADING", "QUEUE_FULL", "RATE_LIMIT"}:
                    self.assertEqual(row["reason"], "ValueError")  # A second SDK POST reached the fence only.

    def test_complete_empty_and_malformed_results_are_known(self):
        for response, expected in [
            (terminal_response(""), "ok"),
            (httpx.Response(200, content=b"invalid MessagePack"), "failed"),
            (
                httpx.Response(
                    200, content=msgpack.packb({"items": [{"error": {"code": "FAILED", "message": "known"}}]})
                ),
                "failed",
            ),
        ]:
            with self.subTest(expected=expected):
                row, _, halt = self.worker(lambda _, response=response: response)
                self.assertEqual(row["status"], expected)
                self.assertEqual(row["text"], "")
                self.assertFalse(halt.is_set())

    def test_same_origin_continuation_get_succeeds_with_one_post(self):
        seen = []

        def handler(request):
            seen.append(request.method)
            if request.method == "POST":
                return httpx.Response(303, headers={"Location": "/result?__modal_attempt_token=opaque"})
            self.assertEqual(request.url.path, "/edge/base/result")
            return terminal_response()

        row, folder, halt = self.worker(handler)
        self.assertEqual(seen, ["POST", "GET"])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["physical_posts"], 1)
        self.assertFalse(halt.is_set())
        self.assertNotIn("opaque", b"".join(path.read_bytes() for path in folder.iterdir()).decode())

    def test_pending_partial_or_lost_continuation_is_unknown(self):
        for failure in ("cross-origin", "lost", "partial"):
            with self.subTest(failure=failure):
                seen = []

                def handler(request, seen=seen, failure=failure):
                    seen.append(request.method)
                    if request.method == "POST":
                        location = (
                            "https://elsewhere.invalid/result?__modal_attempt_token=opaque"
                            if failure == "cross-origin"
                            else "/result?__modal_attempt_token=opaque"
                        )
                        return httpx.Response(303, headers={"Location": location})
                    if failure == "lost":
                        raise httpx.ReadError("secret-api-token")
                    return httpx.Response(200, stream=BrokenStream())

                row, folder, halt = self.worker(handler)
                self.assertEqual(row["status"], "UNKNOWN")
                self.assertTrue(halt.is_set())
                self.assertEqual(seen.count("POST"), 1)
                (folder / "terminal.json").unlink()
                self.assertEqual(transport.recover(folder)["status"], "UNKNOWN")

    def test_partial_200_is_unknown_and_preserves_partial_bytes(self):
        row, folder, halt = self.worker(lambda _: httpx.Response(200, stream=BrokenStream()))
        self.assertEqual(row["status"], "UNKNOWN")
        self.assertTrue(halt.is_set())
        evidence = json.loads((folder / "wire-000.json").read_bytes())
        self.assertFalse(evidence["complete"])
        self.assertEqual(base64.b64decode(evidence["body_base64"]), b"partial response")

    def test_compression_is_bounded_and_not_decoded_twice(self):
        raw = msgpack.packb(terminal_payload(), use_bin_type=True)
        for encoding, encoded in [
            ("gzip", gzip.compress(raw)),
            ("deflate", zlib.compress(raw)),
            ("deflate", zlib.compress(raw)[2:-4]),
        ]:
            with self.subTest(encoding=encoding):
                response = httpx.Response(
                    200,
                    stream=httpx.ByteStream(encoded),
                    headers={
                        "Content-Encoding": encoding,
                        "Content-Type": "application/msgpack",
                        "Content-Length": str(len(encoded)),
                    },
                )
                row, folder, _ = self.worker(lambda _, response=response: response)
                self.assertEqual(row["status"], "ok")
                record = json.loads((folder / "wire-000.json").read_bytes())
                self.assertEqual(record["encoded_sha256"], sha256(encoded))
                self.assertEqual(record["decoded_sha256"], sha256(raw))
        with patch.object(transport, "MAX_RESPONSE_BYTES", 128):
            response = httpx.Response(
                200, stream=httpx.ByteStream(gzip.compress(b"x" * 1024)), headers={"Content-Encoding": "gzip"}
            )
            row, _, halt = self.worker(lambda _: response)
            self.assertEqual(row["status"], "UNKNOWN")
            self.assertTrue(halt.is_set())

    def test_intent_and_fsync_failures_send_nothing(self):
        original_fsync = common.os.fsync
        for target in ("transport.save_new", "common.os.fsync"):
            with self.subTest(target=target):
                effects = [OSError("fixture failure")]

                def fail_once(*args, effects=effects):
                    if effects:
                        raise effects.pop()
                    return original_fsync(*args)

                effect = fail_once if target == "common.os.fsync" else OSError("fixture failure")
                with patch(target, side_effect=effect):
                    row, folder, halt = self.worker(lambda _: self.fail("Undurable intent reached egress"))
                self.assertEqual(row["physical_posts"], 0)
                self.assertTrue(halt.is_set())
        folder = self.root / "reused-intent"
        folder.mkdir()
        (folder / "intent.json").write_bytes(b"{}")
        settings = {**self.settings, "case_start_monotonic": time.monotonic(), "case_deadline_at": time.monotonic() + 3}
        halt = multiprocessing.get_context("fork").Event()
        run.piece_worker(
            self.pieces[0], folder, settings, halt, httpx.MockTransport(lambda _: self.fail("Reused intent sent"))
        )
        self.assertEqual(json.loads((folder / "terminal.json").read_bytes())["physical_posts"], 0)

    def test_response_and_terminal_persistence_failures_halt(self):
        original = common.save_new

        def fail_response(path, raw):
            if path.name == "wire-000.json":
                raise PersistenceFailure("injected")
            original(path, raw)

        with patch("transport.save_new", side_effect=fail_response):
            row, _, halt = self.worker(lambda _: terminal_response())
        self.assertEqual(row["status"], "UNKNOWN")
        self.assertTrue(halt.is_set())

        def fail_terminal(path, raw):
            if path.name == "terminal.json":
                raise PersistenceFailure("injected")
            original(path, raw)

        with patch("run.save_new", side_effect=fail_terminal):
            with self.assertRaises(PersistenceFailure):
                self.worker(lambda _: terminal_response())
        folder = sorted(self.root.glob("direct-*"))[-1]
        self.assertEqual(transport.recover(folder)["status"], "ok")
        self.assertTrue(transport.recover(folder)["halt"])

    def test_security_url_validation_and_known_secret_redaction(self):
        for url in (
            "https://user:pass@host/path",
            "https://host/path?key=secret-api-token",
            "https://host/path#fragment",
            "file:///tmp/x",
            "https://host:bad",
        ):
            with self.assertRaises(ValueError):
                common.normalize_url(url)
        self.assertEqual(common.normalize_url("http://localhost:8081/a/b/"), "http://localhost:8081/a/b")
        response = httpx.Response(
            400,
            json={"error": {"message": "secret-api-token"}},
            headers={
                "Set-Cookie": "secret-api-token",
                "Authorization": "secret-api-token",
                "X-SIE-Request-ID": "secret-api-token",
            },
        )
        row, folder, _ = self.worker(lambda _: response)
        records = b"".join(path.read_bytes() for path in folder.iterdir())
        self.assertNotIn(b"secret-api-token", records)
        record = json.loads((folder / "wire-000.json").read_bytes())
        self.assertTrue(record["known_secret_redacted"])
        self.assertNotEqual(record["stored_sha256"], record["decoded_sha256"])
        self.assertEqual(row["status"], "failed")

    def test_direct_fence_rejects_unrelated_requests(self):
        folder = self.root / "fence"
        folder.mkdir()
        halt = multiprocessing.get_context("fork").Event()
        settings = {**self.settings, "case_deadline_at": time.monotonic() + 3}
        fence = transport.OnePost(
            self.pieces[0], folder, settings, halt, httpx.MockTransport(lambda _: self.fail("Unexpected egress"))
        )
        for request in [
            httpx.Request("GET", self.settings["url"]),
            httpx.Request("DELETE", self.settings["url"]),
            httpx.Request("POST", "https://other.invalid/v1/extract/x"),
            httpx.Request("POST", self.settings["url"] + "/wrong", content=b"bad"),
        ]:
            with self.assertRaises(ValueError):
                fence.handle_request(request)

    def test_shared_halt_prevents_new_physical_handoff(self):
        folder = self.root / "halted-fence"
        folder.mkdir()
        halt = multiprocessing.get_context("fork").Event()
        halt.set()
        settings = {**self.settings, "case_start_monotonic": time.monotonic(), "case_deadline_at": time.monotonic() + 3}
        run.piece_worker(
            self.pieces[0], folder, settings, halt, httpx.MockTransport(lambda _: self.fail("Halted egress sent"))
        )
        row = json.loads((folder / "terminal.json").read_bytes())
        self.assertEqual(row["status"], "known_unattempted")
        self.assertEqual(row["physical_posts"], 0)
        self.assertFalse((folder / "intent.json").exists())

    def test_aggregation_keeps_partial_success_and_order(self):
        rows = [
            {**piece, "status": status, "text": text}
            for piece, status, text in zip(
                self.pieces,
                ["ok", "failed", "ok", "UNKNOWN", "known_unattempted"],
                ["first", "bad", "third", "bad", "bad"],
                strict=True,
            )
        ]
        result = run.call_rows(self.calls, rows[::-1])[0]
        self.assertEqual(result["hyp"], "first  third  ")
        self.assertFalse(result["complete"])
        self.assertIsNotNone(result["error"])

    def test_all_denominators_unknown_stop_and_concurrency_bound(self):
        self.calls, self.pieces = fixture_frame([4] + [5] * 43)
        rows, output = self.execute(unknown_worker)
        self.assertEqual(len(rows), 219)
        self.assertEqual(len(json.loads((output / "checkpoint.json").read_bytes())["calls"]), 44)
        attempted = [row for row in rows if row["status"] != "known_unattempted"]
        self.assertLessEqual(len(attempted), 8)
        self.assertEqual(rows[0]["status"], "UNKNOWN")
        self.assertEqual(len((output / "hypotheses.jsonl").read_text().splitlines()), 44)

    def test_out_of_order_closed_loop_and_reuse_rejection(self):
        self.calls, self.pieces = fixture_frame([13])
        rows, output = self.execute()
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertEqual(run.call_rows(self.calls, rows)[0]["hyp"], " ".join(str(index) for index in range(13)))
        events = []
        for piece, row in zip(self.pieces, rows, strict=True):
            start = json.loads((output / "pieces" / piece["id"] / "start.json").read_bytes())["at"]
            events.extend([(start, 1), (row["end"], -1)])
        active, peak = 0, 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        self.assertLessEqual(peak, 8)
        self.assertEqual(active, 0)
        self.assertTrue(json.loads((output / "finished.json").read_bytes())["owned_children_reaped"])
        with self.assertRaises(FileExistsError):
            run.execute(self.calls, self.pieces, output, self.settings, worker=fixture_worker)

    def test_aggregate_failure_stops_replacements_and_reaps(self):
        self.calls, self.pieces = fixture_frame([20])
        original = run.checkpoint
        calls = 0

        def fail_after_initial(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise PersistenceFailure("injected")
            original(*args)

        before = {child.pid for child in multiprocessing.active_children()}
        with patch("run.checkpoint", side_effect=fail_after_initial):
            with self.assertRaises(PersistenceFailure):
                self.execute()
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)
        self.assertLessEqual(len(list((self.root / "output/pieces").glob("*/start.json"))), 8)

    def test_startup_progressive_stream_and_hanging_close_deadlines(self):
        for worker, expected in [
            (sleeping_worker, "known_unattempted"),
            (partial_worker, "UNKNOWN"),
            (close_worker, "ok"),
        ]:
            with self.subTest(worker=worker.__name__):
                self.root = Path(self.temporary.name) / worker.__name__
                self.root.mkdir()
                started = time.monotonic()
                rows, _ = self.execute(worker, concurrency=1, case_seconds=0.2, whole_seconds=1.0)
                self.assertLess(time.monotonic() - started, 0.9)
                self.assertEqual(rows[0]["status"], expected)
                self.assertTrue(all(row["status"] == "known_unattempted" for row in rows[1:]))

    def test_whole_deadline_and_process_group_cleanup(self):
        if os.name != "posix":
            self.skipTest("POSIX process-group control")
        started = time.monotonic()
        rows, output = self.execute(descendant_worker, concurrency=1, case_seconds=60, whole_seconds=0.5)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(rows[0]["status"], "known_unattempted")
        pid = json.loads((output / "pieces" / self.pieces[0]["id"] / "descendant.json").read_bytes())["pid"]
        path = Path(f"/proc/{pid}/stat")
        if path.exists():
            self.assertEqual(path.read_text().split()[2], "Z")  # Exited; the OS owns orphan reaping.
        else:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_finite_settings_before_output_creation(self):
        for key, value in [
            ("case_seconds", float("nan")),
            ("case_seconds", 0),
            ("whole_seconds", float("inf")),
            ("concurrency", 9),
            ("concurrency", -1),
            ("url", "https://key@host"),
            ("model", "bad?model"),
        ]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    run.execute(self.calls, self.pieces, self.root / "never-created", {**self.settings, key: value})
                self.assertFalse((self.root / "never-created").exists())

    def test_default_spawn_with_the_actual_sdk(self):
        calls, pieces = fixture_frame([1])
        rows = run.execute(
            calls,
            pieces,
            self.root / "spawn-output",
            self.settings,
            worker=stock_fixture_worker,
            context=multiprocessing.get_context("spawn"),
        )
        self.assertEqual(rows[0]["status"], "ok")
        self.assertTrue(rows[0]["closure_verified"])
        self.assertGreater(rows[0]["wall_s"], rows[0]["worker_wall_s"])

    def test_mutation_between_worker_read_and_physical_handoff(self):
        original = run.read_piece

        def mutate_after_read(directory, piece):
            audio = original(directory, piece)
            common.piece_path(directory, piece).write_bytes(audio[:-2] + b"xx")
            return audio

        with patch("run.read_piece", side_effect=mutate_after_read):
            row, folder, halt = self.worker(lambda _: self.fail("Mutated bytes reached transport"))
        self.assertEqual(row["physical_posts"], 0)
        self.assertFalse((folder / "intent.json").exists())
        self.assertTrue(halt.is_set())

    def test_prepare_qualifies_all_mp3s_and_suppresses_obsolete_hint(self):
        fixtures = Path(__file__).parent / "fixtures"
        calls = json.loads((fixtures / "calls.json").read_bytes())
        evidence = self.root / "evidence"
        with patch("prepare.load_frame", return_value=(calls, [])), patch("prepare.verify_file") as checked:
            with patch("prepare.subprocess.run") as invoked:
                invoked.side_effect = [
                    subprocess.CompletedProcess([], 0, "fixture ffmpeg version\n", ""),
                    subprocess.CompletedProcess(
                        [], 1, "", "decoded PCM sha256 mismatch (decoder differs; use the published pieces)"
                    ),
                ]
                with self.assertRaisesRegex(ValueError, "No verified derived-WAV archive") as error:
                    prepare.prepare(evidence, self.root / "mp3", self.root / "derived")
                self.assertNotIn("use the published pieces", str(error.exception))
                self.assertEqual(len(checked.call_args_list), 45)
                args = invoked.call_args.args[0]
                self.assertEqual(args[0], sys.executable)
                self.assertEqual(Path(args[1]).name, "make_pieces.py")
                self.assertNotIn("--ids", args)


if __name__ == "__main__":
    unittest.main()
