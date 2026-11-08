import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sie_sdk import RequestError, SIEConnectionError

import latency
import run
from official import NotFound

ITEMS = [
    {"id": "a-1", "domain": "banking", "text": "first message", "labels": ["x", "y"], "gold": "x"},
    {"id": "a-2", "domain": "banking", "text": "second message", "labels": ["y", "z"], "gold": "z"},
    {"id": "b-3", "domain": "travel", "text": "third message", "labels": ["p", "q"], "gold": "p"},
]
ITEMS_RAW = "".join(json.dumps(item) + "\n" for item in ITEMS).encode()
QUESTIONS_RAW = json.dumps(
    {
        "questions": [
            {"id": "q1", "question": "first question", "gold": "g"},
            {"id": "q2", "question": "second question"},
        ]
    }
).encode()


def published_doc(page: str) -> dict:
    classes = {}
    for key, gpu, p50 in (("l4", "L4", 49.09), ("rtx", "RTX PRO 6000", 13.04), ("h100", "H100", 18.6)):
        classes[key] = {
            "gpu": gpu,
            "bundle": "transformers5" if page == "classify" else "sglang-embedding",
            "image_digest": "sha256:" + "a" * 64,
            "server_overlay_revision": "2b3f7d2ed6d79d880c249d9ef9efb72ae3a36135",
            "client_placement": "same container, sidecar local-ingest Unix socket",
            "levels": {
                "c1": {"latency": {"n": 400, "p50_ms": p50, "p95_ms": p50 + 2, "mean_ms": p50 + 0.2}},
                "c8": {"latency": {"n": 400, "p50_ms": 99.0, "p95_ms": 120.0, "mean_ms": 100.0}},
            },
        }
    return {"page": page, "classes": classes, "best_class": {"gpu": "RTX PRO 6000", "key": "rtx"}}


class FakeFetch:
    def __init__(self, *, jobs_sha256: dict | None = None, missing: str | None = None):
        classify_sha = hashlib.sha256(latency.jobs_bytes(latency.classify_jobs(ITEMS_RAW))).hexdigest()
        search_sha = hashlib.sha256(latency.jobs_bytes(latency.search_jobs(QUESTIONS_RAW))).hexdigest()
        sha = jobs_sha256 or {"classify.json": {"sha256": classify_sha}, "search.json": {"sha256": search_sha}}
        study = latency.STUDY_REVISION
        self.files = {
            latency.dataset_url(study, latency.JOBS_SHA256): json.dumps(sha).encode(),
            latency.dataset_url(
                latency.PAGES["classify"].input_revision, latency.PAGES["classify"].input_path
            ): ITEMS_RAW,
            latency.dataset_url(
                latency.PAGES["search"].input_revision, latency.PAGES["search"].input_path
            ): QUESTIONS_RAW,
            latency.dataset_url(study, f"{latency.STUDY}/results/classify.json"): json.dumps(
                published_doc("classify")
            ).encode(),
            latency.dataset_url(study, f"{latency.STUDY}/results/search.json"): json.dumps(
                published_doc("search")
            ).encode(),
        }
        self.missing = missing
        self.urls: list[str] = []

    def __call__(self, url: str, accept: str | None = None) -> bytes:
        self.urls.append(url)
        if self.missing and self.missing in url:
            raise NotFound(url)
        return self.files[url]


class FakeClient:
    def __init__(self, *, fail_ids: frozenset[str] = frozenset(), not_ready: int = 0, retried_ids=frozenset()):
        self.calls: list[tuple[str, str, dict, dict]] = []
        self.fail_ids = fail_ids
        self.not_ready = not_ready
        self.retried_ids = retried_ids
        self.last_retry_count = 0

    def _call(self, op: str, model: str, item: dict, kwargs: dict) -> dict:
        self.calls.append((op, model, item, kwargs))
        if self.not_ready:
            self.not_ready -= 1
            raise SIEConnectionError("connection refused")
        self.last_retry_count = 1 if item["id"] in self.retried_ids and not kwargs["wait_for_capacity"] else 0
        if item["id"] in self.fail_ids and not kwargs["wait_for_capacity"]:
            raise RequestError("bad request", status_code=400)
        return {}

    def extract(self, model, item, **kwargs):
        return self._call("extract", model, item, kwargs)

    def encode(self, model, item, **kwargs):
        return self._call("encode", model, item, kwargs)


def ticking_clock(step_ms: float = 10.0):
    now = [0]

    def clock() -> int:
        now[0] += int(step_ms * latency.NS_PER_MS)
        return now[0]

    return clock


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class JobsTests(unittest.TestCase):
    def test_classify_jobs_are_written_as_the_build_script_writes_them(self):
        doc = latency.classify_jobs(ITEMS_RAW)
        self.assertEqual(list(doc), ["page", "model", "profile", "op", "defaults", "source", "jobs"])
        self.assertEqual(doc["defaults"], {"options": {"overflow_policy": "error"}})
        self.assertEqual(doc["source"]["items_sha256"], hashlib.sha256(ITEMS_RAW).hexdigest())
        self.assertEqual(
            doc["jobs"][0], {"id": "a-1", "text": "first message", "labels": ["x", "y"], "meta": {"domain": "banking"}}
        )
        self.assertTrue(latency.jobs_bytes(doc).startswith(b'{"page": "classify", "model": "fastino/GLiNER2.5-Decide"'))

    def test_search_jobs_are_written_as_the_build_script_writes_them(self):
        doc = latency.search_jobs(QUESTIONS_RAW)
        self.assertEqual(list(doc), ["page", "model", "profile", "op", "defaults", "warm_variant", "source", "jobs"])
        self.assertEqual(doc["defaults"], {"output_types": ["dense"], "is_query": True})
        self.assertEqual(doc["jobs"], [{"id": "q1", "text": "first question"}, {"id": "q2", "text": "second question"}])

    def test_matching_jobs_load_with_their_digest(self):
        doc, digest = latency.load_jobs(latency.PAGES["classify"], FakeFetch())
        self.assertEqual(len(doc["jobs"]), 3)
        self.assertEqual(digest, hashlib.sha256(latency.jobs_bytes(doc)).hexdigest())

    def test_a_digest_mismatch_stops_the_run(self):
        fetch = FakeFetch(jobs_sha256={"classify.json": {"sha256": "0" * 64}})
        with self.assertRaisesRegex(latency.InputMismatch, "0000"):
            latency.load_jobs(latency.PAGES["classify"], fetch)

    def test_inputs_come_from_their_pinned_revisions(self):
        fetch = FakeFetch()
        latency.load_jobs(latency.PAGES["search"], fetch)
        self.assertEqual(
            fetch.urls,
            [
                (
                    "https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/"
                    "2f74a5c09c3cefa0f5f754f92949600fcd4dc1aa/latency-fastest-config/2026-10-11/prereg/jobs-sha256.json"
                ),
                (
                    "https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/"
                    "b939e41046eef6a0c2a9ee60c152161fd3fb7e04/lookalike-search/inputs/questions.json"
                ),
            ],
        )


class StatisticsTests(unittest.TestCase):
    def test_percentiles_are_linear_interpolated(self):
        self.assertEqual(latency.percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertAlmostEqual(latency.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)
        self.assertEqual(latency.percentile([7.0], 0.95), 7.0)

    def test_stats_of_nothing_are_null(self):
        self.assertEqual(latency.latency_stats([]), {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None})

    def test_published_c1_values_are_read_per_class(self):
        published = latency.load_published(latency.PAGES["classify"], FakeFetch())
        self.assertEqual(list(published["classes"]), ["l4", "rtx", "h100"])
        self.assertEqual(published["classes"]["rtx"]["p50_ms"], 13.04)
        self.assertEqual(published["best_class"], "rtx")
        self.assertEqual(published["classes"]["l4"]["bundle"], "transformers5")


class WarmupTests(unittest.TestCase):
    def test_warmups_come_from_the_tail_of_the_list(self):
        warm = latency.warmup_jobs(latency.classify_jobs(ITEMS_RAW))
        self.assertEqual(len(warm), 20)
        self.assertEqual([job["id"] for job in warm[:4]], ["b-3", "a-2", "a-1", "b-3"])
        self.assertEqual(warm[0]["text"], "third message")

    def test_search_warmups_are_tagged(self):
        warm = latency.warmup_jobs(latency.search_jobs(QUESTIONS_RAW))
        self.assertEqual(warm[0]["text"], "[c1-10000000] second question")
        self.assertEqual(warm[19]["text"], "[c1-10000019] first question")

    def test_the_first_warmup_waits_for_the_server(self):
        client, clock = FakeClient(not_ready=3), Clock()
        doc = latency.classify_jobs(ITEMS_RAW)
        latency.wait_until_ready(client, doc, doc["jobs"][0], monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(clock.sleeps, [latency.READY_POLL_S] * 3)
        self.assertTrue(all(call[3]["wait_for_capacity"] for call in client.calls))

    def test_waiting_gives_up_at_the_deadline(self):
        client, clock = FakeClient(not_ready=10_000), Clock()
        doc = latency.classify_jobs(ITEMS_RAW)
        with self.assertRaises(SIEConnectionError):
            latency.wait_until_ready(client, doc, doc["jobs"][0], monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertGreaterEqual(clock.now, latency.READY_TIMEOUT_S)


class RunTests(unittest.TestCase):
    def run_page(self, page: str, client: FakeClient, fetch: FakeFetch | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            record, path = latency.run_latency(
                page,
                client,
                fetch or FakeFetch(),
                endpoint="http://localhost:8080",
                client_description="test client",
                results_dir=Path(tmp),
                clock=ticking_clock(),
            )
            self.assertEqual(json.loads(path.read_text()), record)
            self.assertTrue(path.name.startswith(f"latency-{page}-"))
        return record

    def test_classify_sends_one_extract_per_message_after_the_warmups(self):
        client = FakeClient()
        record = self.run_page("classify", client)
        self.assertEqual(len(client.calls), 20 + 3)
        op, model, item, kwargs = client.calls[20]
        self.assertEqual(
            (op, model, item), ("extract", "fastino/GLiNER2.5-Decide", {"id": "a-1", "text": "first message"})
        )
        self.assertEqual(
            kwargs,
            {
                "labels": ["x", "y"],
                "options": {"overflow_policy": "error"},
                "wait_for_capacity": False,
                "provision_timeout_s": latency.REQUEST_TIMEOUT_S,
                "max_oom_retries": 0,
            },
        )
        self.assertEqual([call[2]["id"] for call in client.calls[20:]], ["a-1", "a-2", "b-3"])
        self.assertTrue(client.calls[0][3]["wait_for_capacity"])
        self.assertFalse(any(call[3]["wait_for_capacity"] for call in client.calls[1:]))
        self.assertEqual(record["kind"], "published-benchmark-reproduction")
        self.assertEqual(record["evaluation_mode"], "latency")
        self.assertEqual(record["run"]["request_count"], 3)
        self.assertEqual(record["run"]["failure_count"], 0)
        self.assertEqual({m["name"]: m["value"] for m in record["metrics"]}["latency_p50_ms"], 10.0)
        self.assertEqual(record["inputs"]["jobs"], 3)
        self.assertEqual(record["published"]["classes"]["h100"]["p50_ms"], 18.6)

    def test_search_sends_one_query_encode_per_question(self):
        client = FakeClient()
        self.run_page("search", client)
        op, model, item, kwargs = client.calls[20]
        self.assertEqual(
            (op, model, item), ("encode", "Qwen/Qwen3-Embedding-4B", {"id": "q1", "text": "first question"})
        )
        self.assertEqual(kwargs["output_types"], ["dense"])
        self.assertIs(kwargs["is_query"], True)
        self.assertNotIn("instruction", kwargs)

    def test_failed_and_retried_requests_are_counted_and_left_out(self):
        client = FakeClient(fail_ids=frozenset({"a-2"}), retried_ids=frozenset({"b-3"}))
        record = self.run_page("classify", client)
        self.assertEqual([row["status"] for row in record["requests"]], ["ok", "error:RequestError", "retried"])
        self.assertEqual(record["run"]["failure_count"], 2)
        report = "\n".join(latency.format_report(record))
        self.assertIn("failed: 2 of 3 (error:RequestError, retried)", report)
        self.assertRegex(report, r"yours\s+1\s+10\.00")

    def test_the_report_sets_the_run_beside_every_published_class(self):
        record = self.run_page("classify", FakeClient())
        lines = latency.format_report(record)
        self.assertRegex(lines[2], r"^yours\s+3\s+10\.00\s+10\.00\s+10\.00$")
        self.assertRegex(lines[3], r"^published L4\s+400\s+49\.09\s+51\.09\s+49\.29$")
        self.assertRegex(lines[4], r"^published RTX PRO 6000\s+400\s+13\.04")
        self.assertRegex(lines[5], r"^published H100\s+400\s+18\.60")
        self.assertIn("the website shows the published RTX PRO 6000 p50, rounded up", lines)
        differs = [line for line in lines if line.startswith("differs: ")]
        self.assertEqual(len(differs), 2)
        self.assertIn("Unix socket, with no HTTP", differs[0])
        self.assertIn("transformers5 image sha256:", differs[1])
        self.assertIn("public main 2b3f7d2e", differs[1])


class CommandTests(unittest.TestCase):
    def command(self, argv: list[str], fetch: FakeFetch, client: FakeClient):
        parser = run.build_parser()
        args = parser.parse_args(argv)
        out, err = io.StringIO(), io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(run, "RESULTS_DIR", Path(tmp)),
            patch.object(run, "SIEClient", return_value=contextlib.nullcontext(client)),
            patch.dict("os.environ", {}, clear=True),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            try:
                run.latency_command(parser, args, fetch)
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_the_endpoint_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.build_parser().parse_args(["latency", "--page", "classify"])

    def test_only_the_two_pages_are_accepted(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.build_parser().parse_args(["latency", "--page", "rerank", "--endpoint", "http://localhost:8080"])

    def test_a_run_prints_the_result_path_and_the_table(self):
        argv = ["latency", "--page", "classify", "--endpoint", "http://LOCALHOST:8080/"]
        code, out, _ = self.command(argv, FakeFetch(), FakeClient())
        self.assertEqual(code, 0)
        self.assertIn("latency-classify-", out.splitlines()[0])
        self.assertIn("published RTX PRO 6000", out)

    def test_a_mismatch_stops_before_any_request(self):
        client = FakeClient()
        fetch = FakeFetch(jobs_sha256={"classify.json": {"sha256": "0" * 64}})
        code, out, err = self.command(
            ["latency", "--page", "classify", "--endpoint", "http://localhost:8080"], fetch, client
        )
        self.assertEqual(code, 1)
        self.assertIn("stopped: the rebuilt classify jobs have sha256", err)
        self.assertEqual(client.calls, [])
        self.assertEqual(out, "")

    def test_a_missing_download_stops_the_run(self):
        fetch = FakeFetch(missing="questions.json")
        code, _, err = self.command(
            ["latency", "--page", "search", "--endpoint", "http://localhost:8080"], fetch, FakeClient()
        )
        self.assertEqual(code, 1)
        self.assertIn("stopped:", err)

    def test_a_run_where_every_request_fails_exits_1(self):
        client = FakeClient(fail_ids=frozenset({"a-1", "a-2", "b-3"}))
        code, out, _ = self.command(
            ["latency", "--page", "classify", "--endpoint", "http://localhost:8080"], FakeFetch(), client
        )
        self.assertEqual(code, 1)
        self.assertIn("failed: 3 of 3", out)


if __name__ == "__main__":
    unittest.main()
