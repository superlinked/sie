import contextlib
import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
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
CASES_RAW = json.dumps(
    {
        "cases": [
            {"id": "c1", "query": "first query", "candidates": [{"text": "d1"}, {"text": "d2"}]},
            {"id": "c2", "query": "second query", "candidates": [{"text": "d3"}]},
        ]
    }
).encode()
RERANK_FROZEN_RAW = json.dumps(
    {"cases_test_sha256": hashlib.sha256(CASES_RAW).hexdigest(), "rules": {"w2": {"proposed": "Judge it."}}}
).encode()
CHAT_SYSTEM_RAW = b"Be helpful."
CHAT_RAW = json.dumps(
    {
        "conversations": [
            {
                "slug": "s1",
                "turns": [
                    {"index": 1, "text": "hi", "reply": "hello", "tokensIn": 5, "tokensOut": 2},
                    {"index": 2, "text": "help", "reply": "sure"},
                ],
            }
        ]
    }
).encode()


def jpeg(color: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (color, color, color)).save(buffer, "JPEG")
    return buffer.getvalue()


PHOTOS = {"p1.jpg": jpeg(10), "p2.jpg": jpeg(200)}
MANIFEST_RAW = json.dumps(
    {
        "records": [
            {"id": "r1", "domain": "receipt", "input_image": {"path": "x/p1.jpg", "size": [8, 8]}},
            {"id": "r2", "domain": "sign", "input_image": {"path": "x/p2.jpg", "size": [8, 8]}},
        ]
    }
).encode()


def png(color: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (color, 0, 0)).save(buffer, "PNG")
    return buffer.getvalue()


DETECT_PHOTOS = {"images/amvd/a.jpg": jpeg(30), "images/neon/t.png": png(60)}
DETECT_SUBSET_RAW = json.dumps(
    {
        "photos": [
            {"dataset": "amvd", "image_id": "a", "bucket": "b1", "hero": True},
            {"dataset": "neon", "image_id": "t", "bucket": "b2", "hero": False},
        ]
    }
).encode()
DETECT_LISTING_RAW = json.dumps(
    [
        {
            "dataset": ds,
            "image_id": iid,
            "path": path,
            "sha256": hashlib.sha256(DETECT_PHOTOS[path]).hexdigest(),
            "size": [8, 8],
        }
        for ds, iid, path in (("amvd", "a", "images/amvd/a.jpg"), ("neon", "t", "images/neon/t.png"))
    ]
).encode()
DETECT_MANIFEST_RAW = json.dumps(
    {
        "rows": [
            {"dataset": "amvd", "image_id": "a", "labels": ["car"]},
            {"dataset": "neon", "image_id": "t", "labels": ["tree"]},
        ]
    }
).encode()


def study_doc(page: str) -> dict:
    classes = {}
    for key, gpu, p50 in (("l4", "L4", 49.09), ("rtx", "RTX PRO 6000", 13.04), ("h100", "H100", 18.6)):
        classes[key] = {
            "gpu": gpu,
            "bundle": "transformers5",
            "image_digest": "sha256:" + "a" * 64,
            "server_overlay_revision": "2b3f7d2ed6d79d880c249d9ef9efb72ae3a36135",
            "client_placement": "same container, sidecar local-ingest Unix socket",
            "levels": {
                "c1": {"latency": {"n": 400, "p50_ms": p50, "p95_ms": p50 + 2, "mean_ms": p50 + 0.2}},
                "c8": {"latency": {"n": 400, "p50_ms": 99.0, "p95_ms": 120.0, "mean_ms": 100.0}},
            },
        }
    best = {"ocr": "h100"}.get(page, "rtx")
    return {"page": page, "classes": classes, "best_class": {"key": best}}


def url(ref: latency.Ref) -> str:
    return latency.dataset_url(ref.revision, ref.path, ref.dataset)


INPUTS = {
    "classify": (ITEMS_RAW,),
    "search": (QUESTIONS_RAW,),
    "rerank": (CASES_RAW, RERANK_FROZEN_RAW),
    "chat": (CHAT_SYSTEM_RAW, CHAT_RAW),
    "ocr": (MANIFEST_RAW,),
    "detect": (DETECT_SUBSET_RAW, DETECT_LISTING_RAW, DETECT_MANIFEST_RAW),
}


class FakeFetch:
    """The pages' inputs, hashes and published files for the pages tested end to end."""

    def __init__(self, *, jobs_sha256: dict | None = None, missing: str | None = None):
        self.files: dict[str, bytes] = {}
        sha = {}
        with patch.object(latency, "OCR_MANIFEST_SHA256", hashlib.sha256(MANIFEST_RAW).hexdigest()):
            for name, raws in INPUTS.items():
                page = latency.PAGES[name]
                for ref, raw in zip(page.inputs, raws, strict=True):
                    self.files[url(ref)] = raw
                sha[page.jobs_key] = {"sha256": hashlib.sha256(latency.jobs_bytes(page.build(*raws))).hexdigest()}
        self.files[url(latency.Ref(latency.STUDY_REVISION, latency.JOBS_SHA256))] = json.dumps(
            jobs_sha256 or sha
        ).encode()
        for name in ("classify", "search", "ocr", "detect"):
            ref = latency.Ref(latency.STUDY_REVISION, f"{latency.STUDY}/results/{name}.json")
            self.files[url(ref)] = json.dumps(study_doc(name)).encode()
        chat = study_doc("chat")
        for entry in chat["classes"].values():
            entry["levels"] = {"chat-c1": entry["levels"]["c1"]}
        del chat["classes"]["l4"]
        chat["best_class"] = {"key": "h100"}
        self.files[url(latency.Ref(latency.STUDY_REVISION, f"{latency.STUDY}/results/chat.json"))] = json.dumps(
            chat
        ).encode()
        self.files[url(latency.CHAT_SITE)] = json.dumps(
            {"complete_reply_s": {"n": 330, "p50": 0.73709, "p95": 1.36999, "mean": 0.80285}}
        ).encode()
        self.files[url(latency.RERANK_SITE)] = json.dumps(rerank_summary()).encode()
        photos = latency.PAGES["ocr"].photos
        for name, data in PHOTOS.items():
            self.files[url(latency.Ref(photos.revision, f"{photos.path}/{name}"))] = data
        detect = latency.PAGES["detect"]
        for path, data in DETECT_PHOTOS.items():
            name = detect.photo_name(f"detect/{path}")
            self.files[url(latency.Ref(detect.photos.revision, f"{detect.photos.path}/{name}"))] = data
        self.missing = missing
        self.urls: list[str] = []

    def __call__(self, address: str, accept: str | None = None) -> bytes:
        self.urls.append(address)
        if self.missing and self.missing in address:
            raise NotFound(address)
        return self.files[address]


def rerank_summary() -> dict:
    def container(p50: float, requests: int = 499) -> dict:
        level = {"p50_ms": p50, "p95_ms": p50 + 10, "mean_ms": p50 + 1, "runs": [{"requests": requests}] * 3}
        return {
            "arms": {"graphs": {"levels": {"rerank-c1": level}}},
            "server_image": "ghcr.io/superlinked/sie-server@sha256:" + "b" * 64,
            "revisions": [["base", "a" * 40], ["pr", "5bfa9ea48a172ea6397194efe703ab8977f95811"]],
        }

    return {"containers": {"h100-4b": container(54.22), "rtx-4b": container(88.22), "l4-4b": container(479.69, 60)}}


class FakeClient:
    def __init__(self, *, fail_ids: frozenset[str] = frozenset(), not_ready: int = 0, retried_ids=frozenset()):
        self.calls: list[tuple] = []
        self.fail_ids = fail_ids
        self.not_ready = not_ready
        self.retried_ids = retried_ids
        self.last_retry_count = 0

    def _call(self, op: str, model: str, key: str, payload, kwargs: dict) -> dict:
        self.calls.append((op, model, payload, kwargs))
        if self.not_ready:
            self.not_ready -= 1
            raise SIEConnectionError("connection refused")
        self.last_retry_count = 1 if key in self.retried_ids and not kwargs["wait_for_capacity"] else 0
        if key in self.fail_ids and not kwargs["wait_for_capacity"]:
            raise RequestError("bad request", status_code=400)
        return {}

    def extract(self, model, item, **kwargs):
        return self._call("extract", model, item["id"], item, kwargs)

    def encode(self, model, item, **kwargs):
        return self._call("encode", model, item["id"], item, kwargs)

    def score(self, model, query, items, **kwargs):
        return self._call("score", model, query["text"], (query, items), kwargs)

    def chat_completions(self, model, messages, **kwargs):
        return self._call("chat", model, messages[-1]["content"], messages, kwargs)


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

    def test_named_entities_take_each_set_with_its_labels_in_the_study_order(self):
        study = {"labels": {name: {"a": f"{name} thing"} for name in latency.NER_SETS}}
        sets = [json.dumps({"id": 1, "text": f"{name} sentence"}).encode() for name in latency.NER_SETS]
        doc = latency.named_entities_jobs(json.dumps(study).encode(), *sets)
        self.assertEqual(doc["defaults"], {"options": {"threshold": 0.5}})
        self.assertEqual(
            doc["jobs"][0], {"id": "ai/1", "text": "ai sentence", "labels": ["ai thing"], "meta": {"set": "ai"}}
        )
        self.assertEqual([job["meta"]["set"] for job in doc["jobs"]], list(latency.NER_SETS))
        reordered = {"labels": dict(reversed(list(study["labels"].items())))}
        with self.assertRaisesRegex(latency.InputMismatch, "sentence sets"):
            latency.named_entities_jobs(json.dumps(reordered).encode(), *sets)

    def test_redact_sends_the_frozen_label_list_and_checks_the_documents(self):
        documents = b'{"index": 0, "text": "Call Ann", "domain": "x"}\n{"index": 1, "text": "Hi", "domain": "y"}\n'
        frozen = {
            "documents": {"sha256": hashlib.sha256(documents).hexdigest()},
            "label_lists": {"L39": ["name", "phone"]},
            "sie": {"model": "superlinked/pii-redact"},
        }
        doc = latency.redact_jobs(documents, json.dumps(frozen).encode())
        self.assertEqual(doc["defaults"], {"labels": ["name", "phone"]})
        self.assertEqual(doc["warm_variant"], "tagged")
        self.assertEqual(doc["jobs"][1], {"id": "1", "text": "Hi", "meta": {"index": 1}})
        frozen["documents"]["sha256"] = "0" * 64
        with self.assertRaises(latency.InputMismatch):
            latency.redact_jobs(documents, json.dumps(frozen).encode())

    def test_typed_decisions_keep_the_first_test_messages_and_split_lines_on_newline_only(self):
        rows = [{"id": f"m{i}", "text": f"text {i}\u0085more"} for i in range(500)]
        gold = [{"id": f"m{i}", "split": "test" if i % 2 else "development"} for i in range(500)]
        data_raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
        gold_raw = "\n".join(json.dumps(g) for g in gold).encode()
        doc = latency.typed_decisions_jobs(data_raw, gold_raw)
        self.assertEqual(len(doc["jobs"]), latency.TYPED_DECISIONS_TIMED + latency.TYPED_DECISIONS_WARMUPS)
        self.assertEqual(doc["jobs"][0], {"id": "m1", "text": "text 1\u0085more"})
        self.assertEqual(
            doc["defaults"],
            {
                "labels": [latency.TYPED_DECISIONS_LABEL],
                "options": {"hypothesis_template": "{}", "multi_label": True},
            },
        )

    def test_rerank_jobs_carry_the_frozen_instruction_and_check_the_cases(self):
        doc = latency.rerank_jobs(CASES_RAW, RERANK_FROZEN_RAW)
        self.assertEqual(doc["defaults"], {"instruction": "Judge it."})
        self.assertEqual(doc["jobs"][0], {"id": "c1", "query": "first query", "docs": ["d1", "d2"]})
        with self.assertRaises(latency.InputMismatch):
            latency.rerank_jobs(CASES_RAW + b" ", RERANK_FROZEN_RAW)

    def test_request_routing_scores_each_request_against_the_description(self):
        specs = {"qwen3_rr4b_q1": {"body": {"instruction": "Judge.", "query": {"text": "A cheap request."}}}}
        inputs = {"queries": [{"id": "r1", "prompt": "What is 2+2?"}]}
        doc = latency.request_routing_jobs(json.dumps(inputs).encode(), json.dumps(specs).encode())
        self.assertEqual(doc["defaults"], {"instruction": "Judge.", "query": "A cheap request."})
        self.assertEqual(doc["jobs"], [{"id": "r1", "docs": ["What is 2+2?"]}])

    def test_ocr_jobs_name_each_photo_and_check_the_manifest(self):
        with self.assertRaises(latency.InputMismatch):
            latency.ocr_jobs(MANIFEST_RAW)
        with patch.object(latency, "OCR_MANIFEST_SHA256", hashlib.sha256(MANIFEST_RAW).hexdigest()):
            doc = latency.ocr_jobs(MANIFEST_RAW)
        self.assertEqual(doc["jobs"][0]["image"], "ocr-photo372/p1.jpg")
        self.assertEqual(doc["defaults"]["instruction"], "Text Recognition:")

    def test_detect_jobs_name_each_photo_with_its_labels_and_hash(self):
        doc = latency.detect_jobs(DETECT_SUBSET_RAW, DETECT_LISTING_RAW, DETECT_MANIFEST_RAW)
        self.assertEqual((doc["model"], doc["op"]), ("facebook/sam3", "extract"))
        self.assertNotIn("defaults", doc)
        self.assertEqual(doc["source"]["subset_sha256"], hashlib.sha256(DETECT_SUBSET_RAW).hexdigest())
        first = doc["jobs"][0]
        self.assertEqual(
            (first["id"], first["image"], first["labels"]), ("amvd/a", "detect/images/amvd/a.jpg", ["car"])
        )
        self.assertEqual(first["meta"]["sha256"], hashlib.sha256(DETECT_PHOTOS["images/amvd/a.jpg"]).hexdigest())
        self.assertEqual((first["meta"]["pixels"], first["meta"]["hero"]), (64, True))

    def test_detect_photos_are_stored_under_dataset_and_file_name(self):
        self.assertEqual(latency.detect_photo_name("detect/images/amvd/DJI-1.jpg.jpg"), "amvd__DJI-1.jpg.jpg")

    def test_a_photo_with_another_hash_stops_the_run(self):
        fetch = FakeFetch()
        page = latency.PAGES["detect"]
        doc = latency.detect_jobs(DETECT_SUBSET_RAW, DETECT_LISTING_RAW, DETECT_MANIFEST_RAW)
        address = url(latency.Ref(page.photos.revision, f"{page.photos.path}/amvd__a.jpg"))
        fetch.files[address] = jpeg(31)
        with self.assertRaises(latency.InputMismatch):
            latency.load_photos(page, doc, fetch)

    def test_chat_turns_carry_the_conversation_so_far_with_the_recorded_replies(self):
        doc = latency.chat_jobs(CHAT_SYSTEM_RAW, CHAT_RAW)
        self.assertEqual([job["id"] for job in doc["jobs"]], ["s1#1", "s1#2"])
        self.assertEqual(
            doc["jobs"][1]["messages"],
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "help"},
            ],
        )
        self.assertEqual(doc["defaults"], {"max_new_tokens": 400, "temperature": 0.0})

    def test_strict_schemas_require_every_property_at_every_depth(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "array", "items": {"type": "object", "properties": {"b": {}}}}},
        }
        strict = latency.strict_schema(schema)
        self.assertEqual(strict["required"], ["a"])
        self.assertIs(strict["additionalProperties"], False)
        self.assertEqual(strict["properties"]["a"]["items"]["required"], ["b"])

    def test_the_sob_draw_is_stratified_and_repeatable(self):
        records = [{"record_id": f"r{i}", "question_difficulty": i % 2, "schema_complexity": "low"} for i in range(10)]
        first = latency.sob_draw(records, 4, 7)
        self.assertEqual(first, latency.sob_draw(records, 4, 7))
        self.assertEqual(sorted(int(r[1:]) % 2 for r in first), [0, 0, 1, 1])

    def test_matching_jobs_load_with_their_digest(self):
        doc, digest = latency.load_jobs(latency.PAGES["classify"], FakeFetch())
        self.assertEqual(len(doc["jobs"]), 3)
        self.assertEqual(digest, hashlib.sha256(latency.jobs_bytes(doc)).hexdigest())

    def test_a_digest_mismatch_stops_the_run(self):
        fetch = FakeFetch(jobs_sha256={"classify.json": {"sha256": "0" * 64}})
        with self.assertRaisesRegex(latency.InputMismatch, "0000"):
            latency.load_jobs(latency.PAGES["classify"], fetch)

    def test_a_page_without_a_published_hash_checks_the_hash_pinned_here(self):
        page = dataclasses.replace(latency.PAGES["search"], jobs_sha256="0" * 64)
        with self.assertRaisesRegex(latency.InputMismatch, "0000"):
            latency.load_jobs(page, FakeFetch())
        self.assertEqual(latency.jobs_hash_source(page), {"pinned": latency.jobs_hash_source(page)["pinned"]})
        self.assertIsInstance(latency.PAGES["redact"].jobs_sha256, str)
        self.assertIsInstance(latency.PAGES["typed-decisions"].jobs_sha256, str)

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

    def test_a_file_name_with_spaces_or_brackets_is_escaped(self):
        address = latency.dataset_url("a" * 40, "x/amvd__DJI 1 (7_42).jpg")
        self.assertTrue(address.endswith("/x/amvd__DJI%201%20%287_42%29.jpg"))

    def test_every_page_pins_full_revisions(self):
        for page in latency.PAGES.values():
            refs = [*page.inputs, *([page.photos] if page.photos else [])]
            if isinstance(page.jobs_sha256, latency.Ref):
                refs.append(page.jobs_sha256)
            else:
                self.assertRegex(page.jobs_sha256, r"^[0-9a-f]{64}$")
            for ref in refs:
                self.assertRegex(ref.revision, r"^[0-9a-f]{40}$", page.name)


class PublishedTests(unittest.TestCase):
    def test_study_pages_read_their_level_per_class(self):
        published = latency.PAGES["classify"].published(FakeFetch())
        self.assertEqual(list(published["classes"]), ["l4", "rtx", "h100"])
        self.assertEqual(published["classes"]["rtx"]["p50_ms"], 13.04)
        self.assertEqual(published["best_class"], "rtx")
        self.assertIn("transformers5 image sha256:", published["server"])

    def test_chat_shows_the_website_run_beside_the_earlier_run(self):
        published = latency.PAGES["chat"].published(FakeFetch())
        self.assertEqual(published["classes"]["h100"]["p50_ms"], 737.09)
        self.assertEqual(published["classes"]["h100-2026-10-11"]["note"], "2026-10-11")
        self.assertEqual(published["path"], latency.CHAT_SITE.path)
        self.assertEqual(published["best_class"], "h100")

    def test_rerank_reads_the_fast_arm_for_each_gpu(self):
        published = latency.PAGES["rerank"].published(FakeFetch())
        self.assertEqual(published["classes"]["h100"]["p50_ms"], 54.22)
        self.assertEqual(published["classes"]["l4"]["note"], "60 questions only")
        self.assertNotIn("note", published["classes"]["rtx"])
        self.assertIn("5bfa9ea4", published["server"])

    def test_request_routing_reads_the_median_of_the_three_runs(self):
        runs = {
            f"promoted-q1-r{i}": {"latency": {"n": 200, "p50_ms": p50, "p95_ms": p50 * 3, "mean_ms": p50 + 4}}
            for i, p50 in enumerate((8.57, 8.49, 8.46), start=1)
        }
        doc = {
            "runs": {**runs, "default-q1": {"latency": {"n": 200, "p50_ms": 24.0, "p95_ms": 30.0, "mean_ms": 24.0}}},
            "promoted_median_of_three_medians_ms": 8.49,
            "image_digest": "sha256:" + "c" * 64,
            "sie_rev": "530dcde89dd0a35500a2346308d30c8e708d7688",
        }
        published = latency.request_routing_published(lambda address, accept=None: json.dumps(doc).encode())
        self.assertEqual(published["classes"], {"h100": latency.row("H100", 200, 8.49, 25.47, 12.49)})

    def test_redact_and_typed_decisions_read_their_one_rtx_figure(self):
        redact = {
            "timing": {
                "sie+L39": {
                    "latency": {"n": 660, "p50_ms": 60.65, "p95_ms": 76.37, "mean_ms": 62.71},
                    "image_digest": "sha256:" + "d" * 64,
                    "sie_rev": "2b3f7d2ed6d79d880c249d9ef9efb72ae3a36135",
                }
            }
        }
        published = latency.redact_published(lambda address, accept=None: json.dumps(redact).encode())
        self.assertEqual(published["classes"]["rtx"]["p50_ms"], 60.65)
        typed = {"tasks": {"comment-spam": {"model_ms": 13.76, "p95_ms": 15.42, "min_ms": 12.23, "max_ms": 15.25}}}
        published = latency.typed_decisions_published(lambda address, accept=None: json.dumps(typed).encode())
        self.assertEqual(
            published["classes"]["rtx"],
            latency.row("RTX PRO 6000", 200, 13.76, 15.42, None, "containers 12.23 to 15.25"),
        )


class StatisticsTests(unittest.TestCase):
    def test_percentiles_are_linear_interpolated(self):
        self.assertEqual(latency.percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertAlmostEqual(latency.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)
        self.assertEqual(latency.percentile([7.0], 0.95), 7.0)

    def test_stats_of_nothing_are_null(self):
        self.assertEqual(latency.latency_stats([]), {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None})


class WarmupTests(unittest.TestCase):
    def test_warmups_come_from_the_tail_of_the_list(self):
        doc = latency.classify_jobs(ITEMS_RAW)
        warm = latency.warmup_jobs(latency.PAGES["classify"], doc, {})
        self.assertEqual(len(warm), 20)
        self.assertEqual([job["id"] for job in warm[:4]], ["b-3", "a-2", "a-1", "b-3"])
        self.assertEqual(warm[0]["text"], "third message")
        self.assertEqual(doc["jobs"][2]["text"], "third message")

    def test_search_warmups_are_tagged(self):
        warm = latency.warmup_jobs(latency.PAGES["search"], latency.search_jobs(QUESTIONS_RAW), {})
        self.assertEqual(warm[0]["text"], "[c1-10000000] second question")
        self.assertEqual(warm[19]["text"], "[c1-10000019] first question")

    def test_generation_warmups_tag_the_first_user_message_with_the_level(self):
        doc = latency.chat_jobs(CHAT_SYSTEM_RAW, CHAT_RAW)
        warm = latency.warmup_jobs(latency.PAGES["chat"], doc, {})
        self.assertEqual(len(warm), 3)
        self.assertEqual(warm[0]["messages"][1]["content"], "[chat-c1-10000000] hi")
        self.assertEqual(doc["jobs"][1]["messages"][1]["content"], "hi")

    def test_score_warmups_are_exact_copies(self):
        doc = latency.rerank_jobs(CASES_RAW, RERANK_FROZEN_RAW)
        warm = latency.warmup_jobs(latency.PAGES["rerank"], doc, {})
        self.assertEqual(warm[:2], [doc["jobs"][1], doc["jobs"][0]])

    def test_image_warmups_change_one_pixel(self):
        with patch.object(latency, "OCR_MANIFEST_SHA256", hashlib.sha256(MANIFEST_RAW).hexdigest()):
            doc = latency.ocr_jobs(MANIFEST_RAW)
        photos = {"ocr-photo372/p1.jpg": PHOTOS["p1.jpg"], "ocr-photo372/p2.jpg": PHOTOS["p2.jpg"]}
        warm = latency.warmup_jobs(latency.PAGES["ocr"], doc, photos)
        self.assertEqual([job["id"] for job in warm], ["r2", "r1", "r2", "r1", "r2"])
        self.assertEqual(Image.open(io.BytesIO(warm[0]["image_bytes"])).size, (8, 8))

    def test_a_pixel_variant_changes_exactly_one_pixel(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "PNG")
        changed = Image.open(io.BytesIO(latency.pixel_variant(buffer.getvalue(), "png", 3)))
        different = [(x, y) for x in range(8) for y in range(8) if changed.getpixel((x, y)) != (10, 20, 30)]
        self.assertEqual(different, [((37 + 3) % 8, (53 + 9) % 8)])
        self.assertEqual(changed.getpixel(different[0]), (11, 21, 31))

    def test_the_first_warmup_waits_for_the_server(self):
        client, clock = FakeClient(not_ready=3), Clock()
        doc = latency.classify_jobs(ITEMS_RAW)

        def send_job(job, *, wait_for_capacity):
            return latency.send(client, doc, job, wait_for_capacity=wait_for_capacity)

        latency.wait_until_ready(send_job, doc["jobs"][0], monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(clock.sleeps, [latency.READY_POLL_S] * 3)
        self.assertTrue(all(call[3]["wait_for_capacity"] for call in client.calls))

    def test_waiting_gives_up_at_the_deadline(self):
        client, clock = FakeClient(not_ready=10_000), Clock()
        doc = latency.classify_jobs(ITEMS_RAW)

        def send_job(job, *, wait_for_capacity):
            return latency.send(client, doc, job, wait_for_capacity=wait_for_capacity)

        with self.assertRaises(SIEConnectionError):
            latency.wait_until_ready(send_job, doc["jobs"][0], monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertGreaterEqual(clock.now, latency.READY_TIMEOUT_S)


class SendTests(unittest.TestCase):
    def test_extract_sends_only_the_fields_the_job_sets(self):
        client = FakeClient()
        doc = latency.redact_jobs(*self.redact_inputs())
        latency.send(client, doc, doc["jobs"][0], wait_for_capacity=False)
        op, model, item, kwargs = client.calls[0]
        self.assertEqual((op, model, item), ("extract", "superlinked/pii-redact", {"id": "0", "text": "Call Ann"}))
        self.assertEqual(set(kwargs), {"labels", "wait_for_capacity", "provision_timeout_s", "max_oom_retries"})

    @staticmethod
    def redact_inputs():
        documents = b'{"index": 0, "text": "Call Ann", "domain": "x"}\n'
        frozen = {
            "documents": {"sha256": hashlib.sha256(documents).hexdigest()},
            "label_lists": {"L39": ["name"]},
            "sie": {"model": "superlinked/pii-redact"},
        }
        return documents, json.dumps(frozen).encode()

    def test_images_go_as_bytes_with_the_instruction_and_options(self):
        client = FakeClient()
        with patch.object(latency, "OCR_MANIFEST_SHA256", hashlib.sha256(MANIFEST_RAW).hexdigest()):
            doc = latency.ocr_jobs(MANIFEST_RAW)
        latency.send(
            client, doc, doc["jobs"][0], wait_for_capacity=False, photos={"ocr-photo372/p1.jpg": PHOTOS["p1.jpg"]}
        )
        _, model, item, kwargs = client.calls[0]
        self.assertEqual(model, "zai-org/GLM-OCR")
        self.assertEqual(item, {"id": "r1", "images": [{"data": PHOTOS["p1.jpg"], "format": "jpeg"}]})
        self.assertEqual(kwargs["instruction"], "Text Recognition:")
        self.assertEqual(kwargs["options"], {"max_new_tokens": 8192, "num_beams": 1})

    def test_score_calls_name_the_served_profile(self):
        client = FakeClient()
        doc = latency.rerank_jobs(CASES_RAW, RERANK_FROZEN_RAW)
        latency.send(client, doc, doc["jobs"][0], wait_for_capacity=False, model=latency.FAST)
        op, model, (query, items), kwargs = client.calls[0]
        self.assertEqual((op, model), ("score", "Qwen/Qwen3-Reranker-4B:fast"))
        self.assertEqual(query, {"text": "first query"})
        self.assertEqual(items, [{"text": "d1"}, {"text": "d2"}])
        self.assertEqual(kwargs["instruction"], "Judge it.")

    def test_structured_output_asks_for_a_strict_json_schema(self):
        client = FakeClient()
        doc = {
            "model": "m",
            "op": "generate",
            "defaults": {"max_new_tokens": 2048, "temperature": 0.0},
            "jobs": [],
        }
        job = {
            "id": "j",
            "messages": [{"role": "user", "content": "x"}],
            "grammar": {"kind": "json_schema", "value": {"type": "object"}, "label": "answer", "strict": True},
        }
        latency.send(client, doc, job, wait_for_capacity=False)
        _, _, messages, kwargs = client.calls[0]
        self.assertEqual(messages, job["messages"])
        self.assertEqual(
            kwargs["response_format"],
            {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True}},
        )
        self.assertEqual((kwargs["max_tokens"], kwargs["temperature"]), (2048, 0.0))


class RunTests(unittest.TestCase):
    def run_page(self, page: str, client: FakeClient, fetch: FakeFetch | None = None):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(latency, "OCR_MANIFEST_SHA256", hashlib.sha256(MANIFEST_RAW).hexdigest()),
        ):
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
        self.assertEqual(record["inputs"]["jobs_sha256_source"]["key"], "classify.json")
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

    def test_rerank_calls_the_fast_profile_once_per_question(self):
        client = FakeClient()
        record = self.run_page("rerank", client)
        self.assertEqual(len(client.calls), 5 + 2)
        self.assertTrue(all(call[1] == latency.FAST for call in client.calls))
        self.assertEqual(record["model"], latency.FAST)
        self.assertIn(latency.FRESH_SERVERS, record["differences"])

    def test_ocr_downloads_each_photo_and_sends_it(self):
        client, fetch = FakeClient(), FakeFetch()
        record = self.run_page("ocr", client, fetch)
        self.assertEqual([call[2]["images"][0]["data"] for call in client.calls[5:]], list(PHOTOS.values()))
        self.assertEqual(record["inputs"]["photos"]["revision"], latency.PAGES["ocr"].photos.revision)
        self.assertTrue(any(address.endswith("images/input/p1.jpg") for address in fetch.urls))

    def test_detect_sends_each_photo_in_its_format_with_its_labels(self):
        client, fetch = FakeClient(), FakeFetch()
        record = self.run_page("detect", client, fetch)
        timed = client.calls[5:]
        self.assertEqual([call[2]["images"][0]["data"] for call in timed], list(DETECT_PHOTOS.values()))
        self.assertEqual([call[2]["images"][0]["format"] for call in timed], ["jpeg", "png"])
        self.assertEqual([call[3]["labels"] for call in timed], [["car"], ["tree"]])
        self.assertNotIn("options", timed[0][3])
        self.assertEqual(client.calls[0][2]["images"][0]["format"], "png")
        self.assertNotEqual(client.calls[0][2]["images"][0]["data"], DETECT_PHOTOS["images/neon/t.png"])
        self.assertEqual(record["inputs"]["photos"]["revision"], latency.PAGES["detect"].photos.revision)
        self.assertTrue(any(address.endswith("images/neon__t.png") for address in fetch.urls))

    def test_chat_times_one_completion_per_turn(self):
        client = FakeClient()
        record = self.run_page("chat", client)
        self.assertEqual([call[0] for call in client.calls], ["chat"] * 5)
        self.assertEqual(client.calls[3][2][-1], {"role": "user", "content": "hi"})
        self.assertNotIn("response_format", client.calls[3][3])
        lines = latency.format_report(record)
        self.assertRegex(lines[3], r"^published H100\s+330\s+737\.09")
        self.assertTrue(any(line.startswith("published H100 (2026-10-11)") for line in lines[4:6]))
        self.assertTrue(any(line.startswith("published RTX PRO 6000 (2026-10-11)") for line in lines[4:6]))

    def test_only_the_first_jobs_are_timed_when_the_page_says_so(self):
        client = FakeClient()
        page = dataclasses.replace(latency.PAGES["classify"], timed=2, warmups=1)
        with patch.dict(latency.PAGES, {"classify": page}):
            record = self.run_page("classify", client)
        self.assertEqual([call[2]["id"] for call in client.calls], ["b-3", "a-1", "a-2"])
        self.assertEqual(record["inputs"]["jobs"], 2)

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
        self.assertIn(f"the website shows {latency.PAGES['classify'].website}", lines)
        self.assertIn("published figure: the p50 of one run per GPU, level c1", lines)
        differs = [line for line in lines if line.startswith("differs: ")]
        self.assertEqual(len(differs), 2)
        self.assertIn("Unix socket, with no HTTP", differs[0])
        self.assertIn("transformers5 image sha256:", differs[1])
        self.assertIn("server package from 2b3f7d2e", differs[1])

    def test_a_published_row_without_a_mean_prints_a_dash(self):
        record = self.run_page("classify", FakeClient())
        record["published"]["classes"] = {"rtx": latency.row("RTX PRO 6000", 200, 13.76, 15.42, None, "note")}
        lines = latency.format_report(record)
        self.assertRegex(lines[3], r"^published RTX PRO 6000 \(note\)\s+200\s+13\.76\s+15\.42\s+-$")


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

    def test_every_page_is_accepted_and_no_other(self):
        parser = run.build_parser()
        for name in latency.PAGES:
            self.assertEqual(parser.parse_args(["latency", "--page", name, "--endpoint", "http://x"]).page, name)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["latency", "--page", "speech-to-text", "--endpoint", "http://localhost:8080"])

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

    def test_the_command_downloads_files_larger_than_the_lookup_limit(self):
        with patch.object(run, "http_fetch", return_value=b"x") as fetch:
            run.latency_fetch("https://example.test/a")
        fetch.assert_called_once_with("https://example.test/a", None, max_bytes=latency.MAX_DOWNLOAD_BYTES)


if __name__ == "__main__":
    unittest.main()
