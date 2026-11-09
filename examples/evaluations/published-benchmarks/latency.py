"""Time one request at a time against your own server, beside the per-request times the SIE website shows."""

import contextlib
import copy
import hashlib
import io
import json
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sie_sdk import ModelLoadingError, ProvisioningError, SIEConnectionError

from official import Fetch
from score import write_json

DATASET = "superlinked/sie-task-evidence"
RESOLVE_URL = "https://huggingface.co/datasets/{dataset}/resolve"
STUDY_REVISION = "2f74a5c09c3cefa0f5f754f92949600fcd4dc1aa"
STUDY = "latency-fastest-config/2026-10-11"
JOBS_SHA256 = f"{STUDY}/prereg/jobs-sha256.json"
LEVEL = "c1"
WARMUP_REQUESTS = 20
WARMUP_TAG_BASE = 10_000_000
CLASSES = ("l4", "rtx", "h100")
REQUEST_TIMEOUT_S = 300.0
READY_TIMEOUT_S = 900.0
READY_POLL_S = 5.0
NOT_READY = (SIEConnectionError, ModelLoadingError, ProvisioningError)
NS_PER_MS = 1_000_000
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
SOCKET_DIFFERENCE = (
    "the published client ran in the server's container and sent each request over the sidecar's local "
    "Unix socket, with no HTTP; this run sends HTTP through sie_sdk.SIEClient, so its time also includes "
    "the HTTP server and your network path"
)


class InputMismatch(Exception):
    pass


@dataclass(frozen=True)
class Ref:
    revision: str
    path: str
    dataset: str = DATASET


@dataclass(frozen=True)
class Page:
    """One website time: the study inputs that rebuild its jobs file, where its hash and its result are, and how it ran.

    ``jobs_sha256`` is the published hash file (read at key ``jobs_key``), or a hash pinned here when the study
    published none. ``model`` is the id to call when it differs from the jobs file's model (a profile such as
    ``:fast``). ``timed`` keeps only the first jobs for timing when the study sent the rest as warm-ups. ``photos``
    is the folder that holds the images the jobs name.
    """

    name: str
    inputs: tuple[Ref, ...]
    build: Callable[..., dict[str, Any]]
    jobs_sha256: Ref | str
    jobs_key: str
    published: Callable[[Fetch], dict[str, Any]]
    website: str
    level: str = LEVEL
    warmups: int = WARMUP_REQUESTS
    timed: int | None = None
    model: str | None = None
    photos: Ref | None = None
    differences: tuple[str, ...] = ()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl(raw: bytes, *, split_on_newline: bool = False) -> list[dict[str, Any]]:
    text = raw.decode("utf-8")
    lines = text.split("\n") if split_on_newline else text.splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---- jobs files, each written exactly as the study's builder wrote it ------------------------------------------


def classify_jobs(raw: bytes) -> dict[str, Any]:
    items = jsonl(raw)
    jobs = [
        {"id": it["id"], "text": it["text"], "labels": list(it["labels"]), "meta": {"domain": it["domain"]}}
        for it in items
    ]
    return {
        "page": "classify",
        "model": "fastino/GLiNER2.5-Decide",
        "profile": "default",
        "op": "extract",
        "defaults": {"options": {"overflow_policy": "error"}},
        "source": {
            "items": "classify-current-arms-2026-10-11/prereg/items.jsonl @4e9f6498",
            "items_sha256": sha256(raw),
            "request": "one message per extract call with its 8 labels, options {overflow_policy: error}",
        },
        "jobs": jobs,
    }


def search_jobs(raw: bytes) -> dict[str, Any]:
    questions = json.loads(raw.decode("utf-8"))["questions"]
    return {
        "page": "search",
        "model": "Qwen/Qwen3-Embedding-4B",
        "profile": "default",
        "op": "encode",
        "defaults": {"output_types": ["dense"], "is_query": True},
        "warm_variant": "tagged",
        "source": {
            "questions": "lookalike-search/inputs/questions.json @b939e410 (440 questions)",
            "sha256": sha256(raw),
            "request": "one question per encode call, is_query true, output_types [dense], no instruction sent (server default)",
        },
        "jobs": [{"id": q["id"], "text": q["question"]} for q in questions],
    }


NER_SETS = ("ai", "literature", "music", "politics", "science", "mit_movie", "mit_restaurant")


def named_entities_jobs(study_raw: bytes, *set_raws: bytes) -> dict[str, Any]:
    study = json.loads(study_raw)
    if tuple(study["labels"]) != NER_SETS:
        raise InputMismatch(f"the study lists the sentence sets {list(study['labels'])}, not {list(NER_SETS)}")
    sentences = dict(zip(NER_SETS, set_raws, strict=True))
    jobs = []
    for set_name, labels in study["labels"].items():
        for row in jsonl(sentences[set_name]):
            jobs.append(
                {
                    "id": f"{set_name}/{row['id']}",
                    "text": row["text"],
                    "labels": list(labels.values()),
                    "meta": {"set": set_name},
                }
            )
    return {
        "page": "named-entities",
        "model": "numind/NuNER_Zero-span",
        "profile": "default",
        "op": "extract",
        "defaults": {"options": {"threshold": 0.5}},
        "source": {
            "sentences": "custom-entity-types/data/main/*.jsonl @8736554b (652 sentences)",
            "labels": "reproducibility/nuner652-current/2026-10-10/STUDY.json labels",
            "study_sha256": sha256(study_raw),
            "request": "one sentence per extract call with its set's label list, options {threshold: 0.5}",
        },
        "jobs": jobs,
    }


def redact_jobs(documents_raw: bytes, frozen_raw: bytes) -> dict[str, Any]:
    frozen = json.loads(frozen_raw)
    if sha256(documents_raw) != frozen["documents"]["sha256"]:
        raise InputMismatch("the documents do not have the SHA-256 the study froze")
    labels = frozen["label_lists"]["L39"]
    return {
        "page": "redact",
        "model": frozen["sie"]["model"],
        "profile": "default",
        "op": "extract",
        "warm_variant": "tagged",
        "source": {
            "documents": "redact/inputs/gretel-main.jsonl @31383ae9",
            "sha256": frozen["documents"]["sha256"],
            "request": f"one document per extract call; labels = the {len(labels)}-label list L39; no options",
        },
        "defaults": {"labels": labels},
        "jobs": [
            {"id": str(d["index"]), "text": d["text"], "meta": {"index": d["index"]}} for d in jsonl(documents_raw)
        ],
    }


TYPED_DECISIONS_LABEL = (
    "The comment promotes something unrelated to the video: a channel, a link, a website or a product, "
    "or asks people to subscribe, follow, vote or send money."
)
TYPED_DECISIONS_TIMED = 200
TYPED_DECISIONS_WARMUPS = 5


def typed_decisions_jobs(data_raw: bytes, gold_raw: bytes) -> dict[str, Any]:
    split = {row["id"]: row["split"] for row in jsonl(gold_raw, split_on_newline=True)}
    test = [row for row in jsonl(data_raw, split_on_newline=True) if split[row["id"]] == "test"]
    return {
        "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "profile": "default",
        "op": "extract",
        "defaults": {
            "labels": [TYPED_DECISIONS_LABEL],
            "options": {"hypothesis_template": "{}", "multi_label": True},
        },
        "jobs": [{"id": r["id"], "text": r["text"]} for r in test[: TYPED_DECISIONS_TIMED + TYPED_DECISIONS_WARMUPS]],
    }


def rerank_jobs(cases_raw: bytes, frozen_raw: bytes) -> dict[str, Any]:
    frozen = json.loads(frozen_raw)
    if frozen["cases_test_sha256"] != sha256(cases_raw):
        raise InputMismatch("the rerank cases do not have the SHA-256 the study froze")
    rule = frozen["rules"]["w2"]["proposed"]
    cases = json.loads(cases_raw)["cases"]
    return {
        "page": "rerank",
        "model": "Qwen/Qwen3-Reranker-4B",
        "profile": "default",
        "op": "score",
        "defaults": {"instruction": rule},
        "source": {
            "cases": "rerank-relevance-rules/inputs/cases_test.json @e5b19d75 (499 cases, 20 candidates each)",
            "sha256": sha256(cases_raw),
            "request": (
                "one score call per case: the query, its 20 candidate texts, instruction = wording w2, "
                "proposed-rule arm (the arm the rivals' latency pass used)"
            ),
            "instruction": rule,
        },
        "jobs": [{"id": c["id"], "query": c["query"], "docs": [x["text"] for x in c["candidates"]]} for c in cases],
    }


def request_routing_jobs(inputs_raw: bytes, specs_raw: bytes) -> dict[str, Any]:
    body = json.loads(specs_raw)["qwen3_rr4b_q1"]["body"]
    requests = json.loads(inputs_raw)["queries"]
    return {
        "page": "request-routing",
        "model": "Qwen/Qwen3-Reranker-4B",
        "profile": "default",
        "op": "score",
        "defaults": {"instruction": body["instruction"], "query": body["query"]["text"]},
        "source": {
            "requests": (
                "evaluations/2026-10-11/request-routing-cost-results-v1/direct-latency-2026-10-11/"
                "inputs.json, the 200 requests of the study's latency pass, in its order"
            ),
            "request": (
                "arm qwen3_rr4b_q1 of "
                "evaluations/2026-10-13/request-routing-decision-routers-v1: one score call, query = the "
                "description, one document = the request, with the arm's instruction"
            ),
            "description": body["query"]["text"],
            "instruction": body["instruction"],
        },
        "jobs": [{"id": r["id"], "docs": [r["prompt"]]} for r in requests],
    }


OCR_MANIFEST_SHA256 = "532ae51d590acef816b97662a65fb5ea1819a56311763fcef752bc5139f65c19"


def ocr_jobs(manifest_raw: bytes) -> dict[str, Any]:
    if sha256(manifest_raw) != OCR_MANIFEST_SHA256:
        raise InputMismatch("the photo manifest does not have the SHA-256 the study's builder checks")
    records = json.loads(manifest_raw)["records"]
    jobs = [
        {
            "id": r["id"],
            "image": f"ocr-photo372/{Path(r['input_image']['path']).name}",
            "image_format": "jpeg",
            "meta": {"domain": r["domain"], "pixels": r["input_image"]["size"][0] * r["input_image"]["size"][1]},
        }
        for r in records
    ]
    return {
        "page": "ocr",
        "model": "zai-org/GLM-OCR",
        "profile": "default",
        "op": "extract",
        "defaults": {"instruction": "Text Recognition:", "options": {"max_new_tokens": 8192, "num_beams": 1}},
        "source": {
            "photos": (
                "ocr-photo372-prereg/2026-10-11/images/input/*.jpg @d03e04fd (372 photos; sha256 checked against "
                "ocr-photo372-source-v1.json @6daad480)"
            ),
            "request": "one photo per extract call, instruction 'Text Recognition:', options {max_new_tokens: 8192, num_beams: 1}",
        },
        "jobs": jobs,
    }


def chat_jobs(system_raw: bytes, transcripts_raw: bytes) -> dict[str, Any]:
    system = system_raw.decode("utf-8")
    jobs = []
    for conversation in json.loads(transcripts_raw)["conversations"]:
        history = [{"role": "system", "content": system}]
        for turn in conversation["turns"]:
            messages = [*history, {"role": "user", "content": turn["text"]}]
            jobs.append(
                {
                    "id": f"{conversation['slug']}#{turn['index']}",
                    "messages": messages,
                    "meta": {
                        "turn": turn["index"],
                        "recorded_tokens_in": turn.get("tokensIn"),
                        "recorded_tokens_out": turn.get("tokensOut"),
                    },
                }
            )
            history = [*messages, {"role": "assistant", "content": turn["reply"]}]
    return {
        "page": "chat",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "profile": "default",
        "op": "generate",
        "defaults": {"max_new_tokens": 400, "temperature": 0.0},
        "warm_variant": "tagged",
        "source": {
            "conversations": (
                "support-assistant-policy/transcripts/Qwen_Qwen3.8-27B-FP8.json @4fbd23b7 (55 conversations, 6 turns "
                "each; customer turns and the recorded replies)"
            ),
            "system_prompt": "support-assistant-policy/inputs/system_prompt.md @4fbd23b7",
            "transcript_sha256": sha256(transcripts_raw),
            "request": (
                "one chat completion per turn: system prompt, the conversation so far with the recorded replies, "
                "the new customer turn; max 400 new tokens, temperature 0"
            ),
        },
        "jobs": jobs,
    }


# The Structured Output Benchmark's (SOB) prompt and strict-schema rule, as the study's items were built with them.
SOB_SYSTEM = (
    "Answer the question using only the provided context. "
    "Return a valid JSON object that strictly follows the given JSON schema. "
    "Do not output anything except the JSON object.\n\n"
    "Rules:\n"
    "- No explanations: Do not include reasoning, analysis, or any sentences outside the JSON.\n"
    "- No markdown: Do not wrap the JSON in code blocks or add formatting like ``` or 'json'.\n"
    "- No extra text: Do not add prefixes or suffixes such as 'Answer:' or 'Output:'.\n"
    "- Follow the schema exactly: Use only the keys defined in the schema and ensure correct data types.\n"
    "- Include all required fields: Every field listed as required in the schema must appear in the JSON.\n"
    "- If unknown, return null: If the context does not contain the answer, set the field value to null "
    "instead of guessing."
)
SOB_PARQUET_SHA256 = "b86a475557aa65028a0d2c71aac780c8258a2a8cf15cd76e1300d9cc241c81a5"
SOB_ITEMS_SHA256 = "54b006028f6091c3aff1322c4a5eb1bce881fab30623bc58a0dc0357f699c0e5"
STRUCTURED_TIMED = 100


def strict_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
            schema["required"] = list(schema["properties"].keys())
            schema["additionalProperties"] = False
            for value in schema["properties"].values():
                strict_schema(value)
        if schema.get("type") == "array" and "items" in schema:
            strict_schema(schema["items"])
    return schema


def sob_draw(records: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[str]:
    strata: dict[tuple[Any, Any], list[str]] = {}
    for record in records:
        strata.setdefault((record["question_difficulty"], record["schema_complexity"]), []).append(record["record_id"])
    total = sum(len(ids) for ids in strata.values())
    rng = random.Random(seed)
    keys = sorted(strata)
    quotas = {k: n * len(strata[k]) // total for k in keys}
    for k in sorted(keys, key=lambda k: -(n * len(strata[k]) / total - quotas[k]))[: n - sum(quotas.values())]:
        quotas[k] += 1
    picked: list[str] = []
    for k in keys:
        ids = sorted(strata[k])
        rng.shuffle(ids)
        picked += ids[: quotas[k]]
    rng.shuffle(picked)
    return picked


def structured_output_jobs(sample_raw: bytes, parquet_raw: bytes) -> dict[str, Any]:
    import pyarrow.parquet as pq  # only this page reads a parquet file

    if sha256(parquet_raw) != SOB_PARQUET_SHA256:
        raise InputMismatch("the SOB test split does not have the SHA-256 the study used")
    sample = json.loads(sample_raw)
    rows = pq.read_table(io.BytesIO(parquet_raw)).to_pylist()
    if sob_draw(rows, sample["n"], sample["seed"]) != sample["e1"]:
        raise InputMismatch("the sample is not the one its seed draws from the SOB test split")
    records = {row["record_id"]: row for row in rows}
    items = []
    for record_id in sample["e1"]:
        record = records[record_id]
        schema = json.loads(record["json_schema"]) if isinstance(record["json_schema"], str) else record["json_schema"]
        user = (
            f"Context:\n{record['context']}\n\nQuestion: {record['question']}\n\n"
            f"Respond with JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        items.append(
            {"id": record_id, "system": SOB_SYSTEM, "user": user, "schema": strict_schema(copy.deepcopy(schema))}
        )
    items_sha256 = sha256(json.dumps(items).encode())
    if items_sha256 != SOB_ITEMS_SHA256:
        raise InputMismatch(f"the SOB items have sha256 {items_sha256}, the study's items {SOB_ITEMS_SHA256}")
    jobs = [
        {
            "id": it["id"][:16],
            "messages": [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}],
            "grammar": {"kind": "json_schema", "value": it["schema"], "label": "answer", "strict": True},
        }
        for it in items[:STRUCTURED_TIMED]
    ]
    return {
        "page": "structured-output",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "profile": "default",
        "op": "generate",
        "defaults": {"max_new_tokens": 2048, "temperature": 0.0},
        "warm_variant": "tagged",
        "source": {
            "records": (
                "the first 100 of the 500 SOB records in structured-output-accuracy/inputs/sob_sample.json @0e1fb325 "
                "(texts from interfaze-ai/sob @c118e38a, built by the public example's score.sob_items)"
            ),
            "items_sha256": items_sha256,
            "request": (
                "one chat completion per record: SOB system prompt and user message, strict json_schema grammar, "
                "max 2,048 new tokens, temperature 0"
            ),
        },
        "jobs": jobs,
    }


# ---- published values ------------------------------------------------------------------------------------------


def dataset_url(revision: str, path: str, dataset: str = DATASET) -> str:
    return f"{RESOLVE_URL.format(dataset=dataset)}/{revision}/{path}"


def fetch_ref(ref: Ref, fetch: Fetch) -> bytes:
    return fetch(dataset_url(ref.revision, ref.path, ref.dataset), None)


def fetch_json(ref: Ref, fetch: Fetch) -> Any:
    return json.loads(fetch_ref(ref, fetch))


def row(gpu: str, n: int | None, p50: float, p95: float | None, mean: float | None, note: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {"gpu": gpu, "n": n, "p50_ms": p50, "p95_ms": p95, "mean_ms": mean}
    if note:
        entry["note"] = note
    return entry


def server_text(bundle: str | None, image: str, revision: str) -> str:
    image_name = f"the {bundle} image" if bundle else "the image"
    return f"{image_name} {image} with the server package from {revision[:8]}"


def study_published(page: str, level: str = LEVEL) -> Callable[[Fetch], dict[str, Any]]:
    """A page of the 2026-10-11 study: one run per GPU, read from ``results/<page>.json``."""

    def read(fetch: Fetch) -> dict[str, Any]:
        ref = Ref(STUDY_REVISION, f"{STUDY}/results/{page}.json")
        doc = fetch_json(ref, fetch)
        classes = {}
        for key in CLASSES:
            entry = doc["classes"].get(key)
            if entry is None or level not in entry["levels"]:
                continue
            latency = entry["levels"][level]["latency"]
            classes[key] = {
                **row(entry["gpu"], latency["n"], latency["p50_ms"], latency["p95_ms"], latency["mean_ms"]),
                "bundle": entry["bundle"],
                "image_digest": entry["image_digest"],
                "server_overlay_revision": entry["server_overlay_revision"],
                "client_placement": entry["client_placement"],
            }
        best = classes[doc["best_class"]["key"]]
        return {
            "dataset": ref.dataset,
            "revision": ref.revision,
            "path": ref.path,
            "level": level,
            "figure": f"the p50 of one run per GPU, level {level}",
            "best_class": doc["best_class"]["key"],
            "classes": classes,
            "server": server_text(best["bundle"], best["image_digest"], best["server_overlay_revision"]),
        }

    return read


CHAT_SITE = Ref(
    "f2152b039ae0ce92a3846ba28cbd862625b8b496",
    "chat-value-prop/2026-10-13/results/latency/sie-Qwen3.8-27B-c1-base55.json",
)


def chat_published(fetch: Fetch) -> dict[str, Any]:
    published = study_published("chat", "chat-c1")(fetch)
    site = fetch_json(CHAT_SITE, fetch)["complete_reply_s"]
    shown = row(
        "H100", site["n"], round(site["p50"] * 1000, 2), round(site["p95"] * 1000, 2), round(site["mean"] * 1000, 2)
    )
    earlier = {f"{key}-2026-10-11": {**entry, "note": "2026-10-11"} for key, entry in published["classes"].items()}
    return {
        **published,
        "dataset": CHAT_SITE.dataset,
        "revision": CHAT_SITE.revision,
        "path": CHAT_SITE.path,
        "figure": (
            "the p50 of one run on an H100, 2026-10-13, time to the complete reply; the same requests ran on "
            "2026-10-11 in the latency study, shown below it"
        ),
        "best_class": "h100",
        "classes": {"h100": shown, **earlier},
        "also": {"revision": published["revision"], "path": published["path"]},
    }


RERANK_SITE = Ref("aa6c734c11063105e2c27ade8de5e014f053a0e9", "reranker-serving/2026-10-12/results/summary.json")
RERANK_CONTAINERS = (("h100", "H100", "h100-4b"), ("rtx", "RTX PRO 6000", "rtx-4b"), ("l4", "L4", "l4-4b"))


def rerank_published(fetch: Fetch) -> dict[str, Any]:
    doc = fetch_json(RERANK_SITE, fetch)
    classes = {}
    for key, gpu, container in RERANK_CONTAINERS:
        level = doc["containers"][container]["arms"]["graphs"]["levels"]["rerank-c1"]
        n = level["runs"][0]["requests"]
        note = "" if n == 499 else f"{n} questions only"
        classes[key] = row(gpu, n, level["p50_ms"], level["p95_ms"], level["mean_ms"], note)
    h100 = doc["containers"]["h100-4b"]
    revisions = dict(h100["revisions"])
    return {
        "dataset": RERANK_SITE.dataset,
        "revision": RERANK_SITE.revision,
        "path": RERANK_SITE.path,
        "level": "rerank-c1, arm graphs (the fast profile's options)",
        "figure": "the median of the p50s of 3 runs, each on a fresh server",
        "best_class": "h100",
        "classes": classes,
        "server": f"the image {h100['server_image'].split('@')[-1]} with the server package from {revisions['pr'][:8]}",
    }


ROUTING_SITE = Ref(
    "6cad8837b01bfd3b91e9fc7de47a5a9d46c73e9d", "rerank-fast-default/2026-10-13/results/q1-h100-summary.json"
)


def request_routing_published(fetch: Fetch) -> dict[str, Any]:
    doc = fetch_json(ROUTING_SITE, fetch)
    runs = [run["latency"] for name, run in sorted(doc["runs"].items()) if name.startswith("promoted-q1-r")]
    shown = row(
        "H100",
        runs[0]["n"],
        doc["promoted_median_of_three_medians_ms"],
        statistics.median(r["p95_ms"] for r in runs),
        statistics.median(r["mean_ms"] for r in runs),
    )
    return {
        "dataset": ROUTING_SITE.dataset,
        "revision": ROUTING_SITE.revision,
        "path": ROUTING_SITE.path,
        "level": "promoted-q1 (the fast profile)",
        "figure": "the median of the p50s of 3 runs, each on a fresh server",
        "best_class": "h100",
        "classes": {"h100": shown},
        "server": server_text("default", doc["image_digest"], doc["sie_rev"]),
    }


REDACT_SITE = Ref("f025dd3cd8b916921a05c163525172fb2cf5ab08", "redact-fair-prompt/2026-10-12/results/results.json")


def redact_published(fetch: Fetch) -> dict[str, Any]:
    timing = fetch_json(REDACT_SITE, fetch)["timing"]["sie+L39"]
    latency = timing["latency"]
    return {
        "dataset": REDACT_SITE.dataset,
        "revision": REDACT_SITE.revision,
        "path": REDACT_SITE.path,
        "level": "c1-L39",
        "figure": "the p50 of one run",
        "best_class": "rtx",
        "classes": {"rtx": row("RTX PRO 6000", latency["n"], latency["p50_ms"], latency["p95_ms"], latency["mean_ms"])},
        "server": server_text("default", timing["image_digest"], timing["sie_rev"]),
    }


TYPED_DECISIONS_SITE = Ref(
    "5283bf0751c1f9fce2ef443eb4d0316c68a230d7",
    "typed-decisions-untuned-confirm/2026-10-13/addendum-speed-pin/results/results.json",
)


def typed_decisions_published(fetch: Fetch) -> dict[str, Any]:
    task = fetch_json(TYPED_DECISIONS_SITE, fetch)["tasks"]["comment-spam"]
    note = f"containers {task['min_ms']:.2f} to {task['max_ms']:.2f}"
    return {
        "dataset": TYPED_DECISIONS_SITE.dataset,
        "revision": TYPED_DECISIONS_SITE.revision,
        "path": TYPED_DECISIONS_SITE.path,
        "level": "comment-spam",
        "figure": "the median of 5 containers, each the median of 3 runs' p50s over the first 200 test comments",
        "best_class": "rtx",
        "classes": {"rtx": row("RTX PRO 6000", TYPED_DECISIONS_TIMED, task["model_ms"], task["p95_ms"], None, note)},
        "server": "the v0.10.0 image with the sidecar's local-ingest mode",
    }


# ---- pages -----------------------------------------------------------------------------------------------------

# The redact and typed-decisions studies published no hash of their jobs files. These are the SHA-256s of the
# rebuilds below, whose request ids and order match the studies' published per-request rows.
REDACT_JOBS_SHA256 = "02ac2236ab388e7785d1b165aba2a5eb1bfd5b816c3e0521f7aa5413ebc0cf29"
TYPED_DECISIONS_JOBS_SHA256 = "13064349537f8b33f65ed1f249cd291568e49dd17eeecd244bb601cf5ceb53ae"


def study_hash() -> Ref:
    return Ref(STUDY_REVISION, JOBS_SHA256)


ROUND_UP = "rounded up to 0.1 ms"
FAST = "Qwen/Qwen3-Reranker-4B:fast"
GENERATE = (
    "on a standalone sie-server, a chat completion goes to the SGLang server it runs; the published client "
    "used the sidecar's generation stream and timed it to the final frame"
)
FRESH_SERVERS = "the published figure is a median over 3 fresh servers; this is one run on the server you started"

PAGES = {
    page.name: page
    for page in (
        Page(
            "classify",
            (Ref("4e9f649842dd802d353d10fc8dc2347b9a927189", "classify-current-arms-2026-10-11/prereg/items.jsonl"),),
            classify_jobs,
            study_hash(),
            "classify.json",
            study_published("classify"),
            f"13.1 ms on /classify and its model page: the RTX PRO 6000 p50, {ROUND_UP}",
        ),
        Page(
            "search",
            (Ref("b939e41046eef6a0c2a9ee60c152161fd3fb7e04", "lookalike-search/inputs/questions.json"),),
            search_jobs,
            study_hash(),
            "search.json",
            study_published("search"),
            f"15.7 ms on the home and about pages: the RTX PRO 6000 p50, {ROUND_UP}",
        ),
        Page(
            "named-entities",
            (
                Ref(
                    "531c444b42539228ad0c231c3a0e9b917c6018ba", "reproducibility/nuner652-current/2026-10-10/STUDY.json"
                ),
                *(
                    Ref("8736554b7afd71f05a9e53daef6be72fb8fe176f", f"custom-entity-types/data/main/{name}.jsonl")
                    for name in NER_SETS
                ),
            ),
            named_entities_jobs,
            study_hash(),
            "named-entities.json",
            study_published("named-entities"),
            "24 ms on the industry pages (extract entities, NuNER Zero): the RTX PRO 6000 p50, rounded",
        ),
        Page(
            "redact",
            (
                Ref("31383ae9986d9dc9edd4e3a06ee9fb40b223d6fb", "redact/inputs/gretel-main.jsonl"),
                Ref(
                    "f025dd3cd8b916921a05c163525172fb2cf5ab08", "redact-fair-prompt/2026-10-12/results/code/frozen.json"
                ),
            ),
            redact_jobs,
            REDACT_JOBS_SHA256,
            "redact.json",
            redact_published,
            "61 ms on /redact, its model page and the industry pages: the RTX PRO 6000 p50, rounded up",
            level="c1-L39",
            warmups=10,
        ),
        Page(
            "typed-decisions",
            (
                Ref(
                    "5283bf0751c1f9fce2ef443eb4d0316c68a230d7",
                    "typed-decisions-untuned-confirm/2026-10-13/data/comment-spam.jsonl",
                ),
                Ref(
                    "5283bf0751c1f9fce2ef443eb4d0316c68a230d7",
                    "typed-decisions-untuned-confirm/2026-10-13/gold/comment-spam.jsonl",
                ),
            ),
            typed_decisions_jobs,
            TYPED_DECISIONS_JOBS_SHA256,
            "typed-decisions.json",
            typed_decisions_published,
            "13.8 ms on /typed-decisions and its model page: the RTX PRO 6000 figure, to 0.1 ms",
            warmups=TYPED_DECISIONS_WARMUPS,
            timed=TYPED_DECISIONS_TIMED,
            differences=(
                "the published figure is a median over 15 runs in 5 containers with 4 CPU cores each; this is one run",
            ),
        ),
        Page(
            "rerank",
            (
                Ref("e5b19d7589fc3a982fcfeeabb071cf833c22455f", "rerank-relevance-rules/inputs/cases_test.json"),
                Ref(
                    "aa5d3ed2074afce7bb5b2ed6f9dcd4ed8bcacd5a",
                    "current-comparators/2026-10-11/rerank/prereg/frozen.json",
                ),
            ),
            rerank_jobs,
            study_hash(),
            "rerank.json",
            rerank_published,
            "55 ms on /rerank, its model page and the industry pages: the H100 figure, rounded up",
            warmups=5,
            model=FAST,
            differences=(FRESH_SERVERS,),
        ),
        Page(
            "request-routing",
            (
                Ref(
                    "bdf9f45218b292b60512fa404595b17abc1cacd1",
                    "evaluations/2026-10-11/request-routing-cost-results-v1/direct-latency-2026-10-11/inputs.json",
                ),
                Ref(
                    "bdf9f45218b292b60512fa404595b17abc1cacd1",
                    "evaluations/2026-10-13/request-routing-decision-routers-v1/results/code/specs.json",
                ),
            ),
            request_routing_jobs,
            Ref("6cad8837b01bfd3b91e9fc7de47a5a9d46c73e9d", "rerank-fast-default/2026-10-13/jobs-sha256.json"),
            "routing-q1-latency200.json",
            request_routing_published,
            "8.5 ms on /request-routing: the H100 figure, rounded",
            warmups=5,
            model=FAST,
            differences=(FRESH_SERVERS,),
        ),
        Page(
            "ocr",
            (
                Ref(
                    "6daad480d988ec1596a51ba61462be0e013028f5",
                    "ocr-photo372-results/2026-10-11/source-manifest/ocr-photo372-source-v1.json",
                ),
            ),
            ocr_jobs,
            study_hash(),
            "ocr.json",
            study_published("ocr"),
            "287 ms on the industry pages (OCR a page, GLM-OCR): the H100 p50, rounded",
            warmups=5,
            photos=Ref("d03e04fd2f7f768afcc753b952fb9c9abc47748b", "ocr-photo372-prereg/2026-10-11/images/input"),
        ),
        Page(
            "chat",
            (
                Ref("4fbd23b7a53de46b52532722d2f37fc17feb60d5", "support-assistant-policy/inputs/system_prompt.md"),
                Ref(
                    "4fbd23b7a53de46b52532722d2f37fc17feb60d5",
                    "support-assistant-policy/transcripts/Qwen_Qwen3.8-27B-FP8.json",
                ),
            ),
            chat_jobs,
            study_hash(),
            "chat.json",
            chat_published,
            "0.74 s on /chat and its model page: the 2026-10-13 H100 p50, rounded",
            level="chat-c1",
            warmups=3,
            differences=(GENERATE, "the 2026-10-13 run also sent top_p 1.0"),
        ),
        Page(
            "structured-output",
            (
                Ref("0e1fb325cf25bd403ff678dd6031f649981eb434", "structured-output-accuracy/inputs/sob_sample.json"),
                Ref("c118e38abdef6a8e1beba183405c70b28ff7d5a8", "data/test-00000-of-00001.parquet", "interfaze-ai/sob"),
            ),
            structured_output_jobs,
            study_hash(),
            "structured-output.json",
            study_published("structured-output", "structured-c1"),
            "938 ms on the industry pages (structured output, Qwen3.8 27B): the H100 p50, rounded",
            level="structured-c1",
            warmups=2,
            differences=(GENERATE,),
        ),
    )
}


# ---- the run ---------------------------------------------------------------------------------------------------


def jobs_bytes(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(doc).encode("utf-8")


def expected_jobs_sha256(page: Page, fetch: Fetch) -> str:
    if isinstance(page.jobs_sha256, Ref):
        return fetch_json(page.jobs_sha256, fetch)[page.jobs_key]["sha256"]
    return page.jobs_sha256


def jobs_hash_source(page: Page) -> dict[str, Any]:
    if isinstance(page.jobs_sha256, Ref):
        ref = page.jobs_sha256
        return {"dataset": ref.dataset, "revision": ref.revision, "path": ref.path, "key": page.jobs_key}
    return {"pinned": "this example; the study published no hash of its jobs file"}


def load_jobs(page: Page, fetch: Fetch) -> tuple[dict[str, Any], str]:
    expected = expected_jobs_sha256(page, fetch)
    doc = page.build(*(fetch_ref(ref, fetch) for ref in page.inputs))
    digest = sha256(jobs_bytes(doc))
    if digest != expected:
        raise InputMismatch(f"the rebuilt {page.name} jobs have sha256 {digest}, the published jobs file {expected}")
    return doc, digest


def load_photos(page: Page, doc: Mapping[str, Any], fetch: Fetch) -> dict[str, bytes]:
    if page.photos is None:
        return {}
    folder = page.photos
    return {
        job["image"]: fetch_ref(Ref(folder.revision, f"{folder.path}/{Path(job['image']).name}", folder.dataset), fetch)
        for job in doc["jobs"]
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def latency_stats(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "n": len(values_ms),
        "p50_ms": round(percentile(values_ms, 0.5), 2),
        "p95_ms": round(percentile(values_ms, 0.95), 2),
        "mean_ms": round(statistics.fmean(values_ms), 2),
    }


def pixel_variant(data: bytes, fmt: str, k: int) -> bytes:
    """The same picture with one pixel changed, as the study's client made its image warm-ups."""
    from PIL import Image  # only image pages need it

    image = Image.open(io.BytesIO(data))
    image.load()
    x, y = (37 + k) % image.width, (53 + 3 * k) % image.height
    pixel = image.getpixel((x, y))
    pixel = tuple((c + 1) % 256 for c in pixel) if isinstance(pixel, tuple) else (pixel + 1) % 256
    image.putpixel((x, y), pixel)
    buffer = io.BytesIO()
    if fmt in ("jpeg", "jpg"):
        try:
            image.save(buffer, "JPEG", quality="keep", subsampling="keep")
        except Exception:  # noqa: BLE001
            image.convert("RGB").save(buffer, "JPEG", quality=95)
    else:
        image.save(buffer, fmt.upper())
    return buffer.getvalue()


def warm_variant(doc: Mapping[str, Any]) -> str:
    return doc.get("warm_variant") or ("tagged" if doc["op"] == "generate" else "exact")


def warmup_jobs(page: Page, doc: Mapping[str, Any], photos: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """The last jobs of the list, changed so that none equals a timed request where the server keeps a cache."""
    jobs = doc["jobs"]
    variant = warm_variant(doc)
    warm = []
    for i in range(page.warmups):
        k = WARMUP_TAG_BASE + i
        job = copy.deepcopy(jobs[-1 - (i % len(jobs))])
        tag = f"[{page.level}-{k}] "
        if "image" in job:
            job["image_bytes"] = pixel_variant(photos[job["image"]], job.get("image_format", "jpeg"), k)
        elif variant == "tagged":
            if "messages" in job:
                user = next(m for m in job["messages"] if m["role"] == "user")
                user["content"] = tag + user["content"]
            elif "query" in job:
                job["query"] = tag + job["query"]
            elif "text" in job:
                job["text"] = tag + job["text"]
        warm.append(job)
    return warm


def response_format(grammar: Mapping[str, Any]) -> dict[str, Any]:
    schema = {"name": grammar["label"], "schema": grammar["value"], "strict": grammar["strict"]}
    return {"type": "json_schema", "json_schema": schema}


def send(
    client: Any,
    doc: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    wait_for_capacity: bool,
    model: str | None = None,
    photos: Mapping[str, bytes] | None = None,
) -> Any:
    """One request, built as the study's client built it; fields the job does not set are not sent."""
    spec = {**(doc.get("defaults") or {}), **job}
    model = model or doc["model"]
    calls = {
        "wait_for_capacity": wait_for_capacity,
        "provision_timeout_s": READY_TIMEOUT_S if wait_for_capacity else REQUEST_TIMEOUT_S,
        "max_oom_retries": 0,
    }
    op = doc["op"]
    if op == "generate":
        if spec.get("grammar"):
            calls = {"response_format": response_format(spec["grammar"]), **calls}
        return client.chat_completions(
            model, spec["messages"], max_tokens=spec["max_new_tokens"], temperature=spec["temperature"], **calls
        )
    if op == "score":
        if spec.get("instruction"):
            calls = {"instruction": spec["instruction"], **calls}
        return client.score(model, {"text": spec["query"]}, [{"text": text} for text in spec["docs"]], **calls)
    if "image" in spec:
        data = spec.get("image_bytes") or (photos or {})[spec["image"]]
        item = {"id": spec["id"], "images": [{"data": data, "format": spec.get("image_format", "jpeg")}]}
    else:
        item = {"id": spec["id"], "text": spec["text"]}
    if op == "extract":
        fields = {
            "labels": list(spec["labels"]) if spec.get("labels") else None,
            "instruction": spec.get("instruction"),
            "options": dict(spec["options"]) if spec.get("options") else None,
        }
        return client.extract(model, item, **{k: v for k, v in fields.items() if v is not None}, **calls)
    return client.encode(model, item, output_types=list(spec["output_types"]), is_query=bool(spec["is_query"]), **calls)


def wait_until_ready(
    send_job: Callable[..., Any],
    job: Mapping[str, Any],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + READY_TIMEOUT_S
    while True:
        try:
            send_job(job, wait_for_capacity=True)
            return
        except NOT_READY:
            if monotonic() >= deadline:
                raise
            sleep(READY_POLL_S)


def warm_up(send_job: Callable[..., Any], jobs: Sequence[Mapping[str, Any]], **ready: Any) -> None:
    if not jobs:
        return
    wait_until_ready(send_job, jobs[0], **ready)
    for job in jobs[1:]:
        with contextlib.suppress(Exception):
            send_job(job, wait_for_capacity=False)


def timed_requests(
    client: Any,
    send_job: Callable[..., Any],
    jobs: Sequence[Mapping[str, Any]],
    clock: Callable[[], int] = time.perf_counter_ns,
) -> list[dict]:
    rows = []
    for job in jobs:
        start = clock()
        try:
            send_job(job, wait_for_capacity=False)
        except Exception as exc:  # noqa: BLE001
            status = f"error:{type(exc).__name__}"
        else:
            status = "retried" if getattr(client, "last_retry_count", 0) else "ok"
        end = clock()
        rows.append({"id": job["id"], "status": status, "latency_ms": round((end - start) / NS_PER_MS, 3)})
    return rows


def differences(page: Page, published: Mapping[str, Any]) -> list[str]:
    return [
        SOCKET_DIFFERENCE,
        f"this run uses your GPU and the server you started; the published run used {published['server']}",
        *page.differences,
    ]


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_record(
    page: Page,
    doc: Mapping[str, Any],
    digest: str,
    published: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    client_description: str,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    ok = [r["latency_ms"] for r in rows if r["status"] == "ok"]
    stats = latency_stats(ok)
    return {
        "kind": "published-benchmark-reproduction",
        "schema_version": 1,
        "evaluation_mode": "latency",
        "page": page.name,
        "model": page.model or doc["model"],
        "operation": doc["op"],
        "website": page.website,
        "ingress": {"url": endpoint},
        "inputs": {
            "files": [{"dataset": ref.dataset, "revision": ref.revision, "path": ref.path} for ref in page.inputs],
            "photos": (
                {"dataset": page.photos.dataset, "revision": page.photos.revision, "path": page.photos.path}
                if page.photos
                else None
            ),
            "jobs_sha256": digest,
            "jobs_sha256_source": jobs_hash_source(page),
            "jobs": len(rows),
        },
        "warmup_requests": page.warmups,
        "client": {"description": client_description, "worker_count": 1, "processes_per_worker": 1},
        "run": {
            "started_at": _timestamp(started_at),
            "finished_at": _timestamp(finished_at),
            "endpoint": endpoint,
            "request_count": len(rows),
            "failure_count": len(rows) - len(ok),
        },
        "metrics": [
            {"name": "latency_p50_ms", "unit": "ms", "aggregation": "p50", "value": stats["p50_ms"]},
            {"name": "latency_p95_ms", "unit": "ms", "aggregation": "p95", "value": stats["p95_ms"]},
            {"name": "latency_mean_ms", "unit": "ms", "aggregation": "mean", "value": stats["mean_ms"]},
        ],
        "published": dict(published),
        "differences": differences(page, published),
        "requests": list(rows),
    }


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_report(record: Mapping[str, Any]) -> list[str]:
    published = record["published"]
    metrics = {metric["name"]: metric["value"] for metric in record["metrics"]}
    n = record["run"]["request_count"] - record["run"]["failure_count"]
    rows = [("yours", n, metrics["latency_p50_ms"], metrics["latency_p95_ms"], metrics["latency_mean_ms"])]
    for entry in published["classes"].values():
        label = f"published {entry['gpu']}" + (f" ({entry['note']})" if entry.get("note") else "")
        rows.append((label, entry["n"], entry["p50_ms"], entry["p95_ms"], entry["mean_ms"]))
    width = max(len(name) for name, *_ in rows)
    lines = [
        (
            f"{record['page']}: {record['model']}, {record['operation']}, one request at a time, "
            f"{record['inputs']['jobs']} inputs after {record['warmup_requests']} warm-up requests"
        ),
        f"{'':{width}}  {'n':>5}  {'p50 ms':>8}  {'p95 ms':>8}  {'mean ms':>8}",
    ]
    lines += [
        f"{name:{width}}  {'-' if count is None else count:>5}  {_cell(p50):>8}  {_cell(p95):>8}  {_cell(mean):>8}"
        for name, count, p50, p95, mean in rows
    ]
    lines.append(f"the website shows {record['website']}")
    lines.append(f"published figure: {published['figure']}")
    lines += [f"differs: {line}" for line in record["differences"]]
    failed = record["run"]["failure_count"]
    if failed:
        statuses = sorted({r["status"] for r in record["requests"] if r["status"] != "ok"})
        lines.append(
            f"failed: {failed} of {record['run']['request_count']} ({', '.join(statuses)}), left out of the numbers"
        )
    return lines


def run_latency(
    page_name: str,
    client: Any,
    fetch: Fetch,
    *,
    endpoint: str,
    client_description: str,
    results_dir: Path,
    clock: Callable[[], int] = time.perf_counter_ns,
    **ready: Any,
) -> tuple[dict[str, Any], Path]:
    page = PAGES[page_name]
    doc, digest = load_jobs(page, fetch)
    published = page.published(fetch)
    photos = load_photos(page, doc, fetch)

    def send_job(job: Mapping[str, Any], *, wait_for_capacity: bool) -> Any:
        return send(client, doc, job, wait_for_capacity=wait_for_capacity, model=page.model, photos=photos)

    timed = doc["jobs"][: page.timed] if page.timed else doc["jobs"]
    started_at = datetime.now(UTC)
    warm_up(send_job, warmup_jobs(page, doc, photos), **ready)
    rows = timed_requests(client, send_job, timed, clock)
    finished_at = datetime.now(UTC)
    record = build_record(
        page,
        doc,
        digest,
        published,
        rows,
        endpoint=endpoint,
        client_description=client_description,
        started_at=started_at,
        finished_at=finished_at,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"latency-{page.name}-{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(path, record)
    return record, path
