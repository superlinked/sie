#!/usr/bin/env python3
"""Run SIE's PII composition yourself, on one recorded document or all 660.

    python3 fetch.py
    python3 run.py --show                                  # offline: the requests for one document, not sent
    SIE_API_KEY=... uv run python run.py                   # one document, masked and scored against its gold
    SIE_API_KEY=... uv run python run.py --doc 851         # another document, by index or uid
    SIE_API_KEY=... uv run python run.py --all             # all 660, into run-output/
    python3 score.py --sie-rows run-output                 # score your run beside the recorded arms

Endpoint   https://api.superlinked.com  (set SIE_BASE_URL to use your own server)
Models     urchade/gliner_multi_pii-v1 and numind/NuNER_Zero, through sie_sdk.SIEClient
Labels     the 36 in score.py, the same for every request

A document over 300 words is also sent as 300-word windows with a 50-word
overlap, and each window's offset is added back to its spans; the composition
uses the windows when there are any. That is what the recorded run did. `--all`
also sends every document whole, as the run did, so score.py can report each
model alone.

The SDK import is deferred into the functions that send, so `--show` works on
a bare `python3` with nothing installed. One document costs a fraction of a
cent; `--all` sends about 470,000 tokens per model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from score import (
    EVIDENCE,
    FIRST_MODEL,
    REQUEST_LABELS,
    SECOND_MODEL,
    compose,
    covered,
    load_documents,
    merge,
    read_jsonl,
    windows,
)

if TYPE_CHECKING:
    from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://api.superlinked.com"
MODELS = (FIRST_MODEL, SECOND_MODEL)
ROWS = {
    FIRST_MODEL: "sie__urchade__gliner_multi_pii-v1.jsonl",
    SECOND_MODEL: "sie__numind__NuNER_Zero.jsonl",
}
DEFAULT_DOCUMENT = "1628"  # a health insurance claim form, the document the page opens with


def pick(documents: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for d in documents:
        if key in (str(d["index"]), d["uid"]):
            return d
    raise SystemExit(f"no document with index or uid {key!r}; indexes run {documents[0]['index']} upward")


def requests_for(doc: dict[str, Any], *, whole: bool) -> list[tuple[int, str]]:
    """(offset, text) for every request one model gets for this document."""
    spans = windows(doc["text"])
    parts = [(start, doc["text"][start:end]) for start, end in spans] if len(spans) > 1 else []
    return [(0, doc["text"]), *parts] if whole or not parts else parts


def show(doc: dict[str, Any]) -> None:
    parts = requests_for(doc, whole=False)
    print(f"document {doc['index']} ({doc['uid']}, {doc['domain']}): {len(doc['text']):,} characters")
    print(f"{len(parts)} request(s) per model, {len(parts) * len(MODELS)} in all, to {ENDPOINT}")
    for model in MODELS:
        for offset, text in parts:
            body = {"items": [{"text": text}], "params": {"labels": REQUEST_LABELS}}
            print()
            print(f"POST /v1/extract/{model}   (window at character {offset})")
            print(json.dumps(body, ensure_ascii=False))


def client() -> SIEClient:
    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. python3 run.py --show needs no key.")
    from sie_sdk import SIEClient

    return SIEClient(os.environ.get("SIE_BASE_URL", ENDPOINT), api_key=api_key, timeout_s=300)


def extract(sie: SIEClient, model: str, offset: int, text: str) -> list[dict[str, Any]]:
    from sie_sdk.types import Item

    result = sie.extract(model, Item(text=text), labels=REQUEST_LABELS)
    # Some SDK versions return a per-item failure in result["error"] instead of
    # raising; an empty entity list there is a failure, not a clean document.
    if result.get("error"):
        raise RuntimeError(f"{model}: {result['error']}")
    return [
        {
            "start": e["start"] + offset,
            "end": e["end"] + offset,
            "label": e["label"],
            "score": e.get("score"),
            "text": e.get("text"),
        }
        for e in result.get("entities") or []
    ]


def record(sie: SIEClient, model: str, doc: dict[str, Any], *, whole: bool) -> dict[str, Any]:
    """One model's output for one document, in the recorded rows' shape."""
    out: dict[str, Any] = {"whole": [], "windows": []}
    spans = windows(doc["text"])
    if whole or len(spans) == 1:
        out["whole"] = extract(sie, model, 0, doc["text"])
    if len(spans) > 1:
        for start, end in spans:
            out["windows"].append({"offset": start, "entities": extract(sie, model, start, doc["text"][start:end])})
    return out


def masked(text: str, spans: list[dict[str, Any]]) -> str:
    labels = {}
    for s in spans:
        labels.setdefault((s["start"], s["end"]), s["label"])
    out, at = [], 0
    for a, b in merge(spans):
        label = next((lab for (s, e), lab in labels.items() if s == a), "pii")
        out += [text[at:a], f"[{label.upper()}]"]
        at = b
    return "".join([*out, text[at:]])


def one(doc: dict[str, Any], evidence: Path) -> int:
    sie = client()
    outputs = {model: record(sie, model, doc, whole=False) for model in MODELS}
    spans = compose(doc["text"], outputs[FIRST_MODEL], outputs[SECOND_MODEL])
    masks = merge(spans)
    hits = [covered(doc["text"], masks, g) for g in doc["scoped"]]
    print(masked(doc["text"], spans))
    print()
    print(f"document {doc['index']} ({doc['domain']}): {sum(hits)} of {len(hits)} in-scope gold spans masked")
    for g, hit in zip(doc["scoped"], hits, strict=True):
        if not hit:
            print(f"  left readable: {g['label']} {doc['text'][g['start'] : g['end']]!r}")
    recorded = {m: read_jsonl(evidence / "rows" / ROWS[m]) for m in MODELS}
    then = compose(
        doc["text"],
        *(next(r["output"] for r in recorded[m] if r["index"] == doc["index"]) for m in MODELS),
    )
    then_hits = sum(covered(doc["text"], merge(then), g) for g in doc["scoped"])
    same = {(s["start"], s["end"]) for s in spans} == {(s["start"], s["end"]) for s in then}
    print(f"recorded run on 2026-09-30: {then_hits} of {len(hits)}; spans {'identical' if same else 'differ'}")
    return 0


def run_all(documents: list[dict[str, Any]], out: Path, workers: int) -> int:
    local = threading.local()

    def job(model: str, doc: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "sie"):
            local.sie = client()
        try:
            return {"index": doc["index"], "output": record(local.sie, model, doc, whole=True), "error": None}
        except Exception as error:  # noqa: BLE001 - recorded per row, as the study did
            return {"index": doc["index"], "output": None, "error": f"{type(error).__name__}: {error}"[:300]}

    client()  # fail on a missing key before starting threads
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for model in MODELS:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda d, m=model: job(m, d), documents))
        failed += sum(r["error"] is not None for r in rows)
        path = out / ROWS[model]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{model}: {len(rows)} documents -> {path}")
    if failed:
        print(f"{failed} document(s) failed; they score as no masks", file=sys.stderr)
    print(f"Now run: python3 score.py --sie-rows {out}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", default=DEFAULT_DOCUMENT, help="document index or uid (default %(default)s)")
    parser.add_argument("--show", action="store_true", help="print the requests for --doc without sending them")
    parser.add_argument("--all", action="store_true", help="send all 660 documents and write rows files")
    parser.add_argument("--out", type=Path, default=HERE / "run-output", help="where --all writes")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight for --all")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE, help="the directory fetch.py wrote")
    args = parser.parse_args()

    documents = load_documents(args.evidence)
    if args.all:
        return run_all(documents, args.out, args.workers)
    doc = pick(documents, args.doc)
    if args.show:
        show(doc)
        return 0
    return one(doc, args.evidence)


if __name__ == "__main__":
    sys.exit(main())
