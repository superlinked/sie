#!/usr/bin/env python3
"""Send the /ocr two-stage calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check            # offline, no key, nothing installed
    python3 run.py --show <doc-id>    # offline, prints both stages for one document
    python3 run.py --verify-inputs    # fetches the images and checks their digests
    uv sync && uv run python run.py --record   # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Stage 1       POST /v1/extract/lightonai/LightOnOCR-2-1B, one page image
Stage 2       POST /v1/chat/completions, the stage-1 Markdown under a strict
              JSON schema, once on Qwen/Qwen3.8-27B-FP8 and once on Qwen/Qwen3.5-4B

The page image never reaches stage 2. That is what makes this a pipeline rather
than two demonstrations, and `--check` is where the claim is tested: it rebuilds
each stage-2 request body from the instruction template and the Markdown in the
recorded stage-1 response, then compares it with the stage-2 request that was
actually sent. If stage 2 had seen anything else, the rebuild would differ.

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

The images are not in the dataset. Several carry licences that do not permit
redistribution, and one is a copyrighted investor page used as an attributed
excerpt. Each document in `inputs/inputs.json` carries the URL serving the exact
bytes that were sent, their length and their SHA-256; `--verify-inputs` and
`--record` fetch by that URL and refuse anything that differs. The recorded
stage-1 request stores that digest in place of the base64, so the payload can be
rebuilt from the image rather than shipped with it.

The key only ever goes into the SDK client. It is never printed, and `--record`
refuses to keep a file that contains it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

ENDPOINT = "https://api.superlinked.com"
STAGE1_STORED_IMAGE = "<base64 of apps/site/public/reference/ocr-review/{image}, sha256 {sha256}>"


class CallFailedError(Exception):
    """A call that did not produce a usable result."""


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def to_jsonable(value: Any) -> Any:
    """What the SDK returned, as plain JSON types and nothing else."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return value


# --- the two request bodies, rebuilt from the registered inputs -------------


def stage1_stored_body(doc: dict[str, Any]) -> dict[str, Any]:
    """The stage-1 body as it is recorded, with the digest standing in for the
    base64 so the run can be checked without redistributing the image."""
    stored = STAGE1_STORED_IMAGE.format(image=doc["image"], sha256=doc["image_sha256"])
    return {"items": [{"images": [{"data": stored, "format": doc["format"]}]}]}


def stage2_body(
    inputs: dict[str, Any], doc: dict[str, Any], call: dict[str, Any], model: str, markdown: str
) -> dict[str, Any]:
    stage2 = inputs["stage2"]
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": stage2["instruction_template"].format(document_type=doc["document_type"], markdown=markdown),
            }
        ],
        "temperature": stage2["temperature"],
        "max_tokens": stage2["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": call["schema"], "strict": True, "schema": inputs["schemas"][call["schema"]]},
        },
    }


def stage1_markdown(body: Any) -> str | None:
    """Read entities[0].text, the field the generated OCR snippet prints."""
    if not isinstance(body, dict):
        return None
    items = body.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        body = items[0]
    entities = body.get("entities")
    if isinstance(entities, list) and entities and isinstance(entities[0], dict):
        text = entities[0].get("text")
        return text if isinstance(text, str) else None
    return None


def expected_slugs(inputs: dict[str, Any]) -> list[str]:
    """Every call the registered inputs imply, in recorded order."""
    slugs = []
    for doc in inputs["documents"]:
        slugs.append(f"{doc['id']}__stage1")
    for doc in inputs["documents"]:
        for call in doc["calls"]:
            for model_key in inputs["stage2"]["models"]:
                slugs.append(f"{doc['id']}__{call['id']}__stage2__{model_key}")
    return slugs


# --- offline check ----------------------------------------------------------


def check(data_dir: Path) -> int:
    """Rebuild all 20 recorded request bodies from the registered inputs.

    A bijection, not a walk over what is there: the expected slugs come from
    inputs.json, so a call that is missing, recorded twice or implied by no
    document all fail. Checking only the calls present would pass a calls.json
    with one of them deleted.
    """
    inputs = load(data_dir / "inputs/inputs.json")
    payload = load(data_dir / "calls.json")

    recorded: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for call in payload["calls"]:
        if call["slug"] in recorded:
            duplicates.append(call["slug"])
            continue
        recorded[call["slug"]] = call

    expected = expected_slugs(inputs)
    if len(expected) != len(set(expected)):
        raise SystemExit("inputs.json implies the same call slug twice")
    missing = sorted(set(expected) - set(recorded))
    unexpected = sorted(set(recorded) - set(expected))

    rebuilt = 0
    mismatched: list[str] = []
    stage2_from_stage1 = 0
    for doc in inputs["documents"]:
        slug = f"{doc['id']}__stage1"
        call = recorded.get(slug)
        if call is None:
            continue
        if call["request"]["body"] != stage1_stored_body(doc):
            mismatched.append(f"{slug}: rebuilt body differs from the recorded body")
        elif call["request"]["path"] != inputs["stage1"]["path"]:
            mismatched.append(f"{slug}: path {call['request']['path']} differs from {inputs['stage1']['path']}")
        elif call["model"] != inputs["stage1"]["model"]:
            mismatched.append(f"{slug}: model {call['model']} differs from {inputs['stage1']['model']}")
        else:
            rebuilt += 1

        markdown = stage1_markdown(call["response"]["body"])
        if markdown is None:
            mismatched.append(f"{slug}: recorded response carries no Markdown to feed stage 2")
            continue
        for stage2_call in doc["calls"]:
            for model_key, model in inputs["stage2"]["models"].items():
                stage2_slug = f"{doc['id']}__{stage2_call['id']}__stage2__{model_key}"
                entry = recorded.get(stage2_slug)
                if entry is None:
                    continue
                if entry["request"]["body"] != stage2_body(inputs, doc, stage2_call, model, markdown):
                    mismatched.append(f"{stage2_slug}: rebuilt body differs from the recorded body")
                elif entry["request"]["path"] != inputs["stage2"]["path"]:
                    mismatched.append(
                        f"{stage2_slug}: path {entry['request']['path']} differs from {inputs['stage2']['path']}"
                    )
                else:
                    rebuilt += 1
                    stage2_from_stage1 += 1

    print(f"{rebuilt} of {len(payload['calls'])} recorded requests rebuilt from the inputs and matched")
    print(f"{len(expected)} calls expected from {len(inputs['documents'])} documents, {len(recorded)} recorded")
    print(
        f"{stage2_from_stage1} stage-2 requests rebuilt from the Markdown in the recorded stage-1 response,"
        " so stage 2 saw that text and nothing else"
    )
    for slug in missing:
        print(f"MISSING {slug}: expected from the inputs, absent from calls.json", file=sys.stderr)
    for slug in duplicates:
        print(f"DUPLICATE {slug}: recorded more than once", file=sys.stderr)
    for slug in unexpected:
        print(f"UNEXPECTED {slug}: recorded but no document in inputs.json implies it", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    if missing or duplicates or unexpected or mismatched:
        return 1
    return 0


# --- images -----------------------------------------------------------------


def image_for(doc: dict[str, Any]) -> bytes:
    """Fetch the exact bytes that were sent, and refuse anything else."""
    source = doc["source"]
    request = urllib.request.Request(source["url"], headers={"Accept": "*/*"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        data = response.read()
    if len(data) != source["bytes"]:
        raise SystemExit(f"{doc['id']}: fetched {len(data)} bytes, registered {source['bytes']}")
    if sha256_bytes(data) != doc["image_sha256"]:
        raise SystemExit(f"{doc['id']}: {doc['image']} does not match the digest registered in inputs.json")
    return data


# --- recording --------------------------------------------------------------


def entry(slug: str, doc_id: str, stage: int, model: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": slug,
        "document": doc_id,
        "stage": stage,
        "model": model,
        "status": 200,
        "request": {
            "method": "POST",
            "endpoint": ENDPOINT,
            "path": path,
            "model": model,
            "headers": {"Content-Type": "application/json", "Accept": "application/json"},
            "body": body,
        },
    }


def record_stage1(client: SIEClient, inputs: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    from sie_sdk import Item  # noqa: PLC0415

    model = inputs["stage1"]["model"]
    raw = image_for(doc)
    started = time.monotonic()
    result = client.extract(model, Item(images=[{"data": raw, "format": doc["format"]}]))
    duration_s = round(time.monotonic() - started, 4)
    item = to_jsonable(result)
    if stage1_markdown({"items": [item]}) is None:
        raise CallFailedError(f"{doc['id']}: stage 1 returned no text")
    record = entry(f"{doc['id']}__stage1", doc["id"], 1, model, inputs["stage1"]["path"], stage1_stored_body(doc))
    record.update(
        {
            "model_revision": client.last_model_revision,
            "server_version": None,
            "attempts": client.last_retry_count + 1,
            "timing": {"duration_s": duration_s},
            "response": {
                "status": 200,
                "headers": {},
                "shape": (
                    "rebuilt around the per-item result sie_sdk returned; the SDK "
                    "returns no server envelope and surfaces no response headers"
                ),
                "body": {"items": [item]},
            },
        }
    )
    return record


def record_stage2(
    client: SIEClient,
    inputs: dict[str, Any],
    doc: dict[str, Any],
    call: dict[str, Any],
    model_key: str,
    markdown: str,
) -> dict[str, Any]:
    model = inputs["stage2"]["models"][model_key]
    body = stage2_body(inputs, doc, call, model, markdown)
    started = time.monotonic()
    completion = client.chat_completions(
        model,
        body["messages"],
        temperature=body["temperature"],
        max_tokens=body["max_tokens"],
        response_format=body["response_format"],
    )
    duration_s = round(time.monotonic() - started, 4)
    envelope = {key: value for key, value in to_jsonable(completion).items() if key != "request"}
    # A 200 is not a result. Refuse anything without a usable message.
    choices = envelope.get("choices") or []
    if not choices or not (choices[0].get("message") or {}).get("content"):
        raise CallFailedError(f"{doc['id']}/{call['id']}/{model_key}: response carried no assistant message")
    slug = f"{doc['id']}__{call['id']}__stage2__{model_key}"
    record = entry(slug, doc["id"], 2, model, inputs["stage2"]["path"], body)
    record.update(
        {
            "call": call["id"],
            "model_key": model_key,
            "model_revision": client.last_model_revision,
            "server_version": None,
            "attempts": client.last_retry_count + 1,
            "timing": {"duration_s": duration_s},
            "response": {
                "status": 200,
                "headers": {},
                "shape": "the server's own chat completion envelope as sie_sdk returns it",
                "body": envelope,
            },
        }
    )
    return record


def refuse_if_key_present(key: str, path: Path) -> None:
    if key and key in path.read_text(encoding="utf-8"):
        path.unlink()
        raise SystemExit(f"Refusing to keep {path.name}: it contains the API key")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="DOC", help="print one document's calls and exit, sending nothing")
    parser.add_argument(
        "--verify-inputs", action="store_true", help="fetch every registered image and check its digest"
    )
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--doc", help="record one document id")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    inputs = load(data_dir / "inputs/inputs.json")
    by_id = {doc["id"]: doc for doc in inputs["documents"]}

    if args.show:
        doc = by_id.get(args.show)
        if doc is None:
            raise SystemExit(f"Unknown document: {args.show}. Known: {', '.join(by_id)}")
        calls = load(data_dir / "calls.json")["calls"]
        # A run whose stage 1 failed records no entry for it, and an entry that
        # came back without text yields None, which would format into the
        # stage-2 prompt as the literal "None". Both stop here instead.
        slug = f"{doc['id']}__stage1"
        recorded = next((call for call in calls if call["slug"] == slug), None)
        if recorded is None:
            raise SystemExit(f"no recorded stage-1 call for {slug}; stage 2 has nothing to be built from")
        markdown = stage1_markdown(recorded["response"]["body"])
        if markdown is None:
            raise SystemExit(f"{slug}: the recorded response carries no Markdown to feed stage 2")
        shown = [{"stage": 1, "path": inputs["stage1"]["path"], "body": stage1_stored_body(doc)}]
        for call in doc["calls"]:
            for model_key, model in inputs["stage2"]["models"].items():
                shown.append(
                    {
                        "stage": 2,
                        "model_key": model_key,
                        "path": inputs["stage2"]["path"],
                        "body": stage2_body(inputs, doc, call, model, markdown),
                    }
                )
        print(json.dumps(shown, indent=2, ensure_ascii=False))
        return 0

    if args.verify_inputs:
        for doc in inputs["documents"]:
            image_for(doc)
            print(f"{doc['id']}: {doc['source']['bytes']} bytes match {doc['image_sha256'][:12]}")
        print(f"verified {len(inputs['documents'])} registered images")
        return 0

    if not args.record:
        return check(data_dir)

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY to send these calls, or run score.py on the recorded ones instead")
    base_url = os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT
    # Deferred so --check, --show and --verify-inputs run on a bare `python3`.
    from sie_sdk import SIEClient  # noqa: PLC0415

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    documents = [doc for doc in inputs["documents"] if args.doc in (None, doc["id"])]
    if not documents:
        raise SystemExit(f"No registered document matches {args.doc!r}")

    calls: list[dict[str, Any]] = []
    failed: list[str] = []
    for doc in documents:
        try:
            stage1 = record_stage1(client, inputs, doc)
        except Exception as error:  # noqa: BLE001
            failed.append(f"{doc['id']}__stage1: {type(error).__name__}: {error}")
            print(f"{doc['id']} stage 1: FAILED {type(error).__name__}", file=sys.stderr)
            continue
        calls.append(stage1)
        markdown = stage1_markdown(stage1["response"]["body"])
        print(f"{doc['id']} stage 1: {stage1['timing']['duration_s']:.1f}s, {len(markdown)} chars", file=sys.stderr)
        for call in doc["calls"]:
            for model_key in inputs["stage2"]["models"]:
                try:
                    calls.append(record_stage2(client, inputs, doc, call, model_key, markdown))
                except Exception as error:  # noqa: BLE001
                    failed.append(f"{doc['id']}/{call['id']}/{model_key}: {type(error).__name__}: {error}")
                    print(f"{doc['id']} {call['id']} {model_key}: FAILED {type(error).__name__}", file=sys.stderr)

    slugs = [call["slug"] for call in calls]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("duplicate call slugs; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "about": (
                    "Every call the run made, request and response, in recorded order. "
                    "With inputs/inputs.json and inputs/predictions.json, which carry the "
                    "pre-registered expectations, this is everything score.py needs: no "
                    "API key, no network."
                ),
                "recorded_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "endpoint": base_url,
                "models": sorted({call["model"] for call in calls}),
                "failed_calls": len(failed),
                "complete": not failed,
                "calls": calls,
            },
            indent="\t",
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    refuse_if_key_present(api_key, out_path)
    print(f"wrote {out_path} with {len(calls)} calls")
    if failed:
        # A run that failed must not look like a run that succeeded.
        print(f"{len(failed)} calls FAILED:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
