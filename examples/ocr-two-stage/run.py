#!/usr/bin/env python3
"""Record the /ocr two-stage page evidence against SIE Cloud.

Stage 1 turns each page image into Markdown with LightOnOCR. Stage 2 turns
*that recorded Markdown* into typed fields under a pre-registered JSON schema.
The page image never reaches stage 2, which is what makes this a pipeline
rather than two independent demos.

Usage (from the repository root):

    python3 examples/ocr-two-stage/run.py --probe
    python3 examples/ocr-two-stage/run.py --stage 1
    python3 examples/ocr-two-stage/run.py --stage 2
    python3 examples/ocr-two-stage/run.py                 # both stages
    python3 examples/ocr-two-stage/run.py --doc ID
    python3 examples/ocr-two-stage/run.py --verify-inputs # fetch + digests only

Stage 1 sends the shape packages/tasks/src/codegen.ts generates for the `ocr`
task: POST /v1/extract/<model with / as __> with one base64 image. Stage 2
sends POST /v1/chat/completions with a strict JSON-schema response format and
the stage-1 Markdown pasted into one user message.

Credentials:
    SIE_API_KEY       bearer key; when unset, the script reads SIE_KEY_FILE
    SIE_KEY_FILE      a bare one-line key (default: ~/.secrets/sie_api_key_sep)
    SIE_CLUSTER_URL   endpoint; SIE_BASE_URL is accepted as an alias
                      (default: https://api.superlinked.com)

The key only ever goes into the Authorization header. It is never printed, and
the script deletes any file it wrote that contains it. Stored stage-1 requests
replace the base64 image with the image path and its SHA-256, so the exact
payload can be rebuilt from the committed image.

Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUTS_PATH = ROOT / "data" / "inputs.json"
RUN_DIR = ROOT / "verified-run"
CALLS_PATH = RUN_DIR / "calls.json"
# Per-call working files. Gitignored: the committed artifact is calls.json.
WORK_DIR = RUN_DIR / ".work"
REQUESTS_DIR = WORK_DIR / "requests"
RESPONSES_DIR = WORK_DIR / "responses"
MANIFEST_PATH = WORK_DIR / "manifest.json"
# Diagnostics for --probe, kept out of the committed run.
PROBE_DIR = WORK_DIR / "probe"

DEFAULT_ENDPOINT = "https://api.superlinked.com"
DEFAULT_KEY_FILE = Path.home() / ".secrets" / "sie_api_key_sep"
PROVISION_TIMEOUT_S = 900
ATTEMPT_TIMEOUT_S = 300
RETRY_STATUSES = {202, 429, 502, 503, 504}
PROBE_DOC = "product-gs1-128-label"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wire_primitive_model(model: str) -> str:
    """Byte-identical to wirePrimitiveModel() in packages/tasks/src/codegen.ts."""
    return urllib.parse.quote(model.replace("/", "__"), safe="")


def load_credentials() -> tuple[str, str]:
    key = os.environ.get("SIE_API_KEY", "").strip()
    endpoint = (os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or "").strip()
    if not key:
        key_file = Path(os.environ.get("SIE_KEY_FILE", str(DEFAULT_KEY_FILE))).expanduser()
        key = key_file.read_text(encoding="utf-8").strip()
    if not key or "\n" in key:
        raise SystemExit("No usable SIE API key found (set SIE_API_KEY or SIE_KEY_FILE)")
    resolved = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    if urllib.parse.urlparse(resolved).scheme != "https":
        raise SystemExit(f"Refusing to send the API key to a non-HTTPS endpoint: {resolved}")
    return key, resolved


def post_json(endpoint: str, key: str, path: str, payload: bytes) -> dict[str, Any]:
    deadline = time.monotonic() + PROVISION_TIMEOUT_S
    attempts = 0
    while True:
        attempts += 1
        request = urllib.request.Request(
            endpoint + path,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        # Unredirected: urllib replays normal headers on a redirect, which
        # would hand the bearer token to whatever host the redirect names.
        request.add_unredirected_header("Authorization", f"Bearer {key}")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=ATTEMPT_TIMEOUT_S) as response:
                status = response.status
                headers = dict(response.headers.items())
                raw = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            headers = dict(error.headers.items()) if error.headers else {}
            raw = error.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if time.monotonic() > deadline:
                raise
            print(f"  transport error ({type(error).__name__}), retrying", file=sys.stderr)
            time.sleep(10)
            continue
        latency_s = time.monotonic() - started
        if status in RETRY_STATUSES and time.monotonic() < deadline:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            try:
                wait = min(float(retry_after) if retry_after else 15.0, 30.0)
            except ValueError:
                wait = 15.0
            print(f"  HTTP {status}, waiting {wait:.0f}s for capacity", file=sys.stderr)
            time.sleep(wait)
            continue
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return {
            "status": status,
            "latency_s": round(latency_s, 4),
            "attempts": attempts,
            # Keep model, version, usage and request-id headers; drop the
            # account key identifier so no key metadata reaches the repository.
            "headers": {
                name.lower(): value
                for name, value in headers.items()
                if (name.lower().startswith("x-sie") and name.lower() != "x-sie-key-id")
                or name.lower() in {"content-type", "date"}
            },
            "json": parsed,
            "text": None if parsed is not None else text,
        }


def image_for(doc: dict[str, Any]) -> bytes:
    """Fetch the exact bytes that were sent, and refuse anything else.

    The images are not committed: several carry licences that do not allow
    redistribution here, and one is a copyrighted investor page used as an
    attributed excerpt. `source.url` serves the same bytes the run used, and
    `source.sha256` is what makes the fetch trustworthy.
    """
    source = doc["source"]
    with urllib.request.urlopen(source["url"], timeout=120) as response:
        data = response.read()
    if len(data) != source["bytes"]:
        raise SystemExit(
            f"{doc['id']}: fetched {len(data)} bytes, registered {source['bytes']}"
        )
    digest = sha256_bytes(data)
    if digest != doc["image_sha256"]:
        raise SystemExit(
            f"{doc['id']}: {doc['image']} does not match the checksum registered in inputs.json"
        )
    return data


def verify_inputs(inputs: dict[str, Any]) -> None:
    for doc in inputs["documents"]:
        image_for(doc)
    print(f"verified {len(inputs['documents'])} registered images", file=sys.stderr)


def stage1_body(image_b64: str, fmt: str) -> dict[str, Any]:
    return {"items": [{"images": [{"data": image_b64, "format": fmt}]}]}


def run_stage1(
    invocation: str,
    endpoint: str,
    key: str,
    inputs: dict[str, Any],
    doc: dict[str, Any],
    out_requests: Path,
    out_responses: Path,
) -> tuple[dict[str, Any], list[Path]]:
    model = inputs["stage1"]["model"]
    path = f"/v1/extract/{wire_primitive_model(model)}"
    raw = image_for(doc)
    body = stage1_body(base64.b64encode(raw).decode("ascii"), doc["format"])
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    name = f"{doc['id']}__stage1"
    print(f"{name}: POST {path}", file=sys.stderr)
    started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    response = post_json(endpoint, key, path, payload)

    stored_body = stage1_body(
        f"<base64 of apps/site/public/reference/ocr-review/{doc['image']}, sha256 {doc['image_sha256']}>",
        doc["format"],
    )
    request_record = {
        "method": "POST",
        "endpoint": endpoint,
        "path": path,
        "model": model,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "payload_sha256": sha256_bytes(payload),
        "body": stored_body,
    }
    response_record = {
        "status": response["status"],
        "headers": response["headers"],
        "body": response["json"] if response["json"] is not None else response["text"],
    }
    request_path = out_requests / f"{name}.json"
    response_path = out_responses / f"{name}.json"
    write_json(request_path, request_record)
    write_json(response_path, response_record)

    markdown = stage1_markdown(response_record)
    record = {
        "document": doc["id"],
        "stage": 1,
        "invocation_id": invocation,
        "endpoint": endpoint,
        "call_started_utc": started_utc,
        "model": model,
        "model_revision": response["headers"].get("x-sie-model-revision"),
        "server_version": response["headers"].get("x-sie-server-version"),
        "status": response["status"],
        "latency_s": response["latency_s"],
        "attempts": response["attempts"],
        "image": doc["image"],
        "image_sha256": doc["image_sha256"],
        "markdown_chars": len(markdown) if markdown is not None else None,
        "markdown_sha256": sha256_text(markdown) if markdown is not None else None,
        "payload_sha256": request_record["payload_sha256"],
        "request_sha256": canonical_sha256(request_record),
        "response_sha256": canonical_sha256(response_record),
    }
    print(
        f"  HTTP {response['status']} in {response['latency_s']:.1f}s,"
        f" markdown chars={record['markdown_chars']}",
        file=sys.stderr,
    )
    return record, [request_path, response_path]


def stage1_markdown(response_record: dict[str, Any]) -> str | None:
    """Read entities[0].text, the field the generated OCR snippet prints."""
    body = response_record.get("body")
    if not isinstance(body, dict):
        return None
    items = body.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        body = items[0]
    entities = body.get("entities")
    if isinstance(entities, list) and entities and isinstance(entities[0], dict):
        text = entities[0].get("text")
        return text if isinstance(text, str) else None
    data = body.get("data")
    if isinstance(data, dict):
        for field in ("markdown", "text"):
            if isinstance(data.get(field), str):
                return data[field]
    for field in ("markdown", "text"):
        if isinstance(body.get(field), str):
            return body[field]
    return None


def stage2_body(
    inputs: dict[str, Any], model: str, schema_id: str, document_type: str, markdown: str
) -> dict[str, Any]:
    stage2 = inputs["stage2"]
    content = stage2["instruction_template"].format(
        document_type=document_type, markdown=markdown
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": stage2["temperature"],
        "max_tokens": stage2["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_id,
                "strict": True,
                "schema": inputs["schemas"][schema_id],
            },
        },
    }


def run_stage2(
    invocation: str,
    endpoint: str,
    key: str,
    inputs: dict[str, Any],
    doc: dict[str, Any],
    call: dict[str, Any],
    model: str,
    model_key: str,
    markdown: str,
    markdown_sha256: str,
    out_requests: Path,
    out_responses: Path,
) -> tuple[dict[str, Any], list[Path]]:
    path = inputs["stage2"]["path"]
    body = stage2_body(inputs, model, call["schema"], doc["document_type"], markdown)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    name = f"{doc['id']}__{call['id']}__stage2__{model_key}"
    print(f"{name}: POST {path}", file=sys.stderr)
    started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    response = post_json(endpoint, key, path, payload)

    request_record = {
        "method": "POST",
        "endpoint": endpoint,
        "path": path,
        "model": model,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "payload_sha256": sha256_bytes(payload),
        "stage1_markdown_sha256": markdown_sha256,
        "body": body,
    }
    response_record = {
        "status": response["status"],
        "headers": response["headers"],
        "body": response["json"] if response["json"] is not None else response["text"],
    }
    request_path = out_requests / f"{name}.json"
    response_path = out_responses / f"{name}.json"
    write_json(request_path, request_record)
    write_json(response_path, response_record)

    content = stage2_content(response_record)
    parsed: Any = None
    parse_error: str | None = None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            parse_error = str(error)

    body_json = response["json"] if isinstance(response["json"], dict) else {}
    usage = body_json.get("usage") or {}
    choices = body_json.get("choices") or []
    finish = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    record = {
        "document": doc["id"],
        "call": call["id"],
        "stage": 2,
        "invocation_id": invocation,
        "endpoint": endpoint,
        "call_started_utc": started_utc,
        "model": model,
        "model_key": model_key,
        "schema": call["schema"],
        "model_revision": response["headers"].get("x-sie-model-revision"),
        "server_version": response["headers"].get("x-sie-server-version"),
        "status": response["status"],
        "latency_s": response["latency_s"],
        "attempts": response["attempts"],
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "stage1_markdown_sha256": markdown_sha256,
        "json_parsed": parsed is not None,
        "parse_error": parse_error,
        "content_sha256": sha256_text(content) if isinstance(content, str) else None,
        "payload_sha256": request_record["payload_sha256"],
        "request_sha256": canonical_sha256(request_record),
        "response_sha256": canonical_sha256(response_record),
    }
    print(
        f"  HTTP {response['status']} in {response['latency_s']:.1f}s, finish={finish},"
        f" tokens in/out={usage.get('prompt_tokens')}/{usage.get('completion_tokens')},"
        f" json={'ok' if parsed is not None else parse_error or 'missing'}",
        file=sys.stderr,
    )
    return record, [request_path, response_path]


def stage2_content(response_record: dict[str, Any]) -> str | None:
    body = response_record.get("body")
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return None


def load_stage1_markdown(doc_id: str) -> tuple[str, str]:
    path = RESPONSES_DIR / f"{doc_id}__stage1.json"
    if not path.exists():
        raise SystemExit(f"{doc_id}: no recorded stage-1 response; run --stage 1 first")
    markdown = stage1_markdown(read_json(path))
    if not isinstance(markdown, str):
        raise SystemExit(f"{doc_id}: recorded stage-1 response carries no text")
    return markdown, sha256_text(markdown)


def assert_no_key(key: str, paths: list[Path]) -> None:
    for path in paths:
        if key in path.read_text(encoding="utf-8"):
            path.unlink()
            raise SystemExit(f"Refusing to keep {path.name}: it contains the API key")


def merge_manifest(new_records: list[dict[str, Any]], started: str) -> None:
    manifest: dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
    calls: list[dict[str, Any]] = manifest.get("calls", [])
    index = {(c.get("document"), c.get("call"), c.get("stage"), c.get("model_key")): i for i, c in enumerate(calls)}
    for record in new_records:
        keyed = (record.get("document"), record.get("call"), record.get("stage"), record.get("model_key"))
        if keyed in index:
            calls[index[keyed]] = record
        else:
            index[keyed] = len(calls)
            calls.append(record)
    # A manifest accumulates calls from several invocations, so the top-level
    # time is only the latest write. Provenance lives per call: its invocation
    # id, its endpoint and its start time. The page derives its run date from
    # the recorded response headers, not from any of this.
    endpoints = {call.get("endpoint") for call in calls if call.get("endpoint")}
    manifest.update(
        {
            "last_run_started_utc": started,
            "composite": len({call.get("invocation_id") for call in calls}) > 1,
            "endpoints": sorted(endpoints),
            "inputs_sha256": sha256_bytes(INPUTS_PATH.read_bytes()),
            "calls": calls,
        }
    )
    write_json(MANIFEST_PATH, manifest)


def consolidate() -> None:
    """Fold the per-call working files into verified-run/calls.json.

    This is the file a reader checks our numbers from, and the only one the
    example commits. evaluate.py reads it and nothing else, so the published
    figures can be re-derived with no API key and no network.
    """
    manifest = read_json(MANIFEST_PATH)
    calls = []
    for entry in manifest["calls"]:
        if entry["stage"] == 1:
            slug = f"{entry['document']}__stage1"
        else:
            slug = f"{entry['document']}__{entry['call']}__stage2__{entry['model_key']}"
        calls.append(
            {
                "slug": slug,
                "document": entry["document"],
                "stage": entry["stage"],
                "model": entry["model"],
                "model_revision": entry.get("model_revision"),
                "server_version": entry.get("server_version"),
                "status": entry["status"],
                "attempts": entry.get("attempts"),
                "timing": {"duration_s": entry.get("latency_s")},
                "request": read_json(REQUESTS_DIR / f"{slug}.json"),
                "response": read_json(RESPONSES_DIR / f"{slug}.json"),
            }
        )
    write_json(
        CALLS_PATH,
        {
            "about": (
                "Every call the run made, request and response, in recorded "
                "order. With data/inputs.json and data/predictions.json, "
                "which carry the pre-registered expectations, this is "
                "everything evaluate.py needs: no API key, no network."
            ),
            "recorded_utc": manifest.get("last_run_started_utc"),
            "endpoint": manifest["endpoints"][0] if manifest.get("endpoints") else None,
            "models": sorted({call["model"] for call in calls}),
            "calls": calls,
        },
    )
    print(f"wrote {CALLS_PATH.relative_to(ROOT)} with {len(calls)} calls")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--doc", help="run one document id")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="run one stage")
    parser.add_argument(
        "--stage2-model",
        choices=["primary", "secondary"],
        help="run one stage-2 model (default: both)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=f"both stages on {PROBE_DOC}, stored under diagnostics/probe",
    )
    parser.add_argument(
        "--verify-inputs", action="store_true", help="check every registered image checksum"
    )
    args = parser.parse_args()

    inputs = read_json(INPUTS_PATH)

    if args.verify_inputs:
        verify_inputs(inputs)
        return

    key, endpoint = load_credentials()
    started = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    invocation = uuid.uuid4().hex
    docs = [d for d in inputs["documents"] if args.doc in (None, d["id"])]
    if not docs:
        raise SystemExit(f"No registered document matches {args.doc!r}")

    model_keys = [args.stage2_model] if args.stage2_model else ["primary", "secondary"]
    models = inputs["stage2"]["models"]

    if args.probe:
        doc = next(d for d in inputs["documents"] if d["id"] == PROBE_DOC)
        req_dir, res_dir = PROBE_DIR / "requests", PROBE_DIR / "responses"
        record, written = run_stage1(invocation, endpoint, key, inputs, doc, req_dir, res_dir)
        assert_no_key(key, written)
        markdown = stage1_markdown(read_json(written[1]))
        print("--- stage 1 markdown ---")
        print(markdown)
        if not isinstance(markdown, str):
            raise SystemExit("probe: stage 1 returned no text")
        call = doc["calls"][0]
        record2, written2 = run_stage2(
            invocation, endpoint, key, inputs, doc, call, models["primary"],
            "primary", markdown, sha256_text(markdown), req_dir, res_dir,
        )
        assert_no_key(key, written2)
        print("--- stage 2 content ---")
        print(stage2_content(read_json(written2[1])))
        write_json(PROBE_DIR / "manifest.json", {"run_started_utc": started, "calls": [record, record2]})
        return

    records: list[dict[str, Any]] = []
    written_all: list[Path] = []

    if args.stage in (None, 1):
        for doc in docs:
            record, written = run_stage1(invocation, endpoint, key, inputs, doc, REQUESTS_DIR, RESPONSES_DIR)
            records.append(record)
            written_all.extend(written)
            assert_no_key(key, written)

    if args.stage in (None, 2):
        for doc in docs:
            markdown, markdown_sha = load_stage1_markdown(doc["id"])
            for call in doc["calls"]:
                for model_key in model_keys:
                    record, written = run_stage2(
                        invocation, endpoint, key, inputs, doc, call,
                        models[model_key], model_key, markdown, markdown_sha,
                        REQUESTS_DIR, RESPONSES_DIR,
                    )
                    records.append(record)
                    written_all.extend(written)
                    assert_no_key(key, written)

    merge_manifest(records, started)
    consolidate()
    assert_no_key(key, [MANIFEST_PATH, CALLS_PATH])
    failed = [r for r in records if r["status"] != 200]
    print(
        f"\n{len(records)} calls, {len(records) - len(failed)} HTTP 200, {len(failed)} other",
        file=sys.stderr,
    )
    for record in failed:
        where = f"{record.get('document')} {record.get('call', 'stage1')}"
        print(f"  FAILED {where}: HTTP {record['status']}", file=sys.stderr)


if __name__ == "__main__":
    main()
