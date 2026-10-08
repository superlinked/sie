#!/usr/bin/env python3
"""Read document images into their own JSON schema with SIE: one document, or a whole published set.

    python3 run.py --image form.png --schema fields.json --dry-run
    uv run run.py --image form.png --schema fields.json --base-url http://127.0.0.1:8080

    python3 run.py --set confirm --dry-run
    uv run run.py --set confirm --base-url http://127.0.0.1:8080

--dry-run needs only the Python standard library and sends nothing. For one document it prints the request
with the image replaced by its sha256. For a set it rebuilds every request of the published study from the
pinned Omni rows and images and checks the bodies against the published sha256.

A send uses sie_sdk.SIEClient (sie-sdk 0.9.0 or newer) against any SIE endpoint: a local server, a remote
self-hosted one, or SIE Cloud. SIE_API_KEY (or the variable named by --api-key-env) supplies the key when the
endpoint needs one. A set run writes one grader row per study document to runs/<set>/replies.jsonl and scores
it with score.py over the whole set. A request that fails, returns no content or hits the output cap becomes an
error row, which scores 0.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import score
from fetch import DATA, HERE, PINS, SETS, load_omni_rows, load_set, prepare_images, sha256

MODEL = "Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec"
MAX_COMPLETION_TOKENS = 8192
SYSTEM = (
    "Extract data from the document image into the JSON schema. Copy values as they appear on the document. "
    "Use null for a field the document does not contain. Return only the JSON."
)
# The manifest rows, the verified image files and the Omni rows of one set.
SetInputs = tuple[list[dict[str, Any]], dict[str, Path], dict[str, dict[str, Any]]]


# ---- the recipe ---------------------------------------------------------------------------------------------------


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The study's shared schema conversion: every property required, every leaf nullable."""
    if "$ref" in schema:
        raise ValueError("Schema references are unsupported; inline the referenced property schema")
    kind = schema.get("type")
    if kind == "enum" or (kind is None and "enum" in schema):
        kind = "string"
    if isinstance(kind, list):
        kind = next((value for value in kind if value != "null"), "string")
    output: dict[str, Any] = {}
    if schema.get("description"):
        output["description"] = schema["description"]
    if kind == "object":
        properties = {name: strict_schema(value) for name, value in (schema.get("properties") or {}).items()}
        return output | {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if kind == "array":
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {"type": "string"}
        output |= {"type": "array", "items": strict_schema(items)}
        return {"anyOf": [output, {"type": "null"}]}
    output["type"] = kind or "string"
    enum = schema.get("enum") or schema.get("emum")
    if enum:
        choices = [value for value in enum if value is not None]
        if not choices:
            raise ValueError("A string enum needs at least one non-null choice")
        if all(isinstance(value, str) for value in choices):
            output["enum"] = choices
    return {"anyOf": [output, {"type": "null"}]}


def media_type(data: bytes) -> str:
    """Identify the supported image format from its signature, not the file name."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise ValueError("Use a PNG or JPEG document image")


def build_body(image: bytes, schema: dict[str, Any], model: str = MODEL, *, show: bool = False) -> dict[str, Any]:
    """The measured one-image request, optionally with the image bytes replaced by their sha256.

    The schema text, descriptions included, goes into the system message. SIE applies response_format only as a
    decoding grammar, so without it the model never reads the field descriptions.
    """
    if schema.get("type") != "object":
        raise ValueError("The document schema must have an object root")
    mime = media_type(image)
    encoded = f"<image sha256:{sha256(image)}>" if show else base64.b64encode(image).decode()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM + "\n\nJSON schema:\n" + json.dumps(schema, indent=2)},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]},
        ],
        "temperature": 0,
        "presence_penalty": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fields", "schema": strict_schema(schema), "strict": True},
        },
    }


def readiness_body(model: str) -> dict[str, Any]:
    """The study's text-only readiness request; it is not scored."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the JSON."},
            {"role": "user", "content": "Readiness check (not a study document). Set ok to true."},
        ],
        "temperature": 0,
        "presence_penalty": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ready",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
    }


# ---- the published sets -------------------------------------------------------------------------------------------


def body_line(slot: str, doc_id: str, image_sha256: str, body: dict[str, Any]) -> bytes:
    """One line of the published bodies file, the canonical form its sha256 is taken over."""
    return (json.dumps({"slot": slot, "id": doc_id, "image_sha256": image_sha256, "body": body}) + "\n").encode()


def build_set(set_name: str, model: str, out: Path, cache: Path, *, write_bodies: bool) -> tuple[dict, SetInputs]:
    """Build every body of a set in published order, write the readable form and hash both forms."""
    rows = load_set(set_name)
    images = prepare_images(set_name, cache)
    omni = load_omni_rows(cache)
    full_hash, show_hash = hashlib.sha256(), hashlib.sha256()
    full_file = (out / "bodies.jsonl").open("wb") if write_bodies else None
    try:
        with (out / "bodies.show.jsonl").open("wb") as show_file:
            for row in rows:
                image = images[row["id"]].read_bytes()
                digest = sha256(image)
                if digest != row["image_sha256"]:
                    raise SystemExit(f"{row['id']}: image sha256 {digest} is not the published {row['image_sha256']}")
                schema = json.loads(omni[row["id"]]["json_schema"])
                line = body_line(row["slot"], row["id"], digest, build_body(image, schema, model))
                full_hash.update(line)
                if full_file:
                    full_file.write(line)
                show = body_line(row["slot"], row["id"], digest, build_body(image, schema, model, show=True))
                show_hash.update(show)
                show_file.write(show)
    finally:
        if full_file:
            full_file.close()
    spec = PINS["sets"][set_name]
    comparable = model == MODEL
    check = {
        "set": set_name,
        "model": model,
        "rows": len(rows),
        "study_rows": sum(1 for row in rows if row["slot"] == "study"),
        "images_verified_against_published_manifest": len(rows),
        "bodies_sha256": full_hash.hexdigest(),
        "expected_bodies_sha256": spec["bodies_sha256"],
        "bodies_match": full_hash.hexdigest() == spec["bodies_sha256"] if comparable else None,
        "show_sha256": show_hash.hexdigest(),
        "expected_show_sha256": spec["bodies_show_sha256"],
        "show_match": show_hash.hexdigest() == spec["bodies_show_sha256"] if comparable else None,
        "note": None if comparable else "the model differs from the measured id, so the hashes are not comparable",
    }
    return check, (rows, images, omni)


# ---- sending ------------------------------------------------------------------------------------------------------


class Sender:
    """One SIEClient per worker thread. Only a failure with no HTTP response is resent, at most twice."""

    def __init__(self, base_url: str, api_key: str | None, read_timeout_s: float, *, wait_for_capacity: bool) -> None:
        try:
            version = importlib.metadata.version("sie-sdk")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit("Sending needs sie-sdk>=0.9.0: use `uv run`, or pip install 'sie-sdk>=0.9.0'") from exc
        if tuple(int(part) for part in version.split(".")[:2]) < (0, 9):
            raise SystemExit(f"sie-sdk {version} is installed; sending needs sie-sdk>=0.9.0 (per-call read timeout)")
        from sie_sdk import SIEClient, SIEConnectionError  # optional: dry runs need only the stdlib

        self.sdk_version = version
        self._make = lambda: SIEClient(base_url, api_key=api_key)
        self._transport_error = SIEConnectionError
        self._wait = wait_for_capacity
        self._read_timeout = read_timeout_s
        self._local = threading.local()
        self._clients: list[Any] = []
        self._lock = threading.Lock()

    def client(self) -> Any:
        if not hasattr(self._local, "client"):
            self._local.client = self._make()
            with self._lock:
                self._clients.append(self._local.client)
        return self._local.client

    def send(self, slot: str, doc_id: str, body: dict[str, Any]) -> dict[str, Any]:
        params = {key: value for key, value in body.items() if key not in ("model", "messages")}
        attempts: list[dict[str, Any]] = []
        response = None
        for _ in range(3):
            started_unix, started = time.time(), time.perf_counter()
            error, resend = None, False
            try:
                response = dict(
                    self.client().chat_completions(
                        body["model"],
                        body["messages"],
                        **params,
                        max_oom_retries=0,
                        wait_for_capacity=self._wait,
                        read_timeout_s=self._read_timeout,
                    )
                )
            except self._transport_error as exc:
                error, resend = f"transport: {exc!r}"[:600], True
            except Exception as exc:  # noqa: BLE001 - kept as a scored-zero row, never resent
                error = f"{type(exc).__name__}: {exc}"[:600]
            attempts.append({"t_start_unix": started_unix, "latency_s": time.perf_counter() - started, "error": error})
            if not resend:
                break
        return {
            "slot": slot,
            "id": doc_id,
            "attempts": attempts,
            "latency_s": attempts[-1]["latency_s"],
            "error": attempts[-1]["error"],
            "response": response,
        }

    def close(self) -> None:
        for client in self._clients:
            client.close()


def grader_row(doc_id: str, record: dict[str, Any] | None) -> dict[str, Any]:
    """The published row rule: an error, no content or a capped reply becomes an error row (score 0)."""
    text, error = None, None
    if record is None:
        error = "unattempted"
    elif record["error"]:
        error = record["error"]
    else:
        try:
            choice = record["response"]["choices"][0]
            finish = choice.get("finish_reason")
            text = (choice.get("message") or {}).get("content")
        except (KeyError, IndexError, TypeError):
            finish, error = None, "malformed response"
        if error is None and finish in ("length", "max_tokens"):
            error = "capped"
        elif error is None and text is None:
            error = "no content"
    row: dict[str, Any] = {"id": doc_id, "text": text if not error else (text or "")}
    if error:
        row["error"] = error
    return row


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def run_set(args: argparse.Namespace, sender: Sender, inputs: SetInputs) -> None:
    """Send every study document of the set, write the grader rows and score them over the whole set."""
    out: Path = args.out
    rows, images, omni = inputs
    study = [row for row in rows if row["slot"] == "study"]
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with (out / "outcomes.jsonl").open("w", encoding="utf-8") as events:

        def record(outcome: dict[str, Any]) -> None:
            with lock:
                events.write(json.dumps(outcome, ensure_ascii=False) + "\n")
                events.flush()

        if not args.no_readiness:
            ready = sender.send("readiness", "readiness-1", readiness_body(args.model))
            record(ready)
            if ready["error"]:
                raise SystemExit(f"The readiness request failed: {ready['error']}")

        def one(row: dict[str, Any]) -> dict[str, Any]:
            schema = json.loads(omni[row["id"]]["json_schema"])
            return sender.send("study", row["id"], build_body(images[row["id"]].read_bytes(), schema, args.model))

        started = time.time()
        with ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(one, row) for row in study]
            for n, future in enumerate(as_completed(futures), 1):
                outcome = future.result()
                results[outcome["id"]] = outcome
                record(outcome)
                if n % 25 == 0 or outcome["error"]:
                    status = f"error: {outcome['error']}" if outcome["error"] else "ok"
                    print(f"{n}/{len(study)} {outcome['id']} {status}", flush=True)
        wall = time.time() - started

    ids = [row["id"] for row in study]
    replies = [grader_row(doc_id, results.get(doc_id)) for doc_id in ids]
    with (out / "replies.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(reply) + "\n" for reply in replies)
    per, summary = score.score_set(args.set, score.read_rows(out / "replies.jsonl"), args.cache)
    (out / "scores.json").write_text(json.dumps({"rerun": per}, indent=1) + "\n", encoding="utf-8")
    ok = [results[doc_id] for doc_id in ids if doc_id in results and not results[doc_id]["error"]]
    usage = [outcome["response"].get("usage") or {} for outcome in ok]
    finish: dict[str, int] = {}
    for outcome in ok:
        reason = str((outcome["response"].get("choices") or [{}])[0].get("finish_reason"))
        finish[reason] = finish.get(reason, 0) + 1
    latencies = [outcome["latency_s"] for outcome in ok]
    report = {
        "set": args.set,
        "model": args.model,
        "measured_model": MODEL,
        "sdk": sender.sdk_version,
        "concurrency": args.concurrency,
        "documents": len(ids),
        "omni_mean_failure_inclusive": summary["mean"],
        "published_mean": PINS["sets"][args.set]["published_mean"],
        "parsed": summary["parsed"],
        "error_rows": sum(1 for reply in replies if reply.get("error")),
        "capped": sum(1 for reply in replies if reply.get("error") == "capped"),
        "transport_resends": sum(len(outcome["attempts"]) - 1 for outcome in results.values()),
        "finish_reasons": finish,
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "wall_s": wall,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def run_document(args: argparse.Namespace) -> None:
    """Show or send one document with its own schema."""
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    image = args.image.read_bytes()
    if args.dry_run:
        print(json.dumps(build_body(image, schema, args.model, show=True), indent=2))
        return
    sender = Sender(args.base_url, api_key(args), args.read_timeout, wait_for_capacity=args.wait_for_capacity)
    try:
        outcome = sender.send("document", args.image.name, build_body(image, schema, args.model))
    finally:
        sender.close()
    row = grader_row(args.image.name, outcome)
    if row.get("error") == "capped":
        raise SystemExit("The output limit was reached; this is not a complete extraction")
    if row.get("error"):
        raise SystemExit(f"The request failed: {row['error']}")
    try:
        parsed = json.loads(row["text"])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"The reply is not JSON: {row['text'][:500]!r}") from exc
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


def api_key(args: argparse.Namespace) -> str | None:
    return os.environ.get(args.api_key_env) if args.api_key_env else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--set", choices=SETS, help="rerun a published set: pilot (100) or confirm (546)")
    target.add_argument("--image", type=Path, help="one PNG or JPEG document image")
    parser.add_argument("--schema", type=Path, help="the JSON schema to fill from --image")
    parser.add_argument(
        "--base-url", help="SIE endpoint, for example http://127.0.0.1:8080 or https://api.superlinked.com"
    )
    parser.add_argument(
        "--api-key-env", default="SIE_API_KEY", help="environment variable that holds the API key (default SIE_API_KEY)"
    )
    parser.add_argument("--model", default=MODEL, help=f"model id to request (measured: {MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="build the requests offline and send nothing")
    parser.add_argument("--out", type=Path, help="run folder for --set (default runs/<set>)")
    parser.add_argument("--concurrency", type=int, default=8, help="requests in flight for --set (measured: 8)")
    parser.add_argument("--write-bodies", action="store_true", help="with --set, also write the full bodies (large)")
    parser.add_argument("--cache", type=Path, default=DATA, help="download folder (default data/)")
    parser.add_argument("--read-timeout", type=float, default=1900.0, help="seconds to wait for one reply")
    parser.add_argument(
        "--no-wait-for-capacity",
        dest="wait_for_capacity",
        action="store_false",
        help="fail instead of waiting while the endpoint provisions capacity",
    )
    parser.add_argument("--no-readiness", action="store_true", help="skip the unscored text-only readiness request")
    args = parser.parse_args()
    if args.image and not args.schema:
        parser.error("--image needs --schema")
    if not args.dry_run and not args.base_url:
        parser.error("--base-url is required unless --dry-run")
    if args.image:
        run_document(args)
        return

    args.out = args.out or HERE / "runs" / args.set
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and any((args.out / name).exists() for name in ("outcomes.jsonl", "replies.jsonl")):
        raise SystemExit(f"{args.out} already holds a run; pass a new --out folder")
    sender = None
    if not args.dry_run:  # fail on a missing or old SDK before any download
        sender = Sender(args.base_url, api_key(args), args.read_timeout, wait_for_capacity=args.wait_for_capacity)
    try:
        check, inputs = build_set(args.set, args.model, args.out, args.cache, write_bodies=args.write_bodies)
        (args.out / "dry_run.json").write_text(json.dumps(check, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(check, indent=1), flush=True)
        if args.dry_run:
            if check["bodies_match"] is False or check["show_match"] is False:
                raise SystemExit(1)
            return
        run_set(args, sender, inputs)
    finally:
        if sender:
            sender.close()


if __name__ == "__main__":
    main()
