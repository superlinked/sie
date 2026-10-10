from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from dotenv import load_dotenv
from sie_sdk import SIEClient
from sie_sdk.types import Item

from document_to_markdown.config import ROOT
from document_to_markdown.ocr import DEFAULT_MODEL

OPTIONS = {"max_new_tokens": 12288, "temperature": 0.1, "top_p": 1.0}


class MarkdownOutputError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_path(value: str, suffix: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.suffix != suffix
        or str(path) != value
        or ":" in path.parts[0]
    ):
        raise ValueError(f"Expected a relative {suffix} path")
    return path


def image_bytes(root: Path, page: dict[str, Any]) -> bytes:
    path = root / relative_path(page["png"], ".png")
    if not path.resolve(strict=True).is_relative_to(root.resolve()):
        raise ValueError("Image must stay inside the input directory")
    raw = path.read_bytes()
    if digest(raw) != page["png_sha256"]:
        raise ValueError("Image checksum differs from the frozen manifest")
    if "png_bytes" in page and len(raw) != page["png_bytes"]:
        raise ValueError("Image length differs from the frozen manifest")
    return raw


def load_pages(manifest: Path, root: Path, selection: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = manifest.read_bytes()
    pages = json.loads(raw)["pages"]
    frame = [page for page in pages if page.get("role", "frame") == "frame"]
    ids = [page["pdf"] for page in frame]
    if len(ids) != len(set(ids)):
        raise ValueError("Frame contains duplicate PDF IDs")
    selected_raw = selection.read_bytes() if selection else None
    if selected_raw is not None:
        selected = json.loads(selected_raw)
        if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
            raise ValueError("Selection must be a JSON list of PDF IDs")
        if len(selected) != len(set(selected)) or not set(selected).issubset(ids):
            raise ValueError("Selection contains duplicate or unknown PDF IDs")
        selected_ids = set(selected)
        frame = [page for page in frame if page["pdf"] in selected_ids]
    if not frame:
        raise ValueError("Frame selection is empty")
    for page in frame:
        relative_path(page["pdf"], ".pdf")
        image_bytes(root, page)
    return frame, {
        "manifest_sha256": digest(raw),
        "selection_sha256": digest(selected_raw) if selected_raw is not None else None,
        "pages": frame,
    }


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json_bytes(row))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class OnePost(httpx.BaseTransport):
    """Keep SDK parsing while rejecting a second physical POST for a page."""

    def __init__(self, url: str, model: str, inner: httpx.BaseTransport | None = None) -> None:
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        self.path = httpx.URL(url).path.rstrip("/") + "/v1/extract/" + model
        self.begin()

    def begin(self) -> None:
        self.sends = 0
        self.body_sha256: str | None = None
        self.response: tuple[int, str | None, bytes] | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if request.url.path != self.path or self.sends:
                raise RuntimeError("A page permits only one extraction POST")
            self.sends += 1
            self.body_sha256 = digest(request.read())
        elif request.method != "GET":
            raise RuntimeError("Unexpected request method")
        response = self.inner.handle_request(request)
        if request.method == "POST":
            raw = response.read()
            self.response = (response.status_code, response.headers.get("content-type"), raw)
        return response

    def close(self) -> None:
        self.inner.close()


def markdown_text(reply: dict[str, Any]) -> str:
    if reply.get("error"):
        raise MarkdownOutputError("Extraction returned an error")
    text = reply.get("data", {}).get("markdown")
    if text is None:
        entities = reply.get("entities", [])
        matching = [entity["text"] for entity in entities if entity.get("label") == "markdown"]
        text = matching[0] if len(matching) == 1 else None
    if not isinstance(text, str) or not text.strip():
        raise MarkdownOutputError("Extraction returned no Markdown")
    return text


def run_frame(
    manifest: Path,
    root: Path,
    out: Path,
    *,
    sie_url: str,
    model: str = DEFAULT_MODEL,
    selection: Path | None = None,
    timeout_s: float = 360,
    served_revision: str | None = None,
    inner: httpx.BaseTransport | None = None,
) -> list[dict[str, Any]]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Timeout must be a finite positive number")
    pages, plan = load_pages(manifest, root, selection)
    out.mkdir(parents=True, exist_ok=False)
    sync_directory(out.parent)
    for name in ("intents", "responses", "sdk", "markdown"):
        (out / name).mkdir()
    plan.update(model=model, options=OPTIONS, timeout_s=timeout_s, declared_served_revision=served_revision)
    save_new(out / "plan.json", json_bytes(plan))
    rows = [
        {
            "ordinal": ordinal,
            "pdf": page["pdf"],
            "png_sha256": page["png_sha256"],
            "category": page.get("category"),
            "family": page.get("family"),
            "status": "known_unattempted",
            "reason": "not_started",
            "physical_sends": 0,
        }
        for ordinal, page in enumerate(pages)
    ]
    rows_path = out / "rows.jsonl"
    save_rows(rows_path, rows)
    transport = OnePost(sie_url, model, inner)
    http = httpx.Client(base_url=sie_url, transport=transport, follow_redirects=False, trust_env=False)
    client = None
    try:
        client = SIEClient(
            sie_url, api_key=os.environ.get("SIE_API_KEY") or None, timeout_s=timeout_s, http_client=http
        )
        for page, row in zip(pages, rows, strict=True):
            raw = image_bytes(root, page)
            transport.begin()
            stem = f"{row['ordinal']:06d}"
            row.update(status="attempt_status_unknown", reason="durable_call_intent")
            save_new(out / "intents" / f"{stem}.json", json_bytes(row))
            save_rows(rows_path, rows)
            started = time.perf_counter()
            output_failure = False
            try:
                reply = client.extract(
                    model,
                    Item(id=page["pdf"], images=[raw]),
                    options=dict(OPTIONS),
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=timeout_s,
                )
                save_new(out / "sdk" / f"{stem}.json", json_bytes(reply))
                if reply.get("id") is not None and reply["id"] != page["pdf"]:
                    raise ValueError("Response ID differs from the requested page")
                markdown = markdown_text(reply).encode("utf-8")
                path = PurePosixPath("markdown") / relative_path(page["pdf"], ".pdf").with_suffix(".md")
                destination = out / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                save_new(destination, markdown)
                row.update(status="received", reason=None, markdown=str(path), markdown_sha256=digest(markdown))
            except Exception as error:
                status = (
                    "known_unattempted"
                    if transport.sends == 0
                    else ("failed" if transport.response is not None else "attempt_status_unknown")
                )
                row.update(status=status, reason=type(error).__name__)
                output_failure = isinstance(error, MarkdownOutputError) and (
                    transport.response is not None and transport.response[0] == 200
                )
            finally:
                row.update(
                    physical_sends=transport.sends,
                    request_body_sha256=transport.body_sha256,
                    client_wall_s=time.perf_counter() - started,
                )
                if transport.response is not None:
                    status_code, content_type, response = transport.response
                    save_new(out / "responses" / f"{stem}.bin", response)
                    row.update(
                        http_status=status_code, response_content_type=content_type, response_sha256=digest(response)
                    )
                save_rows(rows_path, rows)
            if row["status"] != "received" and not output_failure:
                for tail in rows[row["ordinal"] + 1 :]:
                    tail["reason"] = "stopped_after_nonreceived_page"
                break
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            http.close()
            save_rows(rows_path, rows)
    return rows


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run a frozen page-image frame against any compatible SIE URL")
    parser.add_argument("manifest", type=Path, help="JSON manifest with a pages list")
    parser.add_argument("--input-root", type=Path, help="Root for manifest PNG paths; defaults to manifest directory")
    parser.add_argument("--selection", type=Path, help="Frozen JSON list of PDF IDs, selected before viewing outputs")
    parser.add_argument(
        "--sie-url", default=os.environ.get("SIE_URL", os.environ.get("SIE_CLUSTER_URL", "http://localhost:8080"))
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-revision", help="Optional operator-declared checkpoint, not remotely verified")
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument(
        "--out", type=Path, required=True, help="Fresh run directory; existing directories are never resumed"
    )
    args = parser.parse_args()
    rows = run_frame(
        args.manifest,
        args.input_root or args.manifest.parent,
        args.out,
        sie_url=args.sie_url,
        model=args.model,
        selection=args.selection,
        timeout_s=args.timeout,
        served_revision=args.served_revision,
    )
    print(f"Received {sum(row['status'] == 'received' for row in rows)}/{len(rows)} planned pages; see rows.jsonl")
    if any(row["status"] != "received" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
