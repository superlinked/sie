from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from sie_sdk import SIEClient
from sie_sdk.types import Item

from document_to_markdown.canonical import canonical_sha256
from document_to_markdown.config import RUNS_DIR, load_config, select_documents
from document_to_markdown.record import MARKDOWN_FILE_TRANSFORM, build_entry, write_calls

console = Console()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _plain(value: Any) -> Any:
    """Round-trip through JSON so the recorded value holds only JSON types."""
    return json.loads(json.dumps(value, default=_json_default))


def run_conversion(slugs: list[str], run_id: str | None = None) -> Path:
    config = load_config()
    documents = select_documents(config, slugs)
    selected_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / selected_run_id
    markdown_dir = run_dir / "markdown"
    payload_dir = run_dir / "payloads"
    run_dir.mkdir(parents=True, exist_ok=False)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    client = SIEClient(
        config.cluster.url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    started_at = datetime.now(UTC)
    try:
        for document in documents:
            if not document.path.exists():
                raise FileNotFoundError(f"Missing {document.path}. Run `uv run fetch-documents` first.")
            source_bytes = document.path.read_bytes()
            started = time.perf_counter()
            result = client.extract(
                config.cluster.model,
                Item(id=document.slug, document=document.path),
                options={"profile": config.cluster.profile},
                provision_timeout_s=config.cluster.provision_timeout_s,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            # Read straight after the call: the SDK keeps these per request.
            model_revision = client.last_model_revision
            retry_count = client.last_retry_count
            if result.get("error"):
                raise RuntimeError(f"{document.slug}: {result['error']}")
            data = result.get("data", {})
            markdown = str(data.get("markdown", ""))
            if not markdown.strip():
                raise RuntimeError(f"{document.slug}: model returned no Markdown")

            markdown_path = markdown_dir / f"{document.slug}.md"
            markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")

            entries.append(
                build_entry(
                    slug=document.slug,
                    request={
                        "method": "SIEClient.extract",
                        "endpoint": config.cluster.url,
                        "model": config.cluster.model,
                        "options": {"profile": config.cluster.profile},
                        "item": {
                            "id": document.slug,
                            "document": str(document.path.relative_to(document.path.parent.parent)),
                            "document_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "document_bytes": len(source_bytes),
                        },
                    },
                    response=_plain(result),
                    duration_ms=duration_ms,
                    model_revision=model_revision,
                    retry_count=retry_count,
                    markdown_sha256=hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
                    payload_dir=payload_dir,
                )
            )
            rows.append(
                {
                    "slug": document.slug,
                    "source_file": str(document.path.relative_to(document.path.parent.parent)),
                    "source_url": document.url,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "model": config.cluster.model,
                    "model_revision": model_revision,
                    "profile": config.cluster.profile,
                    "endpoint": config.cluster.url,
                    "duration_ms": duration_ms,
                    "markdown_characters": len(markdown),
                    "markdown_output": str(markdown_path.relative_to(run_dir)),
                }
            )
    finally:
        client.close()

    calls = write_calls(run_dir, entries)
    revisions = sorted({row["model_revision"] for row in rows if row["model_revision"]})
    manifest = {
        "schema_version": "1.0",
        "run_id": selected_run_id,
        "run_at": datetime.now(UTC).isoformat(),
        "run_started_at": started_at.isoformat(),
        "model": config.cluster.model,
        "profile": config.cluster.profile,
        "model_revisions": revisions,
        "endpoint": config.cluster.url,
        "calls": calls,
        "scored_markdown_transform": MARKDOWN_FILE_TRANSFORM,
        "documents": rows,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    # run-manifest.json, the name verify-run reads. A run bundle fetched from
    # the dataset keeps manifest.json for the dataset's own digests, so the two
    # cannot share a name without one shadowing the other.
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table = Table("Document", "Latency", "Markdown", "Model revision")
    for row in rows:
        table.add_row(
            row["slug"],
            f"{row['duration_ms']:.1f} ms",
            f"{row['markdown_characters']:,} chars",
            (row["model_revision"] or "not reported")[:12],
        )
    console.print(table)
    console.print(f"Run bundle: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert source PDFs through the SIE Docling adapter")
    parser.add_argument("slugs", nargs="*", default=["all"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_conversion(args.slugs, args.run_id)


if __name__ == "__main__":
    main()
