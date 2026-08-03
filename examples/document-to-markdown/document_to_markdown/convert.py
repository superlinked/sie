from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from sie_sdk import SIEClient
from sie_sdk.types import Item

from document_to_markdown.config import RUNS_DIR, load_config, select_documents

console = Console()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_conversion(slugs: list[str], run_id: str | None = None) -> Path:
    config = load_config()
    documents = select_documents(config, slugs)
    selected_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / selected_run_id
    raw_dir = run_dir / "raw"
    markdown_dir = run_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=False)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    client = SIEClient(
        config.cluster.url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    rows = []
    try:
        for document in documents:
            if not document.path.exists():
                raise FileNotFoundError(f"Missing {document.path}. Run `uv run fetch-documents` first.")
            started = time.perf_counter()
            result = client.extract(
                config.cluster.model,
                Item(id=document.slug, document=document.path),
                options={"profile": config.cluster.profile},
                provision_timeout_s=config.cluster.provision_timeout_s,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            if result.get("error"):
                raise RuntimeError(f"{document.slug}: {result['error']}")
            data = result.get("data", {})
            markdown = str(data.get("markdown", ""))
            if not markdown.strip():
                raise RuntimeError(f"{document.slug}: model returned no Markdown")
            raw_path = raw_dir / f"{document.slug}.json"
            raw_path.write_text(
                json.dumps(result, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )
            markdown_path = markdown_dir / f"{document.slug}.md"
            markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            rows.append(
                {
                    "slug": document.slug,
                    "source_file": str(document.path.relative_to(document.path.parent.parent)),
                    "source_url": document.url,
                    "model": config.cluster.model,
                    "profile": config.cluster.profile,
                    "endpoint": config.cluster.url,
                    "duration_ms": duration_ms,
                    "markdown_characters": len(markdown),
                    "raw_output": str(raw_path.relative_to(run_dir)),
                    "markdown_output": str(markdown_path.relative_to(run_dir)),
                }
            )
    finally:
        client.close()

    manifest = {
        "run_id": selected_run_id,
        "run_at": datetime.now(UTC).isoformat(),
        "model": config.cluster.model,
        "endpoint": config.cluster.url,
        "documents": rows,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    table = Table("Document", "Latency", "Markdown")
    for row in rows:
        table.add_row(row["slug"], f"{row['duration_ms']:.1f} ms", f"{row['markdown_characters']:,} chars")
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
