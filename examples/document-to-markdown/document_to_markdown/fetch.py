from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from document_to_markdown.config import DATA_DIR, PDF_DIR, DocumentSource, load_config, select_documents

console = Console()


def _download(document: DocumentSource, destination: Path) -> tuple[bytes, str]:
    request = urllib.request.Request(
        document.url,
        headers={
            "Accept": "application/pdf,*/*",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        retrieval = "publisher"
    except urllib.error.HTTPError as error:
        curl = shutil.which("curl")
        if error.code != 403 or curl is None:
            raise
        try:
            response = subprocess.run(
                [curl, "-fsSL", "--retry", "3", document.url],
                check=True,
                capture_output=True,
                timeout=180,
            )
            payload = response.stdout
            retrieval = "publisher"
        except subprocess.CalledProcessError:
            if document.fixture_path is None or not document.fixture_path.exists():
                raise
            payload = document.fixture_path.read_bytes()
            retrieval = "bundled-government-source"
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"{document.url} did not return a PDF")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload, retrieval


def fetch_documents(slugs: list[str], *, refresh: bool) -> Path:
    config = load_config()
    documents = select_documents(config, slugs)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in documents:
        if document.path.exists() and not refresh:
            payload = document.path.read_bytes()
            status = "cached"
            retrieval = "cache"
        else:
            payload, retrieval = _download(document, document.path)
            status = "downloaded"
        checksum = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "slug": document.slug,
                "title": document.title,
                "publisher": document.publisher,
                "file_name": document.file_name,
                "url": document.url,
                "source_page": document.source_page,
                "rights": document.rights,
                "retrieval": retrieval,
                "bytes": len(payload),
                "sha256": checksum,
            }
        )
        console.print(f"[green]{status:10}[/] {document.slug}  {len(payload) / 1024:.1f} KiB  {checksum[:12]}")

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "documents": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the licensed source PDFs")
    parser.add_argument("slugs", nargs="*", default=["all"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    manifest_path = fetch_documents(args.slugs, refresh=args.refresh)
    console.print(f"\nSource manifest: {manifest_path}")


if __name__ == "__main__":
    main()
