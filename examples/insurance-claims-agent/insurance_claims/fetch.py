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

from insurance_claims.config import DATA_DIR, SOURCE_DIR, Source, load_config

console = Console()


def _download(source: Source) -> tuple[bytes, str]:
    request = urllib.request.Request(
        source.url,
        headers={
            "Accept": f"{source.media_type},*/*",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(), "publisher"
    except urllib.error.HTTPError as error:
        curl = shutil.which("curl")
        if error.code not in {403, 429} or curl is None:
            raise
        try:
            response = subprocess.run(
                [curl, "-fsSL", "--retry", "3", source.url],
                check=True,
                capture_output=True,
                timeout=180,
            )
            return response.stdout, "publisher"
        except subprocess.CalledProcessError:
            if source.fixture_path is None or not source.fixture_path.exists():
                raise
            return source.fixture_path.read_bytes(), "bundled-government-source"


def _validate(source: Source, payload: bytes) -> None:
    if source.media_type == "application/pdf" and not payload.startswith(b"%PDF"):
        raise ValueError(f"{source.url} did not return a PDF")
    if source.media_type == "image/jpeg" and not payload.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"{source.url} did not return a JPEG")


def fetch_sources(*, refresh: bool) -> Path:
    config = load_config()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in config.sources:
        if source.path.exists() and not refresh:
            payload = source.path.read_bytes()
            status = "cached"
            retrieval = "cache"
        else:
            payload, retrieval = _download(source)
            _validate(source, payload)
            source.path.write_bytes(payload)
            status = "downloaded"
        checksum = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "slug": source.slug,
                "title": source.title,
                "file_name": source.file_name,
                "url": source.url,
                "source_page": source.source_page,
                "rights": source.rights,
                "media_type": source.media_type,
                "retrieval": retrieval,
                "bytes": len(payload),
                "sha256": checksum,
            }
        )
        console.print(f"[green]{status:10}[/] {source.slug}  {len(payload) / 1024:.1f} KiB  {checksum[:12]}")

    manifest_path = DATA_DIR / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "sources": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the public claim form, policy, and damage photograph")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    console.print(f"\nSource manifest: {fetch_sources(refresh=args.refresh)}")


if __name__ == "__main__":
    main()
