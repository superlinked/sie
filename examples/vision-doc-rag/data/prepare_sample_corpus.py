"""Prepare a small, deterministic sample corpus without downloading PDFs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = EXAMPLE_ROOT / "data" / "sample"
SOURCE_URL = "https://github.com/superlinked/sie/tree/main/examples/document-ocr/data/samples"


@dataclass(frozen=True)
class Sample:
    filename: str
    sha256: str
    title: str


SAMPLES = (
    Sample(
        filename="receipt.png",
        sha256="feb5f42448901d1094af5266c4bda37fd3ecb0e26404a268a82dcd0434d7ade5",
        title="Synthetic Receipt",
    ),
    Sample(
        filename="invoice.png",
        sha256="2e0a5b12f3d626b52304c6ffe4121d8507d39fd03690350f097b222067223f41",
        title="Synthetic Invoice",
    ),
    Sample(
        filename="slide.png",
        sha256="5df0640aeb85a4adca39afac765f9007ae33edf21e845d209c2416c075a02150",
        title="Synthetic Slide",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_sources() -> list[tuple[Sample, Path]]:
    samples_dir = REPO_ROOT / "examples" / "document-ocr" / "data" / "samples"
    verified: list[tuple[Sample, Path]] = []
    for sample in SAMPLES:
        source = samples_dir / sample.filename
        actual_hash = _sha256(source)
        if actual_hash != sample.sha256:
            raise ValueError(f"sample hash mismatch for {source}: expected {sample.sha256}, got {actual_hash}")
        verified.append((sample, source))
    return verified


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict[str, Any]]:
    """Replace ``output_dir`` with a freshly verified three-document corpus."""
    sources = _verified_sources()
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))

    try:
        pages_dir = staging_dir / "pages" / "synthetic-docs"
        pages_dir.mkdir(parents=True)
        rows: list[dict[str, Any]] = []

        for sample, source in sources:
            destination = pages_dir / sample.filename
            shutil.copyfile(source, destination)
            copied_hash = _sha256(destination)
            if copied_hash != sample.sha256:
                raise ValueError(
                    f"copied sample hash mismatch for {destination}: expected {sample.sha256}, got {copied_hash}"
                )
            slug = source.stem
            rows.append(
                {
                    "page_id": f"synthetic-docs__{slug}__p001",
                    "client": "synthetic-docs",
                    "title": sample.title,
                    "publisher": "Superlinked",
                    "license": "Apache-2.0",
                    "source_url": SOURCE_URL,
                    "source_pdf": sample.filename,
                    "source_pdf_path": str(source.relative_to(REPO_ROOT)),
                    "page_number": 1,
                    "image_path": f"sample/pages/synthetic-docs/{sample.filename}",
                }
            )

        (staging_dir / "pages_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")

        backup_dir: Path | None = None
        if output_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-backup-", dir=output_dir.parent))
            backup_dir.rmdir()
            output_dir.rename(backup_dir)

        try:
            staging_dir.rename(output_dir)
        except OSError:
            if backup_dir is not None:
                backup_dir.rename(output_dir)
            raise
        else:
            if backup_dir is not None:
                shutil.rmtree(backup_dir)
        return rows
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main() -> None:
    rows = prepare()
    manifest = DEFAULT_OUTPUT_DIR / "pages_manifest.json"
    print(f"Prepared {len(rows)} verified sample pages")
    print(f"Wrote page manifest to {manifest}")


if __name__ == "__main__":
    main()
