"""Prepare the three-file ABI course corpus for vision-doc-rag."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PACK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACK_ROOT.parents[1]
PACK_MANIFEST = PACK_ROOT / "course-pack.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output_data_dir: Path) -> list[dict[str, Any]]:
    pack = json.loads(PACK_MANIFEST.read_text())
    project = next(project for project in pack["projects"] if project["id"] == "vision-document-rag")
    fixtures = project["sample"]["fixtures"]
    if len(fixtures) > project["sample"]["max_corpus_items"]:
        raise ValueError("vision sample exceeds the course corpus cap")

    pages_dir = output_data_dir / "pages" / "course"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        source = (PACK_ROOT / fixture["path"]).resolve()
        if _sha256(source) != fixture["sha256"]:
            raise ValueError(f"fixture hash mismatch: {source}")
        destination = pages_dir / source.name
        shutil.copyfile(source, destination)
        slug = source.stem
        rows.append(
            {
                "page_id": f"course-docs__{slug}__p001",
                "client": "course-docs",
                "title": f"Synthetic {slug.replace('-', ' ').title()}",
                "publisher": "Superlinked",
                "license": "Apache-2.0",
                "source_url": ("https://github.com/superlinked/sie/tree/main/examples/document-ocr/data/samples"),
                "source_pdf": source.name,
                "source_pdf_path": str(source.relative_to(REPO_ROOT)),
                "page_number": 1,
                "image_path": f"pages/course/{source.name}",
            }
        )

    output_data_dir.mkdir(parents=True, exist_ok=True)
    (output_data_dir / "pages_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "vision-doc-rag" / "data",
        help="vision-doc-rag data directory (defaults to the repository example)",
    )
    return parser.parse_args()


def main() -> None:
    rows = prepare(parse_args().output_data_dir.resolve())
    print(f"Prepared {len(rows)} verified course pages")


if __name__ == "__main__":
    main()
