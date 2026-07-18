from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "data" / "prepare_sample_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_sample_corpus", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
prepare_sample_corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_sample_corpus
SPEC.loader.exec_module(prepare_sample_corpus)

INGEST_PATH = Path(__file__).resolve().parent.parent / "python" / "ingest.py"
INGEST_SPEC = importlib.util.spec_from_file_location("vision_doc_ingest", INGEST_PATH)
assert INGEST_SPEC is not None
assert INGEST_SPEC.loader is not None
vision_doc_ingest = importlib.util.module_from_spec(INGEST_SPEC)
sys.modules[INGEST_SPEC.name] = vision_doc_ingest
INGEST_SPEC.loader.exec_module(vision_doc_ingest)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepares_deterministic_verified_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample"

    first_rows = prepare_sample_corpus.prepare(output_dir)
    first_manifest = (output_dir / "pages_manifest.json").read_text()
    second_rows = prepare_sample_corpus.prepare(output_dir)

    assert first_rows == second_rows
    assert first_manifest == (output_dir / "pages_manifest.json").read_text()
    assert json.loads(first_manifest) == first_rows
    assert len(first_rows) == 3
    assert output_dir / "pages_manifest.json" != prepare_sample_corpus.EXAMPLE_ROOT / "data" / "pages_manifest.json"

    expected_hashes = {sample.filename: sample.sha256 for sample in prepare_sample_corpus.SAMPLES}
    copied_files = output_dir.glob("pages/synthetic-docs/*.png")
    assert {path.name: sha256(path) for path in copied_files} == expected_hashes


def test_replaces_output_without_leaving_stale_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample"
    prepare_sample_corpus.prepare(output_dir)
    stale_file = output_dir / "pages" / "synthetic-docs" / "stale.png"
    stale_file.write_bytes(b"stale")

    prepare_sample_corpus.prepare(output_dir)

    assert not stale_file.exists()


def test_hash_failure_preserves_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "sample"
    output_dir.mkdir()
    marker = output_dir / "existing.txt"
    marker.write_text("keep")
    monkeypatch.setattr(
        prepare_sample_corpus,
        "SAMPLES",
        (
            prepare_sample_corpus.Sample(
                filename="receipt.png",
                sha256="0" * 64,
                title="Synthetic Receipt",
            ),
        ),
    )

    with pytest.raises(ValueError, match="sample hash mismatch"):
        prepare_sample_corpus.prepare(output_dir)

    assert marker.read_text() == "keep"


def test_ingest_loads_an_explicit_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "sample_manifest.json"
    rows = [{"page_id": "sample__receipt__p001"}]
    manifest.write_text(json.dumps(rows))

    assert vision_doc_ingest.load_pages(manifest) == rows
