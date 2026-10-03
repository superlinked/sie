from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Written by `python3 fetch.py`, which owns this directory and replaces it
# wholesale. Nothing else writes into it, so the digests the fetch checked stay
# true of what is on disk.
EVIDENCE_DIR = ROOT / "data"
# The file `python3 fetch.py` drops in the directory it wrote, naming the
# dataset and revision. One definition, because "is this a fetched bundle" is
# asked in two places and two different answers to it is the bug this guards.
#
# fetch.py spells the name again rather than importing it. That is deliberate:
# it has to run on a bare `python3` with nothing installed, which is the whole
# point of it, and importing this package would need `uv sync` first. The same
# name in the nine other examples' fetch.py is why it is a fixed string rather
# than something this example gets to choose.
EVIDENCE_MARKER = ".sie-evidence"


def is_fetched_bundle(run_dir: Path) -> bool:
    """True when `python3 fetch.py` wrote this directory.

    A fetched bundle's files are pinned by digests the fetch checked, and it
    carries its own source provenance. A bundle `convert-documents` wrote has
    neither.
    """
    return (run_dir / EVIDENCE_MARKER).is_file()


# Written by `uv run fetch-documents`, and kept outside the fetched tree so a
# re-fetch does not delete the PDFs and a PDF download does not overwrite the
# recorded provenance.
PDF_DIR = ROOT / "pdfs"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class DocumentChecks:
    required_text: tuple[str, ...]
    ordered_text: tuple[str, ...]
    minimum_markdown_tables: int


@dataclass(frozen=True)
class DocumentSource:
    slug: str
    title: str
    publisher: str
    file_name: str
    url: str
    source_page: str
    rights: str
    fixture_file: str | None
    checks: DocumentChecks

    @property
    def path(self) -> Path:
        return PDF_DIR / self.file_name

    @property
    def fixture_path(self) -> Path | None:
        return ROOT / self.fixture_file if self.fixture_file else None


@dataclass(frozen=True)
class ClusterConfig:
    url: str
    api_key: str
    model: str
    profile: str
    request_timeout_s: float
    provision_timeout_s: float


@dataclass(frozen=True)
class AppConfig:
    cluster: ClusterConfig
    documents: tuple[DocumentSource, ...]


def _load_yaml() -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain an object")
    return value


def load_config() -> AppConfig:
    load_dotenv(ROOT / ".env")
    raw = _load_yaml()
    cluster_raw = raw["cluster"]
    cluster = ClusterConfig(
        url=os.environ.get("SIE_CLUSTER_URL", cluster_raw["url"]).rstrip("/"),
        api_key=os.environ.get("SIE_API_KEY", cluster_raw.get("api_key", "")),
        model=str(cluster_raw["model"]),
        profile=str(cluster_raw.get("profile", "default")),
        request_timeout_s=float(cluster_raw.get("request_timeout_s", 900)),
        provision_timeout_s=float(cluster_raw.get("provision_timeout_s", 900)),
    )
    documents = tuple(
        DocumentSource(
            slug=row["slug"],
            title=row["title"],
            publisher=row["publisher"],
            file_name=row["file_name"],
            url=row["url"],
            source_page=row["source_page"],
            rights=row["rights"],
            fixture_file=row.get("fixture_file"),
            checks=DocumentChecks(
                required_text=tuple(row["checks"].get("required_text", [])),
                ordered_text=tuple(row["checks"].get("ordered_text", [])),
                minimum_markdown_tables=int(row["checks"].get("minimum_markdown_tables", 0)),
            ),
        )
        for row in raw["documents"]
    )
    return AppConfig(cluster=cluster, documents=documents)


def select_documents(config: AppConfig, slugs: list[str]) -> tuple[DocumentSource, ...]:
    if not slugs or slugs == ["all"]:
        return config.documents
    by_slug = {document.slug: document for document in config.documents}
    missing = sorted(set(slugs) - set(by_slug))
    if missing:
        raise ValueError(f"Unknown document slug(s): {', '.join(missing)}")
    return tuple(by_slug[slug] for slug in slugs)
