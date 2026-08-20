from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from .config import CACHE_DIR
from .models import LinkingCase


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        if sha256(destination) != expected_sha256:
            raise RuntimeError(f"Cached source hash mismatch: {destination}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        actual = sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(f"Downloaded source hash mismatch for {url}: expected {expected_sha256}, got {actual}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination_root and destination_root not in member_path.parents:
                raise RuntimeError(f"Archive member escapes extraction directory: {member.filename}")
        bundle.extractall(destination)


def ensure_sources(config: dict[str, Any], *, force: bool = False) -> dict[str, Path]:
    attack_source = config["sources"]["attack"]
    annoctr_source = config["sources"]["annoctr"]
    attack_path = download(
        str(attack_source["url"]),
        CACHE_DIR / f"enterprise-attack-{attack_source['version']}.json",
        str(attack_source["sha256"]),
        force=force,
    )
    archive_path = download(
        str(annoctr_source["url"]),
        CACHE_DIR / f"annoctr-{str(annoctr_source['commit'])[:12]}.zip",
        str(annoctr_source["sha256"]),
        force=force,
    )
    extracted = CACHE_DIR / f"annoctr-{str(annoctr_source['commit'])[:12]}"
    marker = extracted / ".complete"
    if force and extracted.exists():
        shutil.rmtree(extracted)
    if not marker.is_file():
        staging = Path(tempfile.mkdtemp(prefix=".annoctr-", dir=CACHE_DIR))
        try:
            _safe_extract(archive_path, staging)
            if extracted.exists():
                shutil.rmtree(extracted)
            staging.rename(extracted)
            marker.write_text(f"{sha256(archive_path)}\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {"attack": attack_path, "annoctr": extracted, "annoctr_archive": archive_path}


def _technique_id(label_link: str) -> str | None:
    match = re.search(r"/techniques/(T\d{4})(?:/(\d{3}))?", label_link)
    if match is None:
        return None
    return match.group(1) if match.group(2) is None else f"{match.group(1)}.{match.group(2)}"


def _evidence(row: dict[str, Any]) -> str:
    left = str(row.get("_context_left", row.get("context_left", "")))
    mention = str(row.get("mention", ""))
    right = str(row.get("_context_right", row.get("context_right", "")))
    evidence = f"{left}{mention}{right}".strip()
    return " ".join(evidence.split())


def _case_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("document", "")),
        str(row.get("mention", "")),
        _evidence(row),
        str(row.get("sentence_left", "")),
        str(row.get("sentence_right", "")),
    )


def _case_id(key: Iterable[str]) -> str:
    encoded = json.dumps(list(key), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _find_split_file(annoctr_root: Path, split: str) -> Path:
    matches = list(annoctr_root.glob(f"*/AnnoCTR/linking_mitre_only/{split}.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one AnnoCTR {split} linking file under {annoctr_root}, found {len(matches)}")
    return matches[0]


def find_annoctr_catalog(annoctr_root: Path) -> Path:
    matches = list(annoctr_root.glob("*/AnnoCTR/entities/mitre_entity.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one AnnoCTR MITRE entity catalog under {annoctr_root}, found {len(matches)}")
    return matches[0]


def find_annoctr_report(annoctr_root: Path, split: str, document: str) -> Path:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", document):
        raise ValueError("document must be one safe AnnoCTR document name")
    matches = list(annoctr_root.glob(f"*/AnnoCTR/text/{split}/{document}.txt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one AnnoCTR report named {document} in {split}, found {len(matches)}")
    return matches[0]


def load_linking_cases(annoctr_root: Path, split: str) -> list[LinkingCase]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    rows: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with _find_split_file(annoctr_root, split).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("entity_type") != "TECHNIQUE":
                continue
            technique_id = _technique_id(str(row.get("label_link", "")))
            if technique_id is None:
                continue
            row["technique_id"] = technique_id
            rows[_case_key(row)].append(row)
    cases: list[LinkingCase] = []
    for key, group in rows.items():
        cases.append(
            LinkingCase(
                case_id=_case_id(key),
                document=key[0],
                mention=key[1],
                evidence=key[2],
                left_context=key[3],
                right_context=key[4],
                gold_ids=tuple(sorted({str(row["technique_id"]) for row in group})),
                annotation_classes=tuple(sorted({str(row.get("entity_class", "")) for row in group})),
            )
        )
    return sorted(cases, key=lambda item: item.case_id)
