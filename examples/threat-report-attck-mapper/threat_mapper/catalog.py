from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from .models import Technique

CITATION_RE = re.compile(r"\(Citation:[^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
TAG_RE = re.compile(r"<[^>]+>")


def _plain_text(value: str) -> str:
    value = MARKDOWN_LINK_RE.sub(r"\1", value)
    value = CITATION_RE.sub("", value)
    value = TAG_RE.sub("", value)
    return " ".join(html.unescape(value).split())


def _attack_reference(obj: dict[str, Any]) -> dict[str, Any] | None:
    for reference in obj.get("external_references", []):
        external_id = str(reference.get("external_id", ""))
        if reference.get("source_name") == "mitre-attack" and re.fullmatch(r"T\d{4}(?:\.\d{3})?", external_id):
            return reference
    return None


def load_catalog(path: Path, *, active_only: bool = True) -> list[Technique]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    techniques: list[Technique] = []
    for obj in payload.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if active_only and (obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True):
            continue
        reference = _attack_reference(obj)
        if reference is None:
            continue
        technique_id = str(reference["external_id"])
        techniques.append(
            Technique(
                technique_id=technique_id,
                name=str(obj.get("name", "")).strip(),
                description=_plain_text(str(obj.get("description", ""))),
                tactics=tuple(
                    sorted(
                        {
                            str(phase.get("phase_name", "")).replace("-", " ")
                            for phase in obj.get("kill_chain_phases", [])
                            if phase.get("phase_name")
                        }
                    )
                ),
                platforms=tuple(sorted(str(value) for value in obj.get("x_mitre_platforms", []))),
                is_subtechnique=bool(obj.get("x_mitre_is_subtechnique", False)),
                attack_url=str(reference.get("url", f"https://attack.mitre.org/techniques/{technique_id}")),
                stix_id=str(obj.get("id", "")),
                modified=str(obj.get("modified", "")),
            )
        )
    if not techniques:
        raise ValueError(f"No Enterprise ATT&CK techniques found in {path}")
    return sorted(techniques, key=lambda item: item.technique_id)


def load_annoctr_catalog(path: Path) -> list[Technique]:
    techniques: list[Technique] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("entity_type") != "techniques":
                continue
            attack_url = str(row.get("idx", ""))
            match = re.search(r"/techniques/(T\d{4})(?:/(\d{3}))?", attack_url)
            if match is None:
                continue
            technique_id = match.group(1) if match.group(2) is None else f"{match.group(1)}.{match.group(2)}"
            techniques.append(
                Technique(
                    technique_id=technique_id,
                    name=str(row.get("title", row.get("entity", ""))).strip(),
                    description=_plain_text(str(row.get("text", ""))),
                    tactics=(),
                    platforms=(),
                    is_subtechnique=bool(row.get("is_subtechnique", False)),
                    attack_url=attack_url,
                    stix_id=f"annoctr:{technique_id}",
                    modified="",
                )
            )
    if not techniques:
        raise ValueError(f"No AnnoCTR ATT&CK techniques found in {path}")
    return sorted(techniques, key=lambda item: item.technique_id)


def catalog_by_id(techniques: list[Technique]) -> dict[str, Technique]:
    result = {technique.technique_id: technique for technique in techniques}
    if len(result) != len(techniques):
        raise ValueError("ATT&CK catalog contains duplicate technique IDs")
    return result
