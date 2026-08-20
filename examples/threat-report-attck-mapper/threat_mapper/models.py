from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Technique:
    technique_id: str
    name: str
    description: str
    tactics: tuple[str, ...]
    platforms: tuple[str, ...]
    is_subtechnique: bool
    attack_url: str
    stix_id: str
    modified: str

    @property
    def candidate_text(self) -> str:
        fields = [
            f"ATT&CK technique: {self.technique_id} {self.name}",
            f"Description: {self.description}",
        ]
        if self.tactics:
            fields.append(f"Tactics: {', '.join(self.tactics)}")
        if self.platforms:
            fields.append(f"Platforms: {', '.join(self.platforms)}")
        return "\n".join(fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinkingCase:
    case_id: str
    document: str
    mention: str
    evidence: str
    left_context: str
    right_context: str
    gold_ids: tuple[str, ...]
    annotation_classes: tuple[str, ...]

    @property
    def query_text(self) -> str:
        return f"Observed behavior: {self.evidence}\nAnnotated span: {self.mention}"


@dataclass(frozen=True)
class CandidateScore:
    technique_id: str
    name: str
    dense_score: float
    rerank_score: float | None = None
    rerank_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorEvidence:
    quote: str
    summary: str
    source_start: int
    source_end: int
    entities: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MappingDecision:
    behavior: BehaviorEvidence
    route: str
    status: str
    selected_technique_id: str | None
    support: str
    evidence_quote: str
    rationale: str
    candidates: tuple[CandidateScore, ...]
    verifier_model: str
    escalated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
