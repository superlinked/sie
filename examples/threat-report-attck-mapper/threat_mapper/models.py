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
class LabeledTechniqueExample:
    technique_id: str
    quote: str
    context: str
    document: str
    annotation_class: str

    @property
    def embedding_text(self) -> str:
        return f"Span: {self.quote}\nSentence: {self.context}"


@dataclass(frozen=True)
class ExemplarScore:
    technique_id: str
    score: float
    rank: int
    quote: str
    document: str


@dataclass(frozen=True)
class CandidateScore:
    technique_id: str
    name: str
    dense_score: float
    rerank_score: float | None = None
    rerank_rank: int | None = None
    late_interaction_score: float | None = None
    dense_rank: int | None = None
    late_interaction_rank: int | None = None
    fusion_score: float | None = None
    exemplar_score: float | None = None
    exemplar_rank: int | None = None
    exemplar_quote: str | None = None
    exemplar_document: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorEvidence:
    quote: str
    summary: str
    source_start: int
    source_end: int
    entities: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    actor: str = ""
    action: str = ""
    object: str = ""
    tool: str = ""
    target: str = ""
    assertion: str = "observed"

    @property
    def event_text(self) -> str:
        fields = [
            f"Actor: {self.actor}" if self.actor else "",
            f"Action: {self.action}" if self.action else "",
            f"Object: {self.object}" if self.object else "",
            f"Tool: {self.tool}" if self.tool else "",
            f"Target: {self.target}" if self.target else "",
            f"Assertion: {self.assertion}",
            f"Evidence: {self.quote}",
        ]
        return "\n".join(field for field in fields if field)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldTechniqueMention:
    mention_id: str
    document: str
    technique_id: str
    annotation_class: str
    quote: str
    source_start: int
    source_end: int

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
    exemplar_agreement: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
