"""TraceDecision -- the canonical audit record for one agent action."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class TraceDecision:
    """One recorded agent decision with full audit trail.

    Fields agent_id, action, score, and reasoning are required.
    All others default so that api.py can construct quickly and augment later.
    """

    agent_id: str
    action: str
    score: int
    reasoning: str
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)
    risk_flags: list[str] = field(default_factory=list)
    similar_decision_ids: list[str] = field(default_factory=list)
    #: The gate's actual verdict (ALLOW / WOULD-BLOCK / BLOCK) for this decision, empty when
    #: unknown (e.g. a decision recorded before this field existed).
    verdict: str = ""

    @property
    def risk_level(self) -> str:
        if self.score >= 70:
            return "high"
        if self.score >= 40:
            return "medium"
        return "low"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "action": self.action,
            "score": self.score,
            "reasoning": self.reasoning,
            "risk_flags": self.risk_flags,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "output": {
                "score": self.score,
                "reasoning": self.reasoning,
                "confidence": round(self.score / 100, 2),
                "risk_flags": self.risk_flags,
            },
            "trace": {
                "status": "recorded",
                "risk_level": self.risk_level,
                "similar_decisions": self.similar_decision_ids,
            },
        }
