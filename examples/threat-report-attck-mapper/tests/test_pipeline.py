from __future__ import annotations

import numpy as np

from threat_mapper.models import BehaviorEvidence, CandidateScore, Technique
from threat_mapper.pipeline import _ground_quote, retrieve, verify_mapping


def technique(technique_id: str, name: str) -> Technique:
    return Technique(
        technique_id=technique_id,
        name=name,
        description=f"Description for {name}",
        tactics=(),
        platforms=(),
        is_subtechnique="." in technique_id,
        attack_url=f"https://attack.mitre.org/techniques/{technique_id}",
        stix_id=f"attack-pattern--{technique_id}",
        modified="2026-08-05T00:00:00Z",
    )


class FakeGenerateClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.models: list[str] = []

    def generate(self, model: str, prompt: str, **kwargs):
        import json

        self.models.append(model)
        return {"model": model, "text": json.dumps(self.payloads.pop(0)), "request": {"id": "request"}}


def test_ground_quote_recovers_source_whitespace() -> None:
    source = "The actor used a transparent\nreverse proxy to steal cookies."
    result = _ground_quote(source, "transparent reverse proxy")
    assert result is not None
    assert result[0] == "transparent\nreverse proxy"


def test_behavior_extraction_keeps_the_quote_in_its_source_chunk() -> None:
    from threat_mapper.pipeline import extract_behaviors

    client = FakeGenerateClient(
        [
            {"behaviors": []},
            {"behaviors": [{"quote": "used a proxy", "summary": "AiTM"}]},
        ]
    )
    report = "First paragraph used a proxy.\n\nSecond paragraph used a proxy."

    behaviors, _ = extract_behaviors(
        client,
        "Qwen/Qwen3.5-4B",
        report,
        max_behaviors=4,
        chunk_characters=31,
        provision_timeout_s=60,
    )

    assert len(behaviors) == 1
    assert behaviors[0].source_start == report.rindex("used a proxy")


def test_retrieve_sorts_by_cosine_score() -> None:
    techniques = [technique("T1557", "AiTM"), technique("T1539", "Steal Cookie")]
    catalog = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([0.1, 0.9], dtype=np.float32)

    rows = retrieve(query, catalog, techniques, 2)

    assert [row.technique_id for row in rows] == ["T1539", "T1557"]


def test_verifier_escalates_ambiguous_and_keeps_human_review_boundary() -> None:
    client = FakeGenerateClient(
        [
            {
                "selected_index": 0,
                "support": "ambiguous",
                "evidence_quote": "stolen session cookies",
                "rationale": "Two close techniques",
            },
            {
                "selected_index": 1,
                "support": "supported",
                "evidence_quote": "used stolen session cookies to log in",
                "rationale": "The quote states reuse after theft",
            },
        ]
    )
    behavior = BehaviorEvidence(
        quote="Necrobrowser used stolen session cookies to log in to the target site.",
        summary="reuse cookie",
        source_start=0,
        source_end=70,
    )
    candidates = [
        CandidateScore("T1539", "Steal Web Session Cookie", 0.9, 0.8, 0),
        CandidateScore("T1550.004", "Web Session Cookie", 0.8, 0.7, 1),
    ]
    lookup = {row.technique_id: technique(row.technique_id, row.name) for row in candidates}

    decision, calls = verify_mapping(
        client,
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.6-27B",
        behavior,
        candidates,
        lookup,
        verifier_count=2,
        use_escalation=True,
        provision_timeout_s=60,
    )

    assert decision.selected_technique_id == "T1550.004"
    assert decision.route == "suggested_mapping"
    assert decision.status == "needs_analyst_review"
    assert decision.escalated is True
    assert client.models == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"]
    assert [call["stage"] for call in calls] == ["verify", "escalate"]


def test_verifier_fails_closed_when_evidence_is_not_in_source() -> None:
    client = FakeGenerateClient(
        [
            {
                "selected_index": 0,
                "support": "supported",
                "evidence_quote": "words that are absent",
                "rationale": "unsupported output",
            }
        ]
    )
    behavior = BehaviorEvidence("stole cookies", "", 0, 13)
    candidate = CandidateScore("T1539", "Steal Web Session Cookie", 0.9, 0.8, 0)
    lookup = {"T1539": technique("T1539", "Steal Web Session Cookie")}

    decision, _ = verify_mapping(
        client,
        "small",
        "large",
        behavior,
        [candidate],
        lookup,
        verifier_count=1,
        use_escalation=False,
        provision_timeout_s=60,
    )

    assert decision.route == "abstain"
    assert decision.selected_technique_id is None
