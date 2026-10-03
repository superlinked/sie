from __future__ import annotations

import numpy as np

from threat_mapper.models import BehaviorEvidence, CandidateScore, LabeledTechniqueExample, Technique
from threat_mapper.pipeline import (
    _ground_quote,
    evidence_sha256,
    rerank,
    retrieve,
    retrieve_exemplars,
    retrieve_hybrid,
    split_report,
    verify_mapping,
)


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
    def __init__(self, payloads: list[dict[str, object] | str]) -> None:
        self.payloads = payloads
        self.models: list[str] = []
        self.prompts: list[str] = []

    def generate(self, model: str, prompt: str, **kwargs):
        import json

        self.models.append(model)
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return {"model": model, "text": text, "request": {"id": "request"}}


class FakeScoreClient:
    def score(self, model: str, query: dict[str, object], items: list[dict[str, object]], **kwargs):
        return {
            "model": model,
            "query_id": query["id"],
            "scores": [
                {"item_id": item["id"], "score": 1.0 - index / 10, "rank": index} for index, item in enumerate(items)
            ],
            "request": {"id": "request"},
        }


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


def test_behavior_extraction_preserves_offsets_with_noncanonical_separators() -> None:
    from threat_mapper.pipeline import extract_behaviors

    client = FakeGenerateClient(
        [
            {"behaviors": []},
            {"behaviors": [{"quote": "used a proxy", "summary": "AiTM"}]},
        ]
    )
    report = "First paragraph used a proxy.\r\n\r\n\r\nSecond paragraph used a proxy."

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
    assert report[behaviors[0].source_start : behaviors[0].source_end] == "used a proxy"


def test_behavior_extraction_keeps_distinct_actions_from_the_same_quote() -> None:
    from threat_mapper.pipeline import extract_behaviors

    quote = "The loader stole passwords and downloaded another payload."
    client = FakeGenerateClient(
        [
            {
                "behaviors": [
                    {
                        "quote": quote,
                        "summary": "The loader stole passwords.",
                        "action": "stole",
                        "object": "passwords",
                    },
                    {
                        "quote": quote,
                        "summary": "The loader downloaded another payload.",
                        "action": "downloaded",
                        "object": "another payload",
                    },
                ]
            }
        ]
    )

    behaviors, _ = extract_behaviors(
        client,
        "Qwen/Qwen3.6-27B",
        quote,
        max_behaviors=4,
        chunk_characters=100,
        provision_timeout_s=60,
    )

    assert [(row.action, row.object) for row in behaviors] == [
        ("stole", "passwords"),
        ("downloaded", "another payload"),
    ]


def test_evidence_hash_distinguishes_actions_from_the_same_source_span() -> None:
    base = {
        "quote": "The loader stole passwords and downloaded another payload.",
        "summary": "Atomic behavior",
        "source_start": 10,
        "source_end": 69,
    }
    theft = BehaviorEvidence(**base, action="stole", object="passwords")
    download = BehaviorEvidence(**base, action="downloaded", object="another payload")

    assert evidence_sha256(theft) != evidence_sha256(download)


def test_behavior_extraction_retries_truncated_json_with_smaller_chunks() -> None:
    from threat_mapper.pipeline import extract_behaviors

    first = "A" * 800 + " used a proxy."
    second = "B" * 800 + " stole a cookie."
    report = f"{first}\n\n{second}"
    client = FakeGenerateClient(
        [
            '{"behaviors":[{"quote":"unfinished',
            {"behaviors": [{"quote": "used a proxy", "summary": "Proxy use"}]},
            {"behaviors": [{"quote": "stole a cookie", "summary": "Cookie theft"}]},
        ]
    )

    behaviors, calls = extract_behaviors(
        client,
        "Qwen/Qwen3.6-27B",
        report,
        max_behaviors=24,
        chunk_characters=2000,
        provision_timeout_s=60,
    )

    assert [row.quote for row in behaviors] == ["used a proxy", "stole a cookie"]
    assert calls[0]["outcome"] == "invalid_json_retried_with_smaller_request"
    assert calls[0]["request_payload"]["prompt"].endswith(report)
    assert calls[0]["request_payload"]["max_new_tokens"] == 4096
    assert calls[0]["request_payload"]["grammar"]["json_schema"]["properties"]["behaviors"]["maxItems"] == 24
    assert calls[0]["raw_response"]["text"] == '{"behaviors":[{"quote":"unfinished'
    assert len(calls) == 3


def test_behavior_extraction_reduces_the_row_limit_when_text_cannot_split() -> None:
    from threat_mapper.pipeline import extract_behaviors

    report = "The loader used a proxy."
    client = FakeGenerateClient(
        [
            '{"behaviors":[{"quote":"unfinished',
            {"behaviors": [{"quote": "used a proxy", "summary": "Proxy use"}]},
        ]
    )

    behaviors, calls = extract_behaviors(
        client,
        "Qwen/Qwen3.6-27B",
        report,
        max_behaviors=4,
        chunk_characters=100,
        provision_timeout_s=60,
    )

    assert [row.quote for row in behaviors] == ["used a proxy"]
    assert "Return at most 4 behaviors" in client.prompts[0]
    assert "Return at most 2 behaviors" in client.prompts[1]
    assert calls[1]["request_payload"]["grammar"]["json_schema"]["properties"]["behaviors"]["maxItems"] == 2
    assert calls[0]["outcome"] == "invalid_json_retried_with_smaller_request"


def test_behavior_extraction_caps_rows_across_the_complete_report() -> None:
    from threat_mapper.pipeline import extract_behaviors

    report = "First actor stole passwords.\n\nSecond actor used a proxy.\n\nThird actor downloaded a payload."
    client = FakeGenerateClient(
        [
            {
                "behaviors": [
                    {
                        "quote": "stole passwords",
                        "summary": "Password theft",
                        "action": "stole",
                        "object": "passwords",
                    }
                ]
            },
            {
                "behaviors": [
                    {
                        "quote": "used a proxy",
                        "summary": "Proxy use",
                        "action": "used",
                        "object": "proxy",
                    },
                    {
                        "quote": "used a proxy",
                        "summary": "Extra row from an invalid response",
                        "action": "accessed",
                        "object": "account",
                    },
                ]
            },
        ]
    )

    behaviors, calls = extract_behaviors(
        client,
        "Qwen/Qwen3.6-27B",
        report,
        max_behaviors=2,
        chunk_characters=35,
        provision_timeout_s=60,
    )

    assert [(row.action, row.object) for row in behaviors] == [
        ("stole", "passwords"),
        ("used", "proxy"),
    ]
    assert len(calls) == 2
    assert "Return at most 2 behaviors" in client.prompts[0]
    assert "Return at most 1 behaviors" in client.prompts[1]
    assert calls[1]["request_payload"]["grammar"]["json_schema"]["properties"]["behaviors"]["maxItems"] == 1


def test_split_report_returns_exact_source_spans_when_paragraphs_merge() -> None:
    report = "  First paragraph.\n\n\nSecond paragraph.  "

    spans = split_report(report, max_characters=80)

    assert spans == [(2, len(report) - 2)]
    assert report[slice(*spans[0])] == "First paragraph.\n\n\nSecond paragraph."


def test_retrieve_sorts_by_cosine_score() -> None:
    techniques = [technique("T1557", "AiTM"), technique("T1539", "Steal Cookie")]
    catalog = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([0.1, 0.9], dtype=np.float32)

    rows = retrieve(query, catalog, techniques, 2)

    assert [row.technique_id for row in rows] == ["T1539", "T1557"]


def test_hybrid_retrieval_unions_dense_and_token_level_maxsim_candidates() -> None:
    techniques = [
        technique("T1000", "Dense match"),
        technique("T2000", "Late interaction match"),
        technique("T3000", "Other"),
    ]
    catalog_dense = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query_dense = np.asarray([1.0, 0.0], dtype=np.float32)
    catalog_multivectors = [
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0]], dtype=np.float32),
    ]
    query_multivector = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    rows = retrieve_hybrid(
        query_dense,
        catalog_dense,
        query_multivector,
        catalog_multivectors,
        techniques,
        dense_count=1,
        late_interaction_count=1,
        candidate_count=2,
    )

    assert {row.technique_id for row in rows} == {"T1000", "T2000"}
    late = next(row for row in rows if row.technique_id == "T2000")
    assert late.late_interaction_rank == 0
    assert late.late_interaction_score == 2.0


def test_hybrid_retrieval_adds_the_best_labeled_example_per_technique() -> None:
    techniques = [
        technique("T1000", "Dense match"),
        technique("T2000", "Example match"),
        technique("T3000", "Other"),
    ]
    lookup = {row.technique_id: row for row in techniques}
    examples = [
        LabeledTechniqueExample("T2000", "used SOCKS5", "The tool used SOCKS5.", "train-a", "CI"),
        LabeledTechniqueExample("T2000", "used a proxy", "The tool used a proxy.", "train-b", "CI"),
        LabeledTechniqueExample("T9999", "old label", "An old label.", "train-c", "CI"),
    ]
    example_vectors = np.asarray([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    exemplar_candidates = retrieve_exemplars(query, example_vectors, examples, lookup, 2)
    rows = retrieve_hybrid(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[-1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        ],
        techniques,
        dense_count=1,
        late_interaction_count=1,
        candidate_count=2,
        exemplar_candidates=exemplar_candidates,
        exemplar_count=1,
        exemplar_rrf_weight=2.0,
    )

    assert [row.technique_id for row in exemplar_candidates] == ["T2000"]
    example_match = next(row for row in rows if row.technique_id == "T2000")
    assert example_match.exemplar_rank == 0
    assert example_match.exemplar_quote == "The tool used SOCKS5."


def test_rerank_records_typed_dict_items_in_the_request_trace() -> None:
    behavior = BehaviorEvidence("stole a cookie", "cookie theft", 0, 14)
    candidates = [CandidateScore("T1539", "Steal Web Session Cookie", 0.9)]
    lookup = {"T1539": technique("T1539", "Steal Web Session Cookie")}

    ranked, call = rerank(
        FakeScoreClient(),
        "Qwen/Qwen3-Reranker-4B",
        behavior,
        candidates,
        lookup,
        rerank_count=1,
        provision_timeout_s=60,
    )

    assert [row.technique_id for row in ranked] == ["T1539"]
    assert call["request_payload"]["query"] == {
        "id": "behavior",
        "text": "Observed adversary behavior: stole a cookie",
    }
    assert call["request_payload"]["items"][0]["id"] == "T1539"


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
        actor="Necrobrowser",
        action="used",
        object="stolen session cookies",
        assertion="defensive",
    )
    candidates = [
        CandidateScore("T1539", "Steal Web Session Cookie", 0.9, 0.8, 0, exemplar_rank=1),
        CandidateScore("T1550.004", "Web Session Cookie", 0.8, 0.7, 1, exemplar_rank=0),
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
    assert decision.exemplar_agreement is True
    assert decision.escalated is True
    assert client.models == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"]
    assert "Actor: Necrobrowser" in client.prompts[0]
    assert "Assertion: defensive" not in client.prompts[0]
    assert "Independently decide whether the quote describes adversary activity" in client.prompts[0]
    assert "Advertising or selling malware proves neither action" in client.prompts[0]
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


def test_verifier_routes_supported_mapping_without_exemplar_agreement_to_review() -> None:
    client = FakeGenerateClient(
        [
            {
                "selected_index": 0,
                "support": "supported",
                "evidence_quote": "downloaded another payload",
                "rationale": "The source states a tool transfer.",
            }
        ]
    )
    behavior = BehaviorEvidence("downloaded another payload", "tool transfer", 0, 26)
    candidate = CandidateScore(
        "T1105",
        "Ingress Tool Transfer",
        0.9,
        0.8,
        0,
        exemplar_rank=2,
    )
    lookup = {"T1105": technique("T1105", "Ingress Tool Transfer")}

    decision, _ = verify_mapping(
        client,
        "large",
        "large",
        behavior,
        [candidate],
        lookup,
        verifier_count=1,
        use_escalation=False,
        provision_timeout_s=60,
    )

    assert decision.support == "supported"
    assert decision.exemplar_agreement is False
    assert decision.route == "analyst_review"


def test_verifier_retries_invalid_json_and_keeps_both_raw_responses() -> None:
    client = FakeGenerateClient(
        [
            '{"selected_index":0,"support":"supported"',
            {
                "selected_index": 0,
                "support": "supported",
                "evidence_quote": "stole cookies",
                "rationale": "The source states cookie theft.",
            },
        ]
    )
    behavior = BehaviorEvidence("stole cookies", "cookie theft", 0, 13)
    candidate = CandidateScore("T1539", "Steal Web Session Cookie", 0.9, 0.8, 0)
    lookup = {"T1539": technique("T1539", "Steal Web Session Cookie")}

    decision, calls = verify_mapping(
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

    assert decision.selected_technique_id == "T1539"
    assert [call["stage"] for call in calls] == ["verify", "verify_retry"]
    assert calls[0]["outcome"] == "invalid_json"
    assert calls[0]["raw_response"]["text"].endswith('"supported"')
    assert calls[1]["request_payload"]["max_new_tokens"] == 1200


def test_verifier_abstains_when_retry_is_also_invalid_json() -> None:
    client = FakeGenerateClient(["{", "{"])
    behavior = BehaviorEvidence("stole cookies", "cookie theft", 0, 13)
    candidate = CandidateScore("T1539", "Steal Web Session Cookie", 0.9, 0.8, 0)
    lookup = {"T1539": technique("T1539", "Steal Web Session Cookie")}

    decision, calls = verify_mapping(
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
    assert decision.rationale == "Verifier returned invalid JSON twice; no mapping emitted."
    assert [call["outcome"] for call in calls] == ["invalid_json", "invalid_json"]
