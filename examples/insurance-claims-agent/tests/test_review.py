from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from insurance_claims.evaluate import evaluate_review
from insurance_claims.review import (
    _extract_claim_facts,
    _json_object_from_text,
    chunk_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFIED_RUN = ROOT / "verified-run"


class FakeExtractClient:
    def __init__(self) -> None:
        self.labels: list[str] | None = None

    def extract(
        self,
        _model: str,
        _item: object,
        **kwargs: object,
    ) -> dict[str, object]:
        self.labels = kwargs.get("labels")  # type: ignore[assignment]
        return {"data": {"entities": []}}


def test_chunk_markdown_keeps_all_paragraphs() -> None:
    markdown = "first paragraph\n\nsecond paragraph is longer\n\nthird"

    chunks = chunk_markdown(markdown, 35)

    assert "\n\n".join(chunks) == markdown
    assert len(chunks) == 2


def test_claim_fact_extraction_passes_domain_labels() -> None:
    client = FakeExtractClient()

    _extract_claim_facts(
        client,
        "fastino/gliner2-large-v1",
        "appeal text",
        60,
    )

    assert client.labels == [
        "proof of loss amount",
        "debris removal estimate",
        "barge transportation estimate",
        "debris volume",
        "date of loss",
        "covered debris removal scope",
        "excluded debris cost",
    ]


def test_review_json_accepts_fenced_model_output() -> None:
    assert _json_object_from_text('```json\n{"route": "scope_review_required"}\n```') == {
        "route": "scope_review_required"
    }


def test_evaluation_accepts_published_appeal_result() -> None:
    review = {
        "route": "scope_review_required",
        "appeal_summary": {
            "proof_of_loss_amount": 182552,
            "removal_estimate": 49500,
            "barge_estimate": 181832.94,
            "debris_cubic_yards_min": 12,
            "debris_cubic_yards_max": 15,
        },
        "decision": {
            "covered_scope": ("Remove flood-borne stones from underneath the insured building to its perimeter."),
            "excluded_scope": ("Barge transport, handling, disposal, and yard removal."),
            "evidence_needed": "Other contractor estimates.",
            "prior_claim_check": "Proof of repairs from previous claims.",
        },
        "findings": [
            {"category": "covered_removal"},
            {"category": "excluded_transport"},
            {"category": "price_support"},
            {"category": "prior_claim_overlap"},
        ],
    }

    assert all(check.passed for check in evaluate_review(review))


def test_verified_evaluation_recomputes_from_the_recorded_review() -> None:
    review = json.loads((VERIFIED_RUN / "review.json").read_text(encoding="utf-8"))
    recorded = json.loads((VERIFIED_RUN / "evaluation.json").read_text(encoding="utf-8"))
    checks = evaluate_review(review)

    assert recorded == {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def test_verified_manifest_pins_every_recorded_artifact() -> None:
    manifest = json.loads((VERIFIED_RUN / "manifest.json").read_text(encoding="utf-8"))
    expected_paths = {
        str(path.relative_to(VERIFIED_RUN))
        for path in VERIFIED_RUN.rglob("*")
        if path.is_file() and path.name not in {"README.md", "manifest.json"}
    }
    artifacts = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}

    assert set(artifacts) == expected_paths
    for path, expected_hash in artifacts.items():
        assert hashlib.sha256((VERIFIED_RUN / path).read_bytes()).hexdigest() == expected_hash
