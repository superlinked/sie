from __future__ import annotations

from insurance_claims.evaluate import evaluate_review
from insurance_claims.review import chunk_markdown


def test_chunk_markdown_keeps_all_paragraphs() -> None:
    markdown = "first paragraph\n\nsecond paragraph is longer\n\nthird"

    chunks = chunk_markdown(markdown, 35)

    assert "\n\n".join(chunks) == markdown
    assert len(chunks) == 2


def test_evaluation_accepts_expected_review() -> None:
    review = {
        "route": "manual_review",
        "claim_summary": {
            "claimed_total": 81060,
            "attachment_total": 80660,
            "difference": 400,
        },
        "findings": [
            {"category": "missing_signature", "severity": "blocking"},
            {"category": "amount_mismatch", "severity": "high"},
        ],
    }

    assert all(check.passed for check in evaluate_review(review))
