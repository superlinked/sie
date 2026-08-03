from __future__ import annotations

from insurance_claims.evaluate import evaluate_review
from insurance_claims.review import (
    _extract_claim_identity,
    _json_object_from_text,
    chunk_markdown,
)


class FakeExtractClient:
    def __init__(self) -> None:
        self.labels: list[str] | None = None

    def extract(self, _model: str, _item: object, **kwargs: object) -> dict[str, object]:
        self.labels = kwargs.get("labels")  # type: ignore[assignment]
        return {"data": {"entities": []}}


def test_chunk_markdown_keeps_all_paragraphs() -> None:
    markdown = "first paragraph\n\nsecond paragraph is longer\n\nthird"

    chunks = chunk_markdown(markdown, 35)

    assert "\n\n".join(chunks) == markdown
    assert len(chunks) == 2


def test_claim_identity_passes_gliner2_labels() -> None:
    client = FakeExtractClient()

    _extract_claim_identity(client, "fastino/gliner2-large-v1", "claim text", 60)

    assert client.labels == [
        "insured name",
        "flood insurance policy number",
        "date and time of loss",
        "insured property address",
    ]


def test_review_json_accepts_fenced_model_output() -> None:
    assert _json_object_from_text('```json\n{"route": "manual_review"}\n```') == {
        "route": "manual_review"
    }


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
