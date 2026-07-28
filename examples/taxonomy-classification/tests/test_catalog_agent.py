from __future__ import annotations

import io
from typing import Any

from PIL import Image

from taxonomy_classification.catalog_agent import (
    CatalogDecision,
    CatalogListing,
    candidate_union,
    classify_listing,
    evaluation_metrics,
)


def listing(*, reference: str) -> CatalogListing:
    return CatalogListing(
        row_idx=7,
        title="Example",
        description="Example listing",
        image_bytes=b"jpeg",
        image_format="jpeg",
        image_sha256="abc",
        candidate_paths=["A > One", "A > Two", "B > Three", "A > Four"],
        ground_truth_path=reference,
    )


def test_candidate_union_preserves_image_ranking_then_adds_text_candidates() -> None:
    paths = ["A", "B", "C", "D"]
    assert candidate_union(
        paths,
        text_scores=[0.9, 0.8, 0.2, 0.1],
        image_plus_copy_scores=[0.1, 0.7, 0.95, 0.4],
    ) == ["C", "B", "A"]


def test_evaluation_reports_exact_top_level_and_macro_hierarchical_f1() -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > Two",
        needs_review=True,
        candidate_union=["A > Two", "A > One"],
        text_scores=[0.1, 0.2, 0.3, 0.4],
        image_plus_copy_scores=[0.4, 0.3, 0.2, 0.1],
        verifier_response_id="chat-1",
    )

    assert evaluation_metrics([source], [decision]) == {
        "sample_size": 1,
        "exact_path_correct": 0,
        "exact_path_accuracy": 0.0,
        "top_level_correct": 1,
        "top_level_accuracy": 1.0,
        "macro_hierarchical_f1": 0.5,
        "needs_review": 1,
    }


def test_classify_listing_runs_two_rankings_then_verifies_the_union() -> None:
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(encoded, format="JPEG")
    source = CatalogListing(
        row_idx=54,
        title="Manual floor sweeper",
        description="Push-powered sweeper for hard floors.",
        image_bytes=encoded.getvalue(),
        image_format="jpeg",
        image_sha256="abc",
        candidate_paths=[
            "Home & Garden > Household Supplies > Carpet Sweepers",
            "Home & Garden > Household Supplies > Power Sweepers",
            "Hardware > Tools > Brooms",
            "Vehicles & Parts > Vehicle Parts",
        ],
        ground_truth_path="Home & Garden > Household Supplies > Power Sweepers",
    )

    class FakeClient:
        def __init__(self) -> None:
            self.score_calls: list[dict[str, Any]] = []
            self.chat_calls: list[dict[str, Any]] = []

        def score(
            self,
            model: str,
            query: dict[str, Any],
            items: list[dict[str, Any]],
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.score_calls.append(
                {
                    "model": model,
                    "query": query,
                    "items": items,
                    "kwargs": kwargs,
                }
            )
            scores = (
                [0.9, 0.4, 0.2, 0.1]
                if len(self.score_calls) == 1
                else [0.3, 0.95, 0.4, 0.1]
            )
            return {
                "scores": [
                    {"item_id": str(index), "score": score}
                    for index, score in enumerate(scores)
                ]
            }

        def chat_completions(
            self,
            model: str,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.chat_calls.append(
                {"model": model, "messages": messages, "kwargs": kwargs}
            )
            return {
                "id": "chat-54",
                "choices": [
                    {
                        "message": {
                            "content": ('{"selected_index": 0, "needs_review": false}')
                        }
                    }
                ],
            }

    client = FakeClient()
    decision = classify_listing(client, source)  # type: ignore[arg-type]

    assert len(client.score_calls) == 2
    assert "images" not in client.score_calls[0]["query"]
    assert client.score_calls[1]["query"]["images"][0]["format"] == "jpeg"
    assert len(client.chat_calls) == 1
    assert decision.candidate_union == [
        "Home & Garden > Household Supplies > Power Sweepers",
        "Hardware > Tools > Brooms",
        "Home & Garden > Household Supplies > Carpet Sweepers",
    ]
    assert decision.selected_path == (
        "Home & Garden > Household Supplies > Power Sweepers"
    )
    assert decision.needs_review is False
    assert decision.verifier_response_id == "chat-54"
