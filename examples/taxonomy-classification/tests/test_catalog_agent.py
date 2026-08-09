from __future__ import annotations

import json
import io
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from taxonomy_classification import catalog_agent
from taxonomy_classification.catalog_agent import (
    CatalogDecision,
    CatalogListing,
    candidate_union,
    classify_listing,
    evaluation_metrics,
    verify_candidates,
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
        verifier_response_id="generate-1",
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
            self.generate_calls: list[dict[str, Any]] = []

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
                "model": catalog_agent.RERANKER_MODEL,
                "request": {"id": f"score-{len(self.score_calls)}"},
                "scores": [
                    {"item_id": str(index), "score": score}
                    for index, score in enumerate(scores)
                ],
            }

        def generate(
            self,
            model: str,
            prompt: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.generate_calls.append(
                {"model": model, "prompt": prompt, "kwargs": kwargs}
            )
            return {
                "model": catalog_agent.VERIFIER_MODEL,
                "text": '{"selected_index": 0, "needs_review": false}',
                "request": {"id": "generate-54"},
            }

    client = FakeClient()
    decision = classify_listing(client, source)  # type: ignore[arg-type]

    assert len(client.score_calls) == 2
    assert "images" not in client.score_calls[0]["query"]
    assert client.score_calls[1]["query"]["images"][0]["format"] == "jpeg"
    assert len(client.generate_calls) == 1
    generate_call = client.generate_calls[0]
    assert generate_call["model"] == catalog_agent.VERIFIER_MODEL
    assert "TITLE\nManual floor sweeper" in generate_call["prompt"]
    assert generate_call["kwargs"]["images"][0]["data"] == source.image_bytes
    assert generate_call["kwargs"]["images"][0]["format"] == source.image_format
    assert (
        generate_call["kwargs"]["grammar"]["json_schema"]["additionalProperties"]
        is False
    )
    assert generate_call["kwargs"]["grammar"]["strict"] is True
    assert decision.candidate_union == [
        "Home & Garden > Household Supplies > Power Sweepers",
        "Hardware > Tools > Brooms",
        "Home & Garden > Household Supplies > Carpet Sweepers",
    ]
    assert decision.selected_path == (
        "Home & Garden > Household Supplies > Power Sweepers"
    )
    assert decision.needs_review is False
    assert decision.verifier_response_id == "generate-54"
    assert [call["request_id"] for call in decision.api_calls] == [
        "score-1",
        "score-2",
        "generate-54",
    ]


def test_verify_candidates_rejects_an_empty_union() -> None:
    source = listing(reference="A > One")

    with pytest.raises(ValueError, match="No candidate paths supplied for row 7"):
        verify_candidates(object(), source, [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ({"text": None}, "SIE verifier returned non-text content"),
        (
            {"text": '{"selected_index": 2, "needs_review": false}'},
            "Invalid selected_index: 2",
        ),
    ],
)
def test_verify_candidates_rejects_invalid_native_output(
    response: dict[str, Any],
    match: str,
) -> None:
    class FakeClient:
        def generate(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return response

    with pytest.raises(ValueError, match=match):
        verify_candidates(
            FakeClient(),  # type: ignore[arg-type]
            listing(reference="A > One"),
            ["A > One", "A > Two"],
        )


def test_image_cache_key_includes_the_dataset_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(encoded, format="JPEG")
    rows = {
        "rows": [
            {
                "row_idx": 7,
                "row": {
                    "product_title": "Example",
                    "product_description": "Example listing",
                    "product_image": {"src": "https://example.com/image.jpg"},
                    "potential_product_categories": ["A > One"],
                    "ground_truth_category": "A > One",
                },
            }
        ]
    }

    def fake_download(url: str) -> bytes:
        if url.startswith(catalog_agent.DATASET_ROWS_URL):
            return json.dumps(rows).encode()
        return encoded.getvalue()

    monkeypatch.setattr(catalog_agent, "_download", fake_download)
    catalog_agent.load_shopify_rows(offset=7, limit=1, cache_dir=tmp_path)

    expected_name = f"shopify-train-{catalog_agent.DATASET_REVISION[:12]}-7.image"
    assert (tmp_path / expected_name).read_bytes() == encoded.getvalue()


def test_eval_resumes_completed_rows_from_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listings = [
        listing(reference="A > One"),
        CatalogListing(
            **{
                **listing(reference="A > Two").__dict__,
                "row_idx": 8,
            }
        ),
    ]
    first_decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="generate-7",
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output(
        listings, {7: first_decision}, offset=7
    )
    catalog_agent._write_evaluation_output(output_path, checkpoint)
    classified_rows: list[int] = []

    def fake_classify(_client: object, source: CatalogListing) -> CatalogDecision:
        classified_rows.append(source.row_idx)
        return CatalogDecision(
            row_idx=source.row_idx,
            selected_path="A > Two",
            needs_review=False,
            candidate_union=["A > Two"],
            text_scores=[0.0, 1.0, 0.0, 0.0],
            image_plus_copy_scores=[0.0, 1.0, 0.0, 0.0],
            verifier_response_id="generate-8",
        )

    @contextmanager
    def fake_client(*, timeout_s: int) -> Any:
        assert timeout_s == 600
        yield object()

    monkeypatch.setattr(catalog_agent, "load_shopify_rows", lambda **_kwargs: listings)
    monkeypatch.setattr(catalog_agent, "classify_listing", fake_classify)
    monkeypatch.setattr(catalog_agent, "create_sie_client", fake_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval-catalog-agent",
            "--offset",
            "7",
            "--limit",
            "2",
            "--output",
            str(output_path),
        ],
    )

    catalog_agent.eval_main()

    assert classified_rows == [8]
    saved = json.loads(output_path.read_text())
    assert [result["row_idx"] for result in saved["results"]] == [7, 8]


def test_predict_honors_the_requested_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = listing(reference="A > One")
    loaded_limits: list[int] = []

    def fake_load(*, offset: int, limit: int, cache_dir: Path) -> list[CatalogListing]:
        assert offset == 0
        assert cache_dir == Path(".cache/catalog-agent")
        loaded_limits.append(limit)
        return [source]

    @contextmanager
    def fake_client(*, timeout_s: int) -> Any:
        assert timeout_s == 600
        yield object()

    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One"],
        text_scores=[1.0],
        image_plus_copy_scores=[1.0],
        verifier_response_id="generate-7",
    )
    monkeypatch.setattr(catalog_agent, "load_shopify_rows", fake_load)
    monkeypatch.setattr(catalog_agent, "create_sie_client", fake_client)
    monkeypatch.setattr(
        catalog_agent, "classify_listing", lambda _client, _listing: decision
    )
    monkeypatch.setattr(sys, "argv", ["predict-catalog-agent", "--limit", "5"])

    catalog_agent.predict_main()

    assert loaded_limits == [5]
