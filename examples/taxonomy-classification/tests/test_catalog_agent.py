from __future__ import annotations

import hashlib
import io
import json
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

ROOT = Path(__file__).resolve().parents[1]


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


def api_calls(row_idx: int) -> list[dict[str, object]]:
    return [
        {
            "stage": stage,
            "requested_model": model,
            "runtime_model": model,
            "request_id": f"{stage}-{row_idx}",
            "timing_ms": 1.0,
            "credits_debited": 1,
            "rate_book_version": "rate-book-v1",
            "execution_identity_sha256": "a" * 64,
        }
        for stage, model in catalog_agent.EXPECTED_API_CALL_MODELS.items()
    ]


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
                "request": {
                    "id": f"score-{len(self.score_calls)}",
                    "credits_debited": 1,
                    "rate_book_version": "rate-book-v1",
                    "execution_identity_sha256": "a" * 64,
                },
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
                "request": {
                    "id": "generate-54",
                    "credits_debited": 1,
                    "rate_book_version": "rate-book-v1",
                    "execution_identity_sha256": "b" * 64,
                },
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
    ("missing_field", "match"),
    [
        ("id", "no request ID"),
        ("rate_book_version", "no rate-book version"),
        ("execution_identity_sha256", "invalid execution identity"),
    ],
)
def test_api_call_record_requires_complete_request_provenance(
    missing_field: str,
    match: str,
) -> None:
    request = {
        "id": "request-1",
        "credits_debited": 1,
        "rate_book_version": "rate-book-v1",
        "execution_identity_sha256": "a" * 64,
    }
    del request[missing_field]

    with pytest.raises(ValueError, match=match):
        catalog_agent._api_call_record(
            stage="copy_rerank",
            requested_model=catalog_agent.RERANKER_MODEL,
            response={"model": catalog_agent.RERANKER_MODEL, "request": request},
            timing_ms=1,
        )


@pytest.mark.parametrize(
    "credits_debited",
    [None, False, -1, "1", float("nan"), float("inf"), float("-inf")],
)
def test_api_call_record_rejects_invalid_credits_debited(
    credits_debited: object,
) -> None:
    with pytest.raises(ValueError, match="invalid credits debited"):
        catalog_agent._api_call_record(
            stage="copy_rerank",
            requested_model=catalog_agent.RERANKER_MODEL,
            response={
                "model": catalog_agent.RERANKER_MODEL,
                "request": {
                    "id": "request-1",
                    "credits_debited": credits_debited,
                    "rate_book_version": "rate-book-v1",
                    "execution_identity_sha256": "a" * 64,
                },
            },
            timing_ms=1,
        )


@pytest.mark.parametrize(
    "execution_identity_sha256",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_api_call_record_rejects_a_malformed_execution_identity(
    execution_identity_sha256: str,
) -> None:
    with pytest.raises(ValueError, match="invalid execution identity"):
        catalog_agent._api_call_record(
            stage="copy_rerank",
            requested_model=catalog_agent.RERANKER_MODEL,
            response={
                "model": catalog_agent.RERANKER_MODEL,
                "request": {
                    "id": "request-1",
                    "credits_debited": 1,
                    "rate_book_version": "rate-book-v1",
                    "execution_identity_sha256": execution_identity_sha256,
                },
            },
            timing_ms=1,
        )


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ({"text": None}, "SIE verifier returned non-text content"),
        (
            {"text": '{"selected_index": 2, "needs_review": false}'},
            "Invalid selected_index: 2",
        ),
        (
            {"text": '{"selected_index": true, "needs_review": false}'},
            "Invalid selected_index: True",
        ),
        (
            {"text": '{"selected_index": 0, "needs_review": "false"}'},
            "Invalid needs_review: 'false'",
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
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output(
        listings, {7: first_decision}, offset=7
    )
    catalog_agent._write_evaluation_output(output_path, checkpoint)
    assert checkpoint["response_schema"]["properties"]["selected_index"]["maximum"] == 3
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
            verifier_response_id="candidate_verification-8",
            api_calls=api_calls(8),
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
    assert saved["run_command"] == (
        "eval-catalog-agent --offset 7 --limit 2 "
        f"--cache-dir .cache/catalog-agent --output {output_path}"
    )


@pytest.mark.parametrize("row_idx", [None, "7", True])
def test_checkpoint_rejects_invalid_row_idx(
    tmp_path: Path,
    row_idx: object,
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    if row_idx is None:
        checkpoint["results"][0].pop("row_idx")
    else:
        checkpoint["results"][0]["row_idx"] = row_idx
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match="invalid row_idx"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


def test_eval_rejects_a_summary_that_overwrites_the_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "evaluation.json"
    equivalent_path = tmp_path / "unused" / ".." / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval-catalog-agent",
            "--output",
            str(output_path),
            "--summary-output",
            str(equivalent_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        catalog_agent.eval_main()

    assert "--summary-output must differ from --output" in capsys.readouterr().err
    assert not output_path.exists()


@pytest.mark.parametrize("changed_field", ["image_sha256", "candidate_paths"])
def test_checkpoint_rejects_changed_listing_source(
    tmp_path: Path,
    changed_field: str,
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    catalog_agent._write_evaluation_output(
        output_path,
        catalog_agent._evaluation_output([source], {7: decision}, offset=7),
    )
    replacement = "changed" if changed_field == "image_sha256" else ["B > Changed"]
    changed = CatalogListing(
        **{
            **source.__dict__,
            changed_field: replacement,
        }
    )

    with pytest.raises(ValueError, match="source changed for row 7"):
        catalog_agent._load_checkpoint(output_path, [changed], offset=7)


def test_checkpoint_rejects_changed_sie_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    catalog_agent._write_evaluation_output(
        output_path,
        catalog_agent._evaluation_output([source], {7: decision}, offset=7),
    )
    monkeypatch.setattr(
        catalog_agent,
        "read_sie_settings",
        lambda: ("https://different.example/", "unused"),
    )

    with pytest.raises(ValueError, match="SIE endpoint changed"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


def test_checkpoint_rejects_incomplete_api_call_provenance(tmp_path: Path) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0]["api_calls"][0]["rate_book_version"] = None
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match="has invalid rate book version"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


@pytest.mark.parametrize(
    "credits_debited",
    [None, False, -1, "1", float("nan"), float("inf"), float("-inf")],
)
def test_checkpoint_rejects_invalid_credits_debited(
    tmp_path: Path,
    credits_debited: object,
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0]["api_calls"][0]["credits_debited"] = credits_debited
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match="invalid credits debited"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


@pytest.mark.parametrize(
    ("score_field", "scores"),
    [
        ("text_scores", [1.0, 0.0]),
        ("image_plus_copy_scores", [True, 0.0, 0.0, 0.0]),
    ],
)
def test_checkpoint_rejects_invalid_score_arrays(
    tmp_path: Path,
    score_field: str,
    scores: list[object],
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0][score_field] = scores
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match=score_field):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


def test_checkpoint_rejects_a_malformed_execution_identity(tmp_path: Path) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0]["api_calls"][0]["execution_identity_sha256"] = (
        "not-a-sha256"
    )
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match="invalid execution identity sha256"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


@pytest.mark.parametrize("field_name", ["verifier_response_id", "api_calls"])
def test_checkpoint_rejects_missing_decision_provenance(
    tmp_path: Path, field_name: str
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0].pop(field_name)
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match=rf"{field_name} missing for row 7"):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("candidate_union", ["A > Two", "A > One"], "candidate union changed"),
        ("selected_path", "B > Three", "selected path changed"),
        ("needs_review", 1, "needs review changed"),
    ],
)
def test_checkpoint_rejects_invalid_decision_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
    match: str,
) -> None:
    source = listing(reference="A > One")
    decision = CatalogDecision(
        row_idx=7,
        selected_path="A > One",
        needs_review=False,
        candidate_union=["A > One", "A > Two"],
        text_scores=[1.0, 0.0, 0.0, 0.0],
        image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
        verifier_response_id="candidate_verification-7",
        api_calls=api_calls(7),
    )
    output_path = tmp_path / "evaluation.json"
    checkpoint = catalog_agent._evaluation_output([source], {7: decision}, offset=7)
    checkpoint["results"][0][field_name] = value
    catalog_agent._write_evaluation_output(output_path, checkpoint)

    with pytest.raises(ValueError, match=match):
        catalog_agent._load_checkpoint(output_path, [source], offset=7)


def test_evaluation_rejects_duplicate_request_ids() -> None:
    sources = [
        listing(reference="A > One"),
        CatalogListing(
            **{
                **listing(reference="A > Two").__dict__,
                "row_idx": 8,
            }
        ),
    ]
    first_calls = api_calls(7)
    second_calls = api_calls(8)
    second_calls[0]["request_id"] = first_calls[0]["request_id"]
    decisions = {
        source.row_idx: CatalogDecision(
            row_idx=source.row_idx,
            selected_path=source.ground_truth_path or "A > One",
            needs_review=False,
            candidate_union=[source.ground_truth_path or "A > One"],
            text_scores=[1.0, 0.0, 0.0, 0.0],
            image_plus_copy_scores=[1.0, 0.0, 0.0, 0.0],
            verifier_response_id=f"candidate_verification-{source.row_idx}",
            api_calls=calls,
        )
        for source, calls in zip(sources, (first_calls, second_calls), strict=True)
    }

    with pytest.raises(ValueError, match="duplicate SIE request IDs"):
        catalog_agent._evaluation_output(sources, decisions, offset=7)


def test_verified_summary_pins_the_exact_evaluation_invocation() -> None:
    evaluation_path = ROOT / "verified-run" / "evaluation.json"
    summary = json.loads(
        (ROOT / "results" / "catalog-agent-summary.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert summary["run_command"] == evaluation["run_command"]
    assert summary["evaluation"] == {
        "path": "verified-run/evaluation.json",
        "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
    }
    assert all(result["source"].get("row_sha256") for result in evaluation["results"])


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
