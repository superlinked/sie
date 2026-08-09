import hashlib
import json
import sys
from pathlib import Path

import pytest

import prior_authorization.evaluate as evaluate_module
import prior_authorization.review as review_module
from prior_authorization.evaluate import evaluate_review, evaluate_run
from prior_authorization.review import (
    FIELD_SPECS,
    GLINER2_GROUP_LABELS,
    GLINER2_REQUIRED_SPANS,
    GROUP_FIELDS,
    OUTCOME_FIELDS,
    REQUIREMENT_FIELDS,
    SUBMISSION_FIELDS,
    _chunks,
    _group_source_scope,
    _rate_book_provenance,
    _require_gliner2_group_evidence,
    _require_ranked_evidence,
    _source_fragments,
    _source_scope,
    build_review,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_docling_line_wraps_stay_in_complete_cms_statements() -> None:
    markdown = """## Billing and Coding Criteria for Lower Limb Orthoses

We require prior authorization for HCPCS

codes L1832 and L1851.

    ### What Happens Next for the Published CMS Claim?

The review contractor completes the claim, and the MAC recoups

payment.
"""
    assert _chunks(markdown) == [
        "## Billing and Coding Criteria for Lower Limb Orthoses",
        "We require prior authorization for HCPCS codes L1832 and L1851.",
        "### What Happens Next for the Published CMS Claim?",
        "The review contractor completes the claim, and the MAC recoups payment.",
    ]


def test_config_uses_canonical_docling_id() -> None:
    assert load_config()["models"]["parse"] == "docling"


def test_docling_bullets_stay_as_distinct_submission_chunks() -> None:
    markdown = """### Documentation submitted

- Standard written order with correct HCPCS coding
- Treating practitioner's medical record that has adequate medical necessity information
- Proof of delivery with face-to-face encounter 7 months ago
"""
    assert _chunks(markdown) == [
        "### Documentation submitted",
        "- Standard written order with correct HCPCS coding",
        "- Treating practitioner's medical record that has adequate medical necessity information",
        "- Proof of delivery with face-to-face encounter 7 months ago",
    ]


def test_docling_chunks_reject_content_without_a_usable_chunk() -> None:
    with pytest.raises(RuntimeError, match="no usable CMS source chunks"):
        _chunks("too short")


def test_joined_docling_bullets_map_to_distinct_source_fragments() -> None:
    joined = (
        "- Standard written order with correct HCPCS coding "
        "- Treating practitioner's medical record that has adequate medical necessity information "
        "- Proof of delivery with face-to-face encounter 7 months ago"
    )
    assert _source_fragments(joined)[1:] == [
        "Standard written order with correct HCPCS coding",
        "Treating practitioner's medical record that has adequate medical necessity information",
        "Proof of delivery with face-to-face encounter 7 months ago",
    ]
    evidence = [{"chunk_id": "chunk-5", "rank": 1, "score": 0.9, "text": joined}]
    assert _source_scope(evidence, "submitted_order")["text"] == "Standard written order with correct HCPCS coding"
    assert _source_scope(evidence, "submitted_medical_record")["text"] == (
        "Treating practitioner's medical record that has adequate medical necessity information"
    )
    assert _source_scope(evidence, "submitted_proof_of_delivery")["text"] == (
        "Proof of delivery with face-to-face encounter 7 months ago"
    )


def requirements_data() -> dict[str, str]:
    return {
        "hcpcs_code": "L1851",
        "authorization_requirement_text": "prior authorization is required",
        "face_to_face_requirement_text": "a face-to-face encounter is required",
        "written_order_requirement_text": "written order prior to delivery is required",
        "face_to_face_window_text": "within the 6 months before prescribing the item",
    }


def submission_data() -> dict[str, str]:
    return {
        "submitted_order": "Standard written order with correct HCPCS coding",
        "submitted_medical_record": (
            "Treating practitioner's medical record that has adequate medical necessity information"
        ),
        "submitted_proof_of_delivery": "Proof of delivery with face-to-face encounter 7 months ago",
        "documented_face_to_face_age": "7 months ago",
    }


def outcome_data() -> dict[str, str]:
    return {
        "missing_documentation": (
            "The doctor didn't document the face-to-face encounter within 6 months of proof of delivery."
        ),
        "review_conclusion": "The review contractor completes the claim as an insufficient documentation error.",
        "payment_action": "The MAC recoups payment.",
    }


def ranked_evidence() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-0",
            "rank": 1,
            "score": 0.99,
            "text": (
                "We require prior authorization, a face-to-face encounter, and written order prior to delivery for "
                "HCPCS codes L1832 and L1851. Conduct the face-to-face encounter within the 6 months before "
                "prescribing the item."
            ),
        },
        {
            "chunk_id": "chunk-1",
            "rank": 2,
            "score": 0.98,
            "text": (
                "A supplier bills the claim for L1851 and submits a standard written order with correct HCPCS coding, "
                "a treating practitioner's medical record with adequate medical necessity information, and proof of "
                "delivery with face-to-face encounter 7 months ago. The review contractor completes the claim as an "
                "insufficient documentation error, and the MAC recoups payment."
            ),
        },
    ]


def source_scope_evidence() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-0",
            "rank": 1,
            "score": 0.99,
            "text": (
                "We require prior authorization, a face-to-face encounter, and written order prior to delivery for "
                "HCPCS codes L1832 and L1851. Conduct the face-to-face encounter within the 6 months before "
                "prescribing the item."
            ),
        },
        {
            "chunk_id": "chunk-1",
            "rank": 2,
            "score": 0.98,
            "text": (
                "A supplier bills the claim for L1851 (Knee orthosis (KO), single upright, thigh and calf, with "
                "adjustable flexion and extension joint) and submits the following documentation."
            ),
        },
        {
            "chunk_id": "chunk-2",
            "rank": 3,
            "score": 0.97,
            "text": (
                "- Standard written order with correct HCPCS coding\n"
                "- Treating practitioner's medical record that has adequate medical necessity information\n"
                "- Proof of delivery with face-to-face encounter 7 months ago"
            ),
        },
        {
            "chunk_id": "chunk-3",
            "rank": 4,
            "score": 0.96,
            "text": "The doctor didn’t document the face-to-face encounter within 6 months of proof of delivery.",
        },
        {
            "chunk_id": "chunk-4",
            "rank": 5,
            "score": 0.95,
            "text": (
                "The review contractor completes the claim as an insufficient documentation error, and the MAC "
                "recoups payment."
            ),
        },
    ]


def build_valid_review() -> dict[str, object]:
    return build_review(requirements_data(), submission_data(), outcome_data(), ranked_evidence())


def test_gliner2_groups_cover_every_canonical_field() -> None:
    canonical_fields = set(REQUIREMENT_FIELDS) | set(SUBMISSION_FIELDS) | set(OUTCOME_FIELDS)
    assert set(FIELD_SPECS) == canonical_fields
    mapped_fields = {field for fields in GROUP_FIELDS.values() for field in fields}
    assert mapped_fields == canonical_fields
    assert set(GLINER2_GROUP_LABELS) == set(GROUP_FIELDS)
    assert set(GLINER2_REQUIRED_SPANS) == set(GROUP_FIELDS)


def test_each_field_gets_the_smallest_exact_ranked_source_scope() -> None:
    evidence = source_scope_evidence()
    scopes = {field: _source_scope(evidence, field) for field in FIELD_SPECS}
    assert scopes["face_to_face_window_text"]["text"] == (
        "Conduct the face-to-face encounter within the 6 months before prescribing the item."
    )
    assert scopes["submitted_order"]["text"] == "Standard written order with correct HCPCS coding"
    assert scopes["documented_face_to_face_age"]["text"] == (
        "Proof of delivery with face-to-face encounter 7 months ago"
    )
    assert scopes["payment_action"]["chunk_id"] == "chunk-4"


def test_source_scope_rejects_evidence_without_the_field_terms() -> None:
    with pytest.raises(RuntimeError, match="omitted the exact source scope"):
        _source_scope(
            [{"chunk_id": "chunk-0", "rank": 1, "score": 0.9, "text": "Unrelated CMS source text."}],
            "hcpcs_code",
        )


def test_group_inputs_deduplicate_the_smallest_exact_field_scopes() -> None:
    evidence = source_scope_evidence()
    requirements = _group_source_scope(evidence, "requirements")
    submission = _group_source_scope(evidence, "submission")
    outcome = _group_source_scope(evidence, "outcome")
    assert len(requirements["text"].split("\n\n")) == 2
    assert len(submission["text"].split("\n\n")) == 4
    assert len(outcome["text"].split("\n\n")) == 2
    assert requirements["chunk_ids"] == ["chunk-0"]
    assert submission["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert outcome["chunk_ids"] == ["chunk-3", "chunk-4"]


def test_gliner2_evidence_gate_accepts_required_exact_spans() -> None:
    response = {
        "entities": [
            {"text": "prior authorization"},
            {"text": "a face-to-face encounter"},
            {"text": "written order prior to delivery"},
            {"text": "within the 6 months before prescribing the item"},
        ]
    }
    _require_gliner2_group_evidence(response, "requirements")


def test_gliner2_evidence_gate_rejects_a_missing_required_span() -> None:
    response = {
        "entities": [
            {"text": "within 6 months"},
            {"text": "insufficient documentation error"},
            {"text": "the MAC recoups"},
        ]
    }
    try:
        _require_gliner2_group_evidence(response, "outcome")
    except RuntimeError as exc:
        assert "payment" in str(exc)
    else:
        raise AssertionError("Accepted GLiNER2 evidence without the payment span")


def test_fixture_preserves_the_published_cms_case() -> None:
    source = (ROOT / "fixtures" / "cms-l1851-insufficient-documentation.html").read_text(encoding="utf-8")
    assert "Proof of delivery with face-to-face encounter 7 months ago" in source
    assert "The doctor didn’t document the face-to-face encounter within 6 months of proof of delivery." in source
    assert "The review contractor completes the claim as an insufficient documentation error" in source
    assert "the MAC recoups" in source


def test_reproduces_cms_route_and_one_month_gap() -> None:
    review = build_valid_review()
    assert review["route"] == "insufficient_documentation"
    assert review["hcpcs_code"] == "L1851"
    assert review["required_face_to_face_within_months"] == 6
    assert review["documented_face_to_face_age_months"] == 7
    assert review["overdue_by_months"] == 1
    assert "MAC recoups payment" in review["payment_action"]
    assert "insufficient documentation error" in review["review_conclusion"]
    assert "within 6 months of proof of delivery" in review["missing_documentation"][0]


def test_review_stays_inside_published_example_boundary() -> None:
    review = build_valid_review()
    assert review["coverage_decision"] is None
    assert review["medical_decision"] is None
    assert all(check.passed for check in evaluate_review(review))


def test_evaluator_records_its_artifact_in_the_manifest(tmp_path: Path) -> None:
    (tmp_path / "review.json").write_text(json.dumps(build_valid_review()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": "review.json", "sha256": "preserved"},
                    {"path": "evaluation.json", "sha256": "stale"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert evaluate_run(tmp_path) is True

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in manifest["artifacts"]] == ["evaluation.json", "review.json"]
    evaluation = tmp_path / "evaluation.json"
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert manifest["artifacts"][1]["sha256"] == "preserved"


def test_evaluator_fails_when_the_one_month_gap_is_removed(tmp_path: Path) -> None:
    review = build_valid_review()
    review["overdue_by_months"] = 0
    checks = evaluate_review(review)
    one_month_gap = next(check for check in checks if check.name == "one-month-gap")
    assert one_month_gap.passed is False

    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    assert evaluate_run(tmp_path) is False
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))["passed"] is False


def test_evaluate_run_reports_a_missing_review_artifact(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Review artifact not found"):
        evaluate_run(tmp_path)


def test_evaluate_main_exits_one_when_checks_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate_module, "evaluate_run", lambda _run_dir: False)
    monkeypatch.setattr(sys, "argv", ["eval-pa", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        evaluate_module.main()
    assert exc_info.value.code == 1


def test_run_reports_an_existing_run_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(review_module, "RUNS_DIR", tmp_path)
    (tmp_path / "local").mkdir()
    with pytest.raises(SystemExit, match="Choose a new --run-id"):
        review_module.run("local")


def test_fails_closed_when_required_window_changes() -> None:
    requirements = requirements_data()
    requirements["face_to_face_window_text"] = "within 8 months"
    try:
        build_review(requirements, submission_data(), outcome_data(), ranked_evidence())
    except RuntimeError as exc:
        assert "wrong face-to-face window" in str(exc)
    else:
        raise AssertionError("build_review accepted a requirement that conflicts with the CMS source")


def test_fails_closed_when_gliner2_omits_a_field() -> None:
    outcome = outcome_data()
    del outcome["payment_action"]
    try:
        build_review(requirements_data(), submission_data(), outcome, ranked_evidence())
    except RuntimeError as exc:
        assert "omitted required fields" in str(exc)
    else:
        raise AssertionError("build_review accepted an incomplete GLiNER2 outcome")


def test_fails_closed_without_ranked_payment_action() -> None:
    evidence = ranked_evidence()
    evidence[1]["text"] = str(evidence[1]["text"]).replace("and the MAC recoups payment.", "")
    try:
        build_review(requirements_data(), submission_data(), outcome_data(), evidence)
    except RuntimeError as exc:
        assert "mac recoups payment" in str(exc)
    else:
        raise AssertionError("build_review accepted ranked evidence without the published payment action")


def test_ranked_evidence_requires_chunk_identity_and_source_text() -> None:
    for malformed in (
        [{"rank": 1, "score": 0.9, "text": "source text"}],
        [{"chunk_id": "chunk-0", "rank": 1, "score": 0.9, "text": ""}],
    ):
        with pytest.raises(RuntimeError, match="chunk identity and source text"):
            _require_ranked_evidence(malformed)


def test_fails_closed_when_submission_age_is_not_seven_months() -> None:
    submission = submission_data()
    submission["documented_face_to_face_age"] = "5 months ago"
    try:
        build_review(requirements_data(), submission, outcome_data(), ranked_evidence())
    except RuntimeError as exc:
        assert "does not establish an overdue" in str(exc)
    else:
        raise AssertionError("build_review accepted a timing observation that conflicts with the CMS source")


def test_verified_manifest_hashes() -> None:
    manifest_path = ROOT / "verified-run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    assert hashlib.sha256((ROOT / fixture["path"]).read_bytes()).hexdigest() == fixture["sha256"]
    for entry in manifest["artifacts"]:
        artifact = manifest_path.parent / entry["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    raw_dir = manifest_path.parent / "raw"
    retrieve = json.loads((raw_dir / "retrieve.json").read_text(encoding="utf-8"))
    rerank_request = json.loads((raw_dir / "rerank-request.json").read_text(encoding="utf-8"))
    rerank = json.loads((raw_dir / "rerank.json").read_text(encoding="utf-8"))
    assert retrieve["query"]["id"] == "cms-l1851-query"
    assert rerank_request["query"]["id"] == retrieve["query"]["id"]
    assert [item["id"] for item in rerank_request["items"]] == [row["chunk_id"] for row in retrieve["ranking"]]
    assert {row["item_id"] for row in rerank["scores"]} == {item["id"] for item in rerank_request["items"]}

    provenance = manifest["rate_book_provenance"]
    assert provenance == _rate_book_provenance(raw_dir)
    assert len(provenance["request_ids"]) == 22
    assert provenance["request_versions"] == {
        request_id: provenance["version"] for request_id in provenance["request_ids"]
    }
    assert "raw/retrieve.json" in provenance["source_artifacts"]


def _write_provenance_payload(path: Path, request: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"request": request}) + "\n",
        encoding="utf-8",
    )


def test_rate_book_provenance_rejects_a_charged_request_without_an_id(
    tmp_path: Path,
) -> None:
    _write_provenance_payload(
        tmp_path / "missing-id.json",
        {"credits_debited": 1, "rate_book_version": "rate-v1"},
    )

    with pytest.raises(RuntimeError, match="charged request without an ID"):
        _rate_book_provenance(tmp_path)


def test_rate_book_provenance_rejects_duplicate_request_ids(tmp_path: Path) -> None:
    request = {
        "id": "request-1",
        "credits_debited": 1,
        "rate_book_version": "rate-v1",
    }
    _write_provenance_payload(tmp_path / "first.json", request)
    _write_provenance_payload(tmp_path / "second.json", request)

    with pytest.raises(RuntimeError, match="duplicate charged request IDs"):
        _rate_book_provenance(tmp_path)


@pytest.mark.parametrize(
    "versions",
    [("rate-v1", "rate-v2")],
)
def test_rate_book_provenance_requires_one_settled_version(
    tmp_path: Path,
    versions: tuple[str, ...],
) -> None:
    for index, version in enumerate(versions):
        request: dict[str, object] = {
            "id": f"request-{index}",
            "credits_debited": 1,
        }
        if version:
            request["rate_book_version"] = version
        _write_provenance_payload(tmp_path / f"request-{index}.json", request)

    with pytest.raises(RuntimeError, match="one settled rate book"):
        _rate_book_provenance(tmp_path)


def test_rate_book_provenance_rejects_a_charged_request_without_its_own_version(
    tmp_path: Path,
) -> None:
    _write_provenance_payload(
        tmp_path / "complete.json",
        {
            "id": "request-1",
            "credits_debited": 1,
            "rate_book_version": "rate-v1",
        },
    )
    _write_provenance_payload(
        tmp_path / "missing.json",
        {"id": "request-2", "credits_debited": 1},
    )

    with pytest.raises(RuntimeError, match="without a rate-book version"):
        _rate_book_provenance(tmp_path)


def test_rate_book_provenance_prefers_request_usage(tmp_path: Path) -> None:
    (tmp_path / "nested-usage.json").write_text(
        json.dumps(
            {
                "request": {
                    "id": "request-1",
                    "credits_debited": 1,
                    "usage": {"rate_book_version": "nested-rate-v1"},
                },
                "usage": {"rate_book_version": "outer-rate-v1"},
            }
        ),
        encoding="utf-8",
    )

    assert _rate_book_provenance(tmp_path)["version"] == "nested-rate-v1"


def test_rate_book_provenance_rejects_conflicting_request_versions(
    tmp_path: Path,
) -> None:
    (tmp_path / "conflict.json").write_text(
        json.dumps(
            {
                "request": {
                    "id": "request-1",
                    "credits_debited": 1,
                    "rate_book_version": "direct-rate-v1",
                    "usage": {"rate_book_version": "usage-rate-v2"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="conflicting rate-book versions"):
        _rate_book_provenance(tmp_path)
