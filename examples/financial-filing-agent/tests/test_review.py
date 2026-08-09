import hashlib
import json
from pathlib import Path

import pytest

from financial_filing.evaluate import evaluate_review
from financial_filing.review import (
    _chunks,
    _original_table_source_value,
    _rate_book_provenance,
    _require_entity_evidence,
    _require_matching_source_values,
    _table_row_context,
    _table_source_values,
    build_review,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_preserves_the_sec_table_rows_and_item_402_sentences() -> None:
    source = (ROOT / "fixtures" / "pathward-filing-packet.html").read_text(encoding="utf-8")
    assert "<td>$45,096</td>" in source
    assert "<td>$36,080</td>" in source
    assert "<td>$1.68</td>" in source
    assert "<td>$1.34</td>" in source
    assert "should no longer be relied upon because of errors identified" in source
    assert "does not impact net income over the life of the portfolio" in source


def test_config_uses_canonical_docling_id() -> None:
    assert load_config()["models"]["parse"] == "docling"


def structured_data() -> dict[str, str]:
    return {
        "company": "Pathward Financial",
        "period": "Three Months Ended June 30, 2023",
        "original_net_income": "$45,096",
        "restated_net_income": "$36,080",
        "original_diluted_eps": "$1.68",
        "restated_diluted_eps": "$1.34",
        "reliance_status": "Affected prior filings should no longer be relied upon.",
    }


def ranked_evidence() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-0",
            "rank": 1,
            "score": 0.99,
            "text": (
                "Pathward Financial, Three Months Ended June 30, 2023. Net income attributable to parent was "
                "$45,096 and diluted EPS was $1.68. The restated table reports $36,080 and $1.34. The affected "
                "periods should no longer be relied upon. The change from net to gross basis presentation does "
                "not impact net income over the life of the portfolio, but changes the timing of when elements "
                "of the programs are recognized for accounting purposes."
            ),
        }
    ]


def test_build_review_computes_source_version_delta() -> None:
    review = build_review(structured_data(), ranked_evidence())
    assert review["change"] == {"value_millions": -9.016, "diluted_eps": -0.34, "percent": -20.0}
    assert review["route"] == "superseded_figure"


def test_verified_review_contract() -> None:
    assert all(check.passed for check in evaluate_review(build_review(structured_data(), ranked_evidence())))


def test_review_fails_closed_on_missing_structured_field() -> None:
    data = structured_data()
    del data["reliance_status"]
    try:
        build_review(data, ranked_evidence())
    except RuntimeError as exc:
        assert "omitted required fields" in str(exc)
    else:
        raise AssertionError("build_review accepted incomplete model evidence")


def test_review_fails_closed_without_exact_company_caveat() -> None:
    evidence = ranked_evidence()
    evidence[0]["text"] = str(evidence[0]["text"]).replace(
        "The change from net to gross basis presentation does not impact net income "
        "over the life of the portfolio, but changes the timing of when elements of "
        "the programs are recognized for accounting purposes.",
        "Net income over the life of the portfolio may change.",
    )
    try:
        build_review(structured_data(), evidence)
    except RuntimeError as exc:
        assert "life-of-portfolio caveat" in str(exc)
    else:
        raise AssertionError("build_review accepted evidence without the exact company caveat")


def test_review_fails_closed_without_ranked_evidence() -> None:
    try:
        build_review(structured_data(), [])
    except RuntimeError as exc:
        assert "no evidence" in str(exc)
    else:
        raise AssertionError("build_review accepted an empty reranker response")


def test_gliner_gate_requires_all_primary_source_amounts() -> None:
    complete_entities = {
        "entities": [
            {"text": "Pathward Financial"},
            {"text": "June 30, 2023"},
            {"text": "$45,096"},
            {"text": "$36,080"},
            {"text": "$1.68"},
            {"text": "$1.34"},
        ]
    }
    _require_entity_evidence(complete_entities)
    complete_entities["entities"] = [entity for entity in complete_entities["entities"] if entity["text"] != "$36,080"]
    try:
        _require_entity_evidence(complete_entities)
    except RuntimeError as exc:
        assert "required source spans" in str(exc)
    else:
        raise AssertionError("GLiNER gate accepted evidence without the restated amount")


def test_table_row_context_keeps_heading_and_isolates_requested_row() -> None:
    source = (
        "## Restated Form 10-K/A. Three Months Ended June 30, 2023 "
        "| Metric | Original | Adjustment | Restated | | --- | --- | --- | --- | "
        "| Net income | $45,096 | $(9,016) | $36,080 | | Diluted | $1.68 | | $1.34 | "
        "| Basic | $1.60 | | $1.20 |"
    )

    context = _table_row_context(source, "Diluted")

    assert "Restated Form 10-K/A" in context
    assert "Three Months Ended June 30, 2023" in context
    assert "| Diluted | $1.68 | | $1.34 |" in context
    assert "Net income" not in context
    assert "Basic" not in context


def test_docling_line_breaks_do_not_split_source_sections() -> None:
    markdown = """# Pathward Q3 FY2023 source packet

## Original Form 10-Q

Three Months Ended June 30, 2023. Net income

was $45,096 and diluted EPS was $1.68.

## Restated Form 10-K/A

Three Months Ended June 30, 2023. Restated net income

was $36,080 and diluted EPS was $1.34.
"""
    assert _chunks(markdown) == [
        "# Pathward Q3 FY2023 source packet",
        "## Original Form 10-Q Three Months Ended June 30, 2023. Net income was $45,096 and diluted EPS was $1.68.",
        (
            "## Restated Form 10-K/A Three Months Ended June 30, 2023. "
            "Restated net income was $36,080 and diluted EPS was $1.34."
        ),
    ]


def test_original_values_are_read_from_the_original_filing_and_cross_checked() -> None:
    original = (
        "## Original Form 10-Q | Net income attributable to parent | $45,096 | $22,391 | | Diluted | $1.68 | $0.76 |"
    )
    restated = (
        "## Restated Form 10-K/A "
        "| Net income attributable to parent | $45,096 | $(9,016) | | $36,080 | "
        "| Diluted | $1.68 | | | $1.34 |"
    )
    original_income = _original_table_source_value(original, "Net income attributable to parent")
    original_eps = _original_table_source_value(original, "Diluted")
    previously_reported_income, restated_income = _table_source_values(restated, "Net income attributable to parent")
    previously_reported_eps, restated_eps = _table_source_values(restated, "Diluted")
    _require_matching_source_values(original_income, previously_reported_income, "net income")
    _require_matching_source_values(original_eps, previously_reported_eps, "diluted EPS")
    assert (original_income, restated_income, original_eps, restated_eps) == ("$45,096", "$36,080", "$1.68", "$1.34")


def test_cross_filing_check_rejects_a_mismatched_previously_reported_value() -> None:
    try:
        _require_matching_source_values("$45,096", "$45,097", "net income")
    except RuntimeError as exc:
        assert "values disagree" in str(exc)
    else:
        raise AssertionError("Accepted a previously reported value that differs from the original Form 10-Q")


def test_verified_manifest_hashes() -> None:
    manifest_path = ROOT / "verified-run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    assert hashlib.sha256((ROOT / fixture["path"]).read_bytes()).hexdigest() == fixture["sha256"]
    for entry in manifest["artifacts"]:
        path = Path(entry["path"])
        resolved = ROOT / path if path.parts[0] == "verified-run" else manifest_path.parent / path
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == entry["sha256"]

    raw_dir = manifest_path.parent / "raw"
    retrieve = json.loads((raw_dir / "retrieve.json").read_text(encoding="utf-8"))
    rerank_request = json.loads((raw_dir / "rerank-request.json").read_text(encoding="utf-8"))
    rerank = json.loads((raw_dir / "rerank.json").read_text(encoding="utf-8"))
    assert rerank_request["query"]["id"] == retrieve["query"]["id"]
    assert [item["id"] for item in rerank_request["items"]] == [row["chunk_id"] for row in retrieve["ranking"]]
    assert {row["item_id"] for row in rerank["scores"]} == {item["id"] for item in rerank_request["items"]}

    provenance = manifest["rate_book_provenance"]
    assert provenance == _rate_book_provenance(raw_dir)
    assert len(provenance["request_ids"]) == 13
    assert provenance["request_versions"] == {
        request_id: provenance["version"] for request_id in provenance["request_ids"]
    }
    assert "raw/retrieve.json" in provenance["source_artifacts"]


def test_rate_book_provenance_rejects_a_charged_request_without_its_own_version(
    tmp_path: Path,
) -> None:
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "request": {
                    "id": "request-1",
                    "credits_debited": 1,
                    "rate_book_version": "rate-v1",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "missing.json").write_text(
        json.dumps({"request": {"id": "request-2", "credits_debited": 1}}),
        encoding="utf-8",
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
