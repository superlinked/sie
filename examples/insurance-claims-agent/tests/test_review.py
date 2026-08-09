from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import insurance_claims.review as review_module
from insurance_claims.evaluate import ARTIFACT_EXCLUDED_PATHS, evaluate_review, evaluate_run
from insurance_claims.review import (
    REVIEW_SCHEMA,
    _extract_claim_facts,
    _final_review,
    _json_object_from_text,
    _require_sources,
    chunk_markdown,
    run_default_stage,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFIED_RUN = ROOT / "verified-run"


class FakeExtractClient:
    def __init__(self) -> None:
        self.labels: list[str] | None = None
        self.item: object | None = None

    def extract(
        self,
        _model: str,
        _item: object,
        **kwargs: object,
    ) -> dict[str, object]:
        self.item = _item
        self.labels = kwargs.get("labels")  # type: ignore[assignment]
        return {"data": {"entities": []}}


class FakeGenerateClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def generate(self, _model: str, _prompt: str, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return {"text": '{"route":"scope_review_required"}'}


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
        "amended proof of loss total",
        "debris removal estimate",
        "barge transportation estimate",
        "debris volume",
    ]


def test_claim_fact_extraction_starts_at_appeal_when_issue_heading_is_absent() -> None:
    client = FakeExtractClient()

    _extract_claim_facts(
        client,
        "fastino/gliner2-large-v1",
        "irrelevant policy preface\n\nThe insurer reviewed the amended claim.",
        60,
    )

    assert isinstance(client.item, dict)
    assert client.item["text"] == "The insurer reviewed the amended claim."


def test_review_json_accepts_fenced_model_output() -> None:
    assert _json_object_from_text('```json\n{"route": "scope_review_required"}\n```') == {
        "route": "scope_review_required"
    }


def test_final_review_uses_native_strict_json_schema() -> None:
    client = FakeGenerateClient()

    _, parsed, _ = _final_review(
        client,
        "Qwen/Qwen3.5-4B",
        appeal_markdown="appeal",
        claim_facts={},
        policy_chunks=[],
        provision_timeout_s=60,
    )

    assert parsed == {"route": "scope_review_required"}
    assert client.kwargs["grammar"] == {
        "json_schema": REVIEW_SCHEMA,
        "label": "insurance_appeal_review",
        "strict": True,
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
            # Synthetic schema-valid category used only to test evaluator tolerance.
            {"category": "other"},
        ],
    }

    assert all(check.passed for check in evaluate_review(review))


def test_evaluation_rejects_the_wrong_proof_of_loss_amount() -> None:
    review = {
        "route": "scope_review_required",
        "appeal_summary": {
            "proof_of_loss_amount": 1,
            "removal_estimate": 49500,
            "barge_estimate": 181832.94,
            "debris_cubic_yards_min": 12,
            "debris_cubic_yards_max": 15,
        },
        "decision": {
            "covered_scope": "Remove flood-borne stones from underneath the insured building to its perimeter.",
            "excluded_scope": "Barge transport, handling, disposal, and yard removal.",
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

    checks = {check.name: check for check in evaluate_review(review)}
    assert {name for name, check in checks.items() if not check.passed} == {"proof-of-loss-amount"}


def test_existing_run_id_has_an_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_module, "RUNS_DIR", tmp_path)
    (tmp_path / "local").mkdir()

    with pytest.raises(FileExistsError, match="Choose a new --run-id or remove that directory"):
        run_default_stage("local")


def test_source_manifest_is_required_during_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"source")
    monkeypatch.setattr(review_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        review_module,
        "load_config",
        lambda: SimpleNamespace(sources=[SimpleNamespace(path=source_path)]),
    )

    with pytest.raises(FileNotFoundError, match="source-manifest.json"):
        _require_sources()


def test_artifact_exclusions_only_apply_at_the_run_root(tmp_path: Path) -> None:
    review = (VERIFIED_RUN / "review.json").read_text(encoding="utf-8")
    (tmp_path / "review.json").write_text(review, encoding="utf-8")
    (tmp_path / "README.md").write_text("run documentation", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"artifacts": []}), encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "README.md").write_text("nested artifact", encoding="utf-8")
    (nested / "manifest.json").write_text("nested artifact manifest", encoding="utf-8")

    assert evaluate_run(tmp_path) is True

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "README.md" not in artifact_paths
    assert "manifest.json" not in artifact_paths
    assert {"nested/README.md", "nested/manifest.json"} <= artifact_paths


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
        if path.is_file() and path.relative_to(VERIFIED_RUN) not in ARTIFACT_EXCLUDED_PATHS
    }
    artifacts = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}

    assert set(artifacts) == expected_paths
    for path, expected_hash in artifacts.items():
        assert hashlib.sha256((VERIFIED_RUN / path).read_bytes()).hexdigest() == expected_hash
