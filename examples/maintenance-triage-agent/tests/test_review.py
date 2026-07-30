import hashlib
import json
from pathlib import Path

from maintenance_triage.evaluate import evaluate_review
from maintenance_triage.review import (
    _map_exact_source_fields,
    _require_entity_evidence,
    _require_gliner2_evidence,
    _require_ranked_evidence,
    _row_for_location,
    build_review,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_is_the_exact_ntsb_page_spread() -> None:
    source = ROOT / "fixtures" / "east-palestine-bearing-spread.pdf"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "0556a971978198a493acda277dfdfd75a837be4ba5c626d6f03565cb17cee8ca"
    )


def test_config_uses_canonical_docling_id() -> None:
    assert load_config()["models"]["parse"] == "docling"


def structured_data() -> dict[str, str]:
    return {
        "bearing": "L1",
        "sebring_time": "7:37 p.m.",
        "sebring_temperature": "38°F above ambient",
        "salem_time": "8:13 p.m.",
        "salem_temperature": "103°F above ambient",
        "east_palestine_time": "8:52 p.m.",
        "east_palestine_temperature": "253°F above ambient",
        "sebring_alert_status_text": "not high enough to trigger an alert",
        "salem_alert_level_text": "noncritical alert",
        "salem_alert_recipient_text": "Wayside Help Desk",
        "salem_crew_notification_text": "not to the crew",
        "salem_camera_observation_text": "fire near the bearing",
        "east_palestine_alert_text": "critical alarm broadcast in the locomotive cab",
        "engineer_action": "began to slow the train before 8:54 p.m.",
        "derailment_statement": "the hopper car and 37 others derailed",
        "cause_statement": "an overheated bearing burned off the accident hopper car",
    }


def ranked_evidence() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-0",
            "rank": 1,
            "score": 0.99,
            "text": (
                "At 7:37 p.m., at the Sebring HBD, the L1 bearing was 38°F above ambient, not high enough to "
                "trigger an alert. At 8:13 p.m., at the Salem HBD, it was 103°F above ambient and triggered a "
                "noncritical alert to the Wayside Help Desk, but not to the crew. A camera showed fire near the "
                "bearing. At East Palestine it reached 253°F above ambient, and a critical alarm was broadcast "
                "in the locomotive cab. The engineer began to slow the train before 8:54 p.m. The hopper car and "
                "37 others derailed. The East Palestine derailment began when an overheated bearing burned off "
                "the accident hopper car."
            ),
        }
    ]


def exact_source_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detectors = [
        {
            "chunk_id": "chunk-11",
            "rank": 1,
            "score": 0.99,
            "text": (
                "At 7:37 p.m. , at the Sebring HBD, the reading for the L1 bearing was only 38°F above ambient "
                "-not high enough to trigger an alert. The R1 bearing measured 20°F above ambient. "
                "But the L1 bearing was hotter."
            ),
        },
        {
            "chunk_id": "chunk-12",
            "rank": 2,
            "score": 0.98,
            "text": (
                "At 8:13 p.m. , at the Salem HBD, the reading for the L1 bearing was 103°F above ambient . "
                "This reading was high enough to trigger a noncritical alert to the Wayside Help Desk ( see box ), "
                "but not to the crew. A surveillance camera picture from Salem showed fire near the bearing."
            ),
        },
        {
            "chunk_id": "chunk-13",
            "rank": 3,
            "score": 0.97,
            "text": (
                "About 8:52 p.m ., the train went over the East Palestine HBD. Less than a minute later, the HBD "
                "recorded a temperature of 253°F above ambient at the L1 bearing; bearing R1 remained at 20°F "
                "above ambient. The HBD immediately transmitted a critical alarm, which was broadcast in the "
                "locomotive cab."
            ),
        },
    ]
    outcome = [
        {
            "chunk_id": "chunk-0",
            "rank": 4,
            "score": 0.96,
            "text": "The East Palestine derailment began when an overheated bearing burned off the accident hopper car.",
        },
        {
            "chunk_id": "chunk-14",
            "rank": 5,
            "score": 0.95,
            "text": (
                "Under NS rules, the engineer began to slow the train before 8:54 p.m. But it was too late; "
                "the hopper car and 37 others derailed as the train's emergency braking system activated."
            ),
        },
    ]
    return detectors, outcome


def test_maps_only_exact_ranked_source_text_after_model_gates() -> None:
    detectors, outcome = exact_source_rows()
    mapped, scopes = _map_exact_source_fields(detectors, outcome)
    assert mapped["bearing"] == "L1"
    assert mapped["east_palestine_time"] == "8:52 p.m ."
    assert mapped["sebring_temperature"] == "38°F above ambient"
    assert mapped["salem_alert_recipient_text"] == "Wayside Help Desk"
    assert mapped["east_palestine_alert_text"] == ("critical alarm, which was broadcast in the locomotive cab")
    assert mapped["cause_statement"].startswith("The East Palestine derailment began")
    assert scopes["salem_alert_recipient_text"]["chunk_id"] == "chunk-12"
    assert scopes["cause_statement"]["chunk_id"] == "chunk-0,chunk-14"


def test_gliner2_gate_requires_recorded_detector_spans() -> None:
    complete = {
        "entities": [
            {"text": "8:13 p.m."},
            {"text": "Salem HBD"},
            {"text": "L1 bearing"},
            {"text": "103°F"},
            {"text": "noncritical"},
            {"text": "Wayside Help Desk"},
            {"text": "fire near the bearing"},
        ]
    }
    _require_gliner2_evidence(complete, "salem")
    complete["entities"] = [entity for entity in complete["entities"] if entity["text"] != "Wayside Help Desk"]
    try:
        _require_gliner2_evidence(complete, "salem")
    except RuntimeError as exc:
        assert "wayside help desk" in str(exc)
    else:
        raise AssertionError("GLiNER2 gate accepted Salem evidence without the alert recipient")


def test_preserves_the_recorded_output_schema_request_and_response() -> None:
    request = json.loads((ROOT / "fixtures" / "output-schema-probe-request.json").read_text(encoding="utf-8"))
    response = json.loads((ROOT / "fixtures" / "output-schema-probe-response.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / request["item"]["source_artifact"]).read_text(encoding="utf-8"))
    by_id = {row["chunk_id"]: row["text"] for row in source["ranking"]}
    reconstructed_text = request["item"]["join_separator"].join(
        by_id[chunk_id] for chunk_id in request["item"]["source_chunk_ids"]
    )
    assert reconstructed_text == request["item"]["text"]
    assert hashlib.sha256(reconstructed_text.encode()).hexdigest() == request["item"]["text_sha256"]
    assert set(request["output_schema"]["required"]) == set(response["data"])
    assert response["model"] == "fastino/gliner2-large-v1"
    assert response["data"]["salem_recipient_source_text"] == ["Wayside Help Desk"]
    assert response["data"]["sebring_source_phrase"] == []
    assert response["data"]["east_palestine_alarm_source_text"] == []


def test_build_review_reconstructs_published_detector_trend() -> None:
    review = build_review(structured_data(), ranked_evidence())
    assert review["trend"] == {
        "successive_increases_degrees_f": [65, 150],
        "total_increase_degrees_f": 215,
        "sebring_to_salem_minutes": 36,
        "salem_to_east_palestine_minutes": 39,
    }
    assert review["derailment"]["total_cars"] == 38
    assert review["new_causal_inferences"] == []
    assert review["control_writes"] == []


def test_review_boundary_passes_evaluation() -> None:
    assert all(check.passed for check in evaluate_review(build_review(structured_data(), ranked_evidence())))


def test_review_fails_closed_on_changed_temperature() -> None:
    data = structured_data()
    data["salem_temperature"] = "130°F above ambient"
    try:
        build_review(data, ranked_evidence())
    except RuntimeError as exc:
        assert "temperatures do not match" in str(exc)
    else:
        raise AssertionError("build_review accepted a detector reading that differs from the NTSB source")


def test_review_fails_closed_without_exact_ntsb_cause_sentence() -> None:
    evidence = ranked_evidence()
    evidence[0]["text"] = str(evidence[0]["text"]).replace(
        "The East Palestine derailment began when an overheated bearing burned off the accident hopper car.",
        "The derailment began after a bearing problem.",
    )
    try:
        build_review(structured_data(), evidence)
    except RuntimeError as exc:
        assert "exact cause statement" in str(exc)
    else:
        raise AssertionError("build_review accepted evidence without the NTSB's exact cause statement")


def test_review_fails_closed_without_ranked_evidence() -> None:
    try:
        build_review(structured_data(), [])
    except RuntimeError as exc:
        assert "no evidence" in str(exc)
    else:
        raise AssertionError("build_review accepted an empty reranker response")


def test_ranked_evidence_requires_chunk_identity_and_source_text() -> None:
    for malformed in (
        [{"rank": 1, "score": 0.9, "text": "source text"}],
        [{"chunk_id": "chunk-0", "rank": 1, "score": 0.9, "text": ""}],
    ):
        try:
            _require_ranked_evidence(malformed)
        except RuntimeError as exc:
            assert "chunk identity and source text" in str(exc)
        else:
            raise AssertionError("Accepted malformed ranked evidence")


def test_detector_location_lookup_rejects_ambiguous_rows() -> None:
    rows = [
        {"chunk_id": "chunk-0", "rank": 1, "text": "Sebring HBD reading"},
        {"chunk_id": "chunk-1", "rank": 2, "text": "Another Sebring HBD reading"},
    ]
    try:
        _row_for_location(rows, "sebring hbd")
    except RuntimeError as exc:
        assert "found 2" in str(exc)
    else:
        raise AssertionError("Accepted ambiguous detector evidence")


def test_gliner_gate_requires_all_detector_spans() -> None:
    complete = {
        "entities": [
            {"text": "Sebring"},
            {"text": "Salem"},
            {"text": "East Palestine"},
            {"text": "7:37 p.m."},
            {"text": "8:13 p.m."},
            {"text": "8:52 p.m."},
            {"text": "L1 bearing"},
            {"text": "noncritical alert"},
            {"text": "critical alarm"},
            {"text": "37 others"},
        ]
    }
    _require_entity_evidence(complete)
    complete["entities"] = [entity for entity in complete["entities"] if entity["text"] != "8:13 p.m."]
    try:
        _require_entity_evidence(complete)
    except RuntimeError as exc:
        assert "NTSB source spans" in str(exc)
    else:
        raise AssertionError("GLiNER gate accepted evidence without the Salem event time")


def test_verified_manifest_hashes() -> None:
    manifest_path = ROOT / "verified-run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    assert hashlib.sha256((ROOT / fixture["path"]).read_bytes()).hexdigest() == fixture["sha256"]
    for entry in manifest["diagnostic_fixtures"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    for entry in manifest["artifacts"]:
        artifact = manifest_path.parent / entry["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
