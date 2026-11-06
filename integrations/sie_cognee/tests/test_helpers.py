"""Fail-closed boundaries for literal sources and native inference outputs."""

import json
from datetime import UTC, datetime
from enum import Enum
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sie_cognee.decision import EvidenceOverflowError, decide, evidence_packet
from sie_cognee.documents import SourceDocument, ocr_page
from sie_cognee.retrieval import SourceChunk, normalize_chunks, rerank_chunks


def test_source_document_preserves_literal_text_and_provenance():
    text = "https://example.invalid/not-an-ingestion-target"
    document = SourceDocument(text, "manual-page-2", "manual.pdf", 2)
    item = document.to_data_item()
    assert item.literal_text is True
    assert item.data == text
    assert item.external_metadata == document.metadata
    assert item.external_metadata["source_id"] == "manual-page-2"
    assert len(item.external_metadata["content_sha256"]) == 64


@pytest.mark.parametrize("reply", [{}, {"entities": []}, {"entities": [{"text": 1}]}, {"entities": [{"text": " "}]}])
def test_ocr_rejects_empty_or_malformed_text(reply):
    client = MagicMock()
    client.extract.return_value = reply
    with pytest.raises(ValueError, match=r"OCR|Source text"):
        ocr_page(client, model="ocr", image=b"image", source_id="p1", filename="image.png", page_number=1)


def test_ocr_preserves_page_and_checks_input_budget():
    client = MagicMock()
    client.extract.return_value = {"entities": [{"text": "literal"}, {"text": "page"}]}
    result = ocr_page(client, model="ocr", image=b"image", source_id="p2", filename="manual.pdf", page_number=2)
    assert result.text == "literal\npage"
    assert result.metadata["page_number"] == 2
    assert client.extract.call_args.args[1]["images"] == [b"image"]
    client.reset_mock()
    with pytest.raises(ValueError, match="max_bytes"):
        ocr_page(client, model="ocr", image=b"too big", source_id="p1", filename="a.png", page_number=1, max_bytes=1)
    client.extract.assert_not_called()


def test_normalizes_actual_released_cognee_chunk_objects():
    from cognee.api.v1.search import SearchType
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry

    entry = ResponseGraphEntry(
        source="graph",
        kind="chunk",
        search_type=SearchType.CHUNKS,
        text="Source, not a completion",
        metadata={"chunk_id": "c1", "data_id": "d1"},
        raw={"external_metadata": {"filename": "manual.pdf"}},
    )
    chunk = normalize_chunks([entry])[0]
    assert chunk.chunk_id == "c1"
    assert chunk.metadata["data_id"] == "d1"
    assert chunk.raw == entry.raw
    assert normalize_chunks([entry.model_dump()]) == [chunk]
    with pytest.raises(ValueError, match="unique"):
        normalize_chunks([entry, entry])
    with pytest.raises(ValueError, match="source chunks"):
        normalize_chunks([entry.model_copy(update={"kind": "graph_completion"})])


def test_chunk_metadata_remains_json_serializable():
    from cognee.api.v1.search import SearchType
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry

    class SourceKind(Enum):
        NOTE = "note"

    entry = ResponseGraphEntry(
        source="graph",
        kind="chunk",
        search_type=SearchType.CHUNKS,
        text="Source text",
        metadata={"chunk_id": "c1"},
        raw={
            "data_id": UUID("bfa87d58-78b9-446c-a044-d371103e82f6"),
            "created_at": datetime(2026, 1, 2, tzinfo=UTC),
            "source_kind": SourceKind.NOTE,
        },
    )
    chunk = normalize_chunks([entry])[0]
    assert json.loads(json.dumps(chunk.raw)) == {
        "data_id": "bfa87d58-78b9-446c-a044-d371103e82f6",
        "created_at": "2026-01-02T00:00:00Z",
        "source_kind": "note",
    }


@pytest.mark.parametrize("result", ["generated context", [{"kind": "chunk", "text": "text", "metadata": {}}]])
def test_rejects_unidentified_or_generated_evidence(result):
    with pytest.raises(ValueError, match=r"Expected|Source chunks"):
        normalize_chunks(result)


@pytest.fixture
def chunks():
    return [SourceChunk("a", "Projector confirmed"), SourceChunk("b", "Audio confirmed"), SourceChunk("c", "Unrelated")]


def test_rerank_aligns_ids_and_keeps_stable_ties(chunks):
    client = MagicMock()
    client.score.return_value = {
        "scores": [
            {"item_id": "c", "score": -1},
            {"item_id": "b", "score": 0.8},
            {"item_id": "a", "score": 0.8},
        ]
    }
    result = rerank_chunks(client, "equipment", chunks, model="reranker", top_k=2)
    assert [entry.chunk.chunk_id for entry in result] == ["a", "b"]
    assert [entry["id"] for entry in client.score.call_args.args[2]] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [{"item_id": "a", "score": True}],
        [{"item_id": "a", "score": float("nan")}],
        [{"item_id": "a", "score": float("inf")}],
        [{"item_id": "a"}],
        [{"item_id": "unknown", "score": 1}],
        [{"item_id": "a", "score": 1}, {"item_id": "a", "score": 1}],
    ],
)
def test_rerank_rejects_incomplete_or_invalid_scores(chunks, scores):
    client = MagicMock()
    client.score.return_value = {"scores": scores}
    with pytest.raises(ValueError, match="Reranker"):
        rerank_chunks(client, "equipment", chunks, model="reranker")


def test_packet_is_exact_and_overflow_never_truncates(chunks):
    packet = evidence_packet("ready?", chunks, max_chars=200)
    assert packet == "Question: ready?\n[a] Projector confirmed\n[b] Audio confirmed\n[c] Unrelated"
    with pytest.raises(EvidenceOverflowError, match="select fewer"):
        evidence_packet("ready?", chunks, max_chars=10)


def _decision(client):
    return decide(
        client,
        "Evidence",
        model="decide",
        choices={"proceed": "ready", "review": "uncertain"},
        instructions="Recommend a next step",
        max_chars=100,
    )


def test_decision_preserves_model_choice_and_error_overflow_policy():
    client = MagicMock()
    client.extract.return_value = {
        "data": {"action": {"choice": "review", "probabilities": {"proceed": 0.2, "review": 0.8}}}
    }
    result = _decision(client)
    assert result.choice == "review"
    assert result.probabilities == {"proceed": 0.2, "review": 0.8}
    assert client.extract.call_args.kwargs["options"] == {"overflow_policy": "error"}
    assert client.extract.call_args.kwargs["output_schema"]["action"]["type"] == "choice"


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        {},
        {"review": 1},
        {"review": True, "proceed": 0},
        {"review": float("nan"), "proceed": 0},
        {"review": float("inf"), "proceed": 0},
        {"review": -0.1, "proceed": 1.1},
        {"review": 0.2, "proceed": 0.2},
        {"review": 1, "proceed": 0, "other": 0},
    ],
)
def test_decision_rejects_malformed_probabilities(probabilities):
    client = MagicMock()
    client.extract.return_value = {"data": {"action": {"choice": "review", "probabilities": probabilities}}}
    with pytest.raises(ValueError, match=r"Decision|Choice"):
        _decision(client)


def test_decision_requires_review_label_and_checks_budget_before_call():
    client = MagicMock()
    with pytest.raises(ValueError, match="review or unknown"):
        decide(client, "a", model="d", choices={"go": "go"}, instructions="i", max_chars=100)
    with pytest.raises(EvidenceOverflowError):
        decide(client, "x" * 100, model="d", choices={"review": "review"}, instructions="i", max_chars=10)
    client.extract.assert_not_called()
