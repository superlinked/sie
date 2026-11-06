"""Offline boundary checks. No model weights, network requests or recorded replies."""

import asyncio
import importlib.util
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from sie_cognee.decision import Decision
from sie_cognee.retrieval import RankedChunk, SourceChunk

from common import ROOT, Demo, endpoint_for, load_demo


def script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"experiment_{name[:2]}", ROOT / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_keys_do_not_cross_origins() -> None:
    env = {
        "SIE_BASE_URL": "https://api.superlinked.com",
        "SIE_API_KEY": "test-only",
        "SIE_LLM_BASE_URL": "http://localhost:8081",
    }
    assert endpoint_for("llm", env).api_key is None
    assert endpoint_for("embedding", env).api_key == "test-only"
    env["SIE_EMBEDDING_API_KEY"] = ""
    with pytest.raises(ValueError, match="SIE_EMBEDDING_API_KEY"):
        endpoint_for("embedding", env)


@pytest.mark.parametrize("url", ["http://user:password@localhost:8080", "http://localhost:8080?key=secret"])
def test_endpoint_does_not_print_secrets(url: str) -> None:
    with pytest.raises(ValueError, match="without credentials") as error:
        endpoint_for("llm", {"SIE_BASE_URL": url})
    assert "secret" not in str(error.value)
    assert "password" not in str(error.value)


def test_config_isolation_and_embedding_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import common

    original = (ROOT / "config.toml").read_text()
    (tmp_path / "config.toml").write_text(original)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("SIE_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("WORKSHOP_NAMESPACE", "test-workshop")
    monkeypatch.setenv("SIE_LLM_MODEL", "Qwen/Qwen3.8-27B-FP8")
    demo = load_demo("test")
    assert demo.models["llm"] == "Qwen/Qwen3.8-27B-FP8"
    assert demo.storage == tmp_path / ".state/test-workshop"
    assert demo.dataset == "sie-test-workshop"
    (tmp_path / "config.toml").write_text(original.replace("embedding_dimensions = 1024", "embedding_dimensions = 768"))
    with pytest.raises(ValueError, match="fresh WORKSHOP_NAMESPACE"):
        load_demo("changed")
    monkeypatch.setenv("WORKSHOP_NAMESPACE", "../unowned")
    with pytest.raises(ValueError, match="WORKSHOP_NAMESPACE"):
        load_demo("escape")


def test_document_limits_and_pdf_page_provenance(tmp_path: Path) -> None:
    documents = script("02_read_documents.py")
    path = tmp_path / "pages.pdf"
    first = Image.new("RGB", (100, 60), "white")
    second = Image.new("RGB", (100, 60), "gray")
    first.save(path, save_all=True, append_images=[second])
    limits = {"max_bytes": 100000, "max_pages": 2, "max_pixels": 10000, "scale": 1}
    rendered = list(documents.pages(path, **limits))
    assert [number for number, _ in rendered] == [1, 2]
    assert all(data.startswith(b"\x89PNG") for _, data in rendered)
    with pytest.raises(ValueError, match="document_pages"):
        list(documents.pages(path, **{**limits, "max_pages": 1}))
    with pytest.raises(ValueError, match="document_bytes"):
        list(documents.pages(path, **{**limits, "max_bytes": 1}))
    with pytest.raises(ValueError, match="page_pixels"):
        list(documents.pages(path, **{**limits, "max_pixels": 1}))


@pytest.mark.parametrize("status", ["no_gated_entries", "no_new_entries", "no_proposed_lessons", "no_accepted_lessons"])
def test_distillation_empty_status_is_not_success(status: str) -> None:
    learn = script("04_learn.py")
    result = learn.distillation_summary({"status": status, "documents": [], "session_id": "test-session"})
    assert result["persisted_lessons"] is False
    assert result["accepted_document_count"] == 0


def test_distillation_requires_actual_documents() -> None:
    learn = script("04_learn.py")
    assert learn.distillation_summary({"status": "completed", "documents": []})["persisted_lessons"] is False
    document = "# Session learning\n\nA hand-authored unit-test stand-in lesson."
    result = learn.distillation_summary({"status": "completed", "documents": [document]})
    assert result["persisted_lessons"] is True
    assert result["documents"] == [document]
    with pytest.raises(ValueError, match="lesson texts"):
        learn.distillation_summary({"status": "completed", "documents": [123]})
    with pytest.raises(ValueError, match="lesson texts"):
        learn.distillation_summary({"status": "completed", "documents": [""]})


def test_lesson_matching_requires_complete_text() -> None:
    learn = script("04_learn.py")
    lesson = "# Session learning\n\nA workshop requires confirmed access."
    chunks = [
        SourceChunk("heading-only", "# Session learning"),
        SourceChunk("different-lesson", "# Session learning\n\nCheck the microphone."),
        SourceChunk("matching-lesson", "# Session learning\nA workshop requires confirmed access."),
    ]
    assert learn.exact_lesson_matches([lesson], chunks) == [
        {"accepted_document_index": 0, "chunk_id": "matching-lesson"}
    ]


@pytest.mark.parametrize("status", ["completed", "no_gated_entries"])
def test_learning_invokes_distillation_and_queries_fresh_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    learn = script("04_learn.py")
    cognee = ModuleType("cognee")
    cognee.recall = AsyncMock(return_value=[])  # type: ignore[attr-defined]
    documents = ["# Session learning\n\nSynthetic accepted lesson."] if status == "completed" else []
    cognee.session = SimpleNamespace(  # type: ignore[attr-defined]
        get_session=AsyncMock(return_value=[]),
        distill_session=AsyncMock(return_value={"status": status, "documents": documents}),
    )
    search = ModuleType("cognee.api.v1.search")
    search.SearchType = SimpleNamespace(HYBRID_COMPLETION="hybrid")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cognee.api.v1.search", search)
    monkeypatch.setattr(learn, "initialize", lambda *args, **kwargs: cognee)
    retrieval = AsyncMock(return_value=[SourceChunk("lesson-chunk", document) for document in documents])
    monkeypatch.setattr(learn, "retrieve_chunks", retrieval)
    monkeypatch.setattr(sys, "argv", ["04_learn.py"])
    demo = Demo(
        {}, {"distill_batches": 1, "retrieve_top_k": 5}, {}, {}, "test", tmp_path, tmp_path, {"query": "new input"}
    )
    asyncio.run(learn.main(demo))
    session_ids = json.loads((tmp_path / "session-ids.json").read_text())
    assert session_ids["teaching"] != session_ids["fresh"]
    cognee.session.distill_session.assert_awaited_once_with(session_ids["teaching"], demo.dataset, max_batches=1)
    assert cognee.recall.await_args_list[-1].kwargs["session_id"] == session_ids["fresh"]
    assert cognee.recall.await_args_list[-1].args[0] == "new input"
    retrieval.assert_awaited_once_with("new input", dataset=demo.dataset, top_k=5)
    assert json.loads((tmp_path / "actual-distillation.json").read_text())["documents"] == documents
    assert bool(json.loads((tmp_path / "learning-evidence.json").read_text())["exact_text_matches"]) == bool(documents)


@pytest.mark.parametrize("evidence", ["present", "missing", "oversized"])
def test_focused_decision_config_is_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str) -> None:
    import common

    async def direct(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(common.asyncio, "to_thread", direct)
    chunk = SourceChunk("test-source", "An authored stand-in policy requires an access check.")
    monkeypatch.setattr(common, "retrieve_chunks", AsyncMock(return_value=[] if evidence == "missing" else [chunk]))
    rerank = Mock(return_value=[RankedChunk(chunk, 0.8)])
    decision = Mock(return_value=Decision("confirmed", {"confirmed": 0.8, "not_required": 0.1, "unknown": 0.1}))
    monkeypatch.setattr(common, "rerank_chunks", rerank)
    monkeypatch.setattr(common, "decide", decision)
    monkeypatch.setattr(Demo, "native", lambda *args: nullcontext(object()))
    default = {"choices": {"confirmed": "passed", "failed": "failed", "unknown": "no result"}}
    original = deepcopy(default)
    learning = {
        "instructions": "Does the policy require access confirmation?",
        "choices": {"confirmed": "required", "not_required": "explicitly unnecessary", "unknown": "unstated"},
        "rerank_top_k": 1,
    }
    demo = Demo(
        {"rerank": "test-reranker", "decision": "test-decision"},
        {
            "retrieve_top_k": 5,
            "rerank_top_k": 3,
            "evidence_chars": 1 if evidence == "oversized" else 500,
            "provision_timeout_seconds": 1,
        },
        default,
        {},
        "test",
        tmp_path,
        tmp_path,
    )
    outcome = asyncio.run(common.run_decision(demo, "policy question", decision_config=learning))
    assert demo.decision == original
    if evidence == "present":
        assert outcome["choice"] == "confirmed"
        assert decision.call_args.kwargs["choices"] == learning["choices"]
        assert decision.call_args.kwargs["instructions"] == learning["instructions"]
    else:
        assert outcome["choice"] == "unknown"
        assert outcome["probabilities"] is None
        decision.assert_not_called()
    if evidence != "missing":
        assert rerank.call_args.kwargs["top_k"] == 1
