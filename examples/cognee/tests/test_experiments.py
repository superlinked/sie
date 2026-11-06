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
def test_teaching_distills_without_verifying_in_same_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: str
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
    initialization = Mock(return_value=cognee)
    monkeypatch.setattr(learn, "initialize", initialization)
    seed = AsyncMock()
    monkeypatch.setattr(learn, "seed_memory", seed)
    retrieval = AsyncMock()
    monkeypatch.setattr(learn, "retrieve_chunks", retrieval)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--seed"])
    demo = Demo(
        {}, {"distill_batches": 1, "retrieve_top_k": 5}, {}, {}, "test", tmp_path, tmp_path, {"query": "new input"}
    )
    asyncio.run(learn.main(demo))
    session_ids = json.loads((tmp_path / "session-ids.json").read_text())
    assert set(session_ids) == {"teaching"}
    initialization.assert_called_once_with(demo, learning=True)
    seed.assert_awaited_once_with(demo, cognee)
    cognee.session.distill_session.assert_awaited_once_with(session_ids["teaching"], demo.dataset, max_batches=1)
    assert all(call.kwargs["session_id"] == session_ids["teaching"] for call in cognee.recall.await_args_list)
    retrieval.assert_not_awaited()
    assert json.loads((tmp_path / "actual-distillation.json").read_text())["documents"] == documents
    assert not (tmp_path / "fresh-session.json").exists()
    assert f"WORKSHOP_NAMESPACE=test uv run 04_learn.py --verify-run {tmp_path}" in capsys.readouterr().out


def verification_fixture(tmp_path: Path, *, status: str = "completed") -> tuple[Demo, Path, list[str]]:
    prior = tmp_path / "teaching"
    prior.mkdir()
    storage = tmp_path / "state"
    (storage / "cognee").mkdir(parents=True)
    run = tmp_path / "verification"
    run.mkdir()
    models = {"embedding": "test-embedding", "embedding_dimensions": 1024}
    demo = Demo(models, {"retrieve_top_k": 5}, {}, {}, "test", storage, run, {"query": "new input"})
    documents = ["A complete, hand-authored stand-in lesson."] if status == "completed" else []
    (prior / "configuration.json").write_text(json.dumps({"dataset": demo.dataset, "models": models}))
    (prior / "actual-distillation.json").write_text(json.dumps({"status": status, "documents": documents}))
    (storage / "embedding.json").write_text(json.dumps(models))
    (storage / "cognee/sie-cognee.json").write_text(
        json.dumps({"embedding_model": "test-embedding", "embedding_dimensions": 1024})
    )
    return demo, prior, documents


@pytest.mark.parametrize("status", ["completed", "no_proposed_lessons"])
def test_verify_reads_prior_lessons_without_teaching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    learn = script("04_learn.py")
    demo, prior, documents = verification_fixture(tmp_path, status=status)
    original = (prior / "actual-distillation.json").read_bytes()
    cognee = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        remember=AsyncMock(),
        session=SimpleNamespace(get_session=AsyncMock(return_value=[]), distill_session=AsyncMock()),
    )
    search = ModuleType("cognee.api.v1.search")
    search.SearchType = SimpleNamespace(HYBRID_COMPLETION="hybrid")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cognee.api.v1.search", search)
    initialization = Mock(return_value=cognee)
    monkeypatch.setattr(learn, "initialize", initialization)
    seed = AsyncMock()
    monkeypatch.setattr(learn, "seed_memory", seed)
    retrieval = AsyncMock(return_value=[SourceChunk("stored-lesson", document) for document in documents])
    monkeypatch.setattr(learn, "retrieve_chunks", retrieval)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(prior)])
    asyncio.run(learn.main(demo))
    initialization.assert_called_once_with(demo, learning=False)
    seed.assert_not_awaited()
    cognee.remember.assert_not_awaited()
    cognee.session.distill_session.assert_not_awaited()
    assert cognee.recall.await_count == 1
    assert cognee.recall.await_args.args[0] == "new input"
    assert cognee.recall.await_args.kwargs["session_id"].startswith("test-verify-")
    retrieval.assert_awaited_once_with("new input", dataset=demo.dataset, top_k=5)
    evidence = json.loads((demo.run / "learning-evidence.json").read_text())
    assert evidence["distillation_status"] == status
    assert bool(evidence["exact_text_matches"]) == bool(documents)
    assert (prior / "actual-distillation.json").read_bytes() == original


@pytest.mark.parametrize(
    "mismatch", ["dataset", "recorded-model", "recorded-dimensions", "stored-model", "owned-model"]
)
def test_verify_rejects_identity_mismatch_before_cognee_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    learn = script("04_learn.py")
    demo, prior, _ = verification_fixture(tmp_path)
    if mismatch.startswith("stored"):
        path = demo.storage / "embedding.json"
        data = {"embedding": "other-model", "embedding_dimensions": 1024}
    elif mismatch.startswith("owned"):
        path = demo.storage / "cognee/sie-cognee.json"
        data = {"embedding_model": "other-model", "embedding_dimensions": 1024}
    else:
        path = prior / "configuration.json"
        data = json.loads(path.read_text())
        if mismatch == "dataset":
            data["dataset"] = "sie-other"
        else:
            key = "embedding" if mismatch == "recorded-model" else "embedding_dimensions"
            data["models"][key] = "other-model" if key == "embedding" else 768
    path.write_text(json.dumps(data))
    initialization = Mock(side_effect=AssertionError("Must validate before Cognee import"))
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(prior)])
    with pytest.raises(ValueError):
        asyncio.run(learn.main(demo))
    initialization.assert_not_called()


def test_verify_cannot_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    learn = script("04_learn.py")
    demo, prior, _ = verification_fixture(tmp_path)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--seed", "--verify-run", str(prior)])
    with pytest.raises(SystemExit) as error:
        asyncio.run(learn.main(demo))
    assert error.value.code == 2
    initialization.assert_not_called()


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
