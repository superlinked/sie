"""Offline boundary checks. No model weights, network requests or recorded replies."""

import asyncio
import importlib.util
import io
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
    monkeypatch.setenv("SIE_COGNEE_NAMESPACE", "test-example")
    monkeypatch.setenv("SIE_LLM_MODEL", "Qwen/Qwen3.8-27B-FP8")
    demo = load_demo("test")
    assert demo.models["llm"] == "Qwen/Qwen3.8-27B-FP8"
    assert demo.storage == tmp_path / ".state/test-example"
    assert demo.dataset == "sie-test-example"
    (tmp_path / "config.toml").write_text(original.replace("embedding_dimensions = 1024", "embedding_dimensions = 768"))
    with pytest.raises(SystemExit, match="fresh SIE_COGNEE_NAMESPACE"):
        load_demo("changed")
    monkeypatch.setenv("SIE_COGNEE_NAMESPACE", "../unowned")
    with pytest.raises(ValueError, match="SIE_COGNEE_NAMESPACE"):
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
    (tmp_path / "data").mkdir()
    (tmp_path / "data/correction.txt").write_text("Hand-authored test correction.", encoding="utf-8")
    monkeypatch.setattr(learn, "ROOT", tmp_path)
    demo = Demo(
        {}, {"distill_batches": 1, "retrieve_top_k": 5}, {}, {}, "test", tmp_path, tmp_path, {"query": "new input"}
    )
    marker = tmp_path / "cognee/sie-cognee.json"
    marker.parent.mkdir()
    marker.write_text('{"embedding_model": "test-embedding", "embedding_dimensions": 1024}')
    marker_before = marker.read_bytes()
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
    assert marker.read_bytes() == marker_before
    assert f"SIE_COGNEE_NAMESPACE=test uv run 04_learn.py --verify-run {tmp_path}" in capsys.readouterr().out


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
    with pytest.raises(SystemExit) as error:
        asyncio.run(learn.main(demo))
    assert error.value.code == 2
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


@pytest.mark.parametrize("missing", ["embedding.json", "cognee/sie-cognee.json"])
def test_verify_checks_missing_markers_before_reading_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    learn = script("04_learn.py")
    demo, prior, _ = verification_fixture(tmp_path)
    (demo.storage / missing).unlink()
    reads: list[Path] = []
    original_read = Path.read_text

    def track_read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track_read)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    with pytest.raises(learn.MemoryStateError, match="Restore the original intact memory"):
        asyncio.run(learn.verify(demo, prior))
    assert demo.storage / "embedding.json" not in reads
    assert demo.storage / "cognee/sie-cognee.json" not in reads
    initialization.assert_not_called()


@pytest.mark.parametrize("missing", ["embedding.json", "cognee/sie-cognee.json"])
def test_missing_state_cli_displays_safe_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    import common

    learn = script("04_learn.py")
    demo, prior, _ = verification_fixture(tmp_path)
    demo.limits["experiment_timeout_seconds"] = 1
    (demo.storage / missing).unlink()
    monkeypatch.setattr(common, "load_demo", lambda experiment: demo)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(prior)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert learn.MEMORY_STATE_RECOVERY in stderr
    assert "fresh SIE_COGNEE_NAMESPACE" in stderr
    assert "FileNotFoundError" not in stderr
    assert str(tmp_path) not in stderr
    initialization.assert_not_called()


def test_verify_cli_still_redacts_upstream_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import common

    learn = script("04_learn.py")
    demo, prior, _ = verification_fixture(tmp_path)
    demo.limits["experiment_timeout_seconds"] = 1
    monkeypatch.setattr(common, "load_demo", lambda experiment: demo)
    monkeypatch.setattr(learn, "initialize", Mock(side_effect=ValueError("upstream-private-detail")))
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(prior)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    assert "ValueError" in str(error.value)
    output = capsys.readouterr()
    assert "upstream-private-detail" not in str(error.value) + output.out + output.err


def bootstrap_verification_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Demo:
    import common

    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text())
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "load_dotenv", lambda *args, **kwargs: None)
    for name in tuple(common.os.environ):
        if name.startswith("SIE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SIE_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("SIE_COGNEE_NAMESPACE", "test")
    demo = load_demo("04-learn")
    demo.show("actual-distillation", {"status": "no_proposed_lessons", "documents": []})
    marker = demo.storage / "cognee/sie-cognee.json"
    marker.parent.mkdir()
    marker.write_text(
        json.dumps(
            {"embedding_model": demo.models["embedding"], "embedding_dimensions": demo.models["embedding_dimensions"]}
        )
    )
    return demo


@pytest.mark.parametrize("role", ["llm", "embedding", "ocr", "rerank", "decision"])
def test_real_bootstrap_missing_key_displays_only_authored_configuration_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], role: str
) -> None:
    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("SIE_API_KEY", "private-key-value")
    monkeypatch.setenv(f"SIE_{role.upper()}_BASE_URL", "https://api.superlinked.com")
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    displayed = str(error.value) + capsys.readouterr().out
    assert f"Set SIE_{role.upper()}_API_KEY or SIE_API_KEY for the Serverless {role} endpoint." in displayed
    assert "private-key-value" not in displayed
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text()) == {"type": "ConfigurationError", "completed": False}
    initialization.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://user:private-url-value@localhost:8080",
        "http://localhost:8080?key=private-url-value",
        "http://localhost:private-url-value",
        "http://[private-url-value",
    ],
)
def test_real_bootstrap_invalid_url_displays_safe_configuration_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], url: str
) -> None:
    import common

    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("SIE_LLM_BASE_URL", url)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    output = capsys.readouterr()
    displayed = str(error.value) + output.out + output.err
    assert common.CONFIGURATION_MESSAGES["endpoint_url"] in displayed
    assert "private-url-value" not in displayed
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text()) == {"type": "ConfigurationError", "completed": False}
    initialization.assert_not_called()


def test_safe_configuration_diagnostics_never_use_exception_text(tmp_path: Path) -> None:
    import common

    error = common.ConfigurationError("private-error-code")
    error.args = ("private-error-message",)
    with pytest.raises(SystemExit) as stopped:
        common.stop_run(tmp_path, error)
    assert "Check the endpoint configuration." in str(stopped.value)
    assert "private-error" not in str(stopped.value)
    assert json.loads((tmp_path / "failure.json").read_text()) == {"type": "ConfigurationError", "completed": False}


def test_json_artifacts_round_trip_with_ascii_console_and_explicit_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = Demo({}, {}, {}, {}, "test", tmp_path, tmp_path)
    payload = {"text": "Caf\u00e9 \u2014 \u65e5\u672c\u8a9e \U0001f680", "metadata": {"\u00e9": "\u00f1"}}
    original_write = Path.write_text

    def utf8_write(path, text, *args, **kwargs):
        assert kwargs.get("encoding") == "utf-8"
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", utf8_write)
    console = io.BytesIO()
    stdout = io.TextIOWrapper(console, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)
    demo.show("unicode", payload)
    stdout.flush()
    encoded = (tmp_path / "unicode.json").read_bytes()
    assert encoded.isascii()
    assert json.loads(encoded.decode("utf-8")) == payload
    assert console.getvalue().isascii()


@pytest.mark.parametrize("filename", ["02_read_documents.py", "04_learn.py"])
def test_missing_default_input_has_fetch_guidance_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    module = script(filename)
    bootstrap_verification_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [filename])
    initialization = Mock()
    monkeypatch.setattr(module, "initialize", initialization)
    with pytest.raises(SystemExit, match="uv run fetch.py"):
        module.launch("missing-input", module.main)
    initialization.assert_not_called()


def test_missing_seed_input_is_checked_before_any_ingestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import common

    (tmp_path / "data").mkdir()
    (tmp_path / "data/readiness-policy.txt").write_text("Minimal stand-in seed.", encoding="utf-8")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    cognee = SimpleNamespace(remember=AsyncMock())
    demo = Demo({}, {}, {}, {}, "test", tmp_path, tmp_path)
    with pytest.raises(common.ConfigurationError, match="uv run fetch.py"):
        asyncio.run(common.seed_memory(demo, cognee))
    cognee.remember.assert_not_awaited()


@pytest.mark.parametrize("missing_outer", [False, True])
def test_real_bootstrap_missing_owned_marker_displays_safe_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing_outer: bool
) -> None:
    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    marker = teaching.storage / "cognee/sie-cognee.json"
    marker.unlink()
    outer = teaching.storage / "embedding.json"
    if missing_outer:
        outer.unlink()
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert learn.MEMORY_STATE_RECOVERY in stderr
    assert "FileNotFoundError" not in stderr
    assert str(tmp_path) not in stderr
    assert outer.is_file()
    assert not marker.exists()
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text()) == {"type": "MemoryStateError", "completed": False}
    initialization.assert_not_called()


@pytest.mark.parametrize("owned_identity", ["matching", "different-model", "different-dimensions"])
def test_real_bootstrap_reconstructs_outer_metadata_but_checks_owned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned_identity: str
) -> None:
    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    marker = teaching.storage / "cognee/sie-cognee.json"
    identity = json.loads(marker.read_text())
    if owned_identity == "different-model":
        identity["embedding_model"] = "other-model"
    elif owned_identity == "different-dimensions":
        identity["embedding_dimensions"] = 768
    marker.write_text(json.dumps(identity))
    marker_before = marker.read_bytes()
    outer = teaching.storage / "embedding.json"
    outer.unlink()
    cognee = SimpleNamespace(
        recall=AsyncMock(return_value=[]), session=SimpleNamespace(get_session=AsyncMock(return_value=[]))
    )
    initialization = Mock(return_value=cognee)
    monkeypatch.setattr(learn, "initialize", initialization)
    search = ModuleType("cognee.api.v1.search")
    search.SearchType = SimpleNamespace(HYBRID_COMPLETION="hybrid")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cognee.api.v1.search", search)
    retrieval = AsyncMock(return_value=[])
    monkeypatch.setattr(learn, "retrieve_chunks", retrieval)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    if owned_identity == "matching":
        learn.launch("04-learn", learn.main)
        initialization.assert_called_once()
        assert initialization.call_args.kwargs == {"learning": False}
        cognee.recall.assert_awaited_once()
        retrieval.assert_awaited_once()
    else:
        with pytest.raises(SystemExit) as error:
            learn.launch("04-learn", learn.main)
        assert error.value.code == 2
        initialization.assert_not_called()
        cognee.recall.assert_not_awaited()
        retrieval.assert_not_awaited()
    assert json.loads(outer.read_text()) == {key: teaching.models[key] for key in ("embedding", "embedding_dimensions")}
    assert marker.read_bytes() == marker_before


@pytest.mark.parametrize("target", ["outer", "owned", "configuration", "distillation"])
@pytest.mark.parametrize("problem", ["malformed", "non-object", "invalid-utf8", "unreadable", "invalid-object"])
def test_real_bootstrap_invalid_state_records_safe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], target: str, problem: str
) -> None:
    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    paths = {
        "outer": teaching.storage / "embedding.json",
        "owned": teaching.storage / "cognee/sie-cognee.json",
        "configuration": teaching.run / "configuration.json",
        "distillation": teaching.run / "actual-distillation.json",
    }
    path = paths[target]
    invalid = {
        "malformed": b'{"private-local-detail":',
        "non-object": b'["private-local-detail"]',
        "invalid-utf8": b"\xffprivate-local-detail",
        "invalid-object": b'{"unexpected": "private-local-detail"}',
    }
    if problem == "unreadable":
        original_read = Path.read_text

        def fail_read(self, *args, **kwargs):
            if self == path:
                raise PermissionError("private-local-detail")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_read)
    else:
        path.write_bytes(invalid[problem])
    before = {key: value.read_bytes() for key, value in paths.items()}
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert failures[0].parent != teaching.run
    assert json.loads(failures[0].read_text()) == {"type": "MemoryStateError", "completed": False}
    output = capsys.readouterr()
    displayed = str(error.value) + output.out + output.err
    assert learn.MEMORY_STATE_RECOVERY in displayed
    assert "private-local-detail" not in displayed + failures[0].read_text()
    assert {key: value.read_bytes() for key, value in paths.items()} == before
    initialization.assert_not_called()


def test_real_bootstrap_failed_outer_metadata_write_records_safe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    outer = teaching.storage / "embedding.json"
    outer.unlink()
    owned = teaching.storage / "cognee/sie-cognee.json"
    owned_before = owned.read_bytes()
    original_write = Path.write_text

    def fail_write(self, *args, **kwargs):
        if self == outer:
            raise PermissionError("private-local-detail")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text()) == {"type": "MemoryStateError", "completed": False}
    output = capsys.readouterr()
    displayed = str(error.value) + output.out + output.err
    assert learn.MEMORY_STATE_RECOVERY in displayed
    assert "private-local-detail" not in displayed + failures[0].read_text()
    assert owned.read_bytes() == owned_before
    assert not outer.exists()
    initialization.assert_not_called()


@pytest.mark.parametrize("stage", ["endpoint", "configuration"])
def test_real_bootstrap_redacts_setup_failure_and_records_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    import common

    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    if stage == "endpoint":
        monkeypatch.setattr(common, "endpoint_for", Mock(side_effect=ValueError("private-local-detail")))
        error_type = "ValueError"
    else:
        original_write = Path.write_text

        def fail_write(self, *args, **kwargs):
            if self.name == "configuration.json":
                raise PermissionError("private-local-detail")
            return original_write(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_write)
        error_type = "PermissionError"
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    failures = list(teaching.run.parent.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text()) == {"type": error_type, "completed": False}
    output = capsys.readouterr()
    displayed = str(error.value) + output.out + output.err
    assert "private-local-detail" not in displayed + failures[0].read_text()
    assert learn.MEMORY_STATE_RECOVERY not in displayed
    initialization.assert_not_called()


@pytest.mark.parametrize("stage", ["outer", "owned", "endpoint"])
def test_unwritable_failure_artifact_does_not_expose_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    import common

    learn = script("04_learn.py")
    teaching = bootstrap_verification_fixture(tmp_path, monkeypatch)
    if stage == "endpoint":
        monkeypatch.setattr(common, "endpoint_for", Mock(side_effect=ValueError("private-local-detail")))
    else:
        name = "embedding.json" if stage == "outer" else "cognee/sie-cognee.json"
        (teaching.storage / name).write_text('{"private-local-detail":')
    original_write = Path.write_text

    def fail_write(self, *args, **kwargs):
        if self.name == "failure.json":
            raise PermissionError("private-write-detail")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    initialization = Mock()
    monkeypatch.setattr(learn, "initialize", initialization)
    monkeypatch.setattr(sys, "argv", ["04_learn.py", "--verify-run", str(teaching.run)])
    with pytest.raises(SystemExit) as error:
        learn.launch("04-learn", learn.main)
    output = capsys.readouterr()
    displayed = str(error.value) + output.out + output.err
    assert "Could not write failure.json." in displayed
    assert "private-local-detail" not in displayed
    assert "private-write-detail" not in displayed
    if stage != "endpoint":
        assert learn.MEMORY_STATE_RECOVERY in displayed
    assert not list(teaching.run.parent.glob("*/failure.json"))
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
