import builtins
import hashlib
import json
import os
import sys
from types import ModuleType

import pytest
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter

from tools.benchmarks.stream_decode import observer
from tools.benchmarks.stream_decode.server import observe
from tools.benchmarks.stream_decode.tests.test_observer import manager_type


@pytest.mark.parametrize("valid_source", [True, False])
def test_child_hook_checks_source_and_restores_import_hook(monkeypatch, tmp_path, valid_source):
    module = ModuleType(observer.MODULE)
    module.TokenizerManager = manager_type()
    source = tmp_path / "srt/managers/tokenizer_manager.py"
    source.parent.mkdir(parents=True)
    source.write_text("# generated test source\n")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, observer.MODULE, module)
    monkeypatch.setattr(
        observer, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest() if valid_source else "wrong"
    )
    monkeypatch.setattr(observer, "SOURCE_FILES", {"managers/tokenizer_manager.py": observer.SOURCE_SHA256})
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _: observer.VERSION)
    monkeypatch.setenv(observer.TRACE_ENV, str(tmp_path))
    observer.save(tmp_path / "server.json", {"cohort": "install-test"})
    previous = builtins.__import__
    try:
        observer.install()
        if valid_source:
            builtins.__import__("math")
            assert builtins.__import__ is previous
            assert getattr(module.TokenizerManager, observer.MARKER)
            marker = json.loads((tmp_path / f"installed-{os.getpid()}.json").read_text())
            assert marker["cohort"] == "install-test"
            assert marker["source_sha256"] == observer.SOURCE_SHA256
        else:
            with pytest.raises(RuntimeError, match="unreviewed"):
                builtins.__import__("math")
            assert not hasattr(module.TokenizerManager, observer.MARKER)
    finally:
        builtins.__import__ = previous


def test_wrong_engine_version_fails_before_installing(monkeypatch, tmp_path):
    monkeypatch.setenv(observer.TRACE_ENV, str(tmp_path))
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _: "0.5.10.post1")
    observer.save(tmp_path / "server.json", {"cohort": "version-test"})
    previous = builtins.__import__
    with pytest.raises(RuntimeError, match="requires SGLang"):
        observer.install()
    assert builtins.__import__ is previous


def test_server_wrapper_preserves_production_hooks_and_restores_on_failure(monkeypatch, tmp_path):
    adapter = SGLangGenerationAdapter("public/checkpoint")
    original = SGLangGenerationAdapter._compat_pythonpath_entries
    production = original(adapter)
    monkeypatch.setenv(observer.TRACE_ENV, "previous-value")
    with pytest.raises(RuntimeError, match="stop"):
        with observe(tmp_path / "trace", "cohort"):
            actual = adapter._compat_pythonpath_entries()
            assert actual[2:] == production
            assert actual[0].endswith("stream_decode/_hooks")
            manifest = json.loads((tmp_path / "trace/server.json").read_text())
            assert manifest["sglang_version"] == "0.5.20"
            assert manifest["cohort"] == "cohort"
            raise RuntimeError("stop")
    assert SGLangGenerationAdapter._compat_pythonpath_entries is original
    assert os.environ[observer.TRACE_ENV] == "previous-value"
