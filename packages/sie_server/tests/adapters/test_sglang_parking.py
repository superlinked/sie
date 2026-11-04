"""Host-RAM parking for SGLang engines: launch flags, park and wake over HTTP."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests
from sie_server.adapters._generation_base import GenerationDrainingError
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.sglang import _server
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.core.host_cache import HOST_CACHE_ENV

URL = "http://localhost:30001"


class _Process:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code

    def poll(self) -> int | None:
        return self.exit_code


class _Response:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _Engine:
    """Records the control calls a park or wake makes."""

    def __init__(self, flush_statuses: list[int] | None = None) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.flush_statuses = list(flush_statuses or [200])
        self.status: dict[str, int] = {}

    def post(self, url: str, *, json: Any = None, params: Any = None, timeout: float) -> _Response:
        path = url.removeprefix(URL)
        self.calls.append(("POST", path, json if json is not None else params))
        if path == "/flush_cache":
            return _Response(self.flush_statuses.pop(0) if self.flush_statuses else 200)
        return _Response(self.status.get(path, 200))

    def get(self, url: str, *, timeout: float) -> _Response:
        path = url.removeprefix(URL)
        self.calls.append(("GET", path, None))
        return _Response(self.status.get(path, 200))


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> _Engine:
    fake = _Engine()
    monkeypatch.setattr(_server.requests, "post", fake.post)
    monkeypatch.setattr(_server.requests, "get", fake.get)
    return fake


def _started(adapter: Any) -> Any:
    """An adapter as ``load`` leaves it, without starting an engine."""
    adapter._memory_saver_launch_args()
    adapter._server_url = URL
    adapter._process = _Process()
    return adapter


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> SGLangGenerationAdapter:
    monkeypatch.delenv(HOST_CACHE_ENV, raising=False)
    monkeypatch.setattr(_server, "checkpoint_weight_bytes", lambda *_: 8 * 1024**3)
    return _started(SGLangGenerationAdapter("org/model", mem_fraction_static=0.8))


def test_launch_flags_follow_the_host_tier_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOST_CACHE_ENV, raising=False)
    assert SGLangGenerationAdapter("org/model")._memory_saver_launch_args() == [
        "--enable-memory-saver",
        "--enable-weights-cpu-backup",
    ]
    assert SGLangEmbeddingAdapter("org/model")._memory_saver_launch_args() == list(_server.MEMORY_SAVER_LAUNCH_ARGS)

    monkeypatch.setenv(HOST_CACHE_ENV, "0")
    assert SGLangGenerationAdapter("org/model")._memory_saver_launch_args() == []

    monkeypatch.setenv(HOST_CACHE_ENV, "16GiB")
    assert SGLangGenerationAdapter("org/model", host_offload=False)._memory_saver_launch_args() == []
    wide = SGLangGenerationAdapter(
        "org/model", tensor_parallel_size=2, request_read_timeout_s=60.0, startup_timeout_s=600.0
    )
    assert wide._memory_saver_launch_args() == []


def test_host_offload_must_be_a_boolean() -> None:
    with pytest.raises(ValueError, match="host_offload"):
        SGLangGenerationAdapter("org/model", host_offload="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host_offload"):
        SGLangEmbeddingAdapter("org/model", host_offload=1)  # type: ignore[arg-type]


def test_only_a_running_engine_launched_with_the_memory_saver_can_park(
    generation: SGLangGenerationAdapter,
) -> None:
    assert generation.host_offload_bytes() == 8 * 1024**3
    generation._process = _Process(exit_code=1)
    assert generation.host_offload_bytes() is None
    generation._process = _Process()
    generation._memory_saver = False
    assert generation.host_offload_bytes() is None
    generation._memory_saver = True
    generation._server_url = None
    assert generation.host_offload_bytes() is None


def test_park_waits_for_idle_then_releases(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    offload = generation.offload_to_host()
    assert offload.host_bytes == 8 * 1024**3
    assert offload.retains_device_memory
    assert [(method, path) for method, path, _ in engine.calls] == [
        ("POST", "/flush_cache"),
        ("POST", "/release_memory_occupation"),
    ]
    assert engine.calls[0][2] == {"timeout": _server.PARK_IDLE_TIMEOUT_S}
    # Parked: requests are told to retry, and it cannot park twice.
    with pytest.raises(GenerationDrainingError):
        generation._check_loaded()
    assert generation.host_offload_bytes() is None


def test_park_aborts_requests_that_outlive_the_wait(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    engine.flush_statuses = [400, 200]
    generation.offload_to_host()
    assert [path for _, path, _ in engine.calls] == [
        "/flush_cache",
        "/abort_request",
        "/flush_cache",
        "/release_memory_occupation",
    ]
    assert engine.calls[1][2] == {"abort_all": True}


def test_park_fails_when_the_engine_stays_busy(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    engine.flush_statuses = [400, 400]
    with pytest.raises(RuntimeError, match="did not become idle"):
        generation.offload_to_host()
    assert "/release_memory_occupation" not in [path for _, path, _ in engine.calls]


def test_park_fails_when_the_release_is_refused(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    engine.status["/release_memory_occupation"] = 400
    with pytest.raises(RuntimeError, match="release_memory_occupation failed"):
        generation.offload_to_host()


def test_wake_resumes_checks_health_and_keeps_the_host_copy(
    generation: SGLangGenerationAdapter, engine: _Engine
) -> None:
    generation.offload_to_host()
    engine.calls.clear()
    assert generation.restore_from_host() == 8 * 1024**3
    assert [(method, path) for method, path, _ in engine.calls] == [
        ("POST", "/resume_memory_occupation"),
        ("GET", "/health"),
    ]
    generation._check_loaded()
    # The engine keeps its host buffer, so parking again costs what it holds.
    assert generation.host_offload_bytes() == 8 * 1024**3


def test_wake_fails_loudly(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    generation.offload_to_host()
    engine.status["/health"] = 503
    with pytest.raises(RuntimeError, match="health check after resume"):
        generation.restore_from_host()
    with pytest.raises(RuntimeError, match="not parked"):
        SGLangGenerationAdapter("org/model").restore_from_host()


def test_wake_refuses_an_engine_that_exited(generation: SGLangGenerationAdapter, engine: _Engine) -> None:
    generation.offload_to_host()
    generation._process = _Process(exit_code=-9)
    with pytest.raises(RuntimeError, match="not running"):
        generation.restore_from_host()


def test_parked_embedding_engine_refuses_requests(monkeypatch: pytest.MonkeyPatch, engine: _Engine) -> None:
    monkeypatch.delenv(HOST_CACHE_ENV, raising=False)
    monkeypatch.setattr(_server, "checkpoint_weight_bytes", lambda *_: 1024)
    adapter = _started(SGLangEmbeddingAdapter("org/model"))
    assert adapter.offload_to_host().host_bytes == 1024
    with pytest.raises(RuntimeError, match=re.escape(ERR_NOT_LOADED)):
        adapter._check_loaded()
    adapter.restore_from_host()
    adapter._check_loaded()


def test_checkpoint_weight_bytes_sums_local_safetensors(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 50)
    (tmp_path / "config.json").write_text("{}")
    assert _server.checkpoint_weight_bytes(str(tmp_path), None) == 150

    bin_only = tmp_path / "bin"
    bin_only.mkdir()
    (bin_only / "pytorch_model.bin").write_bytes(b"x" * 70)
    assert _server.checkpoint_weight_bytes(str(bin_only), None) == 70

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _server.checkpoint_weight_bytes(str(empty), None) is None


def test_checkpoint_weight_bytes_is_none_for_an_uncached_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert _server.checkpoint_weight_bytes("org/never-downloaded", None) is None


def test_requests_errors_propagate_so_the_registry_unloads(
    generation: SGLangGenerationAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*_: Any, **__: Any) -> None:
        raise requests.ConnectionError("engine gone")

    monkeypatch.setattr(_server.requests, "post", refused)
    with pytest.raises(requests.ConnectionError):
        generation.offload_to_host()
    assert isinstance(generation._process, _Process | subprocess.Popen)
