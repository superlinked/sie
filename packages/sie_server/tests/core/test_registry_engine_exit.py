"""A loaded model whose engine process dies is unloaded, marked failed, and reloaded after the cooldown."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.sglang import _server as sglang_server
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import EngineExitedError, LoadErrorClass, LoadFailure
from sie_server.core.memory import MemoryConfig
from sie_server.core.registry import ModelRegistry

MODEL = "test-model"


def _make_config() -> ModelConfig:
    return ModelConfig(
        sie_id=MODEL,
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached() -> Iterator[MagicMock]:
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


class FakeEngine:
    """A real child process standing in for an engine server."""

    def __init__(self) -> None:
        self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])

    def exit_code(self) -> int | None:
        return self.process.poll()

    def die(self) -> int:
        self.process.kill()
        return self.process.wait(timeout=10)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.die()


@pytest.fixture
def engines() -> Iterator[list[FakeEngine]]:
    started: list[FakeEngine] = []
    yield started
    for engine in started:
        engine.stop()


def _engine_adapter(engines: list[FakeEngine]) -> Callable[..., MagicMock]:
    def build(*_: object, **__: object) -> MagicMock:
        engine = FakeEngine()
        engines.append(engine)
        adapter = MagicMock()
        adapter.aclose_client = None
        adapter.memory_footprint.return_value = 1000
        adapter.requires_main_thread = False
        adapter.engine_exit_code.side_effect = engine.exit_code
        return adapter

    return build


class _HeldWorker:
    """A running worker whose drain blocks until the test releases it."""

    def __init__(self) -> None:
        self.is_running = True
        self.stopping = asyncio.Event()
        self.release = asyncio.Event()

    async def stop(self) -> None:
        self.stopping.set()
        await self.release.wait()
        self.is_running = False


def _registry(check_interval_s: float = 1.0) -> ModelRegistry:
    registry = ModelRegistry(memory_config=MemoryConfig(memory_check_interval_s=check_interval_s))
    registry.add_config(_make_config())
    return registry


def _expire_cooldown(registry: ModelRegistry) -> None:
    failure = registry.get_failure(MODEL)
    assert failure is not None
    registry._failed[MODEL] = LoadFailure(
        error_class=failure.error_class,
        message=failure.message,
        attempts=failure.attempts,
        last_attempt_ts=time.monotonic() - 10_000,
        cooldown_s=failure.cooldown_s,
    )


async def _wait_for(predicate: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition not reached before the deadline")
        await asyncio.sleep(0.01)


class TestEngineExitAfterLoad:
    async def test_a_live_engine_is_left_alone(self, engines: list[FakeEngine]) -> None:
        registry = _registry()
        with patch("sie_server.core.model_loader.load_adapter", side_effect=_engine_adapter(engines)):
            await registry._load_model_background(MODEL, "cpu")

        await registry._reap_exited_engines()

        assert registry.is_loaded(MODEL)
        assert registry.get_failure(MODEL) is None

    async def test_a_dead_engine_is_unloaded_marked_failed_and_reloaded_after_the_cooldown(
        self, engines: list[FakeEngine]
    ) -> None:
        registry = _registry()
        with patch("sie_server.core.model_loader.load_adapter", side_effect=_engine_adapter(engines)):
            await registry._load_model_background(MODEL, "cpu")
            first_adapter = registry.get(MODEL)
            exit_code = engines[0].die()

            await registry._reap_exited_engines()

            assert not registry.is_loaded(MODEL)
            first_adapter.unload.assert_called_once()
            failure = registry.get_failure(MODEL)
            assert failure is not None
            assert failure.error_class is LoadErrorClass.ENGINE
            assert not failure.is_permanent
            assert f"exited with code {exit_code}" in failure.message
            assert registry.is_failed(MODEL)
            assert await registry.start_load_async(MODEL, "cpu") is False

            _expire_cooldown(registry)
            assert await registry.start_load_async(MODEL, "cpu") is True
            await _wait_for(lambda: registry.is_loaded(MODEL))

        assert registry.get(MODEL) is not first_adapter
        assert registry.get_failure(MODEL) is None
        assert len(engines) == 2
        assert engines[1].exit_code() is None

    async def test_the_drain_of_a_dead_engine_runs_without_the_registry_lock(self, engines: list[FakeEngine]) -> None:
        registry = _registry()
        with patch("sie_server.core.model_loader.load_adapter", side_effect=_engine_adapter(engines)):
            await registry._load_model_background(MODEL, "cpu")
        worker = _HeldWorker()
        registry._loaded[MODEL].worker = worker  # type: ignore[assignment]
        engines[0].die()

        reaper = asyncio.create_task(registry._reap_exited_engines())
        try:
            await asyncio.wait_for(worker.stopping.wait(), timeout=10)

            assert registry.is_unloading(MODEL)
            async with asyncio.timeout(1):
                async with registry._get_load_lock():
                    pass
        finally:
            worker.release.set()
        await asyncio.wait_for(reaper, timeout=10)
        assert not registry.is_loaded(MODEL)
        assert not registry.is_unloading(MODEL)
        failure = registry.get_failure(MODEL)
        assert failure is not None
        assert failure.error_class is LoadErrorClass.ENGINE

    async def test_the_memory_monitor_reaps_a_dead_engine(self, engines: list[FakeEngine]) -> None:
        registry = _registry(check_interval_s=0.01)
        with ExitStack() as stack:
            stack.enter_context(
                patch("sie_server.core.model_loader.load_adapter", side_effect=_engine_adapter(engines))
            )
            for manager in registry.memory_managers.values():
                stack.enter_context(patch.object(manager, "check_pressure", return_value=False))
            await registry._load_model_background(MODEL, "cpu")
            await registry.start_memory_monitor()
            try:
                engines[0].die()
                await _wait_for(lambda: not registry.is_loaded(MODEL))
            finally:
                await registry.stop_memory_monitor()

        failure = registry.get_failure(MODEL)
        assert failure is not None
        assert failure.error_class is LoadErrorClass.ENGINE

    async def test_a_failing_exit_probe_is_treated_as_a_running_engine(self, engines: list[FakeEngine]) -> None:
        registry = _registry()
        with patch("sie_server.core.model_loader.load_adapter", side_effect=_engine_adapter(engines)):
            await registry._load_model_background(MODEL, "cpu")
        registry.get(MODEL).engine_exit_code.side_effect = OSError("probe failed")

        await registry._reap_exited_engines()

        assert registry.is_loaded(MODEL)


class TestSGLangEngineExitProbe:
    """The SGLang adapters report their engine's exit through the typed error and the probe."""

    @staticmethod
    def _exited_process() -> subprocess.Popen[bytes]:
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        process.wait(timeout=10)
        return process

    def test_the_embedding_adapter_reports_the_exit(self) -> None:
        from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter

        adapter = SGLangEmbeddingAdapter(model_name_or_path="org/embed", dense_dim=768)
        adapter._server_url = "http://localhost:30005"
        adapter._process = self._exited_process()

        assert adapter.engine_exit_code() == 3
        with pytest.raises(EngineExitedError, match="exited with code 3"):
            adapter._check_loaded()

    def test_the_generation_adapter_reports_the_exit(self) -> None:
        from sie_server.adapters.sglang.generation import SGLangGenerationAdapter

        adapter = SGLangGenerationAdapter(
            model_name_or_path="Qwen/Qwen3-4B-Instruct",
            max_seq_length=32768,
            served_model_name="Qwen/Qwen3-4B-Instruct",
        )
        adapter._process = self._exited_process()

        assert adapter.engine_exit_code() == 3
        with pytest.raises(EngineExitedError, match="exited with code 3"):
            adapter._check_engine_alive()

    def test_an_unloaded_adapter_reports_no_exit(self) -> None:
        assert sglang_server.engine_exit_code(None) is None
