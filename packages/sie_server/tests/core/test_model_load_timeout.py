"""Tests for the post-download model-load timeout introduced in sie-internal#846.

Covers:
- ``_resolve_load_timeout`` precedence (kwarg > env > default).
- ``ModelLoader._run_with_timeout`` happy path (no wait_for fires).
- Timeout path: raises ``ModelLoadTimeoutError`` and recreates executor.
- ``classify_load_error`` buckets ``ModelLoadTimeoutError`` into
  ``LoadErrorClass.TIMEOUT`` with a non-permanent cooldown.
- ``hf_env.set_hf_default_timeouts`` is non-clobbering.

The tests use a stub executor function rather than real model loading;
they target the timeout machinery itself, not adapter behaviour.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import AdapterOptions, EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.hf_env import set_hf_default_timeouts
from sie_server.core.load_errors import (
    LoadErrorClass,
    ModelLoadTimeoutError,
    classify_load_error,
)
from sie_server.core.model_loader import (
    DEFAULT_MODEL_LOAD_TIMEOUT_S,
    ModelLoader,
    _resolve_load_timeout,
)
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry


def _make_loader(timeout_s: float | None = None) -> ModelLoader:
    pre = PreprocessorRegistry()
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    return ModelLoader(
        preprocessor_registry=pre,
        postprocessor_registry=PostprocessorRegistry(cpu_pool),
        all_configs={},
        model_load_timeout_s=timeout_s,
    )


def _make_embedding_config(name: str = "t") -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=f"org/{name}",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8,
            )
        },
    )


def test_muvera_runtime_requires_registered_postprocessor() -> None:
    loader = _make_loader()
    config = ModelConfig(
        sie_id="test/visual:muvera",
        hf_id="test/visual",
        tasks=Tasks(encode=EncodeTask(multivector=EmbeddingDim(dim=8))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.base:ModelAdapter",
                max_batch_tokens=8,
                adapter_options=AdapterOptions(
                    runtime={
                        "muvera": {},
                        "output_types": ["dense"],
                    }
                ),
            )
        },
    )
    adapter = MagicMock()
    adapter.get_postprocessors.return_value = None

    with pytest.raises(ValueError, match=r"requests MUVERA.*did not register a 'muvera' postprocessor"):
        loader._finish_load("test/visual:muvera", "cpu", adapter, config)


class TestResolveLoadTimeout:
    """``_resolve_load_timeout`` precedence and parsing."""

    def test_explicit_kwarg_wins_over_env(self) -> None:
        with patch.dict(os.environ, {"SIE_MODEL_LOAD_TIMEOUT_S": "999"}):
            assert _resolve_load_timeout(42.0) == 42.0

    def test_env_used_when_kwarg_none(self) -> None:
        with patch.dict(os.environ, {"SIE_MODEL_LOAD_TIMEOUT_S": "123"}):
            assert _resolve_load_timeout(None) == 123.0

    def test_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SIE_MODEL_LOAD_TIMEOUT_S", None)
            assert _resolve_load_timeout(None) == DEFAULT_MODEL_LOAD_TIMEOUT_S

    def test_zero_disables(self) -> None:
        assert _resolve_load_timeout(0) == 0.0

    def test_negative_clamped_to_zero(self) -> None:
        """Negative values mean "disabled", normalised to 0 for uniform handling."""
        assert _resolve_load_timeout(-5) == 0.0

    def test_invalid_env_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {"SIE_MODEL_LOAD_TIMEOUT_S": "not-a-number"}):
            assert _resolve_load_timeout(None) == DEFAULT_MODEL_LOAD_TIMEOUT_S


class TestRunWithTimeoutHappyPath:
    """Successful execution returns the value and leaves the executor intact."""

    async def test_returns_func_result(self) -> None:
        loader = _make_loader(timeout_s=5.0)
        original_executor = loader._load_executor

        result = await loader._run_with_timeout(
            stage="instantiate",
            name="m",
            func=lambda: "ok",
            args=(),
        )

        assert result == "ok"
        # Executor not swapped on success
        assert loader._load_executor is original_executor

    async def test_disabled_timeout_runs_unbounded(self) -> None:
        """A 0/negative budget skips the ``wait_for`` wrapper entirely."""
        loader = _make_loader(timeout_s=0)

        def _slow() -> str:
            time.sleep(0.05)  # Comfortably exceeds any tight bound
            return "ok"

        # If wait_for were applied with timeout=0 we'd raise immediately.
        result = await loader._run_with_timeout(stage="load", name="m", func=_slow, args=())
        assert result == "ok"


class TestRunWithTimeoutFires:
    """Timeout path: typed error and executor swap."""

    async def test_raises_model_load_timeout_error(self) -> None:
        loader = _make_loader(timeout_s=0.05)

        def _hang() -> None:
            time.sleep(2.0)

        with pytest.raises(ModelLoadTimeoutError) as exc_info:
            await loader._run_with_timeout(stage="load", name="hang-model", func=_hang, args=())

        err = exc_info.value
        assert err.model == "hang-model"
        assert err.stage == "load"
        assert err.timeout_s == pytest.approx(0.05)
        assert err.elapsed_s >= 0.05

    async def test_executor_is_recreated(self) -> None:
        """After a timeout, ``_load_executor`` is a fresh pool so the next
        load is not queued behind the leaked thread.
        """
        loader = _make_loader(timeout_s=0.05)
        original_executor = loader._load_executor

        def _hang() -> None:
            time.sleep(2.0)

        with pytest.raises(ModelLoadTimeoutError):
            await loader._run_with_timeout(stage="load", name="m", func=_hang, args=())

        assert loader._load_executor is not original_executor

        # New executor accepts work
        result = await loader._run_with_timeout(stage="instantiate", name="m", func=lambda: 42, args=())
        assert result == 42

        # Leaked thread cleanup: the orphaned executor is shut down so the
        # interpreter doesn't keep it alive past the test.
        original_executor.shutdown(wait=True)


class TestClassification:
    """``ModelLoadTimeoutError`` is bucketed as ``LoadErrorClass.TIMEOUT``."""

    def test_classified_as_timeout(self) -> None:
        err = ModelLoadTimeoutError(model="m", stage="load", elapsed_s=10.0, timeout_s=5.0)
        result = classify_load_error(err)
        assert result.error_class is LoadErrorClass.TIMEOUT
        # Not permanent — client retry after cooldown is allowed.
        assert result.cooldown_s == 30.0
        assert not result.is_permanent

    def test_generic_timeout_error_still_classifies_as_network(self) -> None:
        """Bare ``TimeoutError`` (not from our wrapper) keeps the existing
        NETWORK bucket; only our typed subclass routes to TIMEOUT.
        """
        result = classify_load_error(TimeoutError("socket read timeout"))
        assert result.error_class is LoadErrorClass.NETWORK


class TestHfEnvDefaults:
    """``set_hf_default_timeouts`` installs sensible HF socket bounds."""

    def test_sets_defaults_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
            os.environ.pop("HF_HUB_ETAG_TIMEOUT", None)
            set_hf_default_timeouts()
            assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60"
            assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"

    def test_does_not_clobber_operator_override(self) -> None:
        """Operators set tighter or looser values via env; we must not overwrite."""
        with patch.dict(os.environ, {"HF_HUB_DOWNLOAD_TIMEOUT": "300"}, clear=False):
            set_hf_default_timeouts()
            assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"


class TestRegistryIntegration:
    """End-to-end: a load timeout records a TIMEOUT-class failure with cooldown."""

    async def test_registry_records_timeout_failure(self) -> None:
        """When ``instantiate_adapter_async`` raises ``ModelLoadTimeoutError``,
        the registry's ``_load_model_background`` records it as a
        ``LoadErrorClass.TIMEOUT`` failure with the 30 s cooldown.
        """
        from pathlib import Path

        from sie_server.config.model import (
            EmbeddingDim,
            EncodeTask,
            ModelConfig,
            ProfileConfig,
            Tasks,
        )
        from sie_server.core.registry import ModelRegistry

        config = ModelConfig(
            sie_id="t",
            hf_id="org/t",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8,
                )
            },
        )
        registry = ModelRegistry()
        registry.add_config(config)

        with (
            patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/fake")),
            patch.object(
                registry._loader,
                "instantiate_adapter_async",
                side_effect=ModelLoadTimeoutError(model="t", stage="instantiate", elapsed_s=601.0, timeout_s=600.0),
            ),
        ):
            await registry._load_model_background("t", "cpu")

        assert registry.is_failed("t")
        failure = registry.get_failure("t")
        assert failure is not None
        assert failure.error_class is LoadErrorClass.TIMEOUT
        assert failure.cooldown_s == 30.0

    async def test_orphan_thread_does_not_corrupt_registry_state(self) -> None:
        """B1 regression: when ``_load_in_executor`` times out, the
        orphaned thread must NOT register pre/postprocessors or set
        model residency on the registry/telemetry for a model the
        registry has marked failed.

        We simulate this by setting a tight timeout and an
        ``adapter.load`` that blocks past the timeout. ``_run_with_timeout``
        fires; the orphan keeps running ``adapter.load`` but
        ``_finish_load`` was lifted out of the executor in the B1 fix —
        so even if the orphan finishes ``adapter.load`` later, no
        registry mutation occurs from the leaked thread.
        """
        from sie_server.config.model import EmbeddingDim, EncodeTask, ProfileConfig, Tasks

        loader = _make_loader(timeout_s=0.05)

        adapter = MagicMock()
        load_finished = False

        def _slow_load(device: str) -> None:
            nonlocal load_finished
            time.sleep(0.3)  # Comfortably past the 0.05 s budget
            load_finished = True

        adapter.load.side_effect = _slow_load
        adapter.warmup.return_value = None
        adapter.requires_main_thread = False

        config = type(
            "C",
            (),
            {
                "sie_id": "ghost",
                "hf_id": "org/ghost",
                "tasks": Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
                "profiles": {
                    "default": ProfileConfig(
                        adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                        max_batch_tokens=8,
                    )
                },
            },
        )()

        telemetry = MagicMock()
        with patch("sie_server.core.model_loader.worker_telemetry", return_value=telemetry):
            with pytest.raises(ModelLoadTimeoutError):
                await loader._load_in_executor("ghost", "cpu", adapter, config)

            # Wait for the orphan thread to "complete" so we can assert it
            # did NOT mutate telemetry state on its way out.
            await asyncio.sleep(0.5)
        assert load_finished, "test bug: orphan didn't reach the post-sleep marker"
        telemetry.model_residency_changed.assert_not_called()

    async def test_adapter_owned_timeout_load_runs_in_executor_without_generic_wait_for(self) -> None:
        """Subprocess-backed adapters own child cleanup and startup timeout.

        They still need to load away from the event loop, but wrapping them in
        the generic ``wait_for`` can orphan a still-loading subprocess. This
        test proves the adapter-owned path runs in the model-load executor and
        is not cut off by ``SIE_MODEL_LOAD_TIMEOUT_S``.
        """
        from unittest.mock import MagicMock

        loader = _make_loader(timeout_s=0.01)
        loop = asyncio.get_running_loop()
        loop_thread_id = threading.get_ident()
        load_started = asyncio.Event()
        release_load = threading.Event()
        load_thread_id: int | None = None

        def _slow_load(device: str) -> None:
            nonlocal load_thread_id
            load_thread_id = threading.get_ident()
            loop.call_soon_threadsafe(load_started.set)
            if not release_load.wait(timeout=1.0):
                raise AssertionError("test bug: load was not released")

        adapter = MagicMock()
        adapter.load.side_effect = _slow_load
        adapter.warmup.return_value = None
        adapter.requires_main_thread = False
        adapter.manages_own_load_timeout = True
        adapter.get_preprocessor.return_value = None
        adapter.get_postprocessors.return_value = None
        adapter.memory_footprint.return_value = 0
        adapter.supports_lora.return_value = False

        started = time.monotonic()
        load_task = asyncio.create_task(
            loader.load_and_register_async("managed", "cpu", adapter, _make_embedding_config("managed"))
        )
        try:
            await asyncio.wait_for(load_started.wait(), timeout=0.5)

            assert load_started.is_set()
            assert time.monotonic() - started < 0.25
            assert not load_task.done()
        finally:
            release_load.set()

        loaded = await asyncio.wait_for(load_task, timeout=1.0)

        assert loaded.adapter is adapter
        assert load_thread_id is not None
        assert load_thread_id != loop_thread_id

    async def test_adapter_owned_startup_timeout_classifies_from_executor(self) -> None:
        from unittest.mock import MagicMock

        loader = _make_loader(timeout_s=0.01)
        adapter = MagicMock()
        adapter.load.side_effect = RuntimeError("SGLang server failed to start within timeout")
        adapter.warmup.return_value = None
        adapter.requires_main_thread = False
        adapter.manages_own_load_timeout = True

        with pytest.raises(ModelLoadTimeoutError) as exc_info:
            await loader.load_and_register_async("sg", "cuda:0", adapter, _make_embedding_config("sg"))

        assert exc_info.value.stage == "load"

    async def test_sglang_startup_timeout_classifies_as_timeout(self) -> None:
        """Regression: ``_load_main_thread`` rewraps SGLang's
        ``RuntimeError('SGLang server failed to start within timeout')``
        into ``ModelLoadTimeoutError`` so the classifier buckets it as
        ``LoadErrorClass.TIMEOUT`` (30 s cooldown) instead of UNKNOWN
        (permanent).
        """
        from unittest.mock import MagicMock

        loader = _make_loader()
        adapter = MagicMock()
        adapter.load.side_effect = RuntimeError("SGLang server failed to start within timeout")
        adapter.warmup.return_value = None

        with pytest.raises(ModelLoadTimeoutError) as exc_info:
            loader._load_main_thread("sg", "cuda:0", adapter, config=None)  # type: ignore[arg-type]

        assert exc_info.value.stage == "load"
        # Non-SGLang RuntimeErrors must NOT be rewrapped.
        adapter2 = MagicMock()
        adapter2.load.side_effect = RuntimeError("CUDA driver error: misaligned address")
        with pytest.raises(RuntimeError) as exc_info2:
            loader._load_main_thread("other", "cuda:0", adapter2, config=None)  # type: ignore[arg-type]
        assert not isinstance(exc_info2.value, ModelLoadTimeoutError)

    async def test_start_load_async_short_circuits_during_cooldown(self) -> None:
        """While the 30 s cooldown is active, ``start_load_async`` is a no-op."""
        from sie_server.config.model import (
            EmbeddingDim,
            EncodeTask,
            ModelConfig,
            ProfileConfig,
            Tasks,
        )
        from sie_server.core.registry import ModelRegistry

        config = ModelConfig(
            sie_id="t",
            hf_id="org/t",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8,
                )
            },
        )
        registry = ModelRegistry()
        registry.add_config(config)

        registry._record_load_failure(
            "t",
            ModelLoadTimeoutError(model="t", stage="load", elapsed_s=601.0, timeout_s=600.0),
        )

        started = await registry.start_load_async("t", "cpu")
        assert started is False
