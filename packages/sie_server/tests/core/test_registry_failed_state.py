"""Tests for the terminal ``failed`` state machine in ModelRegistry.

Covers the regression tracked in sie-test#85: a load failure (gated repo,
missing dependency, etc.) used to silently return the model to
``available``, producing an infinite retry loop. The registry now records
a :class:`LoadFailure` and short-circuits ``start_load_async`` while the
failure is in cooldown, and the API surfaces ``MODEL_LOAD_FAILED`` rather
than a retryable ``MODEL_LOADING``.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_sdk.exceptions import GatedModelError
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import (
    LoadErrorClass,
    classify_load_error,
)
from sie_server.core.registry import ModelRegistry


def _make_config(name: str = "test-model", hf_id: str = "org/test") -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    """Avoid real HF downloads in unit tests."""
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


@pytest.fixture
def registry_with_model() -> ModelRegistry:
    registry = ModelRegistry()
    registry.add_config(_make_config())
    return registry


# ---------------------------------------------------------------------------
# classify_load_error
# ---------------------------------------------------------------------------


class TestClassifyLoadError:
    """Pure unit tests for the classifier."""

    def test_gated_model_error_is_permanent(self) -> None:
        exc = GatedModelError("org/private", RuntimeError("401 unauthorized"))
        result = classify_load_error(exc)
        assert result.error_class is LoadErrorClass.GATED
        assert result.is_permanent
        assert result.cooldown_s is None

    def test_import_error_is_dependency_permanent(self) -> None:
        result = classify_load_error(ImportError("Gemma3TextModel not found in transformers"))
        assert result.error_class is LoadErrorClass.DEPENDENCY
        assert result.is_permanent

    def test_module_not_found_is_dependency(self) -> None:
        result = classify_load_error(ModuleNotFoundError("no module named transformers"))
        assert result.error_class is LoadErrorClass.DEPENDENCY

    def test_oom_runtime_error_is_oom_with_cooldown(self) -> None:
        result = classify_load_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        assert result.error_class is LoadErrorClass.OOM
        assert result.cooldown_s is not None
        assert not result.is_permanent

    def test_connection_error_is_network_with_cooldown(self) -> None:
        result = classify_load_error(ConnectionError("dns failure"))
        assert result.error_class is LoadErrorClass.NETWORK
        assert result.cooldown_s is not None

    def test_unknown_runtime_error_is_unknown_permanent(self) -> None:
        result = classify_load_error(RuntimeError("some unrelated failure"))
        assert result.error_class is LoadErrorClass.UNKNOWN
        assert result.is_permanent


# ---------------------------------------------------------------------------
# Registry: failed state recording
# ---------------------------------------------------------------------------


class TestRegistryFailedState:
    async def test_load_failure_is_recorded_as_terminal(self, registry_with_model: ModelRegistry) -> None:
        """A gated-model failure ends in ``_failed`` not ``_loading``/``_loaded``."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_load.side_effect = GatedModelError("org/test", RuntimeError("401 unauthorized"))

            await registry_with_model._load_model_background("test-model", "cpu")

        assert not registry_with_model.is_loaded("test-model")
        assert not registry_with_model.is_loading("test-model")
        assert registry_with_model.is_failed("test-model")

        failure = registry_with_model.get_failure("test-model")
        assert failure is not None
        assert failure.error_class is LoadErrorClass.GATED
        assert failure.attempts == 1
        assert failure.is_permanent

    async def test_repeated_failures_increment_attempts(self, registry_with_model: ModelRegistry) -> None:
        """Each ``_record_load_failure`` bumps the attempt counter."""
        for _ in range(3):
            registry_with_model._record_load_failure("test-model", RuntimeError("CUDA out of memory"))

        failure = registry_with_model.get_failure("test-model")
        assert failure is not None
        assert failure.attempts == 3
        assert failure.error_class is LoadErrorClass.OOM

    async def test_start_load_async_short_circuits_when_failed(self, registry_with_model: ModelRegistry) -> None:
        """A recorded permanent failure makes ``start_load_async`` a no-op."""
        registry_with_model._record_load_failure("test-model", GatedModelError("org/test", RuntimeError("401")))

        started = await registry_with_model.start_load_async("test-model", "cpu")
        assert started is False
        # No background task was created either
        assert not registry_with_model.is_loading("test-model")

    async def test_oom_cooldown_blocks_then_releases(self, registry_with_model: ModelRegistry) -> None:
        """OOM failures are transient: blocked during cooldown, retryable after."""
        import asyncio

        from sie_server.core.load_errors import LoadFailure

        # Simulate an OOM record with a short cooldown by writing it in
        # with a back-dated timestamp so the cooldown window has already
        # elapsed without an actual sleep — keeps the test deterministic
        # and avoids ``ASYNC251`` violations from ``time.sleep`` in async.
        now = time.monotonic()
        registry_with_model._failed["test-model"] = LoadFailure(
            error_class=LoadErrorClass.OOM,
            message="RuntimeError: CUDA out of memory",
            attempts=1,
            last_attempt_ts=now,
            cooldown_s=10.0,  # long cooldown
        )

        # Inside cooldown — start_load_async no-ops.
        assert await registry_with_model.start_load_async("test-model", "cpu") is False

        # Rewrite the record with the timestamp far in the past so the
        # cooldown is considered expired without sleeping.
        registry_with_model._failed["test-model"] = LoadFailure(
            error_class=LoadErrorClass.OOM,
            message="RuntimeError: CUDA out of memory",
            attempts=1,
            last_attempt_ts=now - 1000.0,
            cooldown_s=10.0,
        )
        # Yield so the event loop is consistent with surrounding async test.
        await asyncio.sleep(0)
        assert not registry_with_model.is_failed("test-model")

    async def test_clear_failure_re_arms(self, registry_with_model: ModelRegistry) -> None:
        registry_with_model._record_load_failure("test-model", GatedModelError("org/test", RuntimeError("401")))
        assert registry_with_model.is_failed("test-model")
        cleared = registry_with_model.clear_failure("test-model")
        assert cleared is True
        assert not registry_with_model.is_failed("test-model")
        # Subsequent clear is a no-op and returns False.
        assert registry_with_model.clear_failure("test-model") is False

    async def test_successful_load_clears_prior_failure(self, registry_with_model: ModelRegistry) -> None:
        """A subsequent successful load wipes the failure record."""
        # Stamp a transient failure first.
        registry_with_model._record_load_failure("test-model", ConnectionError("dns failure"))
        assert registry_with_model.is_failed("test-model")

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            # Force-clear cooldown so load_async runs (would otherwise be
            # blocked in real flow; here we exercise the success-path
            # cleanup directly).
            registry_with_model._failed["test-model"] = registry_with_model._failed["test-model"].__class__(
                error_class=LoadErrorClass.NETWORK,
                message="ConnectionError",
                attempts=1,
                last_attempt_ts=time.monotonic() - 1000,
                cooldown_s=0.0001,
            )

            await registry_with_model._load_model_background("test-model", "cpu")

        assert registry_with_model.is_loaded("test-model")
        assert registry_with_model.get_failure("test-model") is None

    async def test_add_config_clears_failure(self, registry_with_model: ModelRegistry) -> None:
        """A config update is operator intent and clears sticky failures."""
        registry_with_model._record_load_failure("test-model", GatedModelError("org/test", RuntimeError("401")))
        assert registry_with_model.is_failed("test-model")

        # Re-add a (possibly fixed) config.
        registry_with_model.add_config(_make_config(name="test-model"))
        assert not registry_with_model.is_failed("test-model")
