from pathlib import Path

import pytest
import yaml
from sie_router.model_registry import ModelRegistry
from sie_router.registry import WorkerRegistry


def _setup_registry_with_bundle(
    tmp_path: Path,
    adapters: list[str] | None = None,
) -> ModelRegistry:
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()

    if adapters is None:
        adapters = ["sie_server.adapters.bert_flash"]

    data = {"name": "default", "priority": 10, "adapters": adapters}
    (bundles_dir / "default.yaml").write_text(yaml.dump(data))

    return ModelRegistry(bundles_dir, models_dir)


class TestWorkerConfigHashTracking:
    """Tests for worker bundle_config_hash tracking in WorkerRegistry."""

    @pytest.mark.asyncio
    async def test_worker_reports_bundle_config_hash(self) -> None:
        """Worker status with bundle_config_hash is tracked."""
        registry = WorkerRegistry()
        await registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "abc123",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )
        worker = registry.get_worker("http://w1:8080")
        assert worker is not None
        assert worker.bundle_config_hash == "abc123"

    @pytest.mark.asyncio
    async def test_worker_hash_updates_on_status(self) -> None:
        """Worker hash changes when status message has new hash."""
        registry = WorkerRegistry()
        await registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "hash1",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )
        assert registry.get_worker("http://w1:8080").bundle_config_hash == "hash1"

        await registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "hash2",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )
        assert registry.get_worker("http://w1:8080").bundle_config_hash == "hash2"

    @pytest.mark.asyncio
    async def test_worker_empty_hash_by_default(self) -> None:
        """Worker with no hash in status gets empty string."""
        registry = WorkerRegistry()
        await registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )
        assert registry.get_worker("http://w1:8080").bundle_config_hash == ""


class TestBundleConfigHashScoping:
    """Tests for bundle-scoped config hashing."""

    def test_hash_only_includes_models_in_bundle(self, tmp_path: Path) -> None:
        """Hash for bundle A doesn't include models routable only to bundle B."""
        bundles_dir = tmp_path / "bundles"
        models_dir = tmp_path / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        (bundles_dir / "default.yaml").write_text(
            yaml.dump(
                {
                    "name": "default",
                    "priority": 10,
                    "adapters": ["sie_server.adapters.bert_flash"],
                }
            )
        )
        (bundles_dir / "sglang.yaml").write_text(
            yaml.dump(
                {
                    "name": "sglang",
                    "priority": 20,
                    "adapters": ["sie_server.adapters.sglang"],
                }
            )
        )

        registry = ModelRegistry(bundles_dir, models_dir)

        # Add model to default bundle only
        registry.add_model_config(
            {
                "sie_id": "default/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )

        default_hash = registry.compute_bundle_config_hash("default")
        sglang_hash = registry.compute_bundle_config_hash("sglang")

        assert default_hash != ""
        assert sglang_hash == ""  # No models in sglang

    def test_hash_changes_only_for_affected_bundle(self, tmp_path: Path) -> None:
        """Adding model to bundle A doesn't change bundle B's hash."""
        bundles_dir = tmp_path / "bundles"
        models_dir = tmp_path / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        (bundles_dir / "default.yaml").write_text(
            yaml.dump(
                {
                    "name": "default",
                    "priority": 10,
                    "adapters": ["sie_server.adapters.bert_flash"],
                }
            )
        )
        (bundles_dir / "sglang.yaml").write_text(
            yaml.dump(
                {
                    "name": "sglang",
                    "priority": 20,
                    "adapters": ["sie_server.adapters.sglang"],
                }
            )
        )

        registry = ModelRegistry(bundles_dir, models_dir)

        # Add model to sglang
        registry.add_model_config(
            {
                "sie_id": "sglang/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.sglang:S", "max_batch_tokens": 1}},
            }
        )
        sglang_hash_before = registry.compute_bundle_config_hash("sglang")
        default_hash_before = registry.compute_bundle_config_hash("default")

        # Add model to default only
        registry.add_model_config(
            {
                "sie_id": "default/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        sglang_hash_after = registry.compute_bundle_config_hash("sglang")
        default_hash_after = registry.compute_bundle_config_hash("default")

        assert sglang_hash_before == sglang_hash_after  # Unchanged
        assert default_hash_before != default_hash_after  # Changed


class TestServingReadinessWithWorkers:
    """Tests for serving readiness detection with actual workers."""

    @pytest.mark.asyncio
    async def test_workers_with_matching_hash_are_acked(self) -> None:
        """Workers reporting the expected hash count as ACKed."""
        from sie_router.config_api import _wait_for_serving_readiness

        worker_registry = WorkerRegistry()
        await worker_registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "expected_hash",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )

        result = await _wait_for_serving_readiness(
            worker_registry=worker_registry,
            affected_bundles=["default"],
            bundle_config_hashes={"default": "expected_hash"},
            timeout_s=1.0,
        )

        assert result["worker_ack_pending"] is False
        assert result["acked_workers"] == 1
        assert result["total_eligible"] == 1
        assert result["pending_workers"] == 0

    @pytest.mark.asyncio
    async def test_workers_with_stale_hash_are_pending(self) -> None:
        """Workers with old hash count as pending."""
        from sie_router.config_api import _wait_for_serving_readiness

        worker_registry = WorkerRegistry()
        await worker_registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "old_hash",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )

        result = await _wait_for_serving_readiness(
            worker_registry=worker_registry,
            affected_bundles=["default"],
            bundle_config_hashes={"default": "new_hash"},
            timeout_s=0.3,  # Short timeout
        )

        assert result["worker_ack_pending"] is True
        assert result["acked_workers"] == 0
        assert result["pending_workers"] == 1

    @pytest.mark.asyncio
    async def test_mixed_workers_partial_ack(self) -> None:
        """Some workers ACK, others don't — partial readiness."""
        from sie_router.config_api import _wait_for_serving_readiness

        worker_registry = WorkerRegistry()
        await worker_registry.update_worker(
            "http://w1:8080",
            {
                "ready": True,
                "name": "w1",
                "bundle": "default",
                "bundle_config_hash": "new_hash",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )
        await worker_registry.update_worker(
            "http://w2:8080",
            {
                "ready": True,
                "name": "w2",
                "bundle": "default",
                "bundle_config_hash": "old_hash",
                "gpu_count": 1,
                "loaded_models": [],
                "models": [],
                "gpus": [],
            },
        )

        result = await _wait_for_serving_readiness(
            worker_registry=worker_registry,
            affected_bundles=["default"],
            bundle_config_hashes={"default": "new_hash"},
            timeout_s=0.3,
        )

        # The polling loop exits early (>=1 ACK per bundle met), but
        # worker_ack_pending reflects whether ALL eligible workers have
        # ACKed.  With 1/2 ACKed, pending_workers=1 so ack_pending=True.
        assert result["worker_ack_pending"] is True
        assert result["acked_workers"] == 1
        assert result["total_eligible"] == 2
        assert result["pending_workers"] == 1
