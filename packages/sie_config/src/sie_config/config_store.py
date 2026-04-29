from __future__ import annotations

import logging

import yaml
from sie_sdk.storage import StorageBackend, get_storage_backend, is_cloud_path, join_path

from sie_config import metrics as sie_metrics

logger = logging.getLogger(__name__)


class ConfigStore:
    """Config store backed by local filesystem, S3, or GCS.

    The backend is auto-detected from the base_dir URL scheme:
    - Local path (e.g., /tmp/config-store) -> LocalBackend
    - s3://bucket/prefix -> S3Backend
    - gs://bucket/prefix -> GCSBackend

    Layout (same for all backends):
        {base_dir}/models/{model_id}.yaml   # Per-model config files
        {base_dir}/epoch                     # Monotonic counter (plain text integer)

    Args:
        base_dir: Root path for config storage. Accepts local path, s3://, or gs://.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir.rstrip("/")
        self._backend: StorageBackend = get_storage_backend(self._base_dir)
        self._models_path = join_path(self._base_dir, "models")
        self._epoch_path = join_path(self._base_dir, "epoch")
        self._is_cloud = is_cloud_path(self._base_dir)

        # For local backend, ensure directories exist
        if not self._is_cloud:
            from pathlib import Path

            Path(self._models_path).mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> str:
        return self._base_dir

    def read_epoch(self) -> int:
        """Read the current epoch value.

        Returns:
            Current epoch, or 0 if epoch file doesn't exist.
        """
        try:
            if not self._backend.exists(self._epoch_path):
                return 0
            content = self._backend.read_text(self._epoch_path).strip()
            return int(content) if content else 0
        except (ValueError, OSError):
            logger.warning("Failed to read epoch, returning 0")
            return 0

    def increment_epoch(self) -> int:
        """Increment epoch by 1 and return the new value. Single-writer only.

        On success, mirrors the new epoch into the
        `sie_config_epoch` Prometheus gauge. On backend write
        failure, bumps `sie_config_store_writes_total{op="increment_epoch",result="failure"}`
        and re-raises so the caller (the write handler) returns 5xx
        rather than silently losing the epoch bump.
        """
        current = self.read_epoch()
        new_epoch = current + 1
        try:
            self._backend.write_text(self._epoch_path, str(new_epoch))
        except Exception:
            sie_metrics.record_store_write(
                sie_metrics.STORE_OP_INCREMENT_EPOCH,
                sie_metrics.STORE_RESULT_FAILURE,
            )
            raise
        sie_metrics.record_store_write(
            sie_metrics.STORE_OP_INCREMENT_EPOCH,
            sie_metrics.STORE_RESULT_SUCCESS,
        )
        sie_metrics.set_epoch(new_epoch)
        return new_epoch

    def _validate_model_id(self, model_id: str) -> None:
        """Reject model IDs containing the reserved '__' sequence.

        Raises:
            ValueError: If model_id contains '__'.
        """
        if "__" in model_id:
            raise ValueError(
                f"Model ID {model_id!r} contains reserved sequence '__'. Use '/' as the separator (e.g., 'org/model')."
            )

    def write_model(self, model_id: str, config_yaml: str) -> None:
        """Write a model config YAML to the store.

        The model_id is sanitized for path use (/ -> __).

        Args:
            model_id: Model identifier (e.g., "BAAI/bge-m3").
            config_yaml: Raw YAML content to persist.

        Raises:
            ValueError: If model_id contains reserved sequence '__'.
        """
        self._validate_model_id(model_id)
        filename = model_id.replace("/", "__") + ".yaml"
        filepath = join_path(self._models_path, filename)
        try:
            self._backend.write_text(filepath, config_yaml)
        except Exception:
            sie_metrics.record_store_write(
                sie_metrics.STORE_OP_WRITE_MODEL,
                sie_metrics.STORE_RESULT_FAILURE,
            )
            raise
        sie_metrics.record_store_write(
            sie_metrics.STORE_OP_WRITE_MODEL,
            sie_metrics.STORE_RESULT_SUCCESS,
        )
        logger.debug("Wrote model config: %s", filepath)

    def read_model(self, model_id: str) -> str | None:
        """Read a model config YAML from the store.

        Args:
            model_id: Model identifier.

        Returns:
            YAML content string, or None if not found.

        Raises:
            ValueError: If model_id contains reserved sequence '__'.
        """
        self._validate_model_id(model_id)
        filename = model_id.replace("/", "__") + ".yaml"
        filepath = join_path(self._models_path, filename)
        try:
            if not self._backend.exists(filepath):
                return None
            return self._backend.read_text(filepath)
        except Exception:  # noqa: BLE001 -- read failure should not crash
            logger.warning("Failed to read model config: %s", model_id)
            return None

    def list_models(self) -> list[str]:
        """List all stored model IDs.

        Returns:
            List of model IDs (with / restored from __).
        """
        try:
            filenames = list(self._backend.list_files(self._models_path, "*.yaml"))
        except Exception:  # noqa: BLE001
            return []
        return sorted(fn.removesuffix(".yaml").replace("__", "/") for fn in filenames)

    def load_all_models(self) -> dict[str, dict]:
        """Load all model configs from the store.

        Returns:
            Dict mapping model_id to parsed YAML config dict.
        """
        result: dict[str, dict] = {}
        for model_id in self.list_models():
            content = self.read_model(model_id)
            if content:
                try:
                    result[model_id] = yaml.safe_load(content) or {}
                except yaml.YAMLError:
                    logger.exception("Failed to parse stored model config: %s", model_id)
        return result
