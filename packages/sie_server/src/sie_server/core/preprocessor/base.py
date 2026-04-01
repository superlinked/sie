"""Preprocessor protocol and shared utilities.

This module defines the Preprocessor protocol that all modality-specific
preprocessors must implement, plus shared infrastructure for image processing.

See roadmap.md Project 10.4 Phase 2 for design details.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sie_server.core.prepared import PreparedBatch, PreparedItem

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Global thread pool for parallel image preprocessing
# CPU-bound but GIL released during PIL I/O and C extension calls
_IMAGE_PREPROCESS_WORKERS = int(os.environ.get("SIE_IMAGE_WORKERS", "4"))
_image_executor: ThreadPoolExecutor | None = None


def get_image_executor() -> ThreadPoolExecutor:
    """Get or create the image preprocessing thread pool."""
    global _image_executor
    if _image_executor is None:
        _image_executor = ThreadPoolExecutor(
            max_workers=_IMAGE_PREPROCESS_WORKERS,
            thread_name_prefix="img_preproc",
        )
        logger.info("Image preprocessing pool: %d workers", _IMAGE_PREPROCESS_WORKERS)
    return _image_executor


def check_pillow_features() -> dict[str, bool]:
    """Check Pillow optimization features.

    Returns:
        Dict with feature availability.
    """
    try:
        from PIL import features

        return {
            "libjpeg_turbo": bool(features.check("libjpeg_turbo")),
            "webp": bool(features.check("webp")),
        }
    except (ImportError, AttributeError):
        return {"libjpeg_turbo": False, "webp": False}


# Log Pillow features once at import time
_PILLOW_FEATURES = check_pillow_features()
if _PILLOW_FEATURES.get("libjpeg_turbo"):
    logger.debug("Pillow using libjpeg-turbo for fast JPEG decode")


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for modality-specific preprocessing.

    Preprocessors transform raw input items into prepared items ready for
    batching. They run in a thread pool (CPU-bound work) and produce
    PreparedItem instances with cost information for the BatchFormer.

    Implementations must be thread-safe as multiple threads may call
    prepare() concurrently.
    """

    @property
    def modality(self) -> str:
        """Return modality name: 'text', 'image', 'audio'."""
        ...

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[Any]:
        """Prepare items for batching.

        This method is CPU-bound and runs in a thread pool. It should:
        1. Extract relevant data from items (text, images, etc.)
        2. Preprocess (tokenize, resize, normalize, etc.)
        3. Compute cost for each item
        4. Return PreparedBatch with all items

        Args:
            items: Raw input items to prepare.
            config: Model configuration for preprocessing parameters.
            is_query: Whether items are queries (True) or documents (False).
                Used for query-specific preprocessing (e.g., ColBERT query
                expansion, instruction prefixes, query max length limits).
            instruction: Optional instruction text (e.g., DocVQA question).
                Used by vision preprocessors like Florence2Preprocessor.
            task: Optional task token (e.g., "<DocVQA>", "<OCR_WITH_REGION>").
                Used by vision preprocessors to override default task.

        Returns:
            PreparedBatch containing all prepared items.
        """
        ...

    def collate(
        self,
        prepared: list[PreparedItem[Any]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate prepared items into tensors for inference.

        Called just before inference to convert prepared items into
        the tensor format expected by the model.

        Args:
            prepared: List of prepared items to collate.
            device: Target device for tensors.

        Returns:
            Dict of tensors ready for model forward pass.
        """
        ...
