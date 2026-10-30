"""Tokenization utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def image_first_chat_message(*, role: str, text: str, image_count: int) -> dict[str, Any]:
    """Build the canonical image-first chat-template message.

    Native image ``generate`` and queue-backed message generation both use
    this helper, keeping their model-template input byte-for-byte equivalent.
    Image bytes travel separately; this value contains placeholders only.
    """
    content: list[dict[str, str]] = [{"type": "image"} for _ in range(image_count)]
    if text:
        content.append({"type": "text", "text": text})
    return {"role": role, "content": content}


def load_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a HuggingFace tokenizer from a local path or model id.

    When ``revision`` is a pinned commit SHA it is forwarded to
    ``from_pretrained`` so the load resolves the exact ``snapshots/<sha>``
    directory directly. This mirrors how the weights loader pins the
    revision (``core/loader.py``) and — crucially under ``HF_HUB_OFFLINE=1``
    — avoids depending on the ``refs/main`` alias, which the Volume-backed HF
    cache lacks for repos staged by pinned SHA (a no-revision load would try
    to resolve ``refs/main`` and fail offline). ``None`` keeps the legacy
    no-revision behaviour for local-path / unpinned loads.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    logger.debug(
        "Loaded tokenizer from %s@%s (fast=%s)",
        model_path,
        revision or "default",
        getattr(tokenizer, "is_fast", False),
    )
    return tokenizer


def load_image_token_id(model_path: str | Path, *, revision: str | None = None) -> int | None:
    """Return the image placeholder token id the model's config declares.

    A vision processor expands each occurrence of this token into the
    image's visual tokens. Read from the top-level ``image_token_id`` or
    ``image_token_index`` of the model's ``config.json``, resolved like
    :func:`load_tokenizer` resolves the tokenizer. ``None`` when the config
    declares neither as a non-negative integer.
    """
    from transformers import PretrainedConfig

    config, _ = PretrainedConfig.get_config_dict(str(model_path), revision=revision)
    for key in ("image_token_id", "image_token_index"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None
