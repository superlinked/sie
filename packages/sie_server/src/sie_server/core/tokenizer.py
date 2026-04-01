"""Tokenization utilities.

Contains load_tokenizer() for loading HuggingFace tokenizers.
For preprocessing types, see the prepared module (PreparedItem, TextPayload, etc.).

See DESIGN.md Section 5.2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a tokenizer from a model path or HuggingFace ID.

    Attempts to load the fast (Rust) tokenizer first, falls back to Python.

    Args:
        model_path: Local path or HuggingFace model ID.
        trust_remote_code: Whether to trust remote code in the tokenizer.

    Returns:
        Loaded tokenizer instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    logger.debug(
        "Loaded tokenizer from %s (fast=%s)",
        model_path,
        getattr(tokenizer, "is_fast", False),
    )
    return tokenizer
