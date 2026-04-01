"""Text preprocessors for tokenization and cost estimation.

This module contains preprocessors for text modality:
- TextPreprocessor: Full tokenization using HuggingFace tokenizers
- CharCountPreprocessor: Cost estimation for library-wrapped adapters
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sie_server.core.prepared import PreparedBatch, PreparedItem, TextPayload

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from sie_server.config.model import ModelConfig
    from sie_server.types.inputs import Item


class TextPreprocessor:
    """Preprocessor for text tokenization.

    Wraps a HuggingFace tokenizer to produce TextPayload items.
    Thread-safe: tokenizers handle concurrent calls internally.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        model_name: str,
    ) -> None:
        """Initialize with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance.
            model_name: Model name for logging.
        """
        self._tokenizer = tokenizer
        self._model_name = model_name

    @property
    def modality(self) -> str:
        """Return 'text'."""
        return "text"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[TextPayload]:
        """Tokenize text items.

        Args:
            items: Items with text field.
            config: Model config with max_sequence_length.
            is_query: Whether items are queries (True) or documents (False).
                Currently unused - will be used for query-specific tokenization
                in Phase 4 (e.g., ColBERT query expansion, instruction prefixes).
            instruction: Optional instruction (unused for text preprocessing).
            task: Optional task token (unused for text preprocessing).

        Returns:
            PreparedBatch with TextPayload items.
        """
        # Note: is_query available for future query-specific tokenization
        _ = is_query  # Unused for now, will be used in Phase 4
        del instruction, task  # Unused - only needed for vision models

        # Extract texts
        texts = [item.text or "" for item in items]

        # Build tokenizer kwargs
        kwargs: dict[str, Any] = {
            "padding": False,  # Pad later when forming batches
            "truncation": True,
            "return_attention_mask": True,
        }
        if config.max_sequence_length:
            kwargs["max_length"] = config.max_sequence_length

        # Tokenize all at once (HuggingFace handles batches efficiently)
        encoded = self._tokenizer(texts, **kwargs)

        # Build prepared items
        prepared_items: list[PreparedItem[TextPayload]] = []
        total_cost = 0

        for i, (input_ids, attention_mask) in enumerate(
            zip(encoded["input_ids"], encoded["attention_mask"], strict=True)
        ):
            token_count = len(input_ids)
            payload = TextPayload(input_ids=input_ids, attention_mask=attention_mask)
            prepared_items.append(PreparedItem(payload=payload, cost=token_count, original_index=i))
            total_cost += token_count

        return PreparedBatch(
            items=prepared_items,
            total_cost=total_cost,
            modality="text",
        )

    def collate(
        self,
        prepared: list[PreparedItem[TextPayload]],
        *,
        device: str,
        pad_token_id: int = 0,
    ) -> dict[str, Any]:
        """Collate tokenized items into padded tensors.

        Args:
            prepared: List of prepared text items.
            device: Target device.
            pad_token_id: Token ID for padding.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        import torch

        if not prepared:
            return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}

        # Find max length
        max_length = max(p.payload.token_count for p in prepared)

        input_ids_batch = []
        attention_mask_batch = []

        for p in prepared:
            payload = p.payload
            padding_length = max_length - payload.token_count

            padded_ids = payload.input_ids + [pad_token_id] * padding_length
            padded_mask = payload.attention_mask + [0] * padding_length

            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(padded_mask)

        return {
            "input_ids": torch.tensor(input_ids_batch, device=device),
            "attention_mask": torch.tensor(attention_mask_batch, device=device),
        }


class CharCountPreprocessor:
    """Simple cost estimator for library-wrapped adapters.

    Uses character count as a cost proxy instead of actual tokenization.
    This avoids tokenization overhead for adapters that handle tokenization
    internally (e.g., BGE-M3, Qwen embeddings, rerankers).

    The cost multiplier converts characters to approximate token count:
    - English: ~4 chars/token on average
    - With a buffer for safety, we use ~3.5 chars/token (multiplier=0.3)

    Usage:
        # In adapter's get_preprocessor():
        return CharCountPreprocessor(model_name="my-model", chars_per_token=4.0)
    """

    is_trivial: bool = True

    def __init__(
        self,
        model_name: str,
        chars_per_token: float = 4.0,
    ) -> None:
        """Initialize with cost estimation parameters.

        Args:
            model_name: Model name for logging.
            chars_per_token: Average characters per token for cost estimation.
        """
        self._model_name = model_name
        self._chars_per_token = chars_per_token

    @property
    def modality(self) -> str:
        """Return 'text'."""
        return "text"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[TextPayload]:
        """Estimate cost from character count.

        Creates TextPayload items with None for tokenization data since
        the adapter handles tokenization internally.

        Args:
            items: Items with text field.
            config: Model config (unused).
            is_query: Whether items are queries (unused).
            instruction: Optional instruction (unused).
            task: Optional task token (unused).

        Returns:
            PreparedBatch with estimated token counts.
        """
        prepared_items: list[PreparedItem[TextPayload]] = []
        total_cost = 0

        for i, item in enumerate(items):
            text = self._get_text_safe(item)
            char_count = len(text)

            # Estimate token count from character count
            estimated_tokens = max(1, int(char_count / self._chars_per_token))
            total_cost += estimated_tokens

            # Create payload with empty tokenization data
            # The adapter will handle tokenization internally
            # The cost for batching is set in PreparedItem.cost, not here
            payload = TextPayload(
                input_ids=[],  # Empty - adapter tokenizes internally
                attention_mask=[],
            )

            prepared_items.append(
                PreparedItem(
                    cost=estimated_tokens,
                    original_index=i,
                    payload=payload,
                )
            )

        return PreparedBatch(items=prepared_items, total_cost=total_cost)

    def _get_text_safe(self, item: Item) -> str:
        raw = item.text
        if raw is None:
            text = ""
        elif not isinstance(raw, str):
            raise ValueError(f"text item must be a string, got: {raw}")
        else:
            text = raw
        return text

    def collate(
        self,
        prepared: list[PreparedItem[TextPayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate is a no-op for CharCountPreprocessor.

        The adapter handles tokenization and tensor creation internally.

        Args:
            prepared: List of prepared items (unused).
            device: Target device (unused).

        Returns:
            Empty dict - adapter handles its own input preparation.
        """
        return {}
