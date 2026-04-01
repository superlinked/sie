"""Exceptions for SIE operations."""

from __future__ import annotations


class GatedModelError(Exception):
    """Raised when attempting to access a gated model without proper authentication.

    This error is raised when downloading a HuggingFace model that requires:
    1. Accepting the model's license agreement
    2. Providing a valid HuggingFace token with access to the model

    Attributes:
        model_id: The HuggingFace model ID that was attempted
        original_error: The underlying exception from huggingface_hub
    """

    def __init__(self, model_id: str, original_error: Exception) -> None:
        self.model_id = model_id
        self.original_error = original_error
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return f"""
Access denied for model '{self.model_id}'.

This model is gated and requires HuggingFace authentication.

To fix this:
1. Get a token at: https://huggingface.co/settings/tokens
2. Accept the model license at: https://huggingface.co/{self.model_id}
3. Set the HF_TOKEN environment variable:

   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

   Or pass it when starting the server:

   HF_TOKEN=hf_xxx mise run serve

Original error: {self.original_error}
"""
