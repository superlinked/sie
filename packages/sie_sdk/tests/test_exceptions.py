"""Tests for sie_sdk.exceptions module."""

from __future__ import annotations

import pytest
from sie_sdk.exceptions import GatedModelError


class TestGatedModelError:
    """Tests for GatedModelError exception class."""

    def test_creates_exception_with_model_id_and_error(self) -> None:
        """Exception stores model ID and original error."""
        original = ValueError("some HF error")
        error = GatedModelError("org/model", original)

        assert error.model_id == "org/model"
        assert error.original_error is original

    def test_message_contains_model_id(self) -> None:
        """Error message includes the model ID."""
        error = GatedModelError("BAAI/bge-m3", ValueError("test"))
        message = str(error)

        assert "BAAI/bge-m3" in message

    def test_message_contains_token_url(self) -> None:
        """Error message includes HuggingFace token settings URL."""
        error = GatedModelError("org/model", ValueError("test"))
        message = str(error)

        assert "https://huggingface.co/settings/tokens" in message

    def test_message_contains_license_url(self) -> None:
        """Error message includes model license acceptance URL."""
        error = GatedModelError("meta-llama/Llama-2-7b", ValueError("test"))
        message = str(error)

        assert "https://huggingface.co/meta-llama/Llama-2-7b" in message

    def test_message_contains_hf_token_instructions(self) -> None:
        """Error message explains how to set HF_TOKEN."""
        error = GatedModelError("org/model", ValueError("test"))
        message = str(error)

        assert "HF_TOKEN" in message
        assert "export HF_TOKEN=" in message
        assert "HF_TOKEN=hf_xxx mise run serve" in message

    def test_message_contains_original_error(self) -> None:
        """Error message includes the original exception details."""
        original = ValueError("Repository not found")
        error = GatedModelError("org/model", original)
        message = str(error)

        assert "Repository not found" in message
        assert "Original error:" in message

    def test_message_explains_gated_model_requirement(self) -> None:
        """Error message clearly states the model is gated."""
        error = GatedModelError("org/model", ValueError("test"))
        message = str(error)

        assert "gated" in message.lower()
        assert "authentication" in message.lower()

    def test_can_be_caught_as_exception(self) -> None:
        """GatedModelError can be caught as Exception."""
        with pytest.raises(Exception, match="Access denied") as exc_info:
            raise GatedModelError("org/model", ValueError("test"))

        assert isinstance(exc_info.value, GatedModelError)

    def test_can_be_caught_specifically(self) -> None:
        """GatedModelError can be caught by its specific type."""
        with pytest.raises(GatedModelError) as exc_info:
            raise GatedModelError("org/model", ValueError("test"))

        assert exc_info.value.model_id == "org/model"

    def test_different_model_ids_produce_different_messages(self) -> None:
        """Different model IDs result in different error messages."""
        error1 = GatedModelError("org1/model1", ValueError("test"))
        error2 = GatedModelError("org2/model2", ValueError("test"))

        assert "org1/model1" in str(error1)
        assert "org2/model2" in str(error2)
        assert "org1/model1" not in str(error2)
        assert "org2/model2" not in str(error1)

    def test_message_has_clear_numbered_steps(self) -> None:
        """Error message provides clear numbered steps to fix the issue."""
        error = GatedModelError("org/model", ValueError("test"))
        message = str(error)

        assert "1." in message
        assert "2." in message
        assert "3." in message

    def test_preserves_exception_chain(self) -> None:
        """Exception properly chains with original error using 'from'."""
        original = ValueError("HF Hub error")

        with pytest.raises(GatedModelError) as exc_info:
            raise GatedModelError("org/model", original)

        # The original error should be accessible
        assert exc_info.value.original_error is original
