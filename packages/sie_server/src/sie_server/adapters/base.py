"""Base model adapter interface.

See DESIGN.md Section 7.3 for specification.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput

logger = logging.getLogger(__name__)


class ModelCapabilities(BaseModel):
    """Capabilities supported by a model adapter."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[Literal["text", "image", "audio", "video"]] = Field(description="Input modalities supported")
    outputs: list[Literal["dense", "sparse", "multivector", "score", "json"]] = Field(
        description="Output types the adapter can produce"
    )


class ModelDims(BaseModel):
    """Dimension information for model outputs."""

    model_config = ConfigDict(extra="forbid")

    dense: int | None = Field(default=None, description="Dense vector dimensionality")
    sparse: int | None = Field(default=None, description="Sparse vector vocabulary size")
    multivector: int | None = Field(default=None, description="Per-token dimension for multi-vector")


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Each model adapter wraps a specific model architecture and provides
    a consistent interface for loading, inference, and unloading.

    Adapters are resolved in this order (see DESIGN.md Section 7.2):
    1. base_model: inherit adapter from another model's config
    2. adapter: adapter.py:ClassName (file in model directory)
    3. adapter: sie_server.adapters.module:ClassName (built-in adapter)

    Memory Management Contract:
        Adapters are responsible for fully releasing GPU memory in unload().
        After unload() returns, the device memory should be reclaimable.

        For PyTorch-based adapters, this means:
        1. Delete model references (del self._model, etc.)
        2. Run gc.collect() to force Python to release objects
        3. Call torch.cuda.empty_cache() or torch.mps.empty_cache()

        Non-PyTorch adapters (e.g., SGLang) should use their own cleanup
        mechanism (e.g., engine.shutdown()) that fully releases memory.

    Main Thread Requirement:
        Some adapters (e.g., SGLang) use signal handlers that only work in
        the main thread. Set requires_main_thread = True for these adapters.
        The registry will run load() directly in the event loop instead of
        in a thread pool, which blocks but ensures main thread execution.

    Device-Aware Factory Method:
        Adapters can override create_for_device() to return different adapter
        implementations based on device compatibility. Flash adapters use this
        to automatically fall back to standard implementations on non-CUDA devices.
    """

    # Set to True for adapters that need main thread (e.g., SGLang signal handlers)
    requires_main_thread: bool = False

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> "ModelAdapter":
        """Factory method for device-aware adapter instantiation.

        Default implementation creates the adapter normally. Flash adapters
        override this to check device compatibility and return fallback adapters
        when flash-attention is unavailable or device is not CUDA.

        Args:
            device: Target device (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            Adapter instance (may be fallback adapter for incompatible devices).

        Example (Flash adapter with fallback):
            @classmethod
            def create_for_device(cls, device: str, **kwargs) -> ModelAdapter:
                from sie_server.adapters.cross_encoder import CrossEncoderAdapter
                return cls._create_flash_or_fallback(
                    device,
                    fallback_class=CrossEncoderAdapter,
                    fallback_kwargs={**kwargs, "attn_implementation": "sdpa"},
                    **kwargs
                )
        """
        return cls(**kwargs)

    @classmethod
    def _create_flash_or_fallback(
        cls,
        device: str,
        fallback_class: type["ModelAdapter"],
        fallback_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ModelAdapter":
        """Helper for flash adapters to create with fallback logic.

        Checks if device is CUDA and Flash Attention is available (hardware + package).
        If not, returns the fallback adapter instead. This extracts common logic
        shared by all flash adapters to reduce duplication.

        Args:
            device: Target device (e.g., "cuda:0", "mps", "cpu").
            fallback_class: Adapter class to use when Flash Attention unavailable.
            fallback_kwargs: Kwargs for fallback (defaults to same as flash adapter).
            **kwargs: Adapter initialization parameters.

        Returns:
            Flash adapter instance on CUDA with Flash Attention, fallback otherwise.
        """
        # Import here to avoid circular dependency (core.inference imports core.loader imports base)
        from sie_server.core.inference import is_flash_attention_available

        if fallback_kwargs is None:
            fallback_kwargs = kwargs

        # Quick device check before hardware detection
        if not device.startswith("cuda"):
            logger.info(
                "%s requires CUDA. Using %s for device '%s'. "
                "For optimal performance, use a Linux system with NVIDIA GPU.",
                cls.__name__,
                fallback_class.__name__,
                device,
            )
            return fallback_class(**fallback_kwargs)

        # Check Flash Attention availability (hardware capability + package)
        if not is_flash_attention_available():
            logger.warning(
                "Flash Attention unavailable (requires Ampere+ GPU and flash-attn package). "
                "Using %s for device '%s'. "
                "To install on Linux: pip install sie-server[flash-attn]",
                fallback_class.__name__,
                device,
            )
            return fallback_class(**fallback_kwargs)

        return cls(**kwargs)

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return the model's capabilities.

        Returns:
            ModelCapabilities describing what the model can do.
        """

    @property
    @abstractmethod
    def dims(self) -> ModelDims:
        """Return the model's output dimensions.

        Returns:
            ModelDims with dimension info for each output type.
        """

    @abstractmethod
    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources.

        Must release GPU memory and any other resources. After this method
        returns, the device memory should be reclaimable by the system.

        See class docstring for the memory management contract.
        """

    def memory_footprint(self) -> int:
        """Return the GPU memory usage in bytes.

        Default implementation sums the memory of all parameters and buffers
        in the PyTorch model. This gives the actual memory used by the model's
        tensors, not an estimate.

        Looks for self._model or self.model attributes. Adapters can override
        this to provide custom implementations.

        Returns:
            Memory usage in bytes.
        """
        # Try to find a model attribute
        model = getattr(self, "_model", None) or getattr(self, "model", None)
        if model is None:
            return 0

        try:
            import torch

            # Handle wrapped models (e.g., FlagEmbedding's BGEM3FlagModel)
            if hasattr(model, "model"):
                model = model.model

            total_bytes = 0

            # Sum parameters
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, torch.Tensor):
                        total_bytes += param.numel() * param.element_size()

            # Sum buffers (non-parameter tensors like BatchNorm running stats)
            if hasattr(model, "buffers"):
                for buf in model.buffers():
                    if isinstance(buf, torch.Tensor):
                        total_bytes += buf.numel() * buf.element_size()

            return total_bytes
        except (AttributeError, TypeError, RuntimeError):
            pass

        return 0

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> "EncodeOutput":
        """Encode items into embeddings, returning standardized batched output.

        This is the canonical method for encoding. Returns EncodeOutput
        with batched results suitable for postprocessing. All embedding adapters
        must implement this method.

        For LoRA support: The worker calls set_active_lora() before encode() to
        switch the active LoRA adapter. Adapters that support LoRA should read
        the active LoRA from their internal state (set by set_active_lora).

        Args:
            items: List of items to encode.
            output_types: Which outputs to compute ("dense", "sparse", "multivector").
            instruction: Optional instruction for instruction-tuned models.
            is_query: Whether items are queries (True) or documents (False).
            prepared_items: Optional preprocessed items from PreprocessorRegistry.
            options: Runtime adapter options (merged from config, profile, and
                request overrides). Includes query_template, doc_template,
                default_instruction, etc. Adapters may use these to customize
                encoding behavior per-request.

        Returns:
            EncodeOutput with batched embeddings.

        Raises:
            NotImplementedError: If adapter hasn't implemented encode().
        """
        msg = f"{self.__class__.__name__} does not implement encode()"
        raise NotImplementedError(msg)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using a reranker.

        Args:
            query: Query item.
            items: List of items to score against the query.
            instruction: Optional instruction.
            options: Runtime options (config defaults -> profile -> request overrides).

        Returns:
            List of scores, one per item.

        Raises:
            NotImplementedError: If model doesn't support score.
        """
        msg = f"{self.__class__.__name__} does not support score()"
        raise NotImplementedError(msg)

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> "ScoreOutput":
        """Score (query, doc) pairs in a batch.

        This is the batched version of score() used by the worker for
        cross-request batching. Pairs from different requests can be
        batched together for GPU efficiency.

        Args:
            queries: Query items (parallel to docs).
            docs: Document items to score.
            instruction: Optional instruction.
            options: Runtime options (config defaults -> profile -> request overrides).

        Returns:
            ScoreOutput with batched scores.

        Raises:
            NotImplementedError: If model doesn't support score.
        """
        msg = f"{self.__class__.__name__} does not support score_pairs()"
        raise NotImplementedError(msg)

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> "ExtractOutput":
        """Extract structured data from items.

        Args:
            items: List of items to extract from.
            labels: Entity labels for NER (e.g., ["person", "organization"]).
            output_schema: Schema for structured extraction.
            instruction: Optional instruction.
            options: Adapter options to override model config defaults.
                    Common options: threshold, top_k, flat_ner, multi_label.
            prepared_items: Pre-processed items with payloads ready for inference.
                    Used by vision adapters (Florence-2, Donut) for CPU/GPU overlap.

        Returns:
            ExtractOutput with batched extraction results.

        Raises:
            NotImplementedError: If model doesn't support extract.
        """
        msg = f"{self.__class__.__name__} does not support extract()"
        raise NotImplementedError(msg)

    def get_preprocessor(self) -> Any | None:
        """Get the preprocessor for this adapter if available.

        Adapters can return a preprocessor to control how preprocessing is done:

        1. **Text preprocessor** (modality="text"): For text models.
           - Flash adapters can return a TextPreprocessor that wraps their tokenizer.
             This enables using prepared_items directly without re-tokenization.
           - Library-wrapped adapters can return a CharCountPreprocessor that uses
             character count for cost estimation without actual tokenization overhead.

        2. **Image preprocessor** (modality="image"): For vision models.
           - Vision adapters (Florence-2, Donut, NemoColEmbed) return their processor
             for CPU/GPU overlap - heavy preprocessing runs in a thread pool.

        If None is returned, the default behavior applies:
        - Text models: Tokenizer is loaded from model path and registered
        - Image models: Falls back to adapter._processor attribute sniffing

        Returns:
            Preprocessor instance with `modality` property, or None for defaults.
        """
        return None

    def get_postprocessors(self) -> dict[str, Any] | None:
        """Get postprocessors for output transforms.

        Adapters can return postprocessors for converting between output types.
        For example, a ColBERT adapter might return a MUVERA postprocessor that
        converts multivector embeddings to dense vectors.

        Postprocessors are keyed by option name (e.g., "muvera", "quantize").
        The worker checks runtime options for these keys and applies the
        postprocessor if the option is present and not null.

        Example:
            {"muvera": MuveraPostprocessor(token_dim=128, config=...)}

        The MUVERA config (num_repetitions, etc.) should come from loadtime
        options to ensure projection matrices are fixed across requests.

        Returns:
            Dict mapping option names to Postprocessor instances, or None.
        """
        return None

    # -------------------------------------------------------------------------
    # LoRA Support (optional - adapters opt-in by overriding these methods)
    # -------------------------------------------------------------------------

    def supports_lora(self) -> bool:
        """Return True if this adapter supports LoRA adapters.

        Adapters that support LoRA should override this to return True and
        implement load_lora(), unload_lora(), and set_active_lora().

        Returns:
            True if LoRA is supported, False otherwise.
        """
        return False

    def supports_hot_lora_reload(self) -> bool:
        """Return True if LoRAs can be loaded without blocking inference.

        - PEFT-based adapters: True (can load in thread pool)
        - SGLang adapters: False (HTTP API blocks during load)

        This affects whether the server returns 503 LORA_LOADING during
        loading or blocks all requests until loading completes.

        Returns:
            True for non-blocking LoRA loading, False for blocking.
        """
        return False

    def load_lora(self, lora_path: str) -> int:
        """Load a LoRA adapter.

        This is called by the LoRA manager to load a new adapter. The adapter
        should be ready for use after this method returns.

        For PEFT adapters: Uses PeftModel.from_pretrained() or load_adapter().
        For SGLang adapters: Calls /load_lora_adapter HTTP endpoint.

        Args:
            lora_path: HuggingFace path (e.g., "org/lora-name") or local path.

        Returns:
            Memory usage of the loaded LoRA in bytes.

        Raises:
            NotImplementedError: If adapter doesn't support LoRA.
            RuntimeError: If loading fails.
        """
        msg = f"{self.__class__.__name__} does not support LoRA (supports_lora=False)"
        raise NotImplementedError(msg)

    def unload_lora(self, lora_name: str) -> None:
        """Unload a LoRA adapter.

        Called during LRU eviction when max_loras is exceeded.

        For PEFT adapters: Calls peft_model.delete_adapter().
        For SGLang adapters: Calls /unload_lora_adapter HTTP endpoint.

        Args:
            lora_name: The LoRA adapter name to unload.

        Raises:
            NotImplementedError: If adapter doesn't support LoRA.
        """
        msg = f"{self.__class__.__name__} does not support LoRA (supports_lora=False)"
        raise NotImplementedError(msg)

    def set_active_lora(self, lora_name: str | None) -> None:
        """Set the active LoRA for the next inference call.

        Called before each batch to switch to the appropriate LoRA adapter.

        For PEFT adapters: Calls peft_model.set_adapter(lora_name).
        For SGLang adapters: No-op (LoRA is selected per-request via model name).

        Args:
            lora_name: LoRA adapter name, or None for base model.
        """
        # Default is no-op - adapters that need switching should override
