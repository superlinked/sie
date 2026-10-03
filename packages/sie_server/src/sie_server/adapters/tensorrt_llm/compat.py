"""Exact-source compatibility overlay for the TensorRT-LLM 1.3.0rc24 lane.

The selected runtime has three generic T5 direct-HF gaps: its T5 converter
requires ``shared.weight`` even when ``safetensors.torch.save_model`` retained
only a known encoder/decoder embedding key, its T5 model always scales decoder
outputs before the LM head, and Transformers 5.5.4 reconstructs a custom T5
tokenizer pipeline instead of preserving a checkpoint's ``tokenizer.json``.
The child process installs this overlay through a code-owned ``sitecustomize``
directory, while the parent adapter installs the tokenizer-only part before it
counts prompt tokens. Package versions and complete source-file hashes are
checked before patching so a future runtime cannot silently inherit an overlay
written for different code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

TRTLLM_VERSION = "1.3.0rc24"
TRANSFORMERS_VERSION = "5.5.4"
HF_WEIGHT_LOADER_SHA256 = "d8464ac2257b167c8e127606b495b1021ff69f7d3ff1ce663abc09fffa8b932a"
MODELING_T5_SHA256 = "248f80484668b8dee1480b3b49ab81d68193bdd9d4eecd29bd53068ba2dd3d46"
T5_TOKENIZATION_SHA256 = "6fa6696aa2bf6bf40bcd7c7aea81b5b581e365ada45c6d7768d53609174496d5"
TOKENIZERS_BACKEND_SHA256 = "b76df59838488e19539bc50f2a8cf7c8f16c48ef96859fcc1236f5dfe467d29d"
_PATCH_MARKER = "_sie_rc24_compatibility_installed"
_T5_SHARED_EMBEDDING_KEY = "shared.weight"
_T5_INPUT_EMBEDDING_KEYS = ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight")


@runtime_checkable
class _MutableWeights(Protocol):
    """Structural subset shared by dict and rc24 ConsumableWeightsDict."""

    def __contains__(self, key: object) -> bool:
        pass

    def __getitem__(self, key: str) -> Any:
        pass

    def __setitem__(self, key: str, value: Any) -> None:
        pass


def _source_sha256(module: Any) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"cannot locate source for {module.__name__}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def verify_exact_rc24_sources(weight_loader_module: Any, modeling_t5_module: Any) -> None:
    """Fail closed unless the installed distribution and patched files match."""
    installed = importlib.metadata.version("tensorrt-llm")
    if installed != TRTLLM_VERSION:
        raise RuntimeError(f"TensorRT-LLM compatibility requires {TRTLLM_VERSION}, found {installed}")
    expected = (
        (weight_loader_module, HF_WEIGHT_LOADER_SHA256),
        (modeling_t5_module, MODELING_T5_SHA256),
    )
    for module, expected_sha in expected:
        actual_sha = _source_sha256(module)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"refusing TensorRT-LLM compatibility for changed source {module.__name__}: "
                f"expected sha256 {expected_sha}, found {actual_sha}"
            )


def verify_exact_transformers_5_sources(
    t5_tokenization_module: Any,
    tokenizers_backend_module: Any,
) -> None:
    """Fail closed unless the installed Transformers tokenizer sources match."""
    installed = importlib.metadata.version("transformers")
    if installed != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"T5 tokenizer compatibility requires Transformers {TRANSFORMERS_VERSION}, found {installed}"
        )
    expected = (
        (t5_tokenization_module, T5_TOKENIZATION_SHA256),
        (tokenizers_backend_module, TOKENIZERS_BACKEND_SHA256),
    )
    for module, expected_sha in expected:
        actual_sha = _source_sha256(module)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"refusing T5 tokenizer compatibility for changed source {module.__name__}: "
                f"expected sha256 {expected_sha}, found {actual_sha}"
            )


def _restore_t5_shared_embedding(weights: _MutableWeights) -> None:
    """Supply rc24's required T5 key from exact input-embedding semantics."""
    if _T5_SHARED_EMBEDDING_KEY in weights:
        return
    # The guarded rc24 T5 source always shares encoder/decoder input
    # embeddings. ``tie_word_embeddings`` controls only the LM head, which is
    # deliberately not a restoration source here.
    sources = [key for key in _T5_INPUT_EMBEDDING_KEYS if key in weights]
    if not sources:
        return
    source_tensor = weights[sources[0]]
    if any(weights[key] is not source_tensor for key in sources[1:]):
        rendered = ", ".join(repr(key) for key in sources)
        raise RuntimeError(f"ambiguous T5 shared embedding tensors: {rendered}")
    weights[_T5_SHARED_EMBEDDING_KEY] = source_tensor


def _install_t5_weight_patch(modeling_t5_module: Any) -> None:
    model_class = modeling_t5_module.T5ForConditionalGeneration
    original = model_class.load_weights
    if getattr(original, _PATCH_MARKER, False):
        return

    def _load_weights_with_shared_embedding(self: Any, weights: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(weights, _MutableWeights):
            raise RuntimeError("TensorRT-LLM T5 loader received a non-mutable weight mapping")
        _restore_t5_shared_embedding(weights)
        return original(self, weights, *args, **kwargs)

    setattr(_load_weights_with_shared_embedding, _PATCH_MARKER, True)
    model_class.load_weights = _load_weights_with_shared_embedding


def _install_t5_scaling_patch(modeling_t5_module: Any) -> None:
    model_class = modeling_t5_module.T5ForConditionalGeneration
    original = model_class.__init__
    if getattr(original, _PATCH_MARKER, False):
        return

    def _init_with_saved_scaling_semantics(self: Any, model_config: Any, *args: Any, **kwargs: Any) -> None:
        original(self, model_config, *args, **kwargs)
        pretrained_config = getattr(model_config, "pretrained_config", None)
        if pretrained_config is None:
            raise RuntimeError("TensorRT-LLM T5 ModelConfig is missing pretrained_config")
        legacy_default = getattr(pretrained_config, "tie_word_embeddings", True)
        self.rescale_before_lm_head = bool(getattr(pretrained_config, "scale_decoder_outputs", legacy_default))

    setattr(_init_with_saved_scaling_semantics, _PATCH_MARKER, True)
    model_class.__init__ = _init_with_saved_scaling_semantics


def _install_t5_tokenizer_serialization_patch(t5_tokenization_module: Any, tokenizers_backend_module: Any) -> None:
    tokenizer_class = t5_tokenization_module.T5Tokenizer
    original = tokenizer_class.convert_to_native_format
    original_function = original.__func__
    if getattr(original_function, _PATCH_MARKER, False):
        return

    def _convert_to_native_format(
        cls: Any,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tokenizer_file = kwargs.get("tokenizer_file")
        if not trust_remote_code and isinstance(tokenizer_file, str) and Path(tokenizer_file).is_file():
            converted = dict(kwargs)
            converted.pop("tokenizer_file")
            converted["tokenizer_object"] = tokenizers_backend_module.TokenizerFast.from_file(tokenizer_file)
            return converted
        return original_function(cls, trust_remote_code=trust_remote_code, **kwargs)

    setattr(_convert_to_native_format, _PATCH_MARKER, True)
    tokenizer_class.convert_to_native_format = classmethod(_convert_to_native_format)


def install_transformers_5_t5_tokenizer_compatibility() -> None:
    """Make exact Transformers 5.5.4 T5 loads honor ``tokenizer.json``."""
    t5_tokenization = importlib.import_module("transformers.models.t5.tokenization_t5")
    tokenizers_backend = importlib.import_module("transformers.tokenization_utils_tokenizers")

    verify_exact_transformers_5_sources(t5_tokenization, tokenizers_backend)
    _install_t5_tokenizer_serialization_patch(t5_tokenization, tokenizers_backend)


def install_rc24_compatibility() -> None:
    """Verify and install the narrowly scoped direct-HF compatibility patches."""
    modeling_t5 = importlib.import_module("tensorrt_llm._torch.models.modeling_t5")
    weight_loader = importlib.import_module("tensorrt_llm._torch.models.checkpoints.hf.weight_loader")

    verify_exact_rc24_sources(weight_loader, modeling_t5)
    install_transformers_5_t5_tokenizer_compatibility()
    _install_t5_weight_patch(modeling_t5)
    _install_t5_scaling_patch(modeling_t5)
