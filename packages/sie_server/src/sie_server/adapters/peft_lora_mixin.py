"""PEFT-based LoRA mixin for PyTorch model adapters.

This mixin provides LoRA (Low-Rank Adaptation) support using the PEFT library.
Adapters that use PyTorch models can inherit from this mixin to gain LoRA capabilities.

Usage:
    class MyFlashAdapter(PEFTLoRAMixin, ModelAdapter):
        def load(self, device: str) -> None:
            ...
            self._model = AutoModel.from_pretrained(...)
            # Store base model reference for LoRA wrapping
            self._base_model = self._model
"""

from __future__ import annotations

import gc
import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from safetensors import safe_open

from sie_server.adapters._lora_source import resolve_lora_source

if TYPE_CHECKING:
    from collections.abc import Collection

    from peft import PeftConfig, PeftModel

logger = logging.getLogger(__name__)

# String ``init_lora_weights`` values with which PEFT initializes only the
# adapter's own weights (peft 0.19 ``LoraLayer.update_layer``). PiSSA, CorDA,
# OLoRA, LoftQ and LoRA-GA instead rewrite the base layer's weights.
_BASE_PRESERVING_INITIALIZERS = ("gaussian", "eva", "orthogonal")

_ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

# The tensor names PEFT saves for a LoRA's own parameters. PEFT loads any
# other name in an adapter's weights file into the model it wraps, so such a
# name would overwrite the base model.
_ADAPTER_WEIGHT_SUFFIXES = frozenset({("lora_A", "weight"), ("lora_B", "weight"), ("lora_B", "bias")})
_ADAPTER_WEIGHT_PARAMETERS = frozenset({"lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"})


def is_lora_weight_key(key: str) -> bool:
    """True when ``key`` names a LoRA's own parameter in a PEFT adapter weights file."""
    parts = key.split(".")
    if len(parts) < 2 or key.count("lora_") != 1:
        return False
    return tuple(parts[-2:]) in _ADAPTER_WEIGHT_SUFFIXES or parts[-1] in _ADAPTER_WEIGHT_PARAMETERS


def _unwrap_lora_layers(model: Any) -> None:
    """Put back each base module that a PEFT tuner layer wraps inside ``model``, and drop PEFT's config on it."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in list(model.named_modules()):
        if not isinstance(module, BaseTunerLayer):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if getattr(parent, child_name, None) is module:
            setattr(parent, child_name, module.get_base_layer())
    if hasattr(model, "peft_config"):
        del model.peft_config


def validate_lora_target_modules(
    lora_path: str,
    target_modules: Any,
    called_module_names: Collection[str],
) -> None:
    """Reject a LoRA whose ``target_modules`` misses every module the serving
    forward actually calls.

    Manual (module-call) flash forwards invoke a fixed set of submodules
    directly instead of the model's top-level ``forward``. PEFT wraps exactly
    the modules named in the LoRA's ``target_modules`` — so a LoRA targeting
    only modules the manual forward never calls (e.g. ``pooler.dense``) is
    applied to nothing and serves base weights with **zero error** (the
    negative-control failure mode in the LoRA capability audit §3). This check
    makes that mismatch fail loudly at load/staging time instead.

    Args:
        lora_path: Public LoRA id (HF/local path), for error messages.
        target_modules: The PEFT config's ``target_modules`` — typically a
            list/set of module-name suffixes (``"query"``, ``"q_proj"``,
            possibly dotted like ``"encoder.layer.0.attention.self.query"``)
            or a regex string matched against full module paths.
        called_module_names: Leaf module names the adapter's serving forward
            actually invokes (e.g. ``{"query", "key", "value", "dense"}``).

    Raises:
        ValueError: If ``target_modules`` is a collection and none of its
            entries resolve (by leaf name) to a called module. A regex-string
            ``target_modules`` that matches no called leaf name only warns —
            regexes may legitimately target full dotted paths this check
            cannot resolve without the model instance.
    """
    if not target_modules or not called_module_names:
        # Nothing to check (PEFT will resolve/refuse its own defaults).
        return

    called = set(called_module_names)

    if isinstance(target_modules, str):
        # PEFT regex form (fullmatched against full module paths). Leaf names
        # are all we have here, so treat no-match as a warning, not an error.
        if not any(re.fullmatch(target_modules, name) for name in called):
            logger.warning(
                "LoRA '%s' declares regex target_modules %r which matches none "
                "of the modules the serving forward calls (%s) — if it also "
                "matches no full module path, the LoRA will have no effect",
                lora_path,
                target_modules,
                sorted(called),
            )
        return

    targets = [str(t) for t in target_modules]
    # Compare by leaf component: PEFT matches suffixes, and dotted targets
    # ("encoder.layer.0.attention.self.query") end in the module's leaf name.
    if any(t.rsplit(".", 1)[-1] in called for t in targets):
        return

    msg = (
        f"LoRA '{lora_path}' targets modules {sorted(targets)!r}, but the "
        f"serving forward only calls {sorted(called)!r} — the LoRA would be "
        f"silently ignored (zero effect). Retrain/export the LoRA against the "
        f"called projections, or serve it via the merge-at-staging path."
    )
    raise ValueError(msg)


class PEFTLoRAMixin:
    """Mixin providing PEFT-based LoRA support for PyTorch adapters.

    This mixin provides a common implementation of LoRA loading/switching
    using the PEFT library. It works with any adapter that stores its
    PyTorch model in `self._model`.

    The mixin expects:
    - `self._model`: The PyTorch model (will be wrapped by PeftModel)
    - `self._device`: The device string (for memory estimation)

    How it works:
    1. First LoRA load: Creates PeftModel.from_pretrained(base_model, lora_path)
    2. Additional LoRAs: Calls peft_model.load_adapter(lora_path, adapter_name)
    3. Switching: Calls peft_model.set_adapter(peft_adapter_name) before inference
    4. Unloading: Calls peft_model.delete_adapter(peft_adapter_name)

    Memory management:
    - Each LoRA adds ~1-5% of base model memory
    - Memory is estimated from adapter weight sizes
    - LRU eviction is handled by ModelLoader (not this mixin)

    Thread safety:
    - load_lora() is safe to call from thread pool
    - set_active_lora() is lightweight and thread-safe
    - The adapter switching is lock-free
    """

    # Track PEFT state
    _peft_model: PeftModel | None = None
    _active_lora: str | None = None
    _loaded_loras: set[str]  # Track which LoRAs are loaded
    _lora_adapter_names: dict[str, str]  # Public LoRA id -> PEFT-safe adapter name

    #: Leaf names of the submodules the adapter's serving forward actually
    #: invokes (e.g. ``frozenset({"query", "key", "value", "dense"})``).
    #: ``None`` (default) skips the staging-side ``target_modules`` check —
    #: appropriate for adapters that run the model's standard ``forward``,
    #: where every PEFT-wrapped module is reached. Manual/module-call flash
    #: adapters should declare their called set so a LoRA whose
    #: ``target_modules`` intersect none of them is rejected at load instead
    #: of silently serving base weights (LoRA capability audit §3).
    lora_called_module_names: ClassVar[frozenset[str] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize _loaded_loras set for subclasses."""
        super().__init_subclass__(**kwargs)
        # This ensures each subclass gets its own set

    def _ensure_lora_tracking(self) -> None:
        """Ensure LoRA tracking is initialized."""
        if not hasattr(self, "_loaded_loras"):
            self._loaded_loras = set()
        if not hasattr(self, "_lora_adapter_names"):
            self._lora_adapter_names = {}

    @staticmethod
    def _peft_adapter_name(lora_name: str) -> str:
        """Return a deterministic, readable PEFT-safe adapter name."""
        readable = re.sub(r"[^0-9A-Za-z_]+", "_", lora_name).strip("_")
        readable = readable[:48].rstrip("_") or "adapter"
        digest = hashlib.sha256(lora_name.encode("utf-8")).hexdigest()
        return f"sie_lora_{readable}__{digest}"

    def _get_peft_adapter_name(self, lora_name: str) -> str:
        """Return the PEFT adapter name for a loaded public LoRA id."""
        return self._lora_adapter_names.get(lora_name, self._peft_adapter_name(lora_name))

    def supports_lora(self) -> bool:
        """Return True - PEFT adapters support LoRA."""
        return True

    def supports_hot_lora_reload(self) -> bool:
        """Return True - PEFT loading is non-blocking.

        Unlike SGLang which blocks the server during LoRA loading,
        PEFT can load adapters in a thread pool without blocking inference.
        """
        return True

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        """Resolve a LoRA to one local directory, and refuse it there unless it is safe to attach.

        A Hub id uses the configured local/cluster cache and allowed Hub
        fallback at one commit; a local directory is used as it is. The
        required files are ``adapter_config.json`` and
        ``adapter_model.safetensors``. :meth:`load_lora` reads both files
        from the returned directory. Network and disk I/O happen here, so a
        caller can run this outside inference.

        Args:
            lora_path: Hugging Face repo id or local directory.
            revision: Pinned 40-hex commit SHA (#2113); ``None`` resolves the
                default branch.

        Returns:
            The local directory that holds the adapter.

        Raises:
            ValueError: :meth:`_check_lora_config` refuses the adapter config,
                the adapter ships no ``adapter_model.safetensors``, or its
                weights file has a tensor that is not a LoRA parameter.
        """
        directory = resolve_lora_source(lora_path, revision)
        try:
            from peft import PeftConfig
        except ImportError as e:
            msg = "PEFT is required for LoRA support. Install with: pip install peft"
            raise RuntimeError(msg) from e
        self._check_lora_config(lora_path, PeftConfig.from_pretrained(str(directory), local_files_only=True))
        weights = directory / _ADAPTER_WEIGHTS_FILE
        if not weights.is_file():
            msg = f"LoRA '{lora_path}' ships no {_ADAPTER_WEIGHTS_FILE}; adapter weights in other formats are refused."
            raise ValueError(msg)
        with safe_open(str(weights), framework="pt") as handle:
            names = handle.keys()
        foreign = sorted(name for name in names if not is_lora_weight_key(name))
        if foreign:
            msg = (
                f"LoRA '{lora_path}' has {len(foreign)} tensor(s) that are not LoRA parameters, such as "
                f"{foreign[0]!r}; loading them would overwrite the base model every request shares."
            )
            raise ValueError(msg)
        return str(directory)

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        """Load a LoRA adapter using PEFT.

        On first call, wraps the base model with PeftModel.
        On subsequent calls, adds the adapter to the existing PeftModel.

        Args:
            lora_path: HuggingFace path (e.g., "org/lora-name") or local path.
                This path is the public LoRA id used for switching.
            revision: Pinned 40-hex commit SHA (#2113) for the fetch when
                ``source`` is ``None``; ``None`` resolves the default branch.
                The public LoRA id stays the bare ``lora_path``.
            source: The local directory :meth:`fetch_lora` returned for this
                LoRA. ``None`` fetches it first.

        Returns:
            Memory usage of the loaded LoRA in bytes.

        Raises:
            RuntimeError: If model not loaded or PEFT import fails.
            ValueError: If :meth:`fetch_lora` or :meth:`_check_lora_config`
                refuses the LoRA.
        """
        self._ensure_lora_tracking()

        # Check if already loaded
        if lora_path in self._loaded_loras:
            logger.warning("LoRA '%s' is already loaded, skipping", lora_path)
            return 0

        # Get base model - must exist
        base_model = getattr(self, "_model", None)
        if base_model is None:
            msg = "Model not loaded. Call load() before load_lora()."
            raise RuntimeError(msg)

        try:
            from peft import PeftConfig, PeftModel
        except ImportError as e:
            msg = "PEFT is required for LoRA support. Install with: pip install peft"
            raise RuntimeError(msg) from e

        logger.info("Loading LoRA adapter: %s", lora_path)

        if source is None:
            source = self.fetch_lora(lora_path, revision)
        self._check_lora_config(lora_path, PeftConfig.from_pretrained(source))

        peft_adapter_name = self._peft_adapter_name(lora_path)

        try:
            if self._peft_model is None:
                # First LoRA - wrap the base model
                logger.debug("Creating PeftModel from base model")
                self._peft_model = PeftModel.from_pretrained(
                    base_model,
                    source,
                    adapter_name=peft_adapter_name,
                )
                # Update self._model to point to the PEFT-wrapped model
                # This ensures encode() uses the LoRA-enhanced model
                self._model = self._peft_model
            else:
                # Additional LoRA - add to existing PeftModel
                logger.debug("Adding adapter to existing PeftModel")
                self._peft_model.load_adapter(source, adapter_name=peft_adapter_name)
        except Exception:
            self._detach_failed_lora(base_model, peft_adapter_name)
            raise

        self._loaded_loras.add(lora_path)
        self._lora_adapter_names[lora_path] = peft_adapter_name

        # Force-disable adapter layers after loading.
        # PEFT's load_adapter() can re-enable adapter layers, so we must
        # explicitly disable them. Reset _active_lora to force set_active_lora(None)
        # to actually run (not short-circuit via the equality check).
        self._active_lora = "__force_reset__"
        self.set_active_lora(None)

        # Estimate memory usage from adapter parameters
        memory_bytes = self._estimate_lora_memory(lora_path)
        logger.info("LoRA '%s' loaded, estimated memory: %.2f MB", lora_path, memory_bytes / 1024 / 1024)

        return memory_bytes

    def _detach_failed_lora(self, base_model: Any, peft_adapter_name: str) -> None:
        """Remove what PEFT attached for a LoRA whose load raised, so no request runs through it.

        PEFT injects a LoRA's layers into the model before it loads the
        adapter's weights, so a load that fails on the weights leaves them in
        place. A first LoRA's layers wrap the shared base model's own modules,
        which are restored here; a later LoRA's adapter is deleted from the
        existing PeftModel.
        """
        if self._peft_model is None:
            _unwrap_lora_layers(base_model)
            self._model = base_model
            return
        if peft_adapter_name in self._peft_model.peft_config:
            self._peft_model.delete_adapter(peft_adapter_name)

    def _check_lora_config(self, lora_path: str, peft_config: PeftConfig) -> None:
        """Refuse a LoRA from its ``adapter_config.json`` before any weights load.

        Only a LoRA is accepted, and only one whose loading leaves the base
        model, which every request shares, as it was. PEFT rewrites the base
        weights while it loads a LoRA whose ``init_lora_weights`` is not one of
        :data:`_BASE_PRESERVING_INITIALIZERS` or a boolean, restructures the
        base model for ``layer_replication``, replaces whole base modules for
        ``modules_to_save`` and ``trainable_token_indices``, and imports the module named by
        ``megatron_core`` for ``megatron_config``. A LoRA whose ``bias`` is not
        ``"none"`` trains biases that live in the base model's own layers; PEFT
        cannot turn them off with the adapter. A LoRA whose ``target_modules``
        miss every module in :attr:`lora_called_module_names` would have no
        effect (see :func:`validate_lora_target_modules`).
        """
        peft_type = getattr(peft_config, "peft_type", None)
        if getattr(peft_type, "value", peft_type) != "LORA":
            msg = f"LoRA '{lora_path}' is not a LoRA adapter (peft_type={peft_type!r})."
            raise ValueError(msg)

        initializer = getattr(peft_config, "init_lora_weights", True)
        if not isinstance(initializer, bool) and initializer not in _BASE_PRESERVING_INITIALIZERS:
            msg = (
                f"LoRA '{lora_path}' sets init_lora_weights={initializer!r}: loading it would rewrite the "
                "base model's shared weights. Export it as a plain LoRA with init_lora_weights=true."
            )
            raise ValueError(msg)

        for field in ("layer_replication", "megatron_config", "modules_to_save", "trainable_token_indices"):
            if getattr(peft_config, field, None):
                msg = f"LoRA '{lora_path}' sets {field}, which changes the base model every request shares."
                raise ValueError(msg)

        bias = getattr(peft_config, "bias", "none")
        if bias != "none":
            msg = (
                f"LoRA '{lora_path}' sets bias={bias!r}: its trained biases live in the base model's "
                "shared layers, so loading it would change outputs for requests that do not name it. "
                "Retrain or export the LoRA with bias='none'."
            )
            raise ValueError(msg)

        called = self.lora_called_module_names
        if called:
            validate_lora_target_modules(lora_path, getattr(peft_config, "target_modules", None), called)

    def unload_lora(self, lora_name: str) -> None:
        """Unload a LoRA adapter.

        Called during LRU eviction when max_loras is exceeded.

        Args:
            lora_name: The LoRA adapter name to unload.

        Raises:
            ValueError: If LoRA is not loaded.
        """
        self._ensure_lora_tracking()

        if lora_name not in self._loaded_loras:
            msg = f"LoRA '{lora_name}' is not loaded"
            raise ValueError(msg)

        if self._peft_model is None:
            msg = "PeftModel is None but LoRA is in _loaded_loras - inconsistent state"
            raise RuntimeError(msg)

        logger.info("Unloading LoRA adapter: %s", lora_name)

        # If this is the active LoRA, switch to base first
        if self._active_lora == lora_name:
            self.set_active_lora(None)

        # PEFT's get_base_model consults active_peft_config, which disappears
        # when delete_adapter removes the final adapter. Capture it first.
        base_model = self._peft_model.get_base_model()
        # Delete the adapter
        self._peft_model.delete_adapter(self._get_peft_adapter_name(lora_name))
        self._loaded_loras.discard(lora_name)
        self._lora_adapter_names.pop(lora_name, None)

        # If no LoRAs remain, unwrap the model
        if not self._loaded_loras:
            logger.debug("No LoRAs remaining, unwrapping PeftModel")
            _unwrap_lora_layers(base_model)
            self._model = base_model
            del self._peft_model
            self._peft_model = None
            gc.collect()

    def set_active_lora(self, lora_name: str | None) -> None:
        """Set the active LoRA for the next inference call.

        Called by the worker before each batch to switch to the appropriate
        LoRA adapter.

        Args:
            lora_name: LoRA adapter name, or None for base model.
        """
        self._ensure_lora_tracking()

        # Skip if no change
        if lora_name == self._active_lora:
            return

        if self._peft_model is None:
            if lora_name is not None:
                msg = f"LoRA '{lora_name}' is not loaded"
                raise ValueError(msg)
            return

        if lora_name is None:
            # Switch to base model (disable adapter layers)
            # Use PEFT's disable_adapter_layers() instead of transformers' disable_adapters()
            logger.debug("Disabling LoRA adapters (base model)")
            self._peft_model.disable_adapter_layers()
        else:
            if lora_name not in self._loaded_loras:
                msg = f"LoRA '{lora_name}' is not loaded"
                raise ValueError(msg)

            logger.debug("Setting active LoRA: %s", lora_name)
            # Re-enable adapter layers if they were disabled, then set the active adapter
            # Use PEFT's enable_adapter_layers() instead of transformers' enable_adapters()
            self._peft_model.enable_adapter_layers()
            self._peft_model.set_adapter(self._get_peft_adapter_name(lora_name))

        self._active_lora = lora_name

    def _estimate_lora_memory(self, lora_name: str) -> int:
        """Estimate memory usage of a LoRA adapter.

        Sums the memory of all LoRA parameters (lora_A, lora_B matrices).

        Args:
            lora_name: The LoRA adapter name.

        Returns:
            Estimated memory in bytes.
        """
        if self._peft_model is None:
            return 0

        total_bytes = 0
        peft_adapter_name = self._get_peft_adapter_name(lora_name)

        try:
            import torch

            # Iterate through named parameters looking for LoRA weights
            for name, param in self._peft_model.named_parameters():
                # LoRA parameters typically have adapter name in their path
                # and contain "lora_" in the name
                if peft_adapter_name in name and "lora_" in name and isinstance(param, torch.Tensor):
                    total_bytes += param.numel() * param.element_size()

        except (AttributeError, RuntimeError) as e:
            logger.warning("Failed to estimate LoRA memory: %s", e)
            # Fallback: assume ~2% of base model memory
            base_memory = getattr(self, "memory_footprint", lambda: 0)()
            total_bytes = int(base_memory * 0.02)

        return total_bytes
