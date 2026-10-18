"""Opt-in LoRA lifecycle for the ordinary sentence-transformers CrossEncoder.

The outer wrapper, tokenizer, classifier and activation are never replaced.
Transformers injects ordinary LoRA layers into declared BERT/XLM-R encoder Linear
modules. The worker owns selection/mutation exclusion; only background source
preparation needs a separate lock here. Resident layers stay outside the model's
module tree while base is selected. No scoring code calls into this module.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftConfig
from peft.functional import delete_adapter
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file
from sentence_transformers import CrossEncoder
from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification

from sie_server.adapters.lora import LoraRuntimeConfig, LoraStateUncertainError, LoraUnavailableError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin

_FILES = ("adapter_config.json", "adapter_model.safetensors")
_ENCODER_PREFIXES: dict[type[torch.nn.Module], str] = {
    BertForSequenceClassification: "bert.encoder.",
    XLMRobertaForSequenceClassification: "roberta.encoder.",
}
_UNSUPPORTED = (
    "use_dora",
    "use_rslora",
    "use_qalora",
    "use_bdlora",
    "lora_bias",
    "fan_in_fan_out",
    "rank_pattern",
    "alpha_pattern",
    "layers_to_transform",
    "layers_pattern",
    "exclude_modules",
    "target_parameters",
    "alora_invocation_tokens",
    "arrow_config",
    "ensure_weight_tying",
    "loftq_config",
    "eva_config",
    "corda_config",
    "lora_ga_config",
)


@dataclass
class _Prepared:
    directory: tempfile.TemporaryDirectory[str]
    config: LoraConfig
    weights: dict[str, torch.Tensor]


class CrossEncoderLora:
    """Declared, bounded inference adapters; lifecycle calls run under the worker guard."""

    def __init__(
        self,
        paths: dict[str, str],
        revisions: dict[str, str | None] | None,
        max_loaded: int,
    ) -> None:
        if isinstance(max_loaded, bool) or not isinstance(max_loaded, int) or max_loaded < 1:
            raise ValueError("max_loaded_loras must be a positive integer")
        if not isinstance(paths, dict):
            raise ValueError("CrossEncoder lora_paths must map served aliases to declared sources")
        aliases = dict(paths)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in aliases.items()):
            raise ValueError("LoRA declarations must map nonempty names to nonempty sources")
        if any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", alias) for alias in aliases):
            raise ValueError("LoRA served aliases must contain only letters, numbers, underscores, dots and hyphens")
        self._sources = set(aliases.values())
        self._revisions = {source: (revisions or {}).get(source) for source in self._sources}
        for source in self._sources:
            if source in aliases and aliases[source] != source:
                raise ValueError(f"LoRA alias {source!r} shadows a different declared source")
        self._aliases = aliases
        self._max_loaded = max_loaded
        self._source_lock = threading.Lock()
        self._prepared: dict[str, _Prepared] = {}
        self._loaded: dict[str, int] = {}
        self._parameters: dict[str, list[torch.nn.Parameter]] = {}
        self._active: str | None = None
        self._outer: CrossEncoder | None = None
        self._model: Any = None
        self._encoder_prefix: str | None = None
        self._originals: dict[str, torch.nn.Linear] = {}
        # An ordinary dict deliberately keeps resident adapters outside nn.Module's
        # traversal while base is selected. CrossEncoder.predict calls to()/eval()
        # on every batch, even when the model is already on the requested device.
        self._resident_layers: dict[str, LoraLinear] = {}
        self._grad_flags: list[tuple[torch.nn.Parameter, bool]] = []
        self._closed = True
        self._uncertain = False
        self._generation = 0

    def runtime_config(self) -> LoraRuntimeConfig:
        return LoraRuntimeConfig(dict(self._aliases), frozenset(), frozenset(), self._max_loaded)

    def bind(self, outer: CrossEncoder) -> None:
        model = outer.model
        if model is None or type(model) not in _ENCODER_PREFIXES or getattr(model, "is_quantized", False):
            raise ValueError(
                "CrossEncoder LoRA requires a standard, unquantized BertForSequenceClassification "
                "or XLMRobertaForSequenceClassification"
            )
        encoder_prefix = _ENCODER_PREFIXES[type(model)]
        if getattr(model, "peft_config", None) or getattr(model, "_hf_peft_config_loaded", False):
            raise ValueError("The cross-encoder must start from a base checkpoint without attached adapters")
        self._outer, self._model = outer, model
        self._encoder_prefix = encoder_prefix
        self._originals = {name: module for name, module in model.named_modules() if type(module) is torch.nn.Linear}
        self._grad_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        self._active = None
        self._uncertain = False
        with self._source_lock:
            self._generation += 1
            self._closed = False

    def _source(self, name: str) -> str:
        source = self._aliases.get(name, name)
        if source not in self._sources:
            raise ValueError(f"LoRA {name!r} is not declared for this cross-encoder")
        return source

    def _ready(self) -> None:
        if self._uncertain:
            raise LoraStateUncertainError("CrossEncoder LoRA state is uncertain; reload the model")
        if self._closed or self._outer is None:
            raise RuntimeError("CrossEncoder is not loaded")

    def _check_config(self, source: str, config: PeftConfig) -> LoraConfig:
        PEFTLoRAMixin()._check_lora_config(source, config)
        if not isinstance(config, LoraConfig):
            raise ValueError("Only ordinary LoRA configurations are supported")
        for field in _UNSUPPORTED:
            if getattr(config, field, None):
                raise ValueError(f"CrossEncoder LoRA does not support {field}")
        if not isinstance(config.init_lora_weights, bool):
            raise ValueError("CrossEncoder LoRA requires a plain boolean init_lora_weights")
        if isinstance(config.r, bool) or not isinstance(config.r, int) or config.r < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(config.lora_alpha) or config.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(config.lora_dropout) or not 0 <= config.lora_dropout < 1:
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        targets = config.target_modules
        if not isinstance(targets, (set, list)) or not targets or any(not isinstance(t, str) or not t for t in targets):
            raise ValueError("CrossEncoder LoRA requires an explicit list of target module names")
        config.inference_mode = True
        return config

    def _validate_weights(self, config: LoraConfig, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected: dict[str, tuple[int, int]] = {}
        for target in config.target_modules or []:
            matches = [name for name in self._originals if name == target or name.endswith(f".{target}")]
            if (
                self._encoder_prefix is None
                or not matches
                or any(not name.startswith(self._encoder_prefix) for name in matches)
            ):
                raise ValueError(
                    f"LoRA target {target!r} must resolve only to the supported model encoder Linear modules"
                )
            for name in matches:
                module = self._originals[name]
                expected[f"{name}.lora_A.weight"] = (config.r, module.in_features)
                expected[f"{name}.lora_B.weight"] = (module.out_features, config.r)
        normalized: dict[str, torch.Tensor] = {}
        for key, tensor in weights.items():
            name = key.removeprefix("base_model.model.")
            if name in normalized:
                raise ValueError(f"Duplicate LoRA weight {name!r}")
            normalized[name] = tensor
        if set(normalized) != set(expected):
            raise ValueError("LoRA weights must exactly match every declared target's A/B matrices")
        for name, tensor in normalized.items():
            if (
                tuple(tensor.shape) != expected[name]
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid shape, dtype or nonfinite values in LoRA weight {name!r}")
            module_name = name.rsplit(".", 2)[0]
            dtype = self._originals[module_name].weight.dtype
            if not torch.isfinite(tensor.to(dtype=dtype)).all():
                raise ValueError(f"LoRA weight {name!r} overflows the base model's compute dtype")
        return normalized

    def fetch(self, name: str, revision: str | None = None) -> str:
        self._ready()
        source = self._source(name)
        pinned = self._revisions[source]
        if revision is not None and revision != pinned:
            raise ValueError("LoRA revision differs from the declared source revision")
        with self._source_lock:
            generation = self._generation
            if source in self._prepared:
                return self._prepared[source].directory.name
        directory = Path(source)
        if directory.is_dir():
            if pinned is not None:
                raise ValueError("A local LoRA directory cannot honor a Hugging Face revision pin")
        else:
            directory = Path(snapshot_download(source, revision=pinned, allow_patterns=list(_FILES)))
        temporary = tempfile.TemporaryDirectory(prefix="sie-cross-encoder-lora-")
        try:
            # A private snapshot prevents later source edits or alternative pickle files
            # from changing the checkpoint after validation.
            for filename in _FILES:
                shutil.copyfile(directory / filename, Path(temporary.name) / filename)
            config = self._check_config(source, PeftConfig.from_pretrained(temporary.name, local_files_only=True))
            weights = self._validate_weights(config, load_file(str(Path(temporary.name) / _FILES[1])))
            with self._source_lock:
                self._ready()
                if generation != self._generation:
                    raise RuntimeError("CrossEncoder reloaded while LoRA source was being fetched")
                existing = self._prepared.get(source)
                if existing is not None:
                    temporary.cleanup()
                    return existing.directory.name
                self._prepared[source] = _Prepared(temporary, config, weights)
                return temporary.name
        except BaseException:
            temporary.cleanup()
            raise

    def discard(self, name: str) -> None:
        source = self._source(name)
        with self._source_lock:
            prepared = self._prepared.pop(source, None)
        if prepared is not None:
            try:
                prepared.directory.cleanup()
            except OSError:
                # Keep ownership for model teardown to retry, rather than losing
                # the path when filesystem cleanup fails.
                with self._source_lock:
                    self._prepared[source] = prepared
                raise

    def _restore_grad_flags(self) -> None:
        for parameter, requires_grad in self._grad_flags:
            parameter.requires_grad_(requires_grad)

    def _attach_layers(self) -> None:
        """Expose all resident layers only during a guarded adapter operation."""
        for name, layer in self._resident_layers.items():
            original = self._originals[name]
            current = self._model.get_submodule(name)
            if current is not original and current is not layer:
                raise RuntimeError(f"CrossEncoder module {name!r} changed outside the LoRA lifecycle")
            if layer.get_base_layer() is not original:
                raise RuntimeError(f"CrossEncoder LoRA base layer {name!r} changed")
            # Serving keeps the model's device and precision fixed after loading.
            # Do not silently migrate detached weights or invalidate their memory
            # reservation if an external caller moves or casts the base model.
            for parameter in layer.parameters():
                if parameter.device != original.weight.device or parameter.dtype != original.weight.dtype:
                    raise RuntimeError("CrossEncoder device or precision changed with resident LoRAs")
            parent_name, _, child = name.rpartition(".")
            setattr(self._model.get_submodule(parent_name), child, layer)

    def _detach_layers(self) -> None:
        """Restore the exact base forward/tree while retaining adapter residency."""
        for name, original in self._originals.items():
            current = self._model.get_submodule(name)
            if current is original:
                continue
            if not isinstance(current, LoraLinear) or current.get_base_layer() is not original:
                raise RuntimeError(f"CrossEncoder module {name!r} is not its owned LoRA layer")
            if current.lora_A:
                self._resident_layers[name] = current
            else:
                # Upstream deletion leaves empty tuner layers in the tree. A
                # disjoint-target eviction must release those wrappers too.
                self._resident_layers.pop(name, None)
            parent_name, _, child = name.rpartition(".")
            setattr(self._model.get_submodule(parent_name), child, original)
        self._active = None

    def _restore_tree(self) -> None:
        for name, original in self._originals.items():
            parent_name, _, child = name.rpartition(".")
            setattr(self._model.get_submodule(parent_name), child, original)
        if hasattr(self._model, "peft_config"):
            del self._model.peft_config
        self._model._hf_peft_config_loaded = False
        self._resident_layers.clear()
        self._restore_grad_flags()
        self._active = None

    def _remove(self, adapter_name: str) -> None:
        # The low-level deletion also handles an injection that failed before
        # Transformers set its configuration-loaded flag.
        delete_adapter(self._model, adapter_name=adapter_name, prefix="lora_")
        getattr(self._model, "peft_config", {}).pop(adapter_name, None)

    def load(self, name: str, revision: str | None = None, *, source: str | None = None) -> int:
        self._ready()
        canonical = self._source(name)
        if canonical in self._loaded:
            return self._loaded[canonical]
        if len(self._loaded) >= self._max_loaded:
            raise ValueError("CrossEncoder resident LoRA capacity is exhausted; evict an adapter first")
        if source is None:
            source = self.fetch(canonical, revision)
        with self._source_lock:
            prepared = self._prepared.get(canonical)
            if prepared is None or prepared.directory.name != source:
                raise ValueError("LoRA source must be the validated snapshot fetched for this declaration")
        adapter_name = PEFTLoRAMixin._peft_adapter_name(canonical)
        try:
            assert self._outer is not None
            self._attach_layers()
            self._outer.load_adapter(
                adapter_name=adapter_name,
                peft_config=prepared.config,
                adapter_state_dict=prepared.weights,
                is_trainable=False,
            )
            memory = 0
            parameters = []
            for key, weight in prepared.weights.items():
                module_name, part, _ = key.rsplit(".", 2)
                parameter = getattr(self._model.get_submodule(module_name), part)[adapter_name].weight
                # Transformers only warns on missing/unexpected keys. Verify the
                # actual installed tensors before admitting the adapter.
                if not torch.isfinite(parameter).all() or not torch.equal(
                    parameter.detach(), weight.to(device=parameter.device, dtype=parameter.dtype)
                ):
                    raise ValueError(f"LoRA weight {key!r} was not installed exactly")
                parameter.requires_grad_(False)
                parameters.append(parameter)
                memory += parameter.numel() * parameter.element_size()
            self._outer.disable_adapters()
            self._restore_grad_flags()
            self._detach_layers()
            self._loaded[canonical] = memory
            self._parameters[canonical] = parameters
        except BaseException:
            try:
                if self._loaded:
                    # A failed detach may have already saved part of the new
                    # injection off-tree. Include it in rollback deletion too.
                    self._attach_layers()
                    self._remove(adapter_name)
                    assert self._outer is not None
                    self._outer.disable_adapters()
                    self._restore_grad_flags()
                    self._detach_layers()
                else:
                    self._restore_tree()
            except BaseException as rollback:
                self._uncertain = True
                raise LoraStateUncertainError("Failed to roll back CrossEncoder LoRA attachment") from rollback
            raise
        finally:
            try:
                self.discard(canonical)
            except OSError as exc:
                self._uncertain = True
                raise LoraStateUncertainError("LoRA source cleanup failed; retain capacity until model reload") from exc
        return memory

    def select(self, name: str | None) -> None:
        self._ready()
        canonical = self._source(name) if name is not None else None
        if canonical is not None and canonical not in self._loaded:
            raise LoraUnavailableError(f"Declared LoRA {name!r} is not resident")
        if canonical == self._active:
            return
        assert self._outer is not None
        try:
            if canonical is None:
                self._outer.disable_adapters()
                self._detach_layers()
            else:
                self._attach_layers()
                self._outer.set_adapter(PEFTLoRAMixin._peft_adapter_name(canonical))
                self._outer.enable_adapters()
                # Transformers' enable/set hooks default to trainable adapters.
                for parameter in self._parameters[canonical]:
                    parameter.requires_grad_(False)
            self._restore_grad_flags()
            self._active = canonical
        except BaseException as exc:
            self._uncertain = True
            raise LoraStateUncertainError("Failed to select CrossEncoder LoRA; reload the model") from exc

    def unload(self, name: str) -> None:
        self._ready()
        canonical = self._source(name)
        if canonical not in self._loaded:
            raise LoraUnavailableError(f"LoRA {name!r} is not resident")
        try:
            self._attach_layers()
            self._remove(PEFTLoRAMixin._peft_adapter_name(canonical))
            if len(self._loaded) == 1:
                self._restore_tree()
            else:
                assert self._outer is not None
                self._outer.disable_adapters()
                self._restore_grad_flags()
                self._detach_layers()
        except BaseException as exc:
            self._uncertain = True
            raise LoraStateUncertainError("Failed to unload CrossEncoder LoRA; reload the model") from exc
        del self._loaded[canonical]
        del self._parameters[canonical]
        self.discard(canonical)

    def close(self) -> None:
        with self._source_lock:
            self._closed = True
            prepared, self._prepared = self._prepared, {}
        errors = []
        try:
            for item in prepared.values():
                try:
                    item.directory.cleanup()
                except OSError as exc:
                    errors.append(exc)
            if self._model is not None:
                self._restore_tree()
        finally:
            self._loaded.clear()
            self._parameters.clear()
            self._resident_layers.clear()
            self._originals.clear()
            self._grad_flags.clear()
            self._outer = self._model = None
            self._encoder_prefix = None
            self._uncertain = False
        if errors:
            raise ExceptionGroup("Failed to remove owned LoRA source snapshots", errors)
