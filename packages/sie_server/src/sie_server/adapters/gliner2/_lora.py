"""Strict generic-PEFT lifecycle for the opt-in GLiNER2 entity adapter.

The raw extractor owns serving. A separate PEFT controller owns only reversible
LoRA injection; it is never used as the extraction entrypoint. Resident wrappers
stay outside the module tree while base is selected, sharing the exact original
projection weights. Admission and selection still have their own runtime cost.
"""

from __future__ import annotations

import importlib.metadata
import math
import re
import shutil
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftConfig, PeftModel
from peft.functional import delete_adapter
from peft.tuners.lora import LoraModel
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file
from transformers import BertModel, DebertaV2Model

from sie_server.adapters.lora import LoraRuntimeConfig, LoraStateUncertainError, LoraUnavailableError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin

_FILES = ("adapter_config.json", "adapter_model.safetensors")
_PEFT_ATTRIBUTES = ("peft_config", "_hf_peft_config_loaded")
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


class GLiNER2Lora:
    """Declared adapters; worker mutation exclusion plus direct-call protection."""

    def __init__(self, paths: dict[str, str], revisions: dict[str, str | None] | None, max_loaded: int) -> None:
        if isinstance(max_loaded, bool) or not isinstance(max_loaded, int) or max_loaded < 1:
            raise ValueError("max_loaded_loras must be a positive integer")
        if not isinstance(paths, dict) or not paths:
            raise ValueError("Entity LoRA requires a nonempty lora_paths alias-to-source map")
        aliases = dict(paths)
        if any(
            not isinstance(k, str)
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", k)
            or not isinstance(v, str)
            or not v
            for k, v in aliases.items()
        ):
            raise ValueError("LoRA declarations require simple nonempty aliases and nonempty sources")
        self._aliases = aliases
        self._sources = set(aliases.values())
        if any(source in aliases and aliases[source] != source for source in self._sources):
            raise ValueError("A LoRA alias must not shadow a different declared source")
        if revisions is not None and not isinstance(revisions, Mapping):
            raise ValueError("LoRA revisions must be a canonical-source mapping")
        pins = {} if revisions is None else dict(revisions)
        if any(not isinstance(key, str) or not key for key in pins):
            raise ValueError("LoRA revision keys must be nonempty canonical source strings")
        if any(key in self._aliases and key not in self._sources for key in pins):
            raise ValueError("LoRA revision keys must name canonical sources, not served aliases")
        if any(
            pin is not None and (not isinstance(pin, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", pin))
            for pin in pins.values()
        ):
            raise ValueError("LoRA revisions must be immutable 40-hex commits")
        # ModelConfig supplies the canonical-source map for every profile.
        # Validate all pins, but this selected profile only serves its sources.
        self._revisions = {source: pins.get(source) for source in self._sources}
        self._max_loaded = max_loaded
        self.lock = threading.RLock()
        self._source_lock = threading.Lock()
        self._prepared: dict[str, _Prepared] = {}
        self._model: Any = None
        self._controller: PeftModel | None = None
        self._originals: dict[str, torch.nn.Linear] = {}
        # A plain dict retains GPU residency without making raw Extractor.eval()
        # or base inference traverse adapter wrappers and their child modules.
        self._resident_layers: dict[str, LoraLinear] = {}
        self._shapes: dict[str, tuple[int, int, torch.dtype]] = {}
        self._grad: list[tuple[torch.nn.Parameter, bool]] = []
        self._training: list[tuple[torch.nn.Module, bool]] = []
        self._attributes: list[tuple[torch.nn.Module, str, bool, Any, dict | None]] = []
        self._loaded: dict[str, list[torch.nn.Parameter]] = {}
        self._active: str | None = None
        self._closed = True
        self._uncertain = False
        self._generation = 0

    def runtime_config(self) -> LoraRuntimeConfig:
        # Deferred-only: model startup must not fetch a declaration before a
        # request has passed the entity-only admission hook.
        return LoraRuntimeConfig(dict(self._aliases), frozenset(), frozenset(), self._max_loaded)

    def source(self, name: str) -> str:
        source = self._aliases.get(name, name)
        if source not in self._sources:
            raise ValueError("LoRA is not declared for this entity profile")
        return source

    def bind(self, model: Any) -> None:
        with self.lock:
            if importlib.metadata.version("gliner2") not in {"1.3.2", "2.0.0"}:
                raise ValueError("The entity LoRA adapter requires GLiNER2 1.3.2 or 2.0.0")
            if type(model) is not GLiNER2 or type(model.encoder) not in (BertModel, DebertaV2Model):
                raise ValueError("Entity LoRA requires the ordinary GLiNER2 BERT or DeBERTaV2 encoder")
            if getattr(model.encoder, "is_quantized", False) or any(isinstance(m, LoraLinear) for m in model.modules()):
                raise ValueError("Entity LoRA requires an unquantized base without existing adapters")
            if any(
                getattr(m, "peft_config", None) or getattr(m, "_hf_peft_config_loaded", False) for m in model.modules()
            ):
                raise ValueError("Entity LoRA requires a base without existing PEFT state")
            self._model = model
            self._originals = {n: m for n, m in model.named_modules() if type(m) is torch.nn.Linear}
            self._shapes = {n: (m.in_features, m.out_features, m.weight.dtype) for n, m in self._originals.items()}
            self._grad = [(p, p.requires_grad) for p in model.parameters()]
            self._training = [(m, m.training) for m in model.modules()]
            self._attributes = [
                (
                    m,
                    key,
                    key in m.__dict__,
                    m.__dict__.get(key),
                    dict(m.__dict__[key]) if isinstance(m.__dict__.get(key), dict) else None,
                )
                for m in model.modules()
                for key in _PEFT_ATTRIBUTES
            ]
            with self._source_lock:
                self._generation += 1
                self._closed = False
            self._uncertain = False

    def _ready(self) -> None:
        if self._uncertain:
            raise LoraStateUncertainError("Entity LoRA state is uncertain; reload the model")
        if self._closed or self._model is None:
            raise RuntimeError("Entity LoRA model is not loaded")

    @staticmethod
    def _config(source: str, config: PeftConfig) -> LoraConfig:
        PEFTLoRAMixin()._check_lora_config(source, config)
        if not isinstance(config, LoraConfig) or config.task_type is not None:
            raise ValueError("Entity LoRA requires ordinary generic PEFT with task_type=None")
        for field in _UNSUPPORTED:
            if getattr(config, field, None):
                raise ValueError(f"Entity LoRA does not support {field}")
        if not isinstance(config.init_lora_weights, bool):
            raise ValueError("Entity LoRA requires plain boolean init_lora_weights")
        if isinstance(config.r, bool) or not isinstance(config.r, int) or config.r < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(config.lora_alpha) or config.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(config.lora_dropout) or not 0 <= config.lora_dropout < 1:
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        if (
            not isinstance(config.target_modules, (set, list))
            or not config.target_modules
            or any(not isinstance(t, str) or not t for t in config.target_modules)
        ):
            raise ValueError("Entity LoRA requires explicit target module names")
        config.inference_mode = True
        return config

    @staticmethod
    def _weights(
        config: LoraConfig, weights: dict[str, torch.Tensor], shapes: dict[str, tuple[int, int, torch.dtype]]
    ) -> dict[str, torch.Tensor]:
        expected: dict[str, tuple[tuple[int, int], torch.dtype]] = {}
        for target in config.target_modules or []:
            matches = [n for n in shapes if n == target or n.endswith(f".{target}")]
            if not matches or any(not n.startswith(("encoder.encoder.layer.", "span_rep.")) for n in matches):
                raise ValueError(f"LoRA target {target!r} must resolve only to encoder or span Linear projections")
            for name in matches:
                ins, outs, dtype = shapes[name]
                expected[f"{name}.lora_A.weight"] = ((config.r, ins), dtype)
                expected[f"{name}.lora_B.weight"] = ((outs, config.r), dtype)
        normalized: dict[str, torch.Tensor] = {}
        for key, tensor in weights.items():
            name = key.removeprefix("base_model.model.")
            if name in normalized:
                raise ValueError("Duplicate LoRA tensor key")
            normalized[name] = tensor
        if set(normalized) != set(expected):
            raise ValueError("LoRA weights must exactly match all declared A/B projection matrices")
        for name, tensor in normalized.items():
            shape, dtype = expected[name]
            if tuple(tensor.shape) != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid shape, dtype or nonfinite LoRA tensor {name!r}")
            if not torch.isfinite(tensor.to(dtype=dtype)).all():
                raise ValueError(f"LoRA tensor {name!r} overflows the model compute dtype")
        return normalized

    def fetch(self, name: str, revision: str | None = None) -> str:
        canonical = self.source(name)
        pinned = self._revisions[canonical]
        if revision is not None and revision != pinned:
            raise ValueError("LoRA revision differs from the declared source")
        with self._source_lock:
            self._ready()
            generation, shapes = self._generation, dict(self._shapes)
            if canonical in self._prepared:
                return self._prepared[canonical].directory.name
        directory = Path(canonical)
        if directory.is_dir():
            if pinned is not None:
                raise ValueError("A local LoRA directory cannot honor a Hub revision pin")
        else:
            directory = Path(snapshot_download(canonical, revision=pinned, allow_patterns=list(_FILES)))
        temporary = tempfile.TemporaryDirectory(prefix="sie-gliner2-lora-")
        try:
            for filename in _FILES:
                shutil.copyfile(directory / filename, Path(temporary.name) / filename)
            config = self._config(canonical, PeftConfig.from_pretrained(temporary.name, local_files_only=True))
            weights = self._weights(config, load_file(str(Path(temporary.name) / _FILES[1])), shapes)
            with self._source_lock:
                self._ready()
                if generation != self._generation:
                    raise RuntimeError("Model changed while the LoRA source was fetched")
                if canonical in self._prepared:
                    temporary.cleanup()
                    return self._prepared[canonical].directory.name
                self._prepared[canonical] = _Prepared(temporary, config, weights)
                return temporary.name
        except BaseException:
            temporary.cleanup()
            raise

    def discard(self, name: str) -> None:
        canonical = self.source(name)
        with self._source_lock:
            prepared = self._prepared.pop(canonical, None)
        if prepared is not None:
            try:
                prepared.directory.cleanup()
            except OSError:
                with self._source_lock:
                    self._prepared[canonical] = prepared
                raise

    def _restore_flags(self) -> None:
        for parameter, flag in self._grad:
            parameter.requires_grad_(flag)
        # Assign individually: Module.train() recursively overwrites mixed modes.
        for module, flag in self._training:
            module.training = flag

    def _freeze_adapters(self) -> None:
        for module in self._model.modules():
            if isinstance(module, LoraLinear):
                for group in (module.lora_A, module.lora_B, module.lora_dropout):
                    group.eval()
                    for parameter in group.parameters():
                        parameter.requires_grad_(False)
        self._restore_flags()

    def _tuner(self) -> LoraModel:
        if self._controller is None or not isinstance(self._controller.base_model, LoraModel):
            raise RuntimeError("Entity LoRA controller does not own ordinary LoRA layers")
        return self._controller.base_model

    def _attach_layers(self) -> None:
        """Expose all residents before PEFT selection, injection or removal."""
        for name, layer in self._resident_layers.items():
            original = self._originals[name]
            current = self._model.get_submodule(name)
            if current is not original and current is not layer:
                raise RuntimeError("A GLiNER2 module changed outside its owned LoRA lifecycle")
            if layer.get_base_layer() is not original:
                raise RuntimeError("A GLiNER2 LoRA base projection changed")
            # Detached adapters must not silently migrate if an external caller
            # moves/casts the raw model after it was admitted for serving.
            if any(
                parameter.device != original.weight.device or parameter.dtype != original.weight.dtype
                for parameter in layer.parameters()
            ):
                raise RuntimeError("GLiNER2 device or precision changed with resident LoRAs")
            parent, _, leaf = name.rpartition(".")
            setattr(self._model.get_submodule(parent), leaf, layer)

    def _detach_layers(self) -> None:
        """Restore native base projections while retaining nonempty LoRA layers."""
        for name, original in self._originals.items():
            current = self._model.get_submodule(name)
            if current is original:
                continue
            if not isinstance(current, LoraLinear) or current.get_base_layer() is not original:
                raise RuntimeError("A GLiNER2 module changed outside its owned LoRA lifecycle")
            if current.lora_A:
                self._resident_layers[name] = current
            else:
                self._resident_layers.pop(name, None)
            parent, _, leaf = name.rpartition(".")
            setattr(self._model.get_submodule(parent), leaf, original)
        self._active = None

    def _restore_tree(self, *, final: bool) -> None:
        for name, original in self._originals.items():
            current = self._model.get_submodule(name)
            if current is original:
                continue
            if not isinstance(current, LoraLinear) or current.get_base_layer() is not original:
                raise RuntimeError("A GLiNER2 module changed outside its owned LoRA lifecycle")
            if final or not current.lora_A:
                self._resident_layers.pop(name, None)
                parent, _, leaf = name.rpartition(".")
                setattr(self._model.get_submodule(parent), leaf, original)
        if final:
            self._resident_layers.clear()
            for module, name, present, value, contents in self._attributes:
                if present:
                    if contents is not None:
                        value.clear()
                        value.update(contents)
                    setattr(module, name, value)
                else:
                    module.__dict__.pop(name, None)
            self._controller = None
        self._restore_flags()
        self._active = None

    def _remove(self, internal: str) -> None:
        # Works even when injection failed before a PeftModel was returned.
        delete_adapter(self._model, adapter_name=internal, prefix="lora_")
        getattr(self._model, "peft_config", {}).pop(internal, None)
        if self._controller is not None:
            self._controller.peft_config.pop(internal, None)
            # The low-level helper updates leaf layers, not the controller or
            # tuner root. Its returned first-layer selection can also be empty
            # for disjoint targets even while another resident survives.
            # Synchronize all levels to a known resident before callers disable
            # and refreeze them; final removal restores/drops the controller.
            for canonical in self._loaded:
                remaining = PEFTLoRAMixin._peft_adapter_name(canonical)
                if remaining != internal:
                    self._controller.set_adapter(remaining, inference_mode=True)
                    break

    def load(self, name: str, revision: str | None = None, *, source: str | None = None) -> int:
        canonical = self.source(name)
        if revision is not None and revision != self._revisions[canonical]:
            raise ValueError("LoRA revision differs from the declared source")
        with self.lock:
            self._ready()
            if canonical in self._loaded:
                return sum(p.numel() * p.element_size() for p in self._loaded[canonical])
        if source is None:
            source = self.fetch(canonical, revision)
        with self.lock:
            self._ready()
            with self._source_lock:
                prepared = self._prepared.get(canonical)
                if prepared is None or prepared.directory.name != source:
                    # A concurrent direct caller can finish the same load while
                    # this caller fetches. No second attachment is necessary.
                    if canonical in self._loaded:
                        return sum(p.numel() * p.element_size() for p in self._loaded[canonical])
                    raise ValueError("LoRA source must be this declaration's validated snapshot")
            try:
                if canonical in self._loaded:
                    return sum(p.numel() * p.element_size() for p in self._loaded[canonical])
                if len(self._loaded) >= self._max_loaded:
                    raise ValueError("Entity LoRA capacity is exhausted; evict before attaching")
                return self._attach(canonical, source, prepared)
            finally:
                try:
                    self.discard(canonical)
                except OSError as error:
                    self._uncertain = True
                    raise LoraStateUncertainError("Entity LoRA source cleanup failed; reload model") from error

    def _attach(self, canonical: str, source: str, prepared: _Prepared) -> int:
        internal = PEFTLoRAMixin._peft_adapter_name(canonical)
        try:
            self._attach_layers()
            if self._controller is None:
                self._controller = PeftModel.from_pretrained(
                    self._model,
                    source,
                    adapter_name=internal,
                    config=prepared.config,
                    is_trainable=False,
                    autocast_adapter_dtype=False,
                    local_files_only=True,
                )
            else:
                # Reuse the validated config for later attachments too. The
                # loader then reads only the owned, safetensors-only snapshot.
                self._controller.add_adapter(internal, prepared.config, autocast_adapter_dtype=False)
                self._controller.load_adapter(
                    source,
                    adapter_name=internal,
                    is_trainable=False,
                    autocast_adapter_dtype=False,
                    local_files_only=True,
                )
            if self._controller.get_base_model() is not self._model:
                raise RuntimeError("PEFT changed the serving extractor owner")
            parameters = []
            for key, weight in prepared.weights.items():
                target, part, _ = key.rsplit(".", 2)
                layer = self._model.get_submodule(target)
                if not isinstance(layer, LoraLinear) or layer.get_base_layer() is not self._originals[target]:
                    raise RuntimeError("PEFT did not retain the original projection")
                parameter = getattr(layer, part)[internal].weight
                if not torch.isfinite(parameter).all() or not torch.equal(
                    parameter.detach(), weight.to(device=parameter.device, dtype=parameter.dtype)
                ):
                    raise ValueError("LoRA tensors were not installed exactly")
                parameters.append(parameter)
            self._tuner().disable_adapter_layers()
            self._freeze_adapters()
            self._detach_layers()
            self._loaded[canonical] = parameters
        except BaseException:
            try:
                if self._loaded:
                    # A failed detach can have saved some new layers off-tree.
                    # Restore the whole owned set before deleting the injection.
                    self._attach_layers()
                    self._remove(internal)
                    assert self._controller is not None
                    self._tuner().disable_adapter_layers()
                    self._freeze_adapters()
                self._restore_tree(final=not self._loaded)
                if self._loaded:
                    self._detach_layers()
            except BaseException as rollback:
                self._uncertain = True
                raise LoraStateUncertainError("Entity LoRA attachment rollback failed") from rollback
            raise
        return sum(p.numel() * p.element_size() for p in parameters)

    def select(self, name: str | None) -> None:
        with self.lock:
            self._ready()
            canonical = self.source(name) if name is not None else None
            if canonical is not None and canonical not in self._loaded:
                raise LoraUnavailableError("Declared entity LoRA is not resident")
            # Attach/remove/reset already establish this state. Avoid walking
            # every original parameter/module on repeated base selection and
            # the worker's repeated selection before the direct entrypoint.
            if canonical == self._active:
                return
            try:
                if self._controller is not None:
                    if canonical is None:
                        self._tuner().disable_adapter_layers()
                    else:
                        self._attach_layers()
                        self._controller.set_adapter(PEFTLoRAMixin._peft_adapter_name(canonical), inference_mode=True)
                        self._tuner().enable_adapter_layers()
                    self._freeze_adapters()
                    if canonical is None:
                        self._detach_layers()
                self._active = canonical
            except BaseException as error:
                self._uncertain = True
                raise LoraStateUncertainError("Entity LoRA selection failed; reload model") from error

    def unload(self, name: str) -> None:
        with self.lock:
            self._ready()
            canonical = self.source(name)
            if canonical not in self._loaded:
                raise LoraUnavailableError("Entity LoRA is not resident")
            try:
                self._attach_layers()
                self._remove(PEFTLoRAMixin._peft_adapter_name(canonical))
                final = len(self._loaded) == 1
                if not final:
                    assert self._controller is not None
                    self._tuner().disable_adapter_layers()
                    self._freeze_adapters()
                self._restore_tree(final=final)
                if not final:
                    self._detach_layers()
            except BaseException as error:
                self._uncertain = True
                raise LoraStateUncertainError("Entity LoRA removal failed; reload model") from error
            del self._loaded[canonical]
            self.discard(canonical)

    def close(self) -> None:
        with self.lock:
            with self._source_lock:
                self._closed = True
                self._generation += 1
                prepared, self._prepared = self._prepared, {}
            errors = []
            try:
                if self._model is not None:
                    self._restore_tree(final=True)
            except Exception as error:  # noqa: BLE001 — cleanup must attempt every owned resource.
                errors.append(error)
            for item in prepared.values():
                try:
                    item.directory.cleanup()
                except OSError as error:
                    errors.append(error)
            self._controller = self._model = None
            self._originals.clear()
            self._resident_layers.clear()
            self._shapes.clear()
            self._grad.clear()
            self._training.clear()
            self._attributes.clear()
            self._loaded.clear()
            if errors:
                raise ExceptionGroup("Entity LoRA teardown failed", errors)
