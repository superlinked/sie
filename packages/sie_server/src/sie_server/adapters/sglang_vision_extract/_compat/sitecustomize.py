"""Backport SGLang's Transformers 5 LightOnOCR config normalization.

SGLang 0.5.10 reads ``vision_config.rope_theta`` while Transformers 5 stores
the same value in ``vision_config.rope_parameters``. Upstream SGLang fixed this
in #28292. This site hook runs only in the OCR engine subprocess and can be
removed when the shared bundle advances to a release containing that fix. It
also renames LightOnOCR-3-1B's legacy ``language_model.model.*`` weights to the
LightOnOCR-2-1B layout that SGLang 0.5.10 loads. The
same flag installs the ``fix_mistral_regex`` tokenizer backport
(``sie_mistral_regex_compat``), which Transformers 5.16.1 no longer needs.

The SGLang model registry discovers implementations by importing every model
module. Importing LightOnOCR eagerly from ``sitecustomize`` races that registry
bootstrap, so install a one-shot wrapper around the same ``import_module``
boundary and patch the class only after its native module has initialized.

Python imports only the first ``sitecustomize`` on the path, so this hook
also runs the next one on ``PYTHONPATH`` (the generic SGLang generation hook
the adapter places after it).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

_LIGHTON_MODULE = "sglang.srt.models.lightonocr"
_CLASS_PATCH_MARKER = "_sie_rope_parameters_compat"
_IMPORT_HOOK_MARKER = "_sie_lightonocr_deferred_compat"


# LightOnOCR-3-1B ships the older Mistral3 checkpoint layout (``language_model.model.*``);
# SGLang 0.5.10's loader only maps LightOnOCR-2-1B's (``model.language_model.*``).
_LEGACY_WEIGHT_PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head.", "lm_head."),
)


def _normalize_weight_names(weights: Any) -> Any:
    for name, tensor in weights:
        for legacy, current in _LEGACY_WEIGHT_PREFIXES:
            if name.startswith(legacy):
                name = current + name[len(legacy) :]
                break
        yield name, tensor


def _patch_lightonocr_module(module: ModuleType) -> None:
    model_class = getattr(module, "LightOnOCRForConditionalGeneration", None)
    if model_class is None:
        raise RuntimeError(f"{_LIGHTON_MODULE} does not expose LightOnOCRForConditionalGeneration")
    if getattr(model_class, _CLASS_PATCH_MARKER, False):
        return

    original_init = model_class.__init__

    @wraps(original_init)
    def compat_init(self: Any, *, config: Any, prefix: str = "", **kwargs: Any) -> None:
        vision_config = config.vision_config
        config_dict = vision_config.to_dict()
        rope_parameters = getattr(vision_config, "rope_parameters", None)
        if "rope_theta" not in config_dict and isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
            if rope_theta is not None:
                vision_config.rope_theta = rope_theta
        original_init(self, config=config, prefix=prefix, **kwargs)

    original_load_weights = model_class.load_weights

    @wraps(original_load_weights)
    def compat_load_weights(self: Any, weights: Any) -> Any:
        return original_load_weights(self, _normalize_weight_names(weights))

    model_class.__init__ = compat_init
    model_class.load_weights = compat_load_weights
    setattr(model_class, _CLASS_PATCH_MARKER, True)


def _install_lightonocr_compat() -> None:
    loaded = sys.modules.get(_LIGHTON_MODULE)
    if isinstance(loaded, ModuleType):
        _patch_lightonocr_module(loaded)
        return

    current_import_module = importlib.import_module
    if getattr(current_import_module, _IMPORT_HOOK_MARKER, False):
        return

    def deferred_import_module(name: str, package: str | None = None) -> ModuleType:
        module = current_import_module(name, package)
        if module.__name__ != _LIGHTON_MODULE:
            return module
        try:
            _patch_lightonocr_module(module)
        finally:
            if importlib.import_module is deferred_import_module:
                setattr(importlib, "import_module", current_import_module)  # noqa: B010
        return module

    setattr(deferred_import_module, _IMPORT_HOOK_MARKER, True)
    setattr(importlib, "import_module", deferred_import_module)  # noqa: B010


def _run_next_sitecustomize() -> None:
    own_dir = Path(__file__).resolve().parent
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / "sitecustomize.py"
        if not candidate.is_file() or candidate.resolve().parent == own_dir:
            continue
        spec = importlib.util.spec_from_file_location("_sie_chained_sitecustomize", candidate)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return


def _install_tokenizer_compat() -> None:
    """Load the sibling fix_mistral_regex backport (LightOnOCR-3-1B's tokenizer sets it)."""
    spec = importlib.util.spec_from_file_location(
        "_sie_mistral_regex_compat", Path(__file__).resolve().with_name("sie_mistral_regex_compat.py")
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_when_imported()


if os.environ.get("SIE_SGLANG_LIGHTON_OCR_COMPAT") == "1":
    _install_lightonocr_compat()
    _install_tokenizer_compat()
_run_next_sitecustomize()
