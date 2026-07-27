"""Backport SGLang's Transformers 5 LightOnOCR config normalization.

SGLang 0.5.10 reads ``vision_config.rope_theta`` while Transformers 5 stores
the same value in ``vision_config.rope_parameters``. Upstream SGLang fixed this
in #28292. This site hook runs only in the OCR engine subprocess and can be
removed when the shared bundle advances to a release containing that fix.

The SGLang model registry discovers implementations by importing every model
module. Importing LightOnOCR eagerly from ``sitecustomize`` races that registry
bootstrap, so install a one-shot wrapper around the same ``import_module``
boundary and patch the class only after its native module has initialized.
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import wraps
from types import ModuleType
from typing import Any

_LIGHTON_MODULE = "sglang.srt.models.lightonocr"
_CLASS_PATCH_MARKER = "_sie_rope_parameters_compat"
_IMPORT_HOOK_MARKER = "_sie_lightonocr_deferred_compat"


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

    model_class.__init__ = compat_init
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


if os.environ.get("SIE_SGLANG_LIGHTON_OCR_COMPAT") == "1":
    _install_lightonocr_compat()
