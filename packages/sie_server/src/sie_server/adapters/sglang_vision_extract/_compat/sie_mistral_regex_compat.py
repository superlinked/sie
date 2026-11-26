"""Backport Transformers' load fix for tokenizer configs that set ``fix_mistral_regex``.

Transformers 5.3 (the SGLang bundles' version) passes ``fix_mistral_regex`` to
``TokenizersBackend._patch_mistral_regex`` both by name and inside ``**kwargs``, so a
``tokenizer_config.json`` that sets the flag (LightOnOCR-3-1B does) fails to load with
"got multiple values for keyword argument 'fix_mistral_regex'". Upstream pops the key
at that call (Transformers 5.16.1 has the fix). This patch pops it before the
constructor runs and hands it back to ``_patch_mistral_regex`` with the same value and
``init_kwargs`` the fixed release passes, so the flag keeps its meaning.

It installs only on a ``TokenizersBackend`` whose constructor still contains the
duplicated argument, so it is a no-op on fixed releases. This module imports nothing
from SIE: the SGLang child's ``sitecustomize`` arms it with ``install_when_imported`` (no
early Transformers import), and the SIE worker calls ``install`` before building the processor.
"""

from __future__ import annotations

import functools
import importlib.abc
import inspect
import sys
import threading
from typing import Any

_MODULE = "transformers.tokenization_utils_tokenizers"
_KEY = "fix_mistral_regex"
_MARKER = "_sie_fix_mistral_regex_compat"
_DUPLICATED_ARGUMENT = 'fix_mistral_regex=kwargs.get("fix_mistral_regex"),'
_pending = threading.local()


def needs_patch(backend_cls: type) -> bool:
    """Whether ``backend_cls.__init__`` still passes the flag twice."""
    init = backend_cls.__init__
    if getattr(init, _MARKER, False):
        return False
    try:
        source = inspect.getsource(init)
    except (OSError, TypeError):
        return False
    return _DUPLICATED_ARGUMENT in source


def install_on(backend_cls: type) -> bool:
    """Patch ``backend_cls`` in place; return whether this call installed the patch."""
    if not needs_patch(backend_cls):
        return False
    original_init = backend_cls.__init__
    original_patch = backend_cls.__dict__["_patch_mistral_regex"].__func__

    @functools.wraps(original_init)
    def compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if _KEY not in kwargs:
            original_init(self, *args, **kwargs)
            return
        value = kwargs.pop(_KEY)
        stack = getattr(_pending, "values", None)
        if stack is None:
            stack = _pending.values = []
        stack.append(value)
        try:
            original_init(self, *args, **kwargs)
        finally:
            stack.pop()
        self.init_kwargs[_KEY] = value

    def compat_patch(
        cls: Any,
        tokenizer: Any,
        pretrained_model_name_or_path: Any,
        *args: Any,
        init_kwargs: dict[str, Any] | None = None,
        fix_mistral_regex: Any = None,
        **kwargs: Any,
    ) -> Any:
        stack = getattr(_pending, "values", None)
        if stack:
            fix_mistral_regex = stack[-1]
            init_kwargs = {**(init_kwargs or {}), _KEY: fix_mistral_regex}
        return original_patch(
            cls,
            tokenizer,
            pretrained_model_name_or_path,
            *args,
            init_kwargs=init_kwargs,
            fix_mistral_regex=fix_mistral_regex,
            **kwargs,
        )

    setattr(compat_init, _MARKER, True)
    backend_cls.__init__ = compat_init  # ty: ignore[invalid-assignment]
    backend_cls._patch_mistral_regex = classmethod(compat_patch)  # ty: ignore[unresolved-attribute]
    return True


def install() -> bool:
    """Patch the installed Transformers ``TokenizersBackend`` when it needs it."""
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return install_on(TokenizersBackend)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patch ``TokenizersBackend`` right after its module executes, without importing it early."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        original_exec = loader.exec_module

        def exec_module(module: Any) -> None:
            original_exec(module)
            backend = getattr(module, "TokenizersBackend", None)
            if isinstance(backend, type):
                install_on(backend)

        loader.exec_module = exec_module  # ty: ignore[invalid-assignment]
        return spec


def install_when_imported() -> None:
    """Patch now if Transformers' tokenizer module is loaded, else as soon as it loads."""
    module = sys.modules.get(_MODULE)
    if module is not None:
        backend = getattr(module, "TokenizersBackend", None)
        if isinstance(backend, type):
            install_on(backend)
        return
    if not any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
