"""Backport SGLang multimodal processor configuration forwarding.

SGLang 0.5.10 parses ``--mm-process-config`` and stores its per-modality
settings, but its base processor does not forward the image settings to the
Hugging Face processor. Upstream #18467 now passes them as ``images_kwargs``.
This site hook runs only in the SGLang generation subprocess and can be removed
when the shared bundle advances to a release containing that fix.

The hook also applies ``SIE_SGLANG_SINGLE_IMAGE_MAX_PIXELS``: when set, a
request that carries exactly one image is processed with that ``max_pixels``
instead of ``--mm-process-config``'s, so one document page can be read at a
higher resolution while a request with several images keeps the launch bound
(and still fits the context window it fitted before). The raised bound is
applied to a shallow per-call copy of the processor, never to the shared one.

The same hook redacts multimodal load failures: SGLang raises
``Error while loading data {data}`` with the full inline payload, which its
serving layer then logs with a traceback. The redacted error is a
``ValueError`` because that is the only exception SGLang's ``/generate`` turns
into a 400 response; the adapter maps its fixed message to ``invalid_request``.

When the adapter sets ``SIE_SGLANG_JSON_NUMBER_MAX_DIGITS``, the hook also
bounds the digit runs of numbers in XGrammar's JSON-schema grammars (see the
section at the end of this module).
"""

from __future__ import annotations

import builtins
import copy
import importlib
import os
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any

_BASE_PROCESSOR_MODULE = "sglang.srt.multimodal.processors.base_processor"
_CLASS_PATCH_MARKER = "_sie_mm_process_config_compat"
_IMPORT_HOOK_MARKER = "_sie_mm_process_config_deferred_compat"
SINGLE_IMAGE_MAX_PIXELS_ENV = "SIE_SGLANG_SINGLE_IMAGE_MAX_PIXELS"


def _single_image_max_pixels() -> int | None:
    """The raised per-image bound for single-image requests, or ``None``."""
    raw = os.environ.get(SINGLE_IMAGE_MAX_PIXELS_ENV, "").strip()
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def _image_count(images: Any) -> int:
    if images is None:
        return 0
    if isinstance(images, (list, tuple)):
        return len(images)
    return 1


def _patch_base_processor_module(module: ModuleType) -> None:
    processor_class = getattr(module, "BaseMultimodalProcessor", None)
    if processor_class is None:
        raise RuntimeError(f"{_BASE_PROCESSOR_MODULE} does not expose BaseMultimodalProcessor")
    if getattr(processor_class, _CLASS_PATCH_MARKER, False):
        return

    original_process = processor_class.process_mm_data

    @wraps(original_process)
    def compat_process(
        self: Any,
        input_text: Any,
        images: Any = None,
        videos: Any = None,
        audios: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target = self
        image_config = getattr(self, "image_config", None)
        single_max = _single_image_max_pixels()
        if single_max is not None and _image_count(images) == 1 and isinstance(image_config, dict) and image_config:
            target = copy.copy(self)
            target.image_config = {**image_config, "max_pixels": single_max}
            image_config = target.image_config
        if images and isinstance(image_config, dict) and image_config:
            kwargs.setdefault("images_kwargs", {}).update(image_config)
        return original_process(
            target,
            input_text,
            images=images,
            videos=videos,
            audios=audios,
            **kwargs,
        )

    processor_class.process_mm_data = compat_process

    original_load = processor_class.__dict__.get("_load_single_item")
    if isinstance(original_load, classmethod):
        load_function = original_load.__func__

        @wraps(load_function)
        def redacted_load(cls: Any, data: Any, modality: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return load_function(cls, data, modality, *args, **kwargs)
            except RuntimeError as exc:
                cause = exc.__context__ if exc.__context__ is not None else exc
                modality_name = getattr(modality, "name", "multimodal")
                message = f"Error while loading {modality_name} data ({type(cause).__name__})"
                raise ValueError(message) from None

        processor_class._load_single_item = classmethod(redacted_load)

    setattr(processor_class, _CLASS_PATCH_MARKER, True)


def _install_mm_process_config_compat() -> None:
    loaded = sys.modules.get(_BASE_PROCESSOR_MODULE)
    if isinstance(loaded, ModuleType):
        _patch_base_processor_module(loaded)
        return

    current_import = builtins.__import__
    if getattr(current_import, _IMPORT_HOOK_MARKER, False):
        return

    def deferred_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        module = current_import(name, globals, locals, fromlist, level)
        loaded_module = sys.modules.get(_BASE_PROCESSOR_MODULE)
        if not isinstance(loaded_module, ModuleType) or not hasattr(loaded_module, "BaseMultimodalProcessor"):
            return module
        try:
            _patch_base_processor_module(loaded_module)
        finally:
            if builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
        return module

    setattr(deferred_import, _IMPORT_HOOK_MARKER, True)
    setattr(builtins, "__import__", deferred_import)  # noqa: B010


if os.environ.get("SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT") == "1":
    _install_mm_process_config_compat()


# --- JSON number digit bound ---------------------------------------------
#
# XGrammar compiles every numeric JSON-schema value without a range to digit
# runs of unbounded length. Under greedy decoding a model that settles into
# repeating a digit (``8214263.000000...``) has nothing to stop it before
# ``max_new_tokens``, and the reply is truncated JSON. XGrammar has no option
# for this and SGLang 0.5.20 compiles the schema itself, so this hook bounds the
# digit runs where SGLang asks XGrammar for the grammar. Once a number holds the
# maximum digits, the grammar lets the model only close it.

_XGRAMMAR_COMPILER_MODULE = "xgrammar.compiler"
_XGRAMMAR_GRAMMAR_MODULE = "xgrammar.grammar"
_JSON_DIGITS_MARKER = "_sie_json_number_max_digits"
_JSON_DIGITS_IMPORT_HOOK_MARKER = "_sie_json_number_max_digits_deferred"
JSON_NUMBER_MAX_DIGITS_ENV = "SIE_SGLANG_JSON_NUMBER_MAX_DIGITS"


def bound_json_number_rules(ebnf: str, max_digits: int) -> str:
    """Replace XGrammar's shared number rules with digit-bounded ones.

    XGrammar routes every numeric schema without a range through two shared
    rules, ``basic_integer`` and ``basic_number``, which ``basic_any`` also
    uses. A numeric schema with a range compiles to its own finite pattern
    instead. The replacements keep each rule's JSON shape and bound only the
    digit runs: at most ``max_digits`` integer digits, at most ``max_digits``
    fractional digits, and at most three exponent digits. Every other rule,
    including digit runs that come from a string ``pattern``, is unchanged.
    """
    tail = max_digits - 1
    replacements = {
        "basic_integer": f'(("0") | ("-"? [1-9] [0-9]{{0,{tail}}}))',
        "basic_number": (
            f'(("-"? ("0" | [1-9] [0-9]{{0,{tail}}}) ("." [0-9]{{1,{max_digits}}})? ([eE] [+\\-]? [0-9]{{1,3}})?))'
        ),
    }
    lines = []
    for line in ebnf.splitlines():
        name, separator, _ = line.partition(" ::= ")
        if separator and name in replacements:
            line = f"{name} ::= {replacements[name]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def json_number_max_digits_from_env() -> int | None:
    """Return the configured digit bound, or None when the hook is off."""
    raw = os.environ.get(JSON_NUMBER_MAX_DIGITS_ENV)
    if not raw:
        return None
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"{JSON_NUMBER_MAX_DIGITS_ENV} must be a positive integer, got {raw!r}")
    return int(raw)


def _patch_xgrammar_compiler_module(module: ModuleType, max_digits: int) -> None:
    compiler_class = getattr(module, "GrammarCompiler", None)
    if compiler_class is None:
        raise RuntimeError(f"{_XGRAMMAR_COMPILER_MODULE} does not expose GrammarCompiler")
    if getattr(compiler_class, _JSON_DIGITS_MARKER, False):
        return
    grammar_class = importlib.import_module(_XGRAMMAR_GRAMMAR_MODULE).Grammar
    original_compile = compiler_class.compile_json_schema

    @wraps(original_compile)
    def compile_json_schema(
        self: Any,
        schema: Any,
        *,
        any_whitespace: bool = True,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        strict_mode: bool = True,
        max_whitespace_cnt: int | None = None,
    ) -> Any:
        grammar = grammar_class.from_json_schema(
            schema,
            any_whitespace=any_whitespace,
            indent=indent,
            separators=separators,
            strict_mode=strict_mode,
            max_whitespace_cnt=max_whitespace_cnt,
        )
        return self.compile_grammar(bound_json_number_rules(str(grammar), max_digits))

    compiler_class.compile_json_schema = compile_json_schema
    setattr(compiler_class, _JSON_DIGITS_MARKER, True)


def _install_json_number_max_digits() -> None:
    max_digits = json_number_max_digits_from_env()
    if max_digits is None:
        return
    loaded = sys.modules.get(_XGRAMMAR_COMPILER_MODULE)
    if isinstance(loaded, ModuleType) and hasattr(loaded, "GrammarCompiler"):
        _patch_xgrammar_compiler_module(loaded, max_digits)
        return

    current_import = builtins.__import__
    if getattr(current_import, _JSON_DIGITS_IMPORT_HOOK_MARKER, False):
        return

    def deferred_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        module = current_import(name, globals, locals, fromlist, level)
        loaded_module = sys.modules.get(_XGRAMMAR_COMPILER_MODULE)
        if not isinstance(loaded_module, ModuleType) or not hasattr(loaded_module, "GrammarCompiler"):
            return module
        try:
            _patch_xgrammar_compiler_module(loaded_module, max_digits)
        finally:
            if builtins.__import__ is deferred_import:
                setattr(builtins, "__import__", current_import)  # noqa: B010
        return module

    setattr(deferred_import, _JSON_DIGITS_IMPORT_HOOK_MARKER, True)
    setattr(builtins, "__import__", deferred_import)  # noqa: B010


_install_json_number_max_digits()
