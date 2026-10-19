"""Load-only compatibility boundary for Sentence Transformers 5.x custom code.

The supported 5.x releases implicitly trust local model directories. Enforce an
explicit opt-in at the actual class-resolution sites without changing library
globals, native model classes, or their inference methods.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import FunctionType, MethodType, ModuleType
from typing import Any

import sentence_transformers
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

from sie_server.adapters._transformers_compat import install_transformers_security_fixes

_SUPPORTED_VERSIONS = frozenset({"5.4.1", "5.6.0", "5.7.0"})
_ROOT_CLASSES = (SentenceTransformer, SparseEncoder, CrossEncoder)
_RESOLVER = "_load_module_class_from_ref"
_REFUSAL = "Custom Sentence Transformers code requires trust_remote_code=True."
_UNSUPPORTED = "Unsupported Sentence Transformers loading implementation."
_NESTED_IMPORTS = {
    "5.4.1": ("import_from_string", "import_from_string"),
    "5.6.0": ("import_module_class", "import_from_string"),
    "5.7.0": ("import_module_class", "import_module_class"),
}
_KWARG_GROUPS = (
    "model_kwargs",
    "processor_kwargs",
    "config_kwargs",
    "tokenizer_kwargs",
    "automodel_args",
    "model_args",
    "tokenizer_args",
    "config_args",
)


def _native_references(exports: tuple[ModuleType, ...], base: type, aliases: dict[str, str]) -> dict[str, type]:
    """Build a closed name map from installed, trusted library exports.

    Model-supplied names never reach an import function, even for malformed
    names resembling a native namespace. Legacy aliases map to the same class.
    """
    references: dict[str, type] = {}
    for namespace in exports:
        for name in namespace.__all__:
            value = getattr(namespace, name)
            if not inspect.isclass(value) or not issubclass(value, base) or inspect.isabstract(value):
                continue
            references[f"{namespace.__name__}.{name}"] = value
            references[f"{value.__module__}.{value.__name__}"] = value
    for old, new in aliases.items():
        for reference, value in tuple(references.items()):
            if reference.startswith(new + "."):
                references[old + reference[len(new) :]] = value
            # Historical import_from_string also accepts a module alias whose
            # final component is the exported class name (models.Pooling).
            if new == value.__module__ and old.rsplit(".", 1)[-1] == value.__name__:
                references[old] = value
    return references


class _NativeLoadBoundary:
    """Private state used only while one native root constructor executes."""

    def __init__(self) -> None:
        # This private layout is optional across the public ST5 range. Import
        # it only after selecting a supported untrusted path, never for True.
        from sentence_transformers.base import modules as base_modules
        from sentence_transformers.cross_encoder import modules as cross_modules
        from sentence_transformers.sentence_transformer import modules as sentence_modules
        from sentence_transformers.sentence_transformer.modules import tokenizer as tokenizers
        from sentence_transformers.sparse_encoder import modules as sparse_modules
        from sentence_transformers.util.deprecated_import import DEPRECATED_MODULE_PATHS

        self.router = base_modules.Router
        self.word_embeddings = sentence_modules.WordEmbeddings
        self.modules = _native_references(
            (base_modules, sentence_modules, sparse_modules, cross_modules),
            base_modules.Module,
            DEPRECATED_MODULE_PATHS,
        )
        self.tokenizers = _native_references((tokenizers,), tokenizers.WordTokenizer, DEPRECATED_MODULE_PATHS)
        self.facades: dict[type, type] = {}
        self.import_names = _NESTED_IMPORTS[sentence_transformers.__version__]

    def resolve_module(self, class_ref: object, *_args: object, **_kwargs: object) -> type:
        if not isinstance(class_ref, str) or class_ref not in self.modules:
            raise ValueError(_REFUSAL)
        native = self.modules[class_ref]
        if issubclass(native, self.router) or native is self.word_embeddings:
            return self._facade(native)
        return native

    def resolve_tokenizer(self, class_ref: object, *_args: object, **_kwargs: object) -> type:
        if not isinstance(class_ref, str) or class_ref not in self.tokenizers:
            raise ValueError(_REFUSAL)
        return self.tokenizers[class_ref]

    def resolve_root(self, _model: object, class_ref: object, *_args: object, **_kwargs: object) -> type:
        return self.resolve_module(class_ref)

    def _facade(self, native: type) -> type:
        if native in self.facades:
            return self.facades[native]
        method = getattr(native, "load", None)
        if not isinstance(method, MethodType):
            raise ValueError(_UNSUPPORTED)
        loader = method.__func__
        import_name = self.import_names[1 if native is self.word_embeddings else 0]
        if not isinstance(loader, FunctionType) or import_name not in loader.__code__.co_names:
            raise ValueError(_UNSUPPORTED)
        # Copy globals for this function only. The original library function,
        # class and globals remain unchanged, including during concurrent loads.
        globals_copy = dict(loader.__globals__)
        globals_copy[import_name] = self.resolve_tokenizer if native is self.word_embeddings else self.resolve_module
        guarded = FunctionType(loader.__code__, globals_copy, loader.__name__, loader.__defaults__, loader.__closure__)
        guarded.__kwdefaults__ = loader.__kwdefaults__
        guarded.__annotations__ = loader.__annotations__
        # Bind cls to the ORIGINAL native class. The facade is never a model
        # instance; only its load entry point is presented to upstream code.
        facade = type(native.__name__, (native,), {"load": staticmethod(MethodType(guarded, native))})
        self.facades[native] = facade
        return facade


def load_sentence_transformer[T: (SentenceTransformer, SparseEncoder, CrossEncoder)](
    model_class: type[T],
    model_name_or_path: str,
    *,
    trust_remote_code: bool | None = False,
    **kwargs: Any,
) -> T:
    """Construct the native model, enforcing explicit trust for custom classes.

    This compatibility helper intentionally supports only the inspected 5.x
    releases. An unknown version fails before an untrusted model is loaded.
    Explicit True preserves the original constructor path exactly.
    """
    install_transformers_security_fixes()
    if trust_remote_code is True:
        return model_class(model_name_or_path, trust_remote_code=True, **kwargs)
    if trust_remote_code is not False and trust_remote_code is not None:
        raise ValueError(_REFUSAL)
    if sentence_transformers.__version__ not in _SUPPORTED_VERSIONS or model_class not in _ROOT_CLASSES:
        raise ValueError(_UNSUPPORTED)
    # The upstream decorators rename these aliases after this boundary. Check
    # every supported spelling before that happens, including canonical/alias
    # collisions. Do not mutate the caller's dictionaries.
    for name in _KWARG_GROUPS:
        nested = kwargs.get(name)
        if nested is not None and not isinstance(nested, Mapping):
            raise ValueError(_REFUSAL)
        if isinstance(nested, Mapping) and "trust_remote_code" in nested:
            nested_trust = nested["trust_remote_code"]
            if nested_trust is not False and nested_trust is not None:
                raise ValueError(_REFUSAL)
            kwargs[name] = {**nested, "trust_remote_code": False}

    boundary = _NativeLoadBoundary()
    model = model_class.__new__(model_class)
    # nn.Module is not initialized yet; use object assignment deliberately.
    # Native nn.Module initialization preserves unrelated instance attributes.
    object.__setattr__(model, _RESOLVER, MethodType(boundary.resolve_root, model))
    try:
        model_class.__init__(model, model_name_or_path, trust_remote_code=False, **kwargs)
        return model
    finally:
        # Remove even if construction/nn.Module initialization failed. Never
        # leave a bound method cycle or an inference-time compatibility wrapper.
        object.__getattribute__(model, "__dict__").pop(_RESOLVER, None)
