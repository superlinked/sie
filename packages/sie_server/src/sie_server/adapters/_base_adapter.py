from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class BaseAdapter(ModelAdapter):
    """Concrete base with common defaults.

    Provides:
    - ``capabilities`` / ``dims`` properties derived from ``spec``.
    - Standard ``unload()`` driven by ``spec.unload_fields``.
    - Default ``get_preprocessor()`` returning ``CharCountPreprocessor``.
    - ``_resolve_dtype()`` mapping ``compute_precision`` string to dtype.
    - ``_check_loaded()`` guard for encode/score/extract entry points.

    Every concrete subclass must declare a class-level ``spec``.
    """

    spec: ClassVar[AdapterSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Only validate classes that declare their own spec
        if "spec" not in cls.__dict__:
            return

        spec = cls.spec
        if not isinstance(spec, AdapterSpec):
            msg = f"{cls.__name__}.spec must be an AdapterSpec instance"
            raise TypeError(msg)

        if not spec.inputs:
            msg = f"{cls.__name__}.spec.inputs must be non-empty"
            raise TypeError(msg)

        if not spec.outputs:
            msg = f"{cls.__name__}.spec.outputs must be non-empty"
            raise TypeError(msg)

        # Validate output -> method consistency
        encode_outputs = {"dense", "sparse", "multivector"}
        declared_encode = encode_outputs & set(spec.outputs)
        if declared_encode and cls.encode is ModelAdapter.encode:
            msg = f"{cls.__name__} declares {declared_encode} in outputs but does not implement encode()"
            raise TypeError(msg)

        if "score" in spec.outputs:
            if cls.score is ModelAdapter.score and cls.score_pairs is ModelAdapter.score_pairs:
                msg = f"{cls.__name__} declares 'score' in outputs but does not implement score() or score_pairs()"
                raise TypeError(msg)

        if "json" in spec.outputs and cls.extract is ModelAdapter.extract:
            msg = f"{cls.__name__} declares 'json' in outputs but does not implement extract()"
            raise TypeError(msg)

    # -- Properties derived from spec ----------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        # spec stores Literal tuples; cast needed because list() widens type.
        return ModelCapabilities(
            inputs=cast("Any", list(self.spec.inputs)),
            outputs=cast("Any", list(self.spec.outputs)),
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims(
            dense=self.spec.dense_dim or getattr(self, "_dense_dim", None),
            sparse=self.spec.sparse_dim or getattr(self, "_sparse_dim", None),
            multivector=self.spec.multivector_dim or getattr(self, "_multivector_dim", None),
        )

    # -- Standard lifecycle --------------------------------------------------

    def unload(self) -> None:
        """Unload model weights and free device memory.

        Iterates ``spec.unload_fields`` and sets each to ``None``, then
        runs ``gc.collect()`` and clears the device cache.
        """
        device = getattr(self, "_device", None)

        for attr in self.spec.unload_fields:
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._device = None

        gc.collect()

        if device is not None:
            import torch as _torch

            if str(device).startswith("cuda"):
                _torch.cuda.empty_cache()
            elif str(device) == "mps":
                _torch.mps.empty_cache()

    def get_preprocessor(self) -> Any:
        """Return ``CharCountPreprocessor`` for cost estimation."""
        from sie_server.core.preprocessor import CharCountPreprocessor

        return CharCountPreprocessor(
            model_name=getattr(self, "_model_name_or_path", ""),
        )

    # -- Shared helpers ------------------------------------------------------

    def _check_loaded(self) -> None:
        """Raise ``RuntimeError`` if the model is not loaded."""
        if getattr(self, "_model", None) is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def _resolve_dtype(self) -> torch.dtype:
        """Map ``self._compute_precision`` to a ``torch.dtype``."""
        import torch as _torch

        dtype_map: dict[str, torch.dtype] = {
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        return dtype_map.get(
            getattr(self, "_compute_precision", "float16"),
            _torch.float16,
        )
