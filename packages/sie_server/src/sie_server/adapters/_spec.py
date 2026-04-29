from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InputModality = Literal["text", "image", "audio", "video", "document"]
OutputType = Literal["dense", "sparse", "multivector", "score", "json"]


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Class-level adapter contract declaration.

    Every concrete adapter declares a ``spec`` ClassVar. The spec is the
    source of truth for what the adapter supports. The existing
    ``capabilities`` and ``dims`` properties on ``BaseAdapter`` read from
    spec, preserving backward compatibility for callers.

    Attributes:
        inputs: Input modalities (e.g. ``("text",)``, ``("text", "image")``).
        outputs: Output types (e.g. ``("dense",)``, ``("score",)``).
        dense_dim: Static dense vector dimensionality, or ``None`` if
            discovered at load time.
        sparse_dim: Sparse vector vocabulary size, or ``None``.
        multivector_dim: Per-token dimension for multi-vector, or ``None``.
        default_preprocessor: Preprocessor type (``"charcount"`` or
            ``"image"``). Adapters returning a custom preprocessor from
            ``get_preprocessor()`` should still set this for contract tests.
        unload_fields: Instance attributes to set to ``None`` during
            ``unload()``. Every adapter must declare the fields it owns.
    """

    inputs: tuple[InputModality, ...]
    outputs: tuple[OutputType, ...]
    dense_dim: int | None = None
    sparse_dim: int | None = None
    multivector_dim: int | None = None
    default_preprocessor: str = "charcount"
    unload_fields: tuple[str, ...] = ("_model",)
