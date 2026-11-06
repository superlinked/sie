"""Prefix-block attention over the shared packed Qwen3.5 kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from sie_server.adapters._qwen3_5_packed import PackedKernels, PackedTextModel, Packing, to_device


@dataclass(frozen=True)
class DecisionPacking(Packing):
    prefix_indices: torch.Tensor
    suffix_indices: torch.Tensor
    prefix_layout: Packing
    suffix_layout: Packing

    @classmethod
    def from_bounds(cls, width: int, bounds: tuple[tuple[int, int], ...], device: torch.device) -> DecisionPacking:
        if not bounds or any(not 0 <= first < end < width for first, end in bounds):
            raise ValueError("Packed decisions need nonempty prefix and suffix blocks")
        lengths = [width - first for first, _ in bounds]
        prefixes = [end - first for first, end in bounds]
        base = Packing.from_lengths(lengths, device)
        left, right = [], []
        for (first, end), prefix in zip(base.segments(), prefixes, strict=True):
            left.extend(range(first, first + prefix))
            right.extend(range(first + prefix, end))
        return cls(
            **vars(base),
            prefix_indices=to_device(torch.tensor(left), device),
            suffix_indices=to_device(torch.tensor(right), device),
            prefix_layout=Packing.from_lengths(prefixes, device),
            suffix_layout=Packing.from_lengths([width - end for _, end in bounds], device),
        )


def block_attention(attention: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def forward(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, packing: DecisionPacking, scale: float
    ) -> torch.Tensor:
        prefix, suffix = packing.prefix_indices, packing.suffix_indices
        left, right = packing.prefix_layout, packing.suffix_layout
        prefix_output = attention(
            query[prefix],
            key[prefix],
            value[prefix],
            left.cu_seqlens_int32,
            left.cu_seqlens_int32,
            left.max_len,
            left.max_len,
            softmax_scale=scale,
            causal=False,
        )
        suffix_output = attention(
            query[suffix],
            key,
            value,
            right.cu_seqlens_int32,
            packing.cu_seqlens_int32,
            right.max_len,
            packing.max_len,
            softmax_scale=scale,
            causal=False,
        )
        result = torch.empty_like(query)
        result[prefix], result[suffix] = prefix_output, suffix_output
        return result

    return forward


class PackedDecisionReadout(torch.nn.Module):
    def __init__(
        self, backbone: Any, readout: torch.nn.Linear, kernels: PackedKernels, attention: Callable[..., torch.Tensor]
    ) -> None:
        super().__init__()
        self.backbone, self.readout = backbone, readout
        self.packed = PackedTextModel(backbone, replace(kernels, attention=block_attention(attention)))
        self.requires_grad_(False).eval()

    def forward(self, ids: torch.Tensor, present: torch.Tensor, bounds: tuple[tuple[int, int], ...]) -> torch.Tensor:
        _ = present
        layout = DecisionPacking.from_bounds(ids.shape[1], bounds, ids.device)
        tokens = torch.cat([ids[row, first:] for row, (first, _) in enumerate(bounds)])[None]
        positions = torch.cat([torch.arange(ids.shape[1] - first, device=ids.device) for first, _ in bounds])[
            None, None
        ].expand(3, 1, -1)
        hidden = self.packed(self.backbone.embed_tokens(tokens), positions, layout)
        selected = hidden[0, layout.cu_seqlens[1:] - 1]
        logits = self.readout(selected.to(self.readout.weight.dtype)).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Decision head returned nonfinite logits")
        return logits
