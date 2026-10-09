"""Turn pooled ``[batch, vocab]`` sparse weights into per-row ``SparseVector`` outputs.

The rows are compacted on the device: only the nonzero entries cross to the
host, in one copy, instead of the whole ``[batch, vocab]`` matrix followed by a
per-row ``numpy.where``. The result is identical to that path: each row's
indices are ascending ``int32`` and its values are the ``float32`` weights.
"""

from __future__ import annotations

import numpy as np
import torch

from sie_server.core.inference_output import SparseVector


def sparse_rows(weights: torch.Tensor) -> list[SparseVector]:
    """The strictly positive entries of each row of ``weights`` (``[batch, vocab]``)."""
    if weights.shape[0] == 0:
        return []
    mask = weights > 0
    counts = mask.sum(dim=1)
    # Boolean indexing and nonzero() both walk the matrix in row-major order,
    # so columns and values line up and each row's columns come out ascending.
    columns = mask.nonzero()[:, 1].to(torch.int32)
    values = weights[mask].float()
    counts_host = counts.cpu().numpy()
    columns_host = columns.cpu().numpy()
    values_host = values.cpu().numpy()
    bounds = np.cumsum(counts_host)[:-1]
    return [
        SparseVector(indices=indices, values=row_values)
        for indices, row_values in zip(np.split(columns_host, bounds), np.split(values_host, bounds), strict=True)
    ]
