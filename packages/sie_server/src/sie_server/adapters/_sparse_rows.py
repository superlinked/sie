"""Turn pooled ``[batch, vocab]`` sparse weights into per-row ``SparseVector`` outputs.

The rows are compacted on the device, so only the per-row counts and the
nonzero entries cross to the host, instead of the whole ``[batch, vocab]``
matrix followed by a per-row ``numpy.where``. (``nonzero()`` synchronizes with
the device, since its output size depends on the data; copying to the host
would anyway.) The result is identical to that path: each row's
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
    # nonzero() walks the matrix in row-major order, so each row's columns come
    # out ascending; the values are gathered at the same coordinates.
    rows, cols = mask.nonzero(as_tuple=True)
    columns = cols.to(torch.int32)
    values = weights[rows, cols].float()
    counts_host = counts.cpu().numpy()
    columns_host = columns.cpu().numpy()
    values_host = values.cpu().numpy()
    bounds = np.cumsum(counts_host)[:-1]
    return [
        SparseVector(indices=indices, values=row_values)
        for indices, row_values in zip(np.split(columns_host, bounds), np.split(values_host, bounds), strict=True)
    ]
