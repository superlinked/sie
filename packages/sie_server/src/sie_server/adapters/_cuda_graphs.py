"""Process-wide pieces shared by the CUDA graph runners.

A runner records the kernel launches of a forward once per input shape and
replays them with one call. Three runners exist: the GLiClass DeBERTa encoder
(``gliclass/cuda_graphs.py``), the ModernBERT flash-attention varlen encoders
(``_modernbert_flash_graphs.py``) and TopK-Embed's text model
(``topk_embed/graphs.py``). They record different forwards,
but they share one device and one caching allocator, so these rules hold
across every runner in the process:

- at most one recording at a time (``RECORDING_LOCK``): while a graph is
  being recorded, PyTorch's caching allocator will not free cached blocks to
  satisfy another allocation, so another model on the same GPU could run out
  of memory;
- no recording while less than a tenth of the device's memory is free;
- a runner's graphs hold at most 4% of the device's memory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch

# One recording at a time in the process, across every model's runner.
RECORDING_LOCK = threading.Lock()
# Recording waits until at least this share of the device's memory is free.
RECORDING_HEADROOM = 0.1
# Device memory a runner's graphs may hold, as a share of the device's memory.
MEMORY_BUDGET_SHARE = 0.04


@dataclass
class CudaGraphStats:
    """Counters of the forwards a runner was offered since the model loaded.

    Each forward counts once: ``replayed`` (a recorded graph served it),
    ``recorded`` (it recorded a graph and was answered by the first replay)
    or under ``eager`` by why it ran eagerly. Requests that opt out of graphs
    are not offered to the runner. ``recording_failures`` counts shapes that
    failed to record for reasons other than memory; ``drops`` counts the times
    every graph was dropped (out of memory, or over the memory budget).
    """

    replayed: int = 0
    recorded: int = 0
    eager: dict[str, int] = field(default_factory=dict)
    recording_failures: int = 0
    drops: int = 0

    @property
    def forwards(self) -> int:
        return self.replayed + self.recorded + sum(self.eager.values())


def free_memory(device: torch.device) -> int:
    """Free device memory in bytes."""
    return torch.cuda.mem_get_info(device)[0]


def has_headroom(device: torch.device) -> bool:
    """Whether enough device memory is free to record without starving other models."""
    free, total = torch.cuda.mem_get_info(device)
    return free >= RECORDING_HEADROOM * total


def memory_budget(device: torch.device) -> int:
    """Device memory one runner's graphs may hold."""
    return int(MEMORY_BUDGET_SHARE * torch.cuda.mem_get_info(device)[1])
