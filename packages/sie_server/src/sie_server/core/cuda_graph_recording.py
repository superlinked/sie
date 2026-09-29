"""One CUDA graph recording at a time, across every model in the process.

While a graph records, PyTorch's caching allocator will not free cached blocks to
satisfy another allocation, so another model on the same GPU can hit an
out-of-memory error it would otherwise have avoided. Graph runners take this lock
without blocking; a forward that finds another recording in progress runs eagerly.
"""

from __future__ import annotations

import threading

RECORDING_LOCK = threading.Lock()
