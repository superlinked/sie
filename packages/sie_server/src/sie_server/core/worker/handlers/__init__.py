"""Operation handlers for model worker.

Each handler encapsulates the operation-specific logic for:
- Creating config keys for batching
- Running inference via the adapter
- Slicing and assembling outputs

This enables the ModelWorker to remain simple while handlers
contain the complexity of each operation type.
"""

from sie_server.core.worker.handlers.base import OperationHandler, make_hashable
from sie_server.core.worker.handlers.encode import EncodeHandler
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.core.worker.handlers.score import ScoreHandler

__all__ = [
    "EncodeHandler",
    "ExtractHandler",
    "OperationHandler",
    "ScoreHandler",
    "make_hashable",
]
