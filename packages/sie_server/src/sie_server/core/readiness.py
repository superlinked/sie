"""Shared readiness state for worker.

This module provides a single source of truth for worker readiness, used by:
- /readyz endpoint (K8s readiness probe)
- /ws/status WebSocket (gateway health check via `ready` field)

The gateway only routes requests to workers that report ready=True.
This prevents routing to workers that are still starting up.

Usage:
    from sie_server.core.readiness import is_ready, mark_ready, mark_not_ready

    # In lifespan:
    mark_ready()   # After startup completes
    yield
    mark_not_ready()  # During shutdown
"""

import logging

logger = logging.getLogger(__name__)

_ready = False


def is_ready() -> bool:
    """Check if worker is ready to accept traffic.

    Returns:
        True if the worker has completed startup and can handle requests.
    """
    return _ready


def mark_ready() -> None:
    """Mark worker as ready for traffic.

    Called by lifespan after all startup tasks complete (before yield).
    """
    global _ready
    logger.info("Worker is ready for traffic")
    _ready = True


def mark_not_ready() -> None:
    """Mark worker as not ready for traffic.

    Called by lifespan during shutdown (after yield).
    """
    global _ready
    logger.info("Worker shutting down, no longer accepting traffic")
    _ready = False
