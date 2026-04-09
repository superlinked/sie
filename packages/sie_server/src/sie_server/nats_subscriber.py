from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import nats

logger = logging.getLogger(__name__)

_BUNDLE_SUBJECT = "sie.config.models.{bundle_id}"

# Characters that are invalid in NATS subject tokens
_NATS_WILDCARD_CHARS = frozenset({"*", ">", " "})


def _validate_bundle_id(bundle_id: str) -> str:
    """Validate bundle_id does not contain NATS wildcard characters.

    NATS subjects use '*' and '>' as wildcards, and '.' as a token
    separator. If bundle_id contains these, subscriptions would match
    unintended subjects.

    Args:
        bundle_id: The bundle identifier to validate.

    Returns:
        The validated bundle_id.

    Raises:
        ValueError: If bundle_id contains wildcard characters.
    """
    invalid = _NATS_WILDCARD_CHARS & set(bundle_id)
    if invalid:
        msg = f"bundle_id contains invalid NATS characters: {invalid!r}"
        raise ValueError(msg)
    if "." in bundle_id:
        msg = "bundle_id contains '.' which is a NATS subject token separator"
        raise ValueError(msg)
    return bundle_id


class NatsSubscriber:
    """Worker-side NATS subscriber for config notifications.

    Subscribes to the bundle-scoped subject and applies config changes
    to the worker's model registry.

    Args:
        bundle_id: This worker's bundle identifier.
        nats_url: NATS connection URL.
        on_model_config: Async callback when a new model config is received.
            Signature: async def callback(model_id: str, config_yaml: str) -> None
            The callback should parse the YAML and add the config to the
            worker's model registry.
    """

    def __repr__(self) -> str:
        return (
            f"NatsSubscriber(bundle_id={self._bundle_id!r}, nats_url={self._nats_url!r}, connected={self._connected})"
        )

    def __init__(
        self,
        bundle_id: str | None = None,
        nats_url: str | None = None,
        on_model_config: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._bundle_id = _validate_bundle_id(bundle_id or os.environ.get("SIE_BUNDLE", "default"))
        self._nats_url = nats_url or os.environ.get("SIE_NATS_URL", "nats://localhost:4222")
        self._on_model_config: Callable[[str, str], Awaitable[None]] | None = on_model_config
        self._model_epochs: dict[str, int] = {}
        self._nc: Any = None
        self._sub: Any = None
        self._connected = False
        self._extra_reconnect_handlers: list[Callable[[], Awaitable[None]]] = []

    @property
    def connected(self) -> bool:
        """Whether NATS connection is active."""
        return self._connected and self._nc is not None and self._nc.is_connected

    @property
    def nc(self) -> Any:
        """The underlying NATS connection, or None if not connected."""
        return self._nc

    def add_reconnect_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register an additional handler to run after NATS reconnect.

        Handlers are invoked in registration order after the subscriber's
        own reconnect logic (clearing epoch watermarks).
        """
        self._extra_reconnect_handlers.append(handler)

    async def start(self) -> None:
        """Connect to NATS and subscribe to bundle subject.

        Does not raise on failure — worker operates without NATS
        (config changes from API won't be received until reconnect).
        """
        try:
            self._nc = await nats.connect(
                self._nats_url,
                reconnected_cb=self._handle_reconnect,
                disconnected_cb=self._handle_disconnect,
                error_cb=self._handle_error,
                max_reconnect_attempts=-1,
                reconnect_time_wait=2,
            )
            self._connected = True

            subject = _BUNDLE_SUBJECT.format(bundle_id=self._bundle_id)
            self._sub = await self._nc.subscribe(subject, cb=self._handle_notification)

            logger.info(
                "NATS subscriber started: bundle=%s, subject=%s, url=%s",
                self._bundle_id,
                subject,
                self._nats_url,
            )

        except Exception:  # noqa: BLE001 — graceful degradation when NATS unavailable
            self._connected = False
            logger.warning(
                "Failed to connect to NATS at %s — worker will not receive config notifications",
                self._nats_url,
                exc_info=True,
            )

    async def stop(self) -> None:
        """Disconnect from NATS."""
        if self._nc:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to drain NATS connection", exc_info=True)
            self._nc = None
        self._sub = None
        self._connected = False
        logger.info("NATS subscriber stopped")

    async def _handle_notification(self, msg: Any) -> None:
        """Handle incoming config notification on bundle subject."""
        try:
            data = json.loads(msg.data.decode())

            if not isinstance(data, dict):
                logger.warning("Dropping non-dict NATS notification: %s", type(data).__name__)
                return

            model_id = data.get("model_id", "")
            profiles_added = data.get("profiles_added", [])
            model_config_yaml = data.get("model_config", "")
            epoch = data.get("epoch", 0)
            router_id = data.get("router_id", "unknown")

            logger.info(
                "Received config notification: model=%s, profiles=%s, epoch=%d, from=%s",
                model_id,
                profiles_added,
                epoch,
                router_id,
            )

            # Epoch high-water-mark: reject stale notifications that arrive
            # out of order. Since add_model_config is append-only (profiles
            # cannot be removed), this only prevents redundant re-application.
            if self._on_model_config and model_config_yaml:
                last_epoch = self._model_epochs.get(model_id, -1)
                if epoch <= last_epoch:
                    logger.debug(
                        "Dropping stale notification for %s: epoch %d <= %d",
                        model_id,
                        epoch,
                        last_epoch,
                    )
                    return
                try:
                    await self._on_model_config(model_id, model_config_yaml)
                except Exception:
                    logger.exception(
                        "Failed to apply config for model %s (epoch %d) — will retry on next notification",
                        model_id,
                        epoch,
                    )
                    return
                self._model_epochs[model_id] = epoch

        except json.JSONDecodeError:
            logger.warning("Dropping malformed NATS notification (invalid JSON)")
        except Exception:
            logger.exception("Failed to handle config notification")

    async def _handle_reconnect(self) -> None:
        """Handle NATS reconnection.

        Clears per-model epoch watermarks so notifications received after
        reconnect are not rejected as stale.  Workers do not pull config
        from the object store (design non-goal), so any notifications
        missed during the disconnect are lost until the next config
        mutation or a manual resync.

        Also invokes any registered extra reconnect handlers (e.g.,
        NatsPullLoop.handle_reconnect).
        """
        stale_count = len(self._model_epochs)
        self._model_epochs.clear()
        logger.info(
            "NATS subscriber reconnected — cleared %d model epoch watermarks; "
            "notifications missed during disconnect will not be recovered automatically",
            stale_count,
        )
        self._connected = True
        for handler in self._extra_reconnect_handlers:
            try:
                await handler()
            except Exception:
                logger.exception("Extra reconnect handler failed")
                raise

    async def _handle_disconnect(self) -> None:
        """Handle NATS disconnection."""
        self._connected = False
        logger.warning("NATS subscriber disconnected — config notifications paused")

    async def _handle_error(self, e: Exception) -> None:
        """Handle NATS errors."""
        logger.error("NATS subscriber error: %s", e)
