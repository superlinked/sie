from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable, Coroutine
from typing import Any, cast

import nats
import orjson

logger = logging.getLogger(__name__)

# Subject patterns
_BUNDLE_SUBJECT = "sie.config.models.{bundle_id}"
_ALL_SUBJECT = "sie.config.models._all"


def _get_router_id() -> str:
    """Get unique router identifier from hostname or env.

    Prefers POD_NAME (unique per pod in K8s) over HOSTNAME
    (may be shared across containers in the same pod).
    Falls back to a UUID suffix on the hostname to ensure
    uniqueness in container environments.
    """
    pod_name = os.environ.get("POD_NAME")
    if pod_name:
        return pod_name
    hostname = os.environ.get("HOSTNAME", socket.gethostname())
    # In containers, hostname is often a short random hex string
    # which is unique per container. Add PID as extra disambiguation.
    return f"{hostname}-{os.getpid()}"


class NatsManager:
    """NATS connection manager for config distribution.

    Handles connection lifecycle, publishing, and subscription.
    Gracefully degrades when NATS is unavailable — inference continues,
    config mutations are blocked.

    Args:
        nats_url: NATS connection URL. Default: nats://localhost:4222
        on_config_notification: Async callback invoked when a config
            notification is received from another router via _all subject.
            Signature: async def callback(notification: dict) -> None
        on_reconnect: Async callback invoked on NATS reconnect for
            epoch reconciliation. Signature: async def callback() -> None
    """

    def __init__(
        self,
        nats_url: str | None = None,
        on_config_notification: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        on_reconnect: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._nats_url = nats_url or os.environ.get("SIE_NATS_URL", "nats://localhost:4222")
        self._on_config_notification = on_config_notification
        self._on_reconnect = on_reconnect
        self._nc: nats.NATS | None = None
        self._sub: Any = None  # subscription to _all subject
        self._router_id = _get_router_id()
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether NATS connection is active."""
        return self._connected and self._nc is not None and self._nc.is_connected

    @property
    def router_id(self) -> str:
        """This router's unique identifier."""
        return self._router_id

    @property
    def nc(self) -> nats.NATS | None:
        """The underlying NATS connection, or None if not connected."""
        return self._nc if self.connected else None

    def add_reconnect_handler(self, handler: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register an additional handler to run after NATS reconnect.

        The handler is chained with any existing reconnect callback.
        """
        prev = self._on_reconnect

        async def _chained() -> None:
            if prev is not None:
                try:
                    await prev()
                except Exception:
                    logger.exception("Previous reconnect handler failed")
            await handler()

        self._on_reconnect = _chained

    async def connect(self) -> None:
        """Connect to NATS server.

        Does not raise on failure — logs warning and sets connected=False.
        The router can operate without NATS (config mutations blocked).
        """
        try:
            self._nc = await nats.connect(
                self._nats_url,
                reconnected_cb=self._handle_reconnect,
                disconnected_cb=self._handle_disconnect,
                error_cb=self._handle_error,
                max_reconnect_attempts=-1,  # Reconnect forever
                reconnect_time_wait=2,  # 2s between reconnect attempts
            )
            self._connected = True
            logger.info("Connected to NATS at %s (router_id=%s)", self._nats_url, self._router_id)

            # Subscribe to global config notifications
            if self._on_config_notification:
                self._sub = await self._nc.subscribe(
                    _ALL_SUBJECT,
                    cb=self._handle_all_notification,
                )
                logger.info("Subscribed to %s", _ALL_SUBJECT)

        except Exception:  # noqa: BLE001 — graceful degradation when NATS unavailable
            self._connected = False
            logger.warning(
                "Failed to connect to NATS at %s — config mutations will be blocked",
                self._nats_url,
                exc_info=True,
            )

    async def disconnect(self) -> None:
        """Disconnect from NATS."""
        if self._sub:
            try:
                await self._sub.unsubscribe()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to unsubscribe from NATS", exc_info=True)
            self._sub = None

        if self._nc:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to drain NATS connection", exc_info=True)
            self._nc = None

        self._connected = False
        logger.info("Disconnected from NATS")

    async def publish_config_notification(
        self,
        *,
        model_id: str,
        profiles_added: list[str],
        affected_bundles: list[str],
        bundle_config_hashes: dict[str, str],
        epoch: int,
        model_config_yaml: str,
    ) -> None:
        """Publish config change notifications to NATS.

        Publishes to:
        1. Per-bundle subjects for each affected bundle (workers subscribe)
        2. Global _all subject (other routers subscribe)

        Args:
            model_id: The model that was added/updated.
            profiles_added: List of profile names that were created.
            affected_bundles: List of bundle IDs whose adapter list matches.
            bundle_config_hashes: Dict of bundle_id -> new hash for each affected bundle.
            epoch: The new epoch value after this mutation.
            model_config_yaml: Full model config YAML content.

        Raises:
            RuntimeError: If NATS is not connected.
        """
        if not self.connected:
            raise RuntimeError("NATS not connected — cannot publish config notification")

        base_payload = {
            "model_id": model_id,
            "profiles_added": profiles_added,
            "epoch": epoch,
            "router_id": self._router_id,
            "model_config": model_config_yaml,
        }

        # Publish to per-bundle subjects
        for bundle_id in affected_bundles:
            subject = _BUNDLE_SUBJECT.format(bundle_id=bundle_id)
            payload = {
                **base_payload,
                "bundle_config_hash": bundle_config_hashes.get(bundle_id, ""),
            }
            await cast("nats.NATS", self._nc).publish(subject, orjson.dumps(payload))
            logger.debug("Published to %s: model=%s, epoch=%d", subject, model_id, epoch)

        # Publish to global subject
        all_payload = {
            **base_payload,
            "affected_bundles": affected_bundles,
        }
        await cast("nats.NATS", self._nc).publish(_ALL_SUBJECT, orjson.dumps(all_payload))
        logger.debug("Published to %s: model=%s, epoch=%d", _ALL_SUBJECT, model_id, epoch)

    async def _handle_all_notification(self, msg: Any) -> None:
        """Handle incoming notification on _all subject."""
        try:
            data = orjson.loads(msg.data)

            if not isinstance(data, dict):
                logger.warning("Dropping non-dict NATS notification: %s", type(data).__name__)
                return

            # Skip notifications from ourselves
            if data.get("router_id") == self._router_id:
                return

            logger.info(
                "Received config notification from router %s: model=%s, epoch=%d",
                data.get("router_id"),
                data.get("model_id"),
                data.get("epoch", 0),
            )

            if self._on_config_notification:
                await self._on_config_notification(data)

        except orjson.JSONDecodeError:
            logger.warning("Dropping malformed NATS notification (invalid JSON)")
        except Exception:
            logger.exception("Failed to handle config notification")

    async def _handle_reconnect(self) -> None:
        """Handle NATS reconnection — trigger epoch reconciliation."""
        self._connected = True
        logger.info("Reconnected to NATS at %s", self._nats_url)

        if self._on_reconnect:
            try:
                await self._on_reconnect()
            except Exception:
                logger.exception("NATS reconnect callback failed")

    async def _handle_disconnect(self) -> None:
        """Handle NATS disconnection."""
        self._connected = False
        logger.warning("Disconnected from NATS — config mutations blocked until reconnect")

    async def _handle_error(self, e: Exception) -> None:
        """Handle NATS errors."""
        logger.error("NATS error: %s", e)
