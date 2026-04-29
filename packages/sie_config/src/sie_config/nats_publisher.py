from __future__ import annotations

import logging
import os
import socket
from typing import cast

import nats
import orjson

from sie_config import metrics as sie_metrics

logger = logging.getLogger(__name__)

# Subject patterns
_BUNDLE_SUBJECT = "sie.config.models.{bundle_id}"
_ALL_SUBJECT = "sie.config.models._all"


class PartialPublishError(RuntimeError):
    """Raised when `publish_config_notification` failed for a subset of
    affected bundles.

    Some subscribers saw the delta and some didn't. The caller should
    treat this as a hard failure for the write (workers on the failed
    bundles will diverge until the next re-export), but the exception
    carries the specific bundles so operators can act on the warning
    without re-reading logs.
    """

    def __init__(self, model_id: str, epoch: int, failed_bundles: list[str], total: int) -> None:
        self.model_id = model_id
        self.epoch = epoch
        self.failed_bundles = list(failed_bundles)
        self.total = total
        super().__init__(
            f"partial NATS publish for model {model_id!r} (epoch {epoch}): "
            f"{len(failed_bundles)}/{total} bundles failed: {failed_bundles}"
        )


def _get_router_id() -> str:
    """Get unique publisher identifier from hostname or env.

    Prefers POD_NAME (unique per pod in K8s) over HOSTNAME
    (may be shared across containers in the same pod).
    Falls back to ``<hostname>-<pid>`` to ensure uniqueness
    in container environments where HOSTNAME is non-unique.
    """
    pod_name = os.environ.get("POD_NAME")
    if pod_name:
        return pod_name
    hostname = os.environ.get("HOSTNAME", socket.gethostname())
    # In containers, hostname is often a short random hex string
    # which is unique per container. Add PID as extra disambiguation.
    return f"{hostname}-{os.getpid()}"


class NatsPublisher:
    """NATS publish-only client for config distribution.

    Handles connection lifecycle and publishing config change notifications.
    Gracefully degrades when NATS is unavailable -- config mutations are blocked.

    Args:
        nats_url: NATS connection URL. Default: nats://localhost:4222
    """

    def __init__(self, nats_url: str | None = None) -> None:
        self._nats_url = nats_url or os.environ.get("SIE_NATS_URL", "nats://localhost:4222")
        self._nc: nats.NATS | None = None
        self._router_id = _get_router_id()
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether NATS connection is active."""
        return self._connected and self._nc is not None and self._nc.is_connected

    @property
    def router_id(self) -> str:
        """This publisher's unique identifier (kept as router_id for Rust gateway compat)."""
        return self._router_id

    async def connect(self) -> None:
        """Connect to NATS server.

        Does not raise on failure -- logs warning and sets connected=False.
        The config service can operate without NATS (config mutations blocked).
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
            sie_metrics.set_nats_connected(True)
            logger.info("Connected to NATS at %s (router_id=%s)", self._nats_url, self._router_id)

        except Exception:  # noqa: BLE001 -- graceful degradation when NATS unavailable
            self._connected = False
            sie_metrics.set_nats_connected(False)
            logger.warning(
                "Failed to connect to NATS at %s -- config mutations will be blocked",
                self._nats_url,
                exc_info=True,
            )

    async def disconnect(self) -> None:
        """Disconnect from NATS."""
        if self._nc:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                logger.debug("Failed to drain NATS connection", exc_info=True)
            self._nc = None

        self._connected = False
        sie_metrics.set_nats_connected(False)
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

        For each affected bundle, builds one canonical ``ConfigNotification``
        payload and publishes it to two subjects:

        1. ``sie.config.models.{bundle}`` -- consumed by workers
        2. ``sie.config.models._all`` -- consumed by Rust gateways

        The same payload shape goes to both subjects so every subscriber sees
        the full set of fields that ``sie_gateway`` expects
        (``router_id``, ``bundle_id``, ``epoch``, ``bundle_config_hash``,
        ``model_id``, ``profiles_added``, ``model_config``, ``affected_bundles``).
        See ``packages/sie_gateway/src/nats/manager.rs::ConfigNotification``.

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
            raise RuntimeError("NATS not connected -- cannot publish config notification")

        if not affected_bundles:
            logger.debug(
                "No affected bundles for model %s (epoch=%d); skipping publish",
                model_id,
                epoch,
            )
            return

        # Fail fast before publishing anything if any affected bundle is missing
        # its computed hash: wire payloads with an empty `bundle_config_hash`
        # can never match the hash workers/gateways compute locally, so silently
        # defaulting to "" would turn a bookkeeping bug into a permanent config
        # convergence failure.
        missing_hashes = [bundle_id for bundle_id in affected_bundles if bundle_id not in bundle_config_hashes]
        if missing_hashes:
            msg = f"Missing bundle_config_hash for affected bundle(s): {missing_hashes}"
            raise ValueError(msg)

        nc = cast("nats.NATS", self._nc)

        # Publish each bundle delta individually, collecting failures
        # rather than short-circuiting on the first exception. If we
        # aborted on the first failure, an operator reading the logs
        # would see a single "bundle-X publish failed" and have no way
        # to know whether bundles after X got the delta; here we
        # guarantee a deterministic partial state and a single summary
        # error that names every bundle that missed the delta.
        failed_bundles: list[str] = []
        for bundle_id in affected_bundles:
            payload = {
                "router_id": self._router_id,
                "bundle_id": bundle_id,
                "epoch": epoch,
                "bundle_config_hash": bundle_config_hashes[bundle_id],
                "model_id": model_id,
                "profiles_added": profiles_added,
                "model_config": model_config_yaml,
                "affected_bundles": affected_bundles,
            }
            encoded = orjson.dumps(payload)

            bundle_subject = _BUNDLE_SUBJECT.format(bundle_id=bundle_id)
            try:
                await nc.publish(bundle_subject, encoded)
                await nc.publish(_ALL_SUBJECT, encoded)
            except Exception:
                logger.exception(
                    "NATS publish failed for bundle=%s model=%s epoch=%d "
                    "(continuing to remaining bundles; operator will see a partial-publish warning)",
                    bundle_id,
                    model_id,
                    epoch,
                )
                failed_bundles.append(bundle_id)
                continue

            logger.debug(
                "Published config notification: bundle=%s model=%s epoch=%d",
                bundle_id,
                model_id,
                epoch,
            )

        if failed_bundles:
            # "partial" when at least one bundle succeeded, "failure"
            # when every bundle failed. Operators treat the two
            # differently: partial means the gateway poller will
            # close the gap; full means NATS is actually down for
            # this publisher.
            result = (
                sie_metrics.NATS_PUBLISH_PARTIAL
                if len(failed_bundles) < len(affected_bundles)
                else sie_metrics.NATS_PUBLISH_FAILURE
            )
            sie_metrics.record_nats_publish(result)
            raise PartialPublishError(
                model_id=model_id,
                epoch=epoch,
                failed_bundles=failed_bundles,
                total=len(affected_bundles),
            )

        sie_metrics.record_nats_publish(sie_metrics.NATS_PUBLISH_SUCCESS)

    async def _handle_reconnect(self) -> None:
        """Handle NATS reconnection."""
        self._connected = True
        sie_metrics.set_nats_connected(True)
        logger.info("Reconnected to NATS at %s", self._nats_url)

    async def _handle_disconnect(self) -> None:
        """Handle NATS disconnection."""
        self._connected = False
        sie_metrics.set_nats_connected(False)
        logger.warning("Disconnected from NATS -- config mutations blocked until reconnect")

    async def _handle_error(self, e: Exception) -> None:
        """Handle NATS errors."""
        logger.error("NATS error: %s", e)
