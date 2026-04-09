from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import nats
import nats.errors
import nats.js.errors
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from sie_sdk.queue_types import (
    WORK_SUBJECT_PREFIX,
    WorkItem,
    WorkResult,
    denormalize_model_id,
    work_consumer_name,
    work_pool_stream_name,
    work_pool_stream_subjects,
    work_stream_name,
)

from sie_server.core.adaptive_batching import LatencyTracker
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.prepared import ExtractPreparedItem, ScorePreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import Item

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

# Default ACK wait in seconds — nats.py converts to nanoseconds internally.
_ACK_WAIT_S = 30

# Default batch budget — controls how many NATS messages a single fetch()
# returns.  This is the primary driver of GPU batch size in queue mode:
# the pull loop groups fetched messages by model and dispatches them as a
# single GPU forward pass.  Configurable via SIE_NATS_FETCH_BUDGET.
_DEFAULT_BATCH_BUDGET = int(os.environ.get("SIE_NATS_FETCH_BUDGET", "64"))

# Maximum concurrent batch-processing tasks to avoid ACK timeout storms.
_MAX_CONCURRENT_BATCHES = 4

# TTL (seconds) for caching the bundle config hash.
_CONFIG_HASH_CACHE_TTL_S = 5.0

# Interval (seconds) between retries for models whose initial subscription failed.
_FAILED_MODEL_RETRY_INTERVAL = 30.0

# Graceful drain timeout (seconds) — matches _ACK_WAIT_S so in-flight batches
# have time to finish and ACK before the pull loop shuts down.
_DRAIN_TIMEOUT_S = 30.0

# NAK delay (seconds) for items targeting unloaded models.
# JetStream redelivers after this delay, giving time for loading to complete.
# Configurable via SIE_NAK_DELAY_S; with max_deliver=20, the default gives
# ~100s total retry budget — enough for cold model downloads.
_NAK_DELAY_S = float(os.environ.get("SIE_NAK_DELAY_S", "5.0"))

# Maximum delivery attempts before a message is sent to the DLQ.
# Configurable via SIE_MAX_DELIVER; must be high enough to cover model load
# times (10-60s+ for cold HuggingFace downloads).
_MAX_DELIVER = int(os.environ.get("SIE_MAX_DELIVER", "20"))

_MIN_SUBJECT_PARTS = 4  # sie.work.{model}.{pool}

# Dynamic fetch timeout bounds.  With the pool-level stream design (single
# subscription per pool), idle polling cost is just one NATS fetch RPC per
# cycle — much lower than the old O(N-models) design.  The max can be kept
# very tight to minimise head-of-line delay at low concurrency.
#
# At 1ms minimum, the worker checks for new messages every 1ms when busy.
# At 20ms maximum, the worst-case idle delay is 20ms (avg ~10ms).
# This translates to ~50 idle polls/sec to NATS — negligible load.
_MIN_FETCH_TIMEOUT_S = 0.001  # 1ms  — near-instant when messages flow
_MAX_FETCH_TIMEOUT_S = 0.02  # 20ms — tight idle delay
_BACKOFF_GROWTH = 2.0  # 1ms → 2ms → 4ms → 8ms → 16ms → 20ms (5 steps)

# Thrashing detection: if a model is background-loaded this many times
# within this window, log a warning suggesting separate bundles.
_THRASH_WINDOW_S = 300.0  # 5 minutes
_THRASH_THRESHOLD = 4  # 4 loads in 5 minutes = thrashing

# Max age (seconds) for messages in the pool-level stream.
# Configurable via SIE_STREAM_MAX_AGE_S to give headroom for slow model loads.
_DEFAULT_STREAM_MAX_AGE_S = int(os.environ.get("SIE_STREAM_MAX_AGE_S", "120"))

# Thread pool for S3 payload fetches — avoids exhausting the default
# asyncio thread pool when fetching many large payloads concurrently.
_PAYLOAD_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="payload-fetch")


# Map numpy dtype to the wire-format dtype string expected by the SDK.
_NP_DTYPE_MAP = {"float32": "float32", "float16": "float16", "int8": "int8", "uint8": "binary"}


def _wrap_encode_output(output: dict, config: Any) -> dict:
    """Wrap raw numpy arrays from ``EncodeHandler.format_output`` into the
    ``DenseVector``/``SparseVector`` wire format that the SDK expects.

    The HTTP path does this via ``encode.py:_build_response_items`` + Pydantic
    ``EncodeResult``.  In the queue path the worker must produce the same shape
    *before* msgpack-serializing, because the router embeds the blob as-is.
    """
    import numpy as np  # noqa: PLC0415

    wrapped = dict(output)

    if "dense" in wrapped and isinstance(wrapped["dense"], np.ndarray):
        arr = wrapped["dense"]
        encode_task = getattr(config, "tasks", None)
        encode_task = getattr(encode_task, "encode", None)
        dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
        dense_dim = dense_cfg.dim if dense_cfg else None

        is_binary = arr.dtype == np.uint8 and dense_dim and arr.shape[0] < dense_dim
        dims = dense_dim if dense_dim is not None else arr.shape[0]
        dtype = "binary" if is_binary else _NP_DTYPE_MAP.get(str(arr.dtype), "float32")

        wrapped["dense"] = {"dims": int(dims), "dtype": dtype, "values": arr}

    if "sparse" in wrapped and isinstance(wrapped["sparse"], dict):
        # sparse already comes as {"indices": ndarray, "values": ndarray}
        pass

    if "multivector" in wrapped and isinstance(wrapped["multivector"], np.ndarray):
        arr = wrapped["multivector"]
        wrapped["multivector"] = {"values": arr}

    return wrapped


# ---------------------------------------------------------------------------
# Metrics (optional — gracefully degrade if prometheus_client not installed)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram

    PULL_ITEMS_FETCHED = Histogram(
        "sie_pull_loop_items_fetched",
        "Number of items fetched per pull cycle",
        ["model"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    PULL_BATCH_PROCESS_SECONDS = Histogram(
        "sie_pull_loop_batch_process_seconds",
        "Time to process a pulled batch (seconds)",
        ["model", "operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    PULL_QUEUE_WAIT_SECONDS = Histogram(
        "sie_pull_loop_queue_wait_seconds",
        "Time items waited in NATS queue before being pulled (seconds)",
        ["model"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    PULL_CONFIG_HASH_MISMATCHES = Counter(
        "sie_pull_loop_config_hash_mismatches_total",
        "Config hash mismatches detected (log-only, items still processed)",
        ["model"],
    )
    PULL_NAK_UNLOADED = Counter(
        "sie_pull_loop_nak_unloaded_total",
        "Work items NAKed because the target model is not loaded",
        ["model"],
    )
    PULL_MODEL_LOADS = Counter(
        "sie_pull_loop_model_loads_total",
        "Background model loads triggered by demand",
        ["model"],
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


# ---------------------------------------------------------------------------
# Minimal payload store (read-only) — avoids cross-package import from
# ``sie_router.payload_store``.
# ---------------------------------------------------------------------------


class _PayloadStore:
    """Read-only payload fetcher (local filesystem or S3)."""

    async def get(self, key: str) -> bytes:
        raise NotImplementedError


class _LocalPayloadStore(_PayloadStore):
    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)

    def _safe_path(self, key: str) -> Path:
        """Resolve key to a path inside base_dir, rejecting traversal."""
        target = (self._base_dir / key).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Path traversal detected: {key}") from None
        return target

    async def get(self, key: str) -> bytes:
        path = self._safe_path(key)
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(_PAYLOAD_THREAD_POOL, path.read_bytes)
        except FileNotFoundError:
            raise KeyError(f"Payload not found: {key}") from None


class _S3PayloadStore(_PayloadStore):
    def __init__(self, bucket: str, prefix: str = "payloads") -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._client: object | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> object:
        """Get or create a cached boto3 S3 client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3  # noqa: PLC0415

                    self._client = boto3.client("s3")
        return self._client

    async def get(self, key: str) -> bytes:
        full_key = f"{self._prefix}/{key}" if self._prefix else key

        def _fetch() -> bytes:
            client = self._get_client()
            response = client.get_object(Bucket=self._bucket, Key=full_key)  # type: ignore[union-attr]
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_PAYLOAD_THREAD_POOL, _fetch)
        except KeyError:
            raise
        except Exception as e:
            import botocore.exceptions  # noqa: PLC0415 — optional dep, lazy import

            if isinstance(e, botocore.exceptions.ClientError):
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise KeyError(f"Payload not found: {key}") from e
            raise


class _GCSPayloadStore(_PayloadStore):
    """Read-only GCS payload store for the worker pull loop."""

    def __init__(self, bucket: str, prefix: str) -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        self._client: Any = None

    def _get_bucket(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage  # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS payload stores. Install the google-cloud-storage package"
                ) from None
            client = storage.Client()
            self._client = client.bucket(self._bucket_name)
        return self._client

    async def get(self, key: str) -> bytes:
        bucket = self._get_bucket()
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        blob = bucket.blob(full_key)
        try:
            data: bytes = await asyncio.to_thread(blob.download_as_bytes)
            return data
        except Exception as e:
            _not_found = None
            with contextlib.suppress(ImportError):
                from google.api_core.exceptions import NotFound  # noqa: PLC0415

                _not_found = NotFound
            if _not_found is not None and isinstance(e, _not_found):
                raise KeyError(f"Payload not found: {key}") from e
            raise


def _create_payload_store(url: str | None) -> _PayloadStore | None:
    """Create a read-only payload store from a URL."""
    if not url:
        return None
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return _S3PayloadStore(bucket=bucket, prefix=prefix)
    if url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return _GCSPayloadStore(bucket=bucket, prefix=prefix)
    if "://" in url:
        raise ValueError(
            f"Unsupported payload store URL scheme: {url!r}. "
            "Supported: 's3://bucket/prefix', 'gs://bucket/prefix', or a local filesystem path."
        )
    return _LocalPayloadStore(base_dir=url)


class NatsPullLoop:
    """Pull work items from NATS JetStream and feed to ModelWorker.

    One pull loop per worker process. The loop round-robins across models
    in the worker's bundle, pulling items from each model's JetStream
    subject and submitting them to the appropriate ModelWorker.
    """

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        registry: ModelRegistry,
        bundle_id: str,
        pool_name: str,
        payload_store_url: str | None = None,
    ) -> None:
        self._nc = nc
        self._js = js
        self._registry = registry
        self._bundle_id = bundle_id
        self._pool_name = pool_name
        self._payload_store_url = payload_store_url
        self._subscriptions: dict[str, Any] = {}  # model_id → PullSubscription
        self._running = False
        self._pull_task: asyncio.Task[None] | None = None
        self._in_flight_tasks: set[asyncio.Task[None]] = set()
        self._payload_store: _PayloadStore | None = None
        self._worker_id = os.environ.get("HOSTNAME", os.environ.get("POD_NAME", "worker"))
        self._failed_models: set[str] = set()
        self._config_hash_cache: str | None = None
        self._config_hash_cache_time: float = 0.0
        self._batch_sem = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)

        # Reactive model loading state
        self._loading_models: set[str] = set()  # Models currently being loaded
        self._load_tasks: dict[str, asyncio.Task[None]] = {}  # model → background load task
        self._load_history: list[tuple[str, float]] = []  # (model_id, timestamp) for thrash detection

        # Adaptive fetch timeout: tracks queue-path latency (time from
        # router publish to result publish) and adjusts fetch_timeout to
        # balance item accumulation (throughput) vs head-of-line delay.
        # When latency is under the target, the timeout can increase to
        # allow more items to accumulate → larger GPU batches.  When over
        # the target, the timeout shrinks for faster pickup.
        self._queue_latency_tracker = LatencyTracker(window_size=200, min_samples=10)

    async def start(self) -> None:
        """Start the pull loop.

        Creates a single multiplexed JetStream consumer for the worker's
        pool and begins pulling work items for all models.
        """
        self._running = True
        self._payload_store = _create_payload_store(self._payload_store_url)

        # Migrate: remove legacy per-model streams that overlap with the
        # new pool-level stream subjects. ``add_stream`` will reject the
        # pool stream if any per-model stream has an overlapping subject.
        await self._migrate_per_model_streams()

        # Create the single pool-level stream + consumer
        await self._ensure_pool_subscription()

        # Start the main pull loop
        self._pull_task = asyncio.create_task(self._run())
        logger.info(
            "NatsPullLoop started (bundle=%s, pool=%s, models=%d, stream=%s)",
            self._bundle_id,
            self._pool_name,
            len(self._registry.model_names),
            work_pool_stream_name(self._pool_name),
        )

    async def stop(self) -> None:
        """Stop the pull loop gracefully.

        Cancels the pull task and waits for in-flight processing to complete.
        """
        self._running = False

        # Cancel the main pull loop
        if self._pull_task is not None:
            self._pull_task.cancel()
            try:
                await self._pull_task
            except asyncio.CancelledError:
                pass
            self._pull_task = None

        # Wait for in-flight tasks to complete (graceful drain).
        # Do NOT cancel — let current batches finish, ACK, and drain cleanly.
        # Items processed during drain are ACKed normally; no GPU work wasted.
        if self._in_flight_tasks:
            logger.info("Draining %d in-flight tasks", len(self._in_flight_tasks))
            _done, pending = await asyncio.wait(self._in_flight_tasks, timeout=_DRAIN_TIMEOUT_S)
            if pending:
                logger.warning("Drain timeout: %d tasks still pending, cancelling", len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        self._in_flight_tasks.clear()

        # Cancel background model load tasks
        for load_task in self._load_tasks.values():
            load_task.cancel()
        if self._load_tasks:
            await asyncio.gather(*self._load_tasks.values(), return_exceptions=True)
        self._load_tasks.clear()
        self._loading_models.clear()

        for sub in self._subscriptions.values():
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe during shutdown")
        self._subscriptions.clear()

        if hasattr(self, "_pool_sub"):
            try:
                await self._pool_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe pool sub during shutdown")

        logger.info("NatsPullLoop stopped")

    async def handle_reconnect(self) -> None:
        """Re-create pull subscriptions after NATS reconnect.

        After a NATS server restart, existing subscriptions may be invalid.
        This clears them and re-creates consumers for all known models.
        """
        # Unsubscribe stale subscriptions (best-effort)
        for sub in self._subscriptions.values():
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe during reconnect")
        self._subscriptions.clear()

        # Unsubscribe stale pool subscription (best-effort)
        if hasattr(self, "_pool_sub"):
            try:
                await self._pool_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe old pool sub during reconnect")

        # Re-create the pool subscription
        await self._ensure_pool_subscription()
        logger.info("NatsPullLoop reconnected — re-created pool subscription")

    async def add_model(self, model_id: str) -> None:
        """No-op in multiplexed mode — the pool subscription already covers all models.

        Kept for API compatibility.
        """

    # -- Pool-level multiplexed stream/consumer --------------------------------

    async def _migrate_per_model_streams(self) -> None:
        """Delete legacy per-model streams that overlap with the pool stream.

        The old design created ``WORK_{model_id}`` streams per model.
        The new pool-level stream ``WORK_POOL_{pool}`` uses ``sie.work.*.{pool}``
        which overlaps. NATS rejects streams with overlapping subjects, so we
        must clean up old streams before creating the pool stream.
        """
        model_ids = list(self._registry.model_names)
        deleted = 0
        for model_id in model_ids:
            stream_name = work_stream_name(model_id)
            try:
                await self._js.delete_stream(stream_name)
                deleted += 1
            except Exception:  # noqa: BLE001, S110
                pass  # Stream may not exist — that's fine
        if deleted:
            logger.info("Migrated: deleted %d legacy per-model streams", deleted)

    async def _ensure_pool_subscription(self) -> None:
        """Create the single pool-level stream and pull consumer.

        One stream captures ``sie.work.*.{pool}`` (all models for this pool).
        One durable consumer pulls from it. This replaces the old O(N-models)
        per-model subscription loop with a single fetch point.
        """
        stream_name = work_pool_stream_name(self._pool_name)
        subjects = work_pool_stream_subjects(self._pool_name)
        consumer_name = work_consumer_name(self._bundle_id, self._pool_name)

        config = StreamConfig(
            name=stream_name,
            subjects=subjects,
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=_DEFAULT_STREAM_MAX_AGE_S,
            max_msgs=100_000,
            storage=StorageType.MEMORY,
            num_replicas=1,
            discard=DiscardPolicy.NEW,
        )

        try:
            await self._js.add_stream(config)
            logger.info(
                "Pool stream created/verified: %s (subjects=%s, max_age=%ds)",
                stream_name,
                subjects,
                _DEFAULT_STREAM_MAX_AGE_S,
            )
        except Exception:
            logger.exception("Failed to create pool stream: %s", stream_name)
            raise

        # Filter subject: all models for this pool
        filter_subject = f"{WORK_SUBJECT_PREFIX}.*.{self._pool_name}"

        consumer_config = ConsumerConfig(
            durable_name=consumer_name,
            filter_subject=filter_subject,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=_ACK_WAIT_S,
            max_deliver=_MAX_DELIVER,
            max_ack_pending=1000,
        )

        try:
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except nats.js.errors.BadRequestError:
            logger.warning(
                "Consumer %s config mismatch (likely max_deliver upgrade), recreating",
                consumer_name,
            )
            await self._js.delete_consumer(stream_name, consumer_name)
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except Exception:
            logger.exception("Failed to create pool consumer: %s", consumer_name)
            raise

        self._pool_sub = sub
        logger.info(
            "Pool consumer created: %s on stream %s (filter=%s)",
            consumer_name,
            stream_name,
            filter_subject,
        )

    def _get_batch_budget(self, model_id: str) -> int:
        """Get the batch budget for a model from its BatchConfig.

        Accesses ModelWorker._batch_config (private) — falls back to
        _DEFAULT_BATCH_BUDGET if unavailable.
        """
        try:
            worker = self._registry.get_worker(model_id)
            if worker is not None and hasattr(worker, "_batch_config"):
                return worker._batch_config.max_batch_requests
        except (KeyError, AttributeError):
            pass
        return _DEFAULT_BATCH_BUDGET

    def _check_config_hash(self, model_id: str, wi: WorkItem) -> bool:  # noqa: ARG002
        """Log-only soft check of bundle_config_hash — always returns True.

        Compares the hash from the work item against the local config.
        Mismatches are logged but do not reject the item, because hash
        computation differs between router and worker registries.
        """
        expected_hash = wi.get("bundle_config_hash")
        if not expected_hash:
            return True  # No hash set — backward compatible

        # Use cached hash (recompute every _CONFIG_HASH_CACHE_TTL_S seconds)
        now = time.monotonic()
        if self._config_hash_cache is None or (now - self._config_hash_cache_time) > _CONFIG_HASH_CACHE_TTL_S:
            try:
                from sie_server.api.ws import _compute_bundle_config_hash  # noqa: PLC0415

                self._config_hash_cache = _compute_bundle_config_hash(self._registry, self._bundle_id)
                self._config_hash_cache_time = now
            except Exception:  # noqa: BLE001
                logger.debug("Could not compute config hash for validation", exc_info=True)
                return True

        if self._config_hash_cache and self._config_hash_cache != expected_hash:
            # Log-only: hash computation differs between router and worker
            # registries (different model filtering). Allow processing to
            # proceed — the hash is a soft check, not a hard gate.
            logger.debug(
                "Config hash mismatch for %s (router=%s, worker=%s) — processing anyway",
                wi.get("work_item_id"),
                expected_hash[:8],
                self._config_hash_cache[:8],
            )

        return True

    def _extract_model_id(self, msg: Any) -> str | None:
        """Extract model_id from the NATS message subject.

        Subject format: ``sie.work.{normalized_model_id}.{pool_name}``
        We reverse the normalization to recover the original model_id.
        """
        subject = msg.subject  # e.g., "sie.work.BAAI__bge-m3.l4"
        parts = subject.split(".")
        if len(parts) < _MIN_SUBJECT_PARTS:
            return None
        normalized = parts[2]  # e.g., "BAAI__bge-m3"
        return denormalize_model_id(normalized)

    def _adaptive_fetch_timeout(self, current: float) -> float:
        """Adjust fetch timeout based on observed queue-path latency.

        Under load, when latency is below the target SLO, increase the
        timeout to allow more items to accumulate in the queue → bigger
        GPU batches → better throughput.  When over SLO, decrease for
        faster pickup → lower latency at the cost of smaller batches.

        At idle (not enough samples), fall back to the static adaptive
        backoff (caller handles that case).
        """
        observed = self._queue_latency_tracker.p50()
        if observed is None:
            return _MIN_FETCH_TIMEOUT_S  # Not enough samples — reset to minimum

        # Target: 50ms by default (matches EngineConfig default).
        # In production, this should be configurable per pool.
        target = float(os.environ.get("SIE_ADAPTIVE_TARGET_P50_MS", "50"))

        headroom_ms = target - observed
        gain = 0.2  # Conservative gain for fetch timeout

        # Convert headroom to a timeout adjustment:
        # +headroom → increase timeout (we can afford to wait → bigger batches)
        # -headroom → decrease timeout (over SLO → pick up faster)
        adjustment_s = headroom_ms * gain / 1000.0  # ms → seconds
        new_timeout = current + adjustment_s
        return max(_MIN_FETCH_TIMEOUT_S, min(_MAX_FETCH_TIMEOUT_S, new_timeout))

    async def _run(self) -> None:
        """Main pull loop — single-consumer multiplexed design.

        Pulls from the single pool-level consumer, groups messages by
        model_id (extracted from the NATS subject), and dispatches
        per-model batches for processing with fair scheduling.

        This eliminates the O(N-models) sequential scan of the old design.
        One ``fetch()`` call replaces 83+ per-model fetches.

        **Fair dispatch:** When a batch contains multiple models, each
        model's messages are capped at the model's batch budget.  Excess
        messages are NAK'd for redelivery, preventing a hot model from
        monopolising the GPU.

        **Adaptive fetch timeout:** When the latency tracker has enough
        samples, the fetch timeout is adjusted based on observed p50 vs
        the target SLO. Under load with headroom, the timeout grows to
        allow item accumulation (throughput). Over SLO, it shrinks.
        At idle, static backoff applies (1ms → 20ms).
        """
        fetch_timeout = _MIN_FETCH_TIMEOUT_S

        while self._running:
            # Check completed background load tasks
            self._reap_load_tasks()

            try:
                messages = await self._pool_sub.fetch(batch=_DEFAULT_BATCH_BUDGET, timeout=fetch_timeout)
            except nats.errors.TimeoutError:
                # No items — back off
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue
            except Exception:  # noqa: BLE001
                logger.warning("Pool pull error", exc_info=True)
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                await asyncio.sleep(fetch_timeout)
                continue

            if not messages:
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue

            # Adjust fetch timeout — use adaptive controller if we have
            # enough latency samples, otherwise reset to minimum.
            fetch_timeout = self._adaptive_fetch_timeout(fetch_timeout)

            # Group messages by model_id (extracted from subject)
            model_groups: dict[str, list[Any]] = {}
            for msg in messages:
                model_id = self._extract_model_id(msg)
                if model_id is None:
                    logger.warning("Could not extract model_id from subject: %s", msg.subject)
                    try:
                        await msg.nak()
                    except Exception:  # noqa: BLE001, S110
                        pass
                    continue
                model_groups.setdefault(model_id, []).append(msg)

            # Fair dispatch: round-robin across models, capping per-model batch
            for model_id, model_msgs in model_groups.items():
                # Skip models currently loading — NAK for redelivery
                if model_id in self._loading_models:
                    for m in model_msgs:
                        try:
                            await m.nak(delay=_NAK_DELAY_S)
                        except Exception:  # noqa: BLE001, S110
                            pass
                    continue

                # Check if model is loaded
                if not self._registry.is_loaded(model_id):
                    await self._handle_unloaded_model(model_id, model_msgs)
                    continue

                # Apply per-model batch cap for fairness.  If more messages
                # arrived than the model's batch budget, NAK the excess so
                # they are redelivered (possibly to another worker).
                budget = self._get_batch_budget(model_id)
                dispatch_msgs = model_msgs[:budget]
                overflow = model_msgs[budget:]
                for m in overflow:
                    try:
                        await m.nak(delay=0.1)  # fast redeliver
                    except Exception:  # noqa: BLE001, S110
                        pass

                if _HAS_METRICS:
                    PULL_ITEMS_FETCHED.labels(model=model_id).observe(len(dispatch_msgs))

                # Limit concurrent batch processing
                await self._batch_sem.acquire()

                async def _guarded_process(msgs: list[Any], mdl: str) -> None:
                    try:
                        await self._process_messages(mdl, msgs)
                    finally:
                        self._batch_sem.release()

                task = asyncio.create_task(_guarded_process(dispatch_msgs, model_id))
                self._in_flight_tasks.add(task)
                task.add_done_callback(self._in_flight_tasks.discard)

    def _reap_load_tasks(self) -> None:
        """Check completed background load tasks and clean up state."""
        for model_id in list(self._load_tasks):
            task = self._load_tasks[model_id]
            if not task.done():
                continue

            self._loading_models.discard(model_id)
            del self._load_tasks[model_id]

            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.debug("Background model load cancelled for %s", model_id)
                continue
            if exc is not None:
                logger.warning("Background model load failed for %s: %s", model_id, exc)
            else:
                logger.info("Model %s loaded via background demand, now pulling", model_id)

    async def _handle_unloaded_model(self, model_id: str, messages: list[Any]) -> None:
        """Handle items pulled for a model that isn't loaded yet.

        NAKs all items with a delay so JetStream redelivers them after
        the model has had time to load. Triggers a background load if
        one isn't already in progress.

        Uses a longer NAK delay when a load is already in progress to
        conserve the delivery budget (max_deliver) for slow model loads.
        """
        already_loading = model_id in self._loading_models
        nak_delay = _NAK_DELAY_S * 2 if already_loading else _NAK_DELAY_S

        for msg in messages:
            try:
                await msg.nak(delay=nak_delay)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to NAK message for unloaded model %s", model_id)

        if _HAS_METRICS:
            PULL_NAK_UNLOADED.labels(model=model_id).inc(len(messages))

        logger.info(
            "NAKed %d items for unloaded model %s (redeliver in %.0fs, loading=%s)",
            len(messages),
            model_id,
            nak_delay,
            already_loading,
        )

        # Trigger background load if not already in progress
        if model_id not in self._loading_models:
            self._check_thrashing(model_id)
            self._loading_models.add(model_id)
            load_task = asyncio.create_task(self._background_load_model(model_id))
            self._load_tasks[model_id] = load_task

            if _HAS_METRICS:
                PULL_MODEL_LOADS.labels(model=model_id).inc()

            logger.info("Triggered background load for model %s", model_id)

    async def _background_load_model(self, model_id: str) -> None:
        """Load a model in the background without blocking the pull loop.

        ``registry.load_async()`` serializes loads via an internal lock,
        runs the actual GPU loading in a thread pool, and may evict LRU
        models to free memory. If the evicted model was being served,
        the pull loop will discover ``is_loaded() == False`` on the next
        iteration and NAK its items — no race condition.
        """
        try:
            device = self._registry.device
            await self._registry.load_async(model_id, device)
            self._load_history.append((model_id, time.monotonic()))
        except Exception:
            logger.warning("Background load failed for model %s", model_id, exc_info=True)
            self._loading_models.discard(model_id)
            raise

    def _check_thrashing(self, model_id: str) -> None:
        """Detect and warn about model load/evict thrashing.

        If any model has been background-loaded ``_THRASH_THRESHOLD`` or
        more times within ``_THRASH_WINDOW_S``, log a warning.  This
        indicates the GPU cannot hold all in-demand models simultaneously
        and the operator should use separate bundles or add GPU capacity.
        """
        now = time.monotonic()
        cutoff = now - _THRASH_WINDOW_S

        # Prune old entries
        self._load_history = [(m, t) for m, t in self._load_history if t > cutoff]

        # Count recent loads for this model
        recent = sum(1 for m, _ in self._load_history if m == model_id)
        if recent >= _THRASH_THRESHOLD:
            logger.warning(
                "Model thrashing detected: %s loaded %d times in the last %.0fs. "
                "GPU memory cannot hold all in-demand models simultaneously. "
                "Consider using separate bundles per model or adding GPU capacity.",
                model_id,
                recent,
                _THRASH_WINDOW_S,
            )

    async def _process_messages(self, model_id: str, messages: list[Any]) -> None:
        """Process a batch of pulled NATS messages for a model.

        Groups encode/extract items into batches for optimal GPU utilization.
        Score items are submitted concurrently for cross-request batching.
        """
        work_items: list[tuple[WorkItem, Any]] = []  # (work_item, nats_msg)

        for msg in messages:
            try:
                wi: WorkItem = msgpack.unpackb(msg.data, raw=False)
                # Validate reply_subject to prevent injection attacks.
                # Only accept subjects under the _INBOX prefix.
                reply_subj = wi.get("reply_subject", "")
                if reply_subj and not reply_subj.startswith("_INBOX."):
                    logger.warning(
                        "Rejecting work item with suspicious reply_subject: %s",
                        reply_subj[:60],
                    )
                    await msg.ack()  # Consume to prevent redelivery
                    continue
                work_items.append((wi, msg))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to deserialize work item", exc_info=True)
                await msg.nak()

        if not work_items:
            return

        # Record queue wait times
        if _HAS_METRICS:
            for wi, _ in work_items:
                queue_wait_s = time.time() - wi.get("timestamp", time.time())
                if queue_wait_s > 0:
                    PULL_QUEUE_WAIT_SECONDS.labels(model=model_id).observe(queue_wait_s)

        # Validate bundle_config_hash — log-only soft check (never NAKs)
        for wi, _ in work_items:
            self._check_config_hash(model_id, wi)
        valid_items = work_items

        if not valid_items:
            return

        # Group by operation for batch processing
        encode_items: list[tuple[WorkItem, Any]] = []
        score_items: list[tuple[WorkItem, Any]] = []
        extract_items: list[tuple[WorkItem, Any]] = []

        for wi, msg in valid_items:
            op = wi.get("operation", "encode")
            if op == "encode":
                encode_items.append((wi, msg))
            elif op == "score":
                score_items.append((wi, msg))
            elif op == "extract":
                extract_items.append((wi, msg))
            else:
                logger.warning("Unknown operation %s for %s", op, wi.get("work_item_id"))
                await self._publish_error(wi, "unknown_operation", f"Unknown operation: {op}")
                await msg.ack()

        # Process batches concurrently
        tasks: list[Any] = []
        if encode_items:
            tasks.append(self._process_encode_batch(model_id, encode_items))
        if score_items:
            # Score items submitted concurrently for BatchFormer cross-request batching
            tasks.append(self._process_score_batch(model_id, score_items))
        if extract_items:
            tasks.append(self._process_extract_batch(model_id, extract_items))

        await asyncio.gather(*tasks)

    async def _process_encode_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process a batch of encode work items through EncodePipeline."""
        from sie_server.core.encode_pipeline import EncodePipeline  # noqa: PLC0415

        batch_start = time.monotonic()

        # Resolve all item payloads (with fetch timing)
        resolved: list[tuple[WorkItem, Any, Item, float]] = []  # (..., payload_fetch_ms)
        for wi, msg in items_msgs:
            fetch_start = time.monotonic()
            item = await self._resolve_item(wi)
            fetch_ms = (time.monotonic() - fetch_start) * 1000 if wi.get("payload_ref") else 0.0
            if item is None:
                await self._publish_error(wi, "payload_error", "Failed to resolve item payload")
                await msg.ack()
                continue
            resolved.append((wi, msg, item, fetch_ms))

        if not resolved:
            return

        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            # Model config not found or model evicted mid-batch — NAK for redelivery.
            # Another worker (or this one after re-loading) will process them.
            for wi, msg, _, _fm in resolved:
                try:
                    await msg.nak(delay=_NAK_DELAY_S)
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to NAK message for evicted model %s", model_id)
            return

        # Group items by encode params to handle heterogeneous batches.
        # Items from different API requests may have different output_types,
        # instruction, is_query, or options — these cannot be mixed in a
        # single EncodePipeline.run_encode() call.
        groups: dict[tuple, list[tuple[WorkItem, Any, Item, float]]] = {}
        for wi, msg, item, fetch_ms in resolved:
            output_types = tuple(wi.get("output_types") or ["dense"])
            instruction = wi.get("instruction")
            is_query = wi.get("is_query", False)
            # options dict is not hashable — use sorted tuple of items
            options = wi.get("options") or {}
            options_key = msgpack.packb(options, use_bin_type=True) if options else b""
            key = (output_types, instruction, is_query, options_key)
            groups.setdefault(key, []).append((wi, msg, item, fetch_ms))

        # Process each sub-batch
        for (output_types_t, instruction, is_query, options_key), group in groups.items():
            output_types = list(output_types_t)
            options = msgpack.unpackb(options_key, raw=False) if options_key else {}
            all_items = [item for _, _, item, _ in group]
            queue_times = [(time.time() - wi.get("timestamp", time.time())) * 1000 for wi, _, _, _ in group]
            fetch_times = [fm for _, _, _, fm in group]

            try:
                formatted_outputs, timing = await EncodePipeline.run_encode(
                    registry=self._registry,
                    model=model_id,
                    items=all_items,
                    output_types=output_types,
                    instruction=instruction,
                    config=config,
                    is_query=is_query,
                    options=options,
                )

                for idx, (wi, msg, _, _fm) in enumerate(group):
                    output = formatted_outputs[idx] if idx < len(formatted_outputs) else {}
                    # Wrap raw numpy arrays in the DenseVector/SparseVector wire
                    # format that the SDK client expects:
                    #   dense  → {"dims": N, "dtype": str, "values": ndarray}
                    #   sparse → {"indices": ndarray, "values": ndarray}
                    # This matches ``encode.py:_build_response_items`` on the HTTP path.
                    output = _wrap_encode_output(output, config)
                    result_data = msgpack.packb(output, use_bin_type=True)
                    reply_subject = wi.get("reply_subject", "")
                    if reply_subject:
                        result: WorkResult = {
                            "work_item_id": wi.get("work_item_id", ""),
                            "request_id": wi.get("request_id", ""),
                            "item_index": wi.get("item_index", 0),
                            "success": True,
                            "result_msgpack": result_data,
                            "queue_ms": queue_times[idx],
                            "processing_ms": 0.0,
                            "worker_id": self._worker_id,
                        }
                        if timing.inference_ms is not None:
                            result["inference_ms"] = timing.inference_ms
                        if timing.tokenization_ms > 0:
                            result["tokenization_ms"] = timing.tokenization_ms
                        if timing.postprocessing_ms > 0:
                            result["postprocessing_ms"] = timing.postprocessing_ms
                        result["payload_fetch_ms"] = fetch_times[idx]
                        result_bytes = msgpack.packb(result, use_bin_type=True)
                        try:
                            await self._nc.publish(reply_subject, result_bytes)
                        except Exception:  # noqa: BLE001
                            logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                            continue  # Don't ACK — let JetStream redeliver

                        # Feed adaptive latency tracker: total queue-path
                        # latency = queue wait + inference + postprocessing.
                        total_ms = queue_times[idx] + (timing.inference_ms or 0) + timing.postprocessing_ms
                        self._queue_latency_tracker.record(total_ms)

                    await msg.ack()

            except asyncio.CancelledError:
                # Worker was evicted (model unloaded) mid-batch — NAK for redelivery
                logger.info("Encode batch cancelled (model %s likely evicted) — NAKing items", model_id)
                for wi, msg, _, _fm in group:
                    try:
                        await msg.nak(delay=_NAK_DELAY_S)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to NAK message for evicted model %s", model_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("Encode sub-batch failed for model %s: %s", model_id, e)
                for wi, msg, _, _fm in group:
                    await self._publish_error(wi, "inference_error", str(e))
                    await msg.ack()

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="encode").observe(elapsed)

    async def _process_score_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process score items concurrently for BatchFormer cross-request batching."""
        batch_start = time.monotonic()

        async def _process_one(wi: WorkItem, msg: Any) -> None:
            try:
                await self._process_single_score(model_id, wi, msg)
            except Exception:  # noqa: BLE001
                logger.warning("Score failed for %s", wi.get("work_item_id"), exc_info=True)
                try:
                    await self._publish_error(wi, "internal_error", "Unexpected processing failure")
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to publish error result for %s", wi.get("work_item_id"))
                try:
                    await msg.ack()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to ACK message for %s", wi.get("work_item_id"))

        await asyncio.gather(*(_process_one(wi, msg) for wi, msg in items_msgs))

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="score").observe(elapsed)

    async def _process_extract_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process extract items individually but concurrently."""
        batch_start = time.monotonic()

        # Extract items go through worker.submit_extract individually
        # but concurrent submission lets BatchFormer batch them
        async def _process_one(wi: WorkItem, msg: Any) -> None:
            try:
                fetch_start = time.monotonic()
                item = await self._resolve_item(wi)
                fetch_ms = (time.monotonic() - fetch_start) * 1000 if wi.get("payload_ref") else 0.0
                if item is None:
                    await self._publish_error(wi, "payload_error", "Failed to resolve item payload")
                    await msg.ack()
                    return
                await self._process_single_extract(model_id, wi, msg, item, fetch_ms)
            except Exception:  # noqa: BLE001
                logger.warning("Extract failed for %s", wi.get("work_item_id"), exc_info=True)
                try:
                    await self._publish_error(wi, "internal_error", "Unexpected processing failure")
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to publish error result for %s", wi.get("work_item_id"))
                try:
                    await msg.ack()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to ACK message for %s", wi.get("work_item_id"))

        await asyncio.gather(*(_process_one(wi, msg) for wi, msg in items_msgs))

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="extract").observe(elapsed)

    async def _process_single_score(
        self,
        model_id: str,
        wi: WorkItem,
        msg: Any,
    ) -> None:
        """Run the score pipeline for a single item, ACK, and publish result."""
        queue_ms = (time.time() - wi.get("timestamp", time.time())) * 1000

        # Start the model worker (lazy loading)
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for score: %s — NAKing", model_id, e)
            await msg.nak(delay=_NAK_DELAY_S)
            return

        query = wi.get("query_item")
        items = wi.get("score_items")

        # Resolve offloaded score payload
        payload_fetch_ms = 0.0
        if query is None and wi.get("query_payload_ref"):
            ref_key: str = wi["query_payload_ref"]  # type: ignore[assignment]
            fetch_start = time.monotonic()
            payload_data = await self._fetch_payload(ref_key)
            payload_fetch_ms = (time.monotonic() - fetch_start) * 1000
            if payload_data:
                decoded = msgpack.unpackb(payload_data, raw=False)
                query = decoded.get("query")
                items = decoded.get("items")

        if query is None or items is None:
            await self._publish_error(wi, "payload_error", "Missing query or items for score")
            await msg.ack()
            return

        query_item = self._dict_to_item(query) if isinstance(query, dict) else query
        score_items = [self._dict_to_item(it) if isinstance(it, dict) else it for it in items]

        instruction = wi.get("instruction")
        options = wi.get("options") or {}

        try:
            timing = RequestTiming()

            # Build prepared items with cost (query + doc char count)
            query_text = query_item.text
            query_len = len(query_text) if query_text else 0
            timing.start_tokenization()
            prepared_items = []
            for i, it in enumerate(score_items):
                item_text = it.text
                doc_len = len(item_text) if item_text else 0
                prepared_items.append(ScorePreparedItem(cost=query_len + doc_len, original_index=i))
            timing.end_tokenization()

            # Submit to worker and await result
            future = await worker.submit_score(
                prepared_items=prepared_items,
                query=query_item,
                items=score_items,
                instruction=instruction,
                options=options,
                timing=timing,
            )
            worker_result = await future

            # Extract scores and build ranked ScoreEntry list.
            # The router wraps this in {"model": ..., "items": <blob>}, so
            # result_data must be the scores list, not a full ScoreResponse.
            score_output: ScoreOutput = worker_result.output  # type: ignore[assignment]
            raw_scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]

            scored_items = []
            for i, sc in enumerate(raw_scores):
                item_id = score_items[i].id if score_items[i].id is not None else f"item-{i}"
                scored_items.append((item_id, sc))
            scored_items.sort(key=lambda x: x[1], reverse=True)

            score_entries = [
                {"item_id": item_id, "score": sc, "rank": rank} for rank, (item_id, sc) in enumerate(scored_items)
            ]
            result_data = msgpack.packb(score_entries, use_bin_type=True)
            inference_ms = timing.inference_ms
        except Exception as e:  # noqa: BLE001
            logger.warning("Score failed for %s: %s", wi.get("work_item_id"), e)
            await self._publish_error(wi, "inference_error", str(e))
            await msg.ack()
            return

        # Publish result
        reply_subject = wi.get("reply_subject", "")
        if reply_subject and result_data is not None:
            result: WorkResult = {
                "work_item_id": wi.get("work_item_id", ""),
                "request_id": wi.get("request_id", ""),
                "item_index": wi.get("item_index", 0),
                "success": True,
                "result_msgpack": result_data,
                "queue_ms": queue_ms,
                "processing_ms": 0.0,
                "worker_id": self._worker_id,
            }
            if inference_ms is not None:
                result["inference_ms"] = inference_ms
            if timing.tokenization_ms > 0:
                result["tokenization_ms"] = timing.tokenization_ms
            if timing.postprocessing_ms > 0:
                result["postprocessing_ms"] = timing.postprocessing_ms
            if payload_fetch_ms > 0:
                result["payload_fetch_ms"] = payload_fetch_ms
            result_bytes = msgpack.packb(result, use_bin_type=True)
            try:
                await self._nc.publish(reply_subject, result_bytes)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                return  # Don't ACK — let JetStream redeliver

        await msg.ack()

    async def _process_single_extract(
        self,
        model_id: str,
        wi: WorkItem,
        msg: Any,
        item: Item,
        payload_fetch_ms: float = 0.0,
    ) -> None:
        """Run the extract pipeline for a single item, ACK, and publish result."""
        queue_ms = (time.time() - wi.get("timestamp", time.time())) * 1000

        # Start the model worker (lazy loading)
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for extract: %s — NAKing", model_id, e)
            await msg.nak(delay=_NAK_DELAY_S)
            return

        labels = wi.get("labels")
        output_schema = wi.get("output_schema")
        instruction = wi.get("instruction")
        options = wi.get("options") or {}

        try:
            timing = RequestTiming()

            # Build prepared items with cost (character count)
            timing.start_tokenization()
            text = item.text
            char_count = len(text) if text else 0
            prepared_items = [ExtractPreparedItem(cost=char_count, original_index=0)]
            timing.end_tokenization()

            # Submit to worker and await result
            future = await worker.submit_extract(
                prepared_items=prepared_items,
                items=[item],
                labels=labels,
                output_schema=output_schema,
                instruction=instruction,
                options=options,
                timing=timing,
            )
            worker_result = await future

            # Format output using ExtractHandler (matches api/extract.py)
            extraction_results = ExtractHandler.format_output(worker_result.output)  # type: ignore[arg-type]
            result_data = msgpack.packb(extraction_results[0] if extraction_results else {}, use_bin_type=True)
            inference_ms = timing.inference_ms
        except Exception as e:  # noqa: BLE001
            logger.warning("Extract failed for %s: %s", wi.get("work_item_id"), e)
            await self._publish_error(wi, "inference_error", str(e))
            await msg.ack()
            return

        # Publish result
        reply_subject = wi.get("reply_subject", "")
        if reply_subject and result_data is not None:
            result: WorkResult = {
                "work_item_id": wi.get("work_item_id", ""),
                "request_id": wi.get("request_id", ""),
                "item_index": wi.get("item_index", 0),
                "success": True,
                "result_msgpack": result_data,
                "queue_ms": queue_ms,
                "processing_ms": 0.0,
                "worker_id": self._worker_id,
            }
            if inference_ms is not None:
                result["inference_ms"] = inference_ms
            if timing.tokenization_ms > 0:
                result["tokenization_ms"] = timing.tokenization_ms
            if timing.postprocessing_ms > 0:
                result["postprocessing_ms"] = timing.postprocessing_ms
            if payload_fetch_ms > 0:
                result["payload_fetch_ms"] = payload_fetch_ms
            result_bytes = msgpack.packb(result, use_bin_type=True)
            try:
                await self._nc.publish(reply_subject, result_bytes)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                return  # Don't ACK — let JetStream redeliver

        await msg.ack()

    async def _resolve_item(self, wi: WorkItem) -> Item | None:
        """Resolve a work item's payload (inline or from payload store)."""
        item_data = wi.get("item")
        if item_data is not None:
            if isinstance(item_data, dict):
                return self._dict_to_item(item_data)
            return item_data

        # Fetch from payload store
        payload_ref = wi.get("payload_ref")
        if payload_ref:
            payload_bytes = await self._fetch_payload(payload_ref)
            if payload_bytes:
                item_dict = msgpack.unpackb(payload_bytes, raw=False)
                return self._dict_to_item(item_dict)

        return None

    @staticmethod
    def _dict_to_item(d: dict) -> Item:
        """Convert a dict (from SDK wire format) to an Item.

        The SDK sends ``content`` while the server ``Item`` struct expects
        ``text``.  Map the field and drop any other unknown keys so that the
        ``Item`` constructor doesn't raise.
        """
        if "content" in d and "text" not in d:
            d = {**d, "text": d.pop("content")}
        known = set(Item.__struct_fields__)
        return Item(**{k: v for k, v in d.items() if k in known})

    async def _fetch_payload(self, ref_key: str) -> bytes | None:
        """Fetch payload from the cached payload store."""
        if self._payload_store is None:
            logger.warning("Payload ref %s but no payload store configured", ref_key)
            return None

        try:
            return await self._payload_store.get(ref_key)
        except KeyError:
            logger.warning("Payload not found: %s", ref_key)
            return None

    async def _publish_error(self, wi: WorkItem, error_code: str, error_msg: str) -> None:
        """Publish an error result to the reply subject."""
        reply_subject = wi.get("reply_subject", "")
        if not reply_subject:
            return

        result: WorkResult = {
            "work_item_id": wi.get("work_item_id", ""),
            "request_id": wi.get("request_id", ""),
            "item_index": wi.get("item_index", 0),
            "success": False,
            "error": error_msg,
            "error_code": error_code,
            "worker_id": self._worker_id,
        }
        result_bytes = msgpack.packb(result, use_bin_type=True)
        try:
            await self._nc.publish(reply_subject, result_bytes)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish error result for %s", wi.get("work_item_id"))
