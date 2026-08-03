from typing import Any, TypedDict


class _WorkItemRequired(TypedDict):
    work_item_id: str
    request_id: str
    item_index: int
    total_items: int
    operation: str
    model_id: str
    profile_id: str
    pool_name: str
    machine_profile: str
    router_id: str
    reply_subject: str
    timestamp: float


class WorkItem(_WorkItemRequired, total=False):
    """A single inference work item published to NATS JetStream.

    Published by the gateway to
    ``sie.work.{pool_name}.{machine_profile}.{bundle_id}.{model_id_normalized}``.
    Consumed by workers via JetStream pull consumers.

    Attributes:
        work_item_id: Unique ID for this work item. Format: ``{request_id}.{item_index}``.
        request_id: Unique ID for the originating client request. Format: ``{router_id}-{counter}``.
        item_index: Position of this item in the original request (for result ordering).
        total_items: Total number of items in the original request.
        operation: Inference operation: ``"encode"`` | ``"score"`` | ``"extract"``.
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).
        profile_id: Profile name (e.g., ``"default"``).
        pool_name: Target pool (e.g., ``"default"``).
        machine_profile: Required GPU type (e.g., ``"l4"``). Workers validate this.
        bundle_config_hash: Expected bundle config hash from the gateway's ModelRegistry.
            Workers compare this against their own computed hash and NAK items with
            mismatched hashes so they are redelivered to updated workers.

        item: Inline item payload (text, small images). Mutually exclusive with ``payload_ref``.
        payload_ref: S3 key for offloaded payload. Mutually exclusive with ``item``.

        output_types: Encode: which outputs to return (``["dense", "sparse"]``).
        instruction: Optional instruction for instruction-tuned models.
        is_query: Encode: whether items are queries (True) or documents (False).
        options: Runtime options (lora, pooling, output_dtype, etc.).

        query_item: Score: the query item (inline).
        query_payload_ref: Score: S3 key for offloaded query payload.
        score_items: Score: all candidate items (inline list).

        labels: Extract: entity labels.
        output_schema: Extract: structured output schema.

        router_id: Originating gateway identifier (for observability).
        reply_subject: NATS subject where the worker should publish results.
        timestamp: Unix timestamp when the work item was created.
    """

    bundle_config_hash: str

    # Item payload (inline or reference)
    item: dict[str, Any] | None
    payload_ref: str | None

    # Operation-specific context
    output_types: list[str] | None
    instruction: str | None
    is_query: bool
    options: dict[str, Any] | None

    # Score-specific: query + all candidate items travel together
    query_item: dict[str, Any] | None
    query_payload_ref: str | None
    score_items: list[dict[str, Any]] | None

    # Extract-specific
    labels: list[str] | None
    output_schema: dict[str, Any] | None

    # Generate-specific wire shape.
    # Carries prompt + sampling params for one blocking call to
    # ``adapter.generate(...)``. Keys: prompt (str), max_new_tokens (int),
    # temperature (float), top_p (float), stop (list[str]).
    generate: dict[str, Any] | None

    # W3C Trace Context propagation. Populated by the gateway
    # when it has an active span (extracted from the inbound
    # ``traceparent`` HTTP header, or rooted at the gateway when
    # the client did not send one). Both fields are msgpack-
    # skipped on the wire when ``None``, so a worker without trace
    # support sees its expected key set, and a trace-aware worker
    # decoding an older envelope sees both
    # fields default to ``None`` via ``WorkItem.get(...)``.
    #
    # Privacy: the value is two opaque IDs (trace id + span id)
    # plus flags. Do not log it at info-level; debug is fine.
    traceparent: str | None
    tracestate: str | None


class _WorkResultRequired(TypedDict):
    work_item_id: str
    request_id: str
    item_index: int
    success: bool


class WorkResult(_WorkResultRequired, total=False):
    """Result for a single work item, published to the reply subject.

    Published by the worker to the ``reply_subject`` from the corresponding
    ``WorkItem``. The gateway collects results and reassembles the HTTP response.

    Result payloads are serialized as msgpack bytes (opaque blobs) by the worker.
    The gateway embeds them directly into the response without deserializing,
    keeping numpy/msgpack-numpy out of the gateway's dependency graph.

    Attributes:
        work_item_id: Matches the originating ``WorkItem.work_item_id``.
        request_id: Matches the originating ``WorkItem.request_id``.
        item_index: Position in the original request (for result ordering).

        success: Whether inference succeeded.
        result_msgpack: Msgpack-serialized result bytes (opaque to the gateway).
            For encode: a single ``EncodeResult`` dict.
            For score: list of ``ScoreResult`` dicts.
            For extract: a single ``ExtractResult`` dict.
        error: Error message if ``success`` is False.
        error_code: Machine-readable error code (e.g., ``"queue_full"``).

        inference_ms: Time spent on GPU inference.
        queue_ms: Time spent waiting in the NATS queue.
        processing_ms: Time spent on processing (preprocessing + inference).
        worker_id: Worker that processed this item (for observability).

        tokenization_ms: Time spent tokenizing the input (ms).
        postprocessing_ms: Time spent on output post-processing (ms).
        payload_fetch_ms: Time spent fetching from payload store, 0 if inline (ms).

        units: Authoritative billable-unit counts emitted by the engine
            (``{"input_tokens": int, "pages": int, "images": int}``, every
            key optional). ``input_tokens`` is the real tokenizer count
            taken post-tokenization — never a bytes/4-style estimate.
            Optional: absent on results from workers that don't emit it,
            so the wire shape stays backward-compatible.
        execution_identity_sha256: Opaque worker-origin SHA-256 binding the
            successful result to one immutable managed deployment/resource
            identity. Optional for rolling and self-host compatibility.
    """

    result_msgpack: bytes | None
    error: str | None
    error_code: str | None

    inference_ms: float
    queue_ms: float
    processing_ms: float
    worker_id: str

    # Additional timing breakdown (all in milliseconds)
    tokenization_ms: float
    postprocessing_ms: float
    payload_fetch_ms: float

    # Authoritative unit counts for metering (see docstring). Optional —
    # older workers omit it and decoders ignore unknown keys.
    units: dict[str, int] | None

    execution_identity_sha256: str | None


# -- NATS subject helpers ---------------------------------------------------

# NATS JetStream max message size default is 1MB.
INLINE_THRESHOLD_BYTES = 1_048_576  # 1 MB

# Subject prefix for work items
WORK_SUBJECT_PREFIX = "sie.work"

# Dead-letter subject prefix — uses ``sie.dlq`` (not ``sie.work.dead``)
# to avoid subject overlap with pool-level work streams
# (``sie.work.{pool}.*.*.*``).
DEAD_LETTER_PREFIX = "sie.dlq"


def normalize_model_id(model_id: str) -> str:
    """Normalize a model ID for use in NATS subjects and stream names.

    Replaces characters that are invalid in NATS subject tokens:
    ``/`` -> ``__``, ``.`` -> ``_dot_``, ``*`` and ``>`` -> ``_``.
    Consistent with config store file naming convention.

    .. warning::

        The encoding is **not fully reversible**. Model IDs containing
        literal ``__`` (e.g., ``"org/a__b"``) collide with IDs containing
        ``/`` at the same position (``"org/a/b"``).  In practice this is
        safe because HuggingFace model IDs never contain ``__``, but
        callers that need the canonical ID should use the ``model_id``
        field from the deserialized ``WorkItem``, not the NATS subject.

    Args:
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).

    Returns:
        Normalized string (e.g., ``"BAAI__bge-m3"``).
    """
    result = model_id.replace("/", "__")
    # Sanitize additional NATS-invalid characters
    result = result.replace(".", "_dot_")
    result = result.replace("*", "_")
    result = result.replace(">", "_")
    result = result.replace(" ", "_")
    return result


def normalize_worker_id(worker_id: str) -> str:
    """Normalize a worker identity for use as a single NATS subject token.

    Worker IDs flow into three places that all share the same wire contract:

    * the gateway's direct-dispatch publish subject
      (``sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}``),
    * the per-worker JetStream stream subjects
      (``sie.work.{pool}.{machine_profile}.{bundle}.*.{worker_id}``), and
    * the durable consumer name (``gen-{worker_id}``).

    In production, worker IDs come from ``SIE_WORKER_ID`` / ``HOSTNAME`` /
    ``POD_NAME``. Kubernetes pod hostnames (e.g.
    ``sie-worker-7d9f-default-0.sie-worker.default.svc``) contain ``.``
    which is the NATS subject separator — interpolating that raw value
    expands the subject into extra tokens and the gateway and worker bind
    to different subjects, silently breaking direct-dispatch.

    This helper applies the **same** mapping as :func:`normalize_model_id`
    (matching the Rust gateway's ``normalize_model_id`` in
    ``packages/sie_gateway/src/queue/publisher.rs``), so the two sides
    always produce identical subject tokens. Empty input — and input that
    normalizes to the empty string — is rejected: silently substituting a
    default (the previous ``or "worker"`` fallback) caused durable-consumer
    collisions across processes when env vars were missing.

    Args:
        worker_id: Raw worker identity (env var value, hostname, pod name).

    Returns:
        Single NATS-subject-safe token derived from ``worker_id``.

    Raises:
        ValueError: ``worker_id`` is empty or whitespace-only.
    """
    if not worker_id or not worker_id.strip():
        raise ValueError(
            "worker_id is empty after stripping whitespace; refusing to "
            "substitute a default because that would collide durables "
            "across processes"
        )
    return normalize_model_id(worker_id)


def denormalize_model_id(normalized: str) -> str:
    """Best-effort reversal of :func:`normalize_model_id`.

    Recovers the original model ID from a normalized NATS subject token.
    The decoding assumes the original ID did not contain literal ``__``
    or ``_dot_`` sequences — see the warning on :func:`normalize_model_id`.

    Args:
        normalized: Normalized model ID (e.g., ``"BAAI__bge-m3"``).

    Returns:
        Original model ID (e.g., ``"BAAI/bge-m3"``).
    """
    result = normalized.replace("__", "/")
    result = result.replace("_dot_", ".")
    return result


def work_subject(pool_name: str, machine_profile: str, bundle_id: str, model_id: str) -> str:
    """Build the NATS subject for a work queue.

    Args:
        pool_name: Pool name (e.g., ``"default"``).
        machine_profile: Hardware lane (e.g., ``"rtx6000"``).
        bundle_id: Runtime/image bundle identifier (e.g., ``"sglang"``).
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).

    Returns:
        NATS subject string, e.g.
        ``"sie.work.default.rtx6000.sglang.BAAI__bge-m3"``.
    """
    return (
        f"{WORK_SUBJECT_PREFIX}.{pool_name}.{normalize_model_id(machine_profile)}."
        f"{normalize_model_id(bundle_id)}.{normalize_model_id(model_id)}"
    )


def work_consumer_name(pool_name: str, machine_profile: str, bundle_id: str) -> str:
    """Build the JetStream consumer name for a concrete queue lane.

    Args:
        pool_name: Pool name (e.g., ``"default"``).
        machine_profile: Hardware lane (e.g., ``"rtx6000"``).
        bundle_id: Bundle identifier (e.g., ``"default"``).

    Returns:
        Consumer durable name (e.g., ``"default_rtx6000_sglang"``).
    """
    return f"{normalize_model_id(pool_name)}_{normalize_model_id(machine_profile)}_{normalize_model_id(bundle_id)}"


# -- Pool-level (multiplexed) stream helpers --------------------------------
# A single stream per pool captures ALL models for that pool, replacing the
# O(N-models) per-model stream design.  The worker creates one pull consumer
# and dispatches locally by model_id extracted from the message subject.

WORK_POOL_STREAM_PREFIX = "WORK_POOL"


def work_pool_stream_name(pool_name: str) -> str:
    """Stream name for the multiplexed pool-level work queue.

    Args:
        pool_name: Pool name (e.g., ``"l4"``).

    Returns:
        Stream name (e.g., ``"WORK_POOL_l4"``).
    """
    return f"{WORK_POOL_STREAM_PREFIX}_{pool_name}"


def work_pool_stream_subjects(pool_name: str) -> list[str]:
    """Subjects captured by a pool-level stream.

    Matches ``sie.work.<pool_name>.<machine_profile>.<bundle>.<model>`` for
    all machine profiles, bundles, and models in the given logical pool.

    Args:
        pool_name: Pool name (e.g., ``"l4"``).

    Returns:
        Subject list, e.g. ``["sie.work.default.*.*.*"]``.
    """
    return [f"{WORK_SUBJECT_PREFIX}.{pool_name}.*.*.*"]


# -- Per-worker direct-dispatch helpers -----------------------------------
# Each worker subscribes to a second, narrower JetStream stream that
# captures ONLY messages addressed to its own worker_id. The gateway
# computes an HRW pick and publishes to
# ``sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}``; the pool
# stream's subjects (``sie.work.{pool}.*.*.*``) deliberately do not match those
# worker-addressed subjects, so the two paths cannot double-deliver.

WORK_WORKER_STREAM_PREFIX = "WORK_WORKER"


def work_worker_subject(
    pool_name: str,
    machine_profile: str,
    bundle_id: str,
    model_id: str,
    worker_id: str,
) -> str:
    """Build the NATS subject for a per-worker direct-dispatch.

    Args:
        pool_name: Pool name (e.g., ``"default"``).
        machine_profile: Hardware lane (e.g., ``"rtx6000"``).
        bundle_id: Runtime/image bundle identifier (e.g., ``"sglang"``).
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).
        worker_id: Stable worker identity advertised by the worker-sidecar.

    Returns:
        NATS subject like
        ``"sie.work.default.rtx6000.sglang.BAAI__bge-m3.worker-0"``.

    Raises:
        ValueError: ``worker_id`` is empty or whitespace-only — see
            :func:`normalize_worker_id`.
    """
    return (
        f"{WORK_SUBJECT_PREFIX}.{pool_name}.{normalize_model_id(machine_profile)}."
        f"{normalize_model_id(bundle_id)}.{normalize_model_id(model_id)}."
        f"{normalize_worker_id(worker_id)}"
    )


def work_worker_stream_name(worker_id: str) -> str:
    """Stream name for the per-worker direct-dispatch queue.

    Args:
        worker_id: Stable worker identity.

    Returns:
        Stream name (e.g., ``"WORK_WORKER_worker-0"``).

    Raises:
        ValueError: ``worker_id`` is empty or whitespace-only — see
            :func:`normalize_worker_id`.
    """
    return f"{WORK_WORKER_STREAM_PREFIX}_{normalize_worker_id(worker_id)}"


def work_worker_stream_subjects(
    pool_name: str,
    machine_profile: str,
    bundle_id: str,
    worker_id: str,
) -> list[str]:
    """Subjects captured by a per-worker stream.

    Matches
    ``sie.work.<pool>.<machine_profile>.<bundle>.<any-model>.<worker_id>``
    so any model on this concrete lane addressed to this worker lands in
    the stream.

    Args:
        pool_name: Pool name.
        machine_profile: Hardware lane.
        bundle_id: Runtime/image bundle identifier.
        worker_id: Stable worker identity.

    Returns:
        Subject list, e.g.
        ``["sie.work.default.rtx6000.sglang.*.worker-0"]``.

    Raises:
        ValueError: ``worker_id`` is empty or whitespace-only — see
            :func:`normalize_worker_id`.
    """
    return [
        f"{WORK_SUBJECT_PREFIX}.{pool_name}.{normalize_model_id(machine_profile)}."
        f"{normalize_model_id(bundle_id)}.*.{normalize_worker_id(worker_id)}"
    ]


def work_worker_consumer_name(worker_id: str) -> str:
    """Durable JetStream consumer name for the per-worker stream.

    The plan calls for ``gen-{worker_id}``. Collisions corrupt durables
    across worker boots, so callers must use a stable worker_id
    (``SIE_WORKER_ID`` env or hostname-derived).

    Uses :func:`normalize_worker_id` rather than a private sanitizer so
    the consumer name always tracks the subject token the gateway and
    worker actually publish/subscribe on. The previous ``or "worker"``
    fallback silently substituted a constant for an empty input — that
    constant collided durables across processes and is now an explicit
    error.

    Raises:
        ValueError: ``worker_id`` is empty or whitespace-only — see
            :func:`normalize_worker_id`.
    """
    return f"gen-{normalize_worker_id(worker_id)}"
