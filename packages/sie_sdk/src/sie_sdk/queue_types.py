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
    router_id: str
    reply_subject: str
    timestamp: float


class WorkItem(_WorkItemRequired, total=False):
    """A single inference work item published to NATS JetStream.

    Published by the router to ``sie.work.{model_id_normalized}.{pool_name}``.
    Consumed by workers via JetStream pull consumers.

    Attributes:
        work_item_id: Unique ID for this work item. Format: ``{request_id}.{item_index}``.
        request_id: Unique ID for the originating client request. Format: ``{router_id}-{counter}``.
        item_index: Position of this item in the original request (for result ordering).
        total_items: Total number of items in the original request.
        operation: Inference operation: ``"encode"`` | ``"score"`` | ``"extract"``.
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).
        profile_id: Profile name (e.g., ``"default"``).
        pool_name: Target pool (e.g., ``"_default"``).
        machine_profile: Required GPU type (e.g., ``"l4"``). Workers validate this.
        bundle_config_hash: Expected bundle config hash from the router's ModelRegistry.
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

        router_id: Originating router identifier (for observability).
        reply_subject: NATS subject where the worker should publish results.
        timestamp: Unix timestamp when the work item was created.
    """

    machine_profile: str
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


class _WorkResultRequired(TypedDict):
    work_item_id: str
    request_id: str
    item_index: int
    success: bool


class WorkResult(_WorkResultRequired, total=False):
    """Result for a single work item, published to the reply subject.

    Published by the worker to the ``reply_subject`` from the corresponding
    ``WorkItem``. The router collects results and reassembles the HTTP response.

    Result payloads are serialized as msgpack bytes (opaque blobs) by the worker.
    The router embeds them directly into the response without deserializing,
    keeping numpy/msgpack-numpy out of the router's dependency graph.

    Attributes:
        work_item_id: Matches the originating ``WorkItem.work_item_id``.
        request_id: Matches the originating ``WorkItem.request_id``.
        item_index: Position in the original request (for result ordering).

        success: Whether inference succeeded.
        result_msgpack: Msgpack-serialized result bytes (opaque to the router).
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


# -- NATS subject helpers ---------------------------------------------------

# NATS JetStream max message size default is 1MB.
INLINE_THRESHOLD_BYTES = 1_048_576  # 1 MB

# Stream name prefix for work queues
WORK_STREAM_PREFIX = "WORK"

# Subject prefix for work items
WORK_SUBJECT_PREFIX = "sie.work"

# Dead-letter subject prefix — uses ``sie.dlq`` (not ``sie.work.dead``)
# to avoid subject overlap with pool-level work streams (``sie.work.*.{pool}``).
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


def work_subject(model_id: str, pool_name: str) -> str:
    """Build the NATS subject for a work queue.

    Args:
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).
        pool_name: Pool name (e.g., ``"_default"``).

    Returns:
        NATS subject string (e.g., ``"sie.work.BAAI__bge-m3._default"``).
    """
    return f"{WORK_SUBJECT_PREFIX}.{normalize_model_id(model_id)}.{pool_name}"


def work_stream_name(model_id: str) -> str:
    """Build the JetStream stream name for a model's work queue.

    Args:
        model_id: Model identifier (e.g., ``"BAAI/bge-m3"``).

    Returns:
        Stream name (e.g., ``"WORK_BAAI__bge-m3"``).
    """
    return f"{WORK_STREAM_PREFIX}_{normalize_model_id(model_id)}"


def work_consumer_name(bundle_id: str, pool_name: str) -> str:
    """Build the JetStream consumer name for a (bundle, pool) pair.

    Args:
        bundle_id: Bundle identifier (e.g., ``"default"``).
        pool_name: Pool name (e.g., ``"_default"``).

    Returns:
        Consumer durable name (e.g., ``"default__default"``).
    """
    return f"{bundle_id}_{pool_name}"


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

    Matches ``sie.work.<any-single-token>.<pool_name>`` which covers all
    models for the given pool.

    Args:
        pool_name: Pool name (e.g., ``"l4"``).

    Returns:
        Subject list, e.g. ``["sie.work.*.l4"]``.
    """
    return [f"{WORK_SUBJECT_PREFIX}.*.{pool_name}"]
