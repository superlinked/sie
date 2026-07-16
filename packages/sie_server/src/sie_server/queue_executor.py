from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import msgpack
import msgspec
import yaml

from sie_server.api.ws import compute_bundle_config_hash_cached
from sie_server.config.model import ModelConfig
from sie_server.core.oom import is_oom_error
from sie_server.core.prepared import ExtractPreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.runtime_options import merge_runtime_options
from sie_server.core.score_cost import build_score_prepared_items
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.core.worker.model_worker import PreformedExtractRequest, PreformedScoreRequest
from sie_server.ipc_types import (
    ApplyModelConfigRequest,
    ApplyModelConfigResponse,
    BatchOutcome,
    DenseOutput,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    ModelDescriptor,
    MultivectorOutput,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    RawOutput,
    ReadinessState,
    ReplaceModelConfigsRequest,
    ReplaceModelConfigsResponse,
    ScoreBatchItem,
    ScoreOutputRaw,
    SetPinnedModelsRequest,
    SetPinnedModelsResponse,
    SparseOutput,
    UnitCounts,
)
from sie_server.observability.metrics import record_ipc_batch_shape
from sie_server.types.inputs import InvalidMediaError, Item, decode_item
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue-path wire error code
# ---------------------------------------------------------------------------
#
# The queue / sidecar path publishes ``ItemOutcome.error_code`` as a lowercase
# wire string the Rust gateway consumes. The gateway maps any worker code
# outside its stable set to this same ``inference_error`` discriminator (see
# ``stable_code`` in ``handlers/proxy.rs``), so this is the sidecar↔gateway
# contract value — NOT the uppercase ``ErrorCode.INFERENCE_ERROR`` enum, which
# is the *in-process HTTP* surface's generic-failure code. Keep it lowercase
# and keep it here as the single definition every queue-path emitter shares;
# do not "align" it to the HTTP enum or the gateway will fall through to its
# generic arm with a mismatched code.
_INFERENCE_ERROR_CODE: Final[str] = "inference_error"


# ---------------------------------------------------------------------------
# Rust-side output framing: every adapter that emits typed RawOutput
# ---------------------------------------------------------------------------
#
# Per-request safety rules in ``_maybe_dense_raw_output`` /
# ``_maybe_sparse_raw_output`` / ``_maybe_multivector_raw_output`` emit
# ``RawOutput`` only when the adapter's output is the exact (single-key,
# float32, well-shaped) form Rust knows how to frame byte-identically;
# everything else still falls back to the legacy
# ``msgpack.packb(_wrap_encode_output(...))`` path.
#
# Safety rules:
#   * Dense: ONLY when the adapter emits a single ``dense`` output key
#     backed by a float32 ``np.ndarray``. Binary / int8 / float16 dense
#     still go through the Python-framed fallback path so nothing
#     regresses.
#   * Sparse: ONLY when the adapter emits a single ``sparse`` output
#     key with int32 indices + float32 values (the
#     ``SparseVector(indices=..., values=...)`` shape every in-tree
#     sparse adapter produces). float16 values fall back.
#   * Multivector: ONLY for a single ``multivector`` output key with
#     a float32 ``[num_tokens, token_dims]`` ndarray. Bit-packed
#     binary multivector (``shape[1] < mv_dim``) and float16 fall back.
#   * Score: always eligible — Rust mirrors the Python sort + rank
#     assignment byte-for-byte.
#   * Multi-output items (e.g. dense + sparse in one response) always
#     fall back — the wire contract is one variant per ``RawOutput``.
#   * On any shape error the Rust publisher converts the outcome into
#     ``publish_error_and_ack`` with ``error_code="raw_output_shape_error"``.
#     We never silently drop or mis-frame a request.

# ---------------------------------------------------------------------------
# Tokenizer materialisation
# ---------------------------------------------------------------------------

# Treat ``model_max_length`` ≥ this value as "unset" — HF defaults
# slow tokenizers to ``int(1e30)`` when no cap is declared. Anything
# above 1M tokens is implausible for the encoders we run; the sidecar
# falls back to its own default cap rather than truncating after a
# trillion tokens.
_TOKENIZER_MAX_SEQ_LEN_PLAUSIBILITY_CAP: Final[int] = 1_000_000
#
# The worker-sidecar can't read the adapter's HF cache directly (different
# container in the split-image world; in single-container today it's the
# same FS but the path is hard to discover from the model_id alone). On
# first ``EnsureModelReady`` we write the adapter's ``tokenizer.json``
# to a stable, per-model path inside an emptyDir-equivalent staging
# directory and ship that path to the sidecar in
# ``ModelDescriptor.tokenizer_path``. The sidecar loads from there.
#
# The directory is configurable via ``SIE_TOKENIZER_STAGING_DIR`` for
# the split-container deploy (point both containers at the same
# ``emptyDir`` mount); defaults to ``$TMPDIR/sie-tokenizers`` so unit
# tests and dev shells need no extra config.

_TOKENIZER_STAGING_DIR = Path(
    os.environ.get("SIE_TOKENIZER_STAGING_DIR") or (tempfile.gettempdir() + "/sie-tokenizers")
)


def _safe_model_id_for_path(model_id: str) -> str:
    """Map a HF-style ``org/name`` (or worse, ``Org/Name@revision``) to
    a single filesystem-safe path component. We keep the mapping
    intentionally lossy (no reverse) because the Rust side gets the
    full path back via the descriptor, not by reconstruction.
    """
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "__" for ch in model_id)


def _materialise_tokenizer(model_id: str, tokenizer: Any) -> str | None:
    """Write the canonical ``tokenizer.json`` for ``tokenizer`` to a
    per-model path under :data:`_TOKENIZER_STAGING_DIR` and return the
    absolute path. Returns ``None`` when the tokenizer is a slow (Python)
    tokenizer that doesn't expose a ``backend_tokenizer`` — the sidecar
    cannot fast-path those anyway, so there is no value in materialising
    them.

    Idempotent on repeat invocations: if the target file already exists
    with matching size, we trust the cached descriptor (the per-process
    cache in :class:`QueueExecutor` short-circuits before we get here on
    the steady-state hot path; this routine is the cold-start writer
    plus the cross-process recovery path).

    Concurrency-safe: the temporary file name embeds ``os.getpid()`` so
    two adapter processes (or two threads in the same process taking
    different identity locks) writing the same model's tokenizer don't
    stomp each other's tmp file. POSIX ``rename`` then publishes the
    final ``tokenizer.json`` atomically — readers in the sidecar never
    see a partial write.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    try:
        raw = backend.to_str(pretty=False)
    except Exception:  # noqa: BLE001
        logger.debug("materialise_tokenizer: backend_tokenizer.to_str failed for %s", model_id, exc_info=True)
        return None
    # ``to_str`` MUST return ``str`` on a real ``tokenizers.Tokenizer``.
    # Anything else (MagicMock, None, bytes) is treated as "no canonical
    # JSON available" — the sidecar will fall back to Python tokenisation
    # for this model.
    if not isinstance(raw, str):
        return None
    canonical = raw.encode("utf-8")

    target_dir = _TOKENIZER_STAGING_DIR / _safe_model_id_for_path(model_id)
    target_path = target_dir / "tokenizer.json"
    try:
        if target_path.is_file() and target_path.stat().st_size == len(canonical):
            # Size match is a cheap proxy for byte identity. The sidecar
            # additionally hashes on its side and reconciles against the
            # ``tokenizer_id`` on the descriptor, so a false positive
            # here just means the registry refuses the registration and
            # Python keeps tokenising — never a silent mis-frame.
            return str(target_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_dir / f"tokenizer.json.{os.getpid()}.tmp"
        tmp_path.write_bytes(canonical)
        tmp_path.replace(target_path)
        return str(target_path)
    except OSError:
        logger.warning(
            "materialise_tokenizer: failed to write %s — sidecar will fall back to Python tokenisation",
            target_path,
            exc_info=True,
        )
        return None


def _maybe_dense_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Return a ``RawOutput`` for the dense-only fast path, or ``None``
    to fall back to the Python-framed fallback path.

    All conditions must hold — the gate is deliberately strict so
    Rust-side framing only intercepts cases the Rust shaper is known
    to produce byte-identical bytes for.
    """
    import numpy as np  # noqa: PLC0415

    if output_types != ["dense"]:
        return None
    if set(formatted.keys()) - {"dense"}:
        # Sparse / multivector in the same item would need different
        # Rust framers — not in v1.
        return None
    arr = formatted.get("dense")
    if not isinstance(arr, np.ndarray):
        return None
    if arr.dtype != np.float32:
        # Binary (uint8 bit-packed), float16, int8 all need different
        # framers and/or dtype tags; Python still handles them.
        return None
    if arr.ndim != 1:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
    dense_dim = dense_cfg.dim if dense_cfg else None
    dim = int(dense_dim) if dense_dim is not None else int(arr.shape[0])
    if arr.shape[0] != dim:
        # Don't try to hide a shape mismatch; fall back and let the
        # legacy path mis-label it the same way it does today rather
        # than introduce a new error class here.
        return None

    # ``arr.tolist()`` widens each ``np.float32`` to a Python ``float``
    # (f64). The Rust side narrows back to ``f32`` on decode — this
    # round-trip is exact because every f32 has a unique f64
    # representation. The Rust shaper then widens back to f64 before
    # emitting the ``msgpack_numpy`` sentinel's raw-bytes payload,
    # matching Python's ``arr.tobytes()`` bit-for-bit.
    return RawOutput(
        dense=DenseOutput(
            values=arr.tolist(),
            dim=dim,
            # v1 policy: adapter-side normalize stays in Python.
            # Flipping this to ``True`` is a later optimisation.
            normalize=False,
        ),
    )


def _maybe_sparse_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Sparse-only fast path. Returns a ``RawOutput`` carrying a
    ``SparseOutput`` when every v1 invariant holds, otherwise ``None``
    so the Python-framed fallback path takes over.

    The Rust shaper emits the exact bytes that
    ``_wrap_encode_output`` packs today — see
    ``sie_server_sidecar::output::build_sparse_payload`` and the
    byte-identity tests in
    ``test_stage1d_byte_identity.py::test_sparse_legacy_matches_rust_golden``.
    """
    import numpy as np  # noqa: PLC0415

    if output_types != ["sparse"]:
        return None
    if set(formatted.keys()) - {"sparse"}:
        return None
    sparse_in = formatted.get("sparse")
    if not isinstance(sparse_in, dict):
        return None
    indices = sparse_in.get("indices")
    values = sparse_in.get("values")
    if indices is None or values is None:
        return None
    if not isinstance(indices, np.ndarray) or not isinstance(values, np.ndarray):
        return None
    if indices.ndim != 1 or values.ndim != 1:
        return None
    if indices.shape[0] != values.shape[0]:
        return None
    # Adapter-layer convention is ``np.int32`` indices; anything
    # wider / signedness-different would reshape on .tolist() but we
    # stay strict so the gate is obvious at review time.
    if indices.dtype != np.int32:
        return None
    # v1: float32 values only. float16 (the other dtype the legacy
    # path labels) stays on Python until we teach the Rust shaper
    # the ``"<f2"`` sentinel variant.
    if values.dtype != np.float32:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    sparse_cfg = getattr(encode_task, "sparse", None) if encode_task else None
    sparse_dim = sparse_cfg.dim if sparse_cfg else None
    dims = int(sparse_dim) if sparse_dim is not None else None

    return RawOutput(
        sparse=SparseOutput(
            # ``.tolist()`` widens np.int32 → Python int (exact) and
            # np.float32 → Python float (exact round-trip). Rust
            # rehydrates via ``Vec<i32>`` / ``Vec<f32>`` with narrowing.
            indices=indices.tolist(),
            values=values.tolist(),
            dims=dims,
        ),
    )


def _maybe_multivector_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Multivector-only fast path for the Rust output shaper.

    Mirrors the invariants of the ``multivector`` branch of
    ``_wrap_encode_output``:

      * Single output key == ``multivector``.
      * ``np.float32`` 2-D ``[num_tokens, token_dims]`` ndarray.
      * NOT bit-packed binary (``shape[1] < mv_dim`` with uint8
        dtype) — the binary path stays in Python for v1 because the
        Rust shaper does not know the ``"binary"`` dtype tag yet.
      * ``shape[1]`` matches the configured ``token_dims`` when the
        model exposes one, so the Rust shaper's ``num_tokens ×
        token_dims`` invariant holds without Python-side backfill.
    """
    import numpy as np  # noqa: PLC0415

    _MV_NDIM = 2  # `[num_tokens, token_dims]` — the only shape we forward.

    if output_types != ["multivector"]:
        return None
    if set(formatted.keys()) - {"multivector"}:
        return None
    arr = formatted.get("multivector")
    if not isinstance(arr, np.ndarray):
        return None
    if arr.dtype != np.float32:
        return None
    if arr.ndim != _MV_NDIM:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    mv_cfg = getattr(encode_task, "multivector", None) if encode_task else None
    mv_dim = mv_cfg.dim if mv_cfg else None

    num_tokens = int(arr.shape[0])
    if mv_dim is not None:
        # Bit-packed binary has shape[1] == dim/8 with uint8 dtype —
        # the dtype check above already refused uint8, but also
        # refuse a narrower float32 shape just in case an adapter
        # ever pre-flattens/truncates.
        if arr.shape[1] != int(mv_dim):
            return None
        token_dims = int(mv_dim)
    else:
        token_dims = int(arr.shape[1])

    # Values must be contiguous in C order so ``.tobytes()`` (and the
    # Rust ``values.to_le_bytes()`` equivalent) agree. ``tolist()``
    # on a 2-D ndarray returns a nested list; ``ravel()`` flattens
    # row-major first so we stay byte-compatible regardless of input
    # memory layout.
    return RawOutput(
        multivector=MultivectorOutput(
            values=arr.ravel(order="C").tolist(),
            num_tokens=num_tokens,
            token_dims=token_dims,
        ),
    )


# ---------------------------------------------------------------------------
# Wire formatting helpers
# ---------------------------------------------------------------------------

# NOTE: uint8 maps to "uint8", NOT "binary". Bit-packed binary is detected by
# the explicit ``is_binary`` shape-check (``arr.shape < dim``) at each call site,
# which is the SOLE emitter of "binary"; a linear uint8 quantization keeps full
# dimensionality so it must stay labelled "uint8" to match the HTTP path
# (api/encode.py::_format_dense -> np_to_dtype). Mapping uint8->"binary" here
# made the queue path disagree with HTTP for the same request. See #1603.
_NP_DTYPE_MAP = {"float32": "float32", "float16": "float16", "int8": "int8", "uint8": "uint8"}


def _wrap_encode_output(output: dict, config: Any) -> dict:
    """Wrap raw numpy arrays from EncodeHandler.format_output into the
    DenseVector / SparseVector / MultiVector wire format the SDK expects.

    The HTTP path does this via pydantic models (see
    ``api/encode.py::_format_dense`` / ``_format_sparse`` /
    ``_format_multivector``); the queue path publishes msgpack bytes
    directly, so the same wrapping must happen here. Keeping the two
    paths in sync matters — an SDK client that round-trips via HTTP then
    via queue otherwise sees different shapes for ``sparse`` and
    ``multivector``.
    """
    import numpy as np  # noqa: PLC0415

    wrapped = dict(output)

    encode_task = getattr(config, "tasks", None)
    encode_task = getattr(encode_task, "encode", None)

    if "dense" in wrapped and isinstance(wrapped["dense"], np.ndarray):
        arr = wrapped["dense"]
        dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
        dense_dim = dense_cfg.dim if dense_cfg else None

        is_binary = arr.dtype == np.uint8 and dense_dim and arr.shape[0] < dense_dim
        dims = dense_dim if dense_dim is not None else arr.shape[0]
        dtype = "binary" if is_binary else _NP_DTYPE_MAP.get(str(arr.dtype), "float32")

        wrapped["dense"] = {"dims": int(dims), "dtype": dtype, "values": arr}

    if "sparse" in wrapped and isinstance(wrapped["sparse"], dict):
        # Adapter output shape: {"indices": np.ndarray, "values": np.ndarray}
        # SDK wire shape: {"dims": int|None, "dtype": "float32"|"float16",
        #                   "indices": np.ndarray, "values": np.ndarray}
        sparse_in = wrapped["sparse"]
        indices = sparse_in.get("indices")
        values = sparse_in.get("values")
        if indices is not None and values is not None:
            if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)
            sparse_cfg = getattr(encode_task, "sparse", None) if encode_task else None
            sparse_dim = sparse_cfg.dim if sparse_cfg else None
            dtype = _NP_DTYPE_MAP.get(str(values.dtype), "float32")
            # Sparse only supports float{32,16}; fall back rather than
            # silently mislabel.
            if dtype not in {"float32", "float16"}:
                dtype = "float32"
            wrapped["sparse"] = {
                "dims": int(sparse_dim) if sparse_dim is not None else None,
                "dtype": dtype,
                "indices": indices,
                "values": values,
            }

    if "multivector" in wrapped and isinstance(wrapped["multivector"], np.ndarray):
        arr = wrapped["multivector"]
        mv_cfg = getattr(encode_task, "multivector", None) if encode_task else None
        mv_dim = mv_cfg.dim if mv_cfg else None
        # Binary multivector packs `dim/8` bytes per token; detect by
        # `shape[1] < mv_dim` like `_format_multivector` does.
        if arr.dtype == np.uint8 and mv_dim is not None and arr.shape[1] < mv_dim:
            token_dims = int(mv_dim)
            dtype = "binary"
        else:
            token_dims = int(mv_dim if mv_dim is not None else arr.shape[1])
            dtype = _NP_DTYPE_MAP.get(str(arr.dtype), "float32")
        wrapped["multivector"] = {
            "token_dims": token_dims,
            "num_tokens": int(arr.shape[0]),
            "dtype": dtype,
            "values": arr,
        }

    return wrapped


# ---------------------------------------------------------------------------
# QueueExecutor
# ---------------------------------------------------------------------------


class QueueExecutor:
    """NATS-free execution layer fronted by the IPC server for the worker-sidecar.

    The executor owns the path from "decoded work items have arrived for model X"
    to "per-item outcomes ready to be ACKed, NAKed, or replied to". It does NOT
    own JetStream fetch, ACK/NAK, payload store fetch, or reply publish — those
    live in the worker-sidecar.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        # Per-model descriptor cache. Populated on the first
        # ``EnsureModelReady`` for a model and reused on every
        # subsequent batch's handshake (the dispatcher re-handshakes
        # per group). Keeps file I/O off the hot path:
        # ``_materialise_tokenizer`` runs once at cold start, then we
        # just hand back the cached struct. Cleared by
        # :meth:`invalidate_model_descriptor` when a model is unloaded
        # or hot-reloaded.
        self._descriptor_cache: dict[str, ModelDescriptor] = {}

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def loaded_model_names(self) -> list[str]:
        """Return sorted currently loaded model ids for sidecar health heartbeats."""
        return sorted(self._registry.loaded_model_names)

    def invalidate_model_descriptor(self, model_id: str) -> None:
        """Drop the cached descriptor for ``model_id``.

        Called when a model is unloaded or hot-reloaded so the next
        ``EnsureModelReady`` re-materialises the tokeniser and the
        sidecar picks up the new ``tokenizer_id``. Safe to call for
        unknown models (no-op).
        """
        self._descriptor_cache.pop(model_id, None)

    async def apply_model_config(self, req: ApplyModelConfigRequest) -> ApplyModelConfigResponse:
        """Validate and add a bundle-scoped config delta to the local registry."""
        if not req.bundle_id:
            msg = "bundle_id is required"
            raise ValueError(msg)
        if not req.model_config.strip():
            msg = "model_config is required"
            raise ValueError(msg)

        raw = yaml.safe_load(req.model_config)
        if not isinstance(raw, dict):
            msg = "model_config must decode to a YAML mapping"
            raise ValueError(msg)

        model_config = ModelConfig(**raw)
        if req.model_id and model_config.sie_id != req.model_id:
            msg = f"model_id mismatch: notification={req.model_id!r} config={model_config.sie_id!r}"
            raise ValueError(msg)

        updated_model_ids = await self._registry.add_config_async(model_config)
        for model_id in updated_model_ids:
            self.invalidate_model_descriptor(model_id)
        bundle_hash = compute_bundle_config_hash_cached(self._registry, req.bundle_id)
        return ApplyModelConfigResponse(
            applied=True,
            bundle_config_hash=bundle_hash,
            config_version=int(getattr(self._registry, "_config_version", 0)),
        )

    def compute_bundle_config_hash(self, bundle_id: str) -> str:
        """Return the local registry hash for ``bundle_id``."""
        if not bundle_id:
            return ""
        return compute_bundle_config_hash_cached(self._registry, bundle_id)

    async def replace_model_configs(self, req: ReplaceModelConfigsRequest) -> ReplaceModelConfigsResponse:
        """Replace the bundle-scoped registry view from a full export snapshot."""
        if not req.bundle_id:
            msg = "bundle_id is required"
            raise ValueError(msg)

        configs: list[ModelConfig] = []
        for entry in req.models:
            if not entry.model_config.strip():
                msg = "model_config is required"
                raise ValueError(msg)

            raw = yaml.safe_load(entry.model_config)
            if not isinstance(raw, dict):
                msg = "model_config must decode to a YAML mapping"
                raise ValueError(msg)

            model_config = ModelConfig(**raw)
            if entry.model_id and model_config.sie_id != entry.model_id:
                msg = f"model_id mismatch: export={entry.model_id!r} config={model_config.sie_id!r}"
                raise ValueError(msg)
            configs.append(model_config)

        invalidated = await self._registry.replace_configs_async(configs)
        for model_id in invalidated:
            self.invalidate_model_descriptor(model_id)
        bundle_hash = compute_bundle_config_hash_cached(self._registry, req.bundle_id)
        applied_models = sorted(self._registry.get_configs_snapshot(req.bundle_id))
        return ReplaceModelConfigsResponse(
            applied=True,
            bundle_config_hash=bundle_hash,
            config_version=int(getattr(self._registry, "_config_version", 0)),
            applied_models=applied_models,
        )

    async def set_pinned_models(self, req: SetPinnedModelsRequest) -> SetPinnedModelsResponse:
        """Apply the gateway's authoritative pinned-model set to the local registry."""
        pinned = await self._registry.set_pinned_models(req.models)
        return SetPinnedModelsResponse(applied=True, pinned_count=len(pinned))

    # -- Readiness ---------------------------------------------------------

    async def ensure_model_ready(self, model_id: str) -> ReadinessState:
        """Return the current readiness state for a model, triggering a load if needed.

        Mapping (for the Rust side):
        - ``ready``: continue processing
        - ``loading_started``: progress-ACK and recheck (this call triggered a new load)
        - ``loading_in_progress``: progress-ACK and recheck with a longer delay
        - ``retry_later``: NAK with base delay (unknown error path)
        - ``failed``: TERMINAL — dead-letter as ``MODEL_LOAD_FAILED`` (do NOT
          recheck). Emitted when the registry holds a PERMANENT
          ``LoadFailure`` (``cooldown=permanent``).

        #1786 fast-path twin: a permanent load failure (e.g. gated repo,
        missing dependency) otherwise collapses into ``loading_in_progress``
        here — ``start_load_async`` returns ``False`` under the
        ``registry.is_failed`` guard, indistinguishable from "still loading".
        The Rust sidecar then re-drives ``EnsureModelReady`` every ~10s
        forever and the client hangs. We consult the registry's failure
        record DIRECTLY and report the terminal ``failed`` state for the
        *permanent* classes only, using the SAME ``get_failure().is_permanent``
        classification as the direct-HTTP ``check_not_failed`` gate and the
        Modal lane's ``worker_runtime._terminal_load_failure``. Transient
        in-cooldown classes (OOM / NETWORK / TIMEOUT) are intentionally NOT
        terminal — they stay ``loading_in_progress`` so the caller retries
        after the short cooldown.
        """
        if not self._registry.has_model(model_id):
            return "retry_later"

        if self._registry.is_loaded(model_id):
            return "ready"

        failure = self._registry.get_failure(model_id)
        if failure is not None and failure.is_permanent:
            return "failed"

        if self._registry.is_loading(model_id):
            return "loading_in_progress"

        try:
            started = await self._registry.start_load_async(model_id, self._registry.device)
        except KeyError:
            # Unknown model — gateway should not be sending us work for it, but
            # be defensive: tell caller to retry rather than hard-erroring.
            return "retry_later"
        except Exception:  # noqa: BLE001
            logger.warning("ensure_model_ready: start_load_async failed for %s", model_id, exc_info=True)
            return "retry_later"

        if started:
            return "loading_started"
        # ``start_load_async`` refused to start a new attempt. Usually a
        # transient failure still in cooldown or a concurrent load already
        # in flight — both retryable. But it could also be a PERMANENT
        # failure recorded in the narrow window since the up-front check, so
        # re-read the record and dead-letter rather than re-drive forever.
        failure = self._registry.get_failure(model_id)
        if failure is not None and failure.is_permanent:
            return "failed"
        return "loading_in_progress"

    # -- Handshake-driven model descriptor --------------------------------

    def get_model_descriptor(self, model_id: str) -> ModelDescriptor | None:
        """Return the ``ModelDescriptor`` carried on the
        ``EnsureModelReadyResponse`` for this model, or ``None`` if the
        adapter has nothing structured to declare yet (slow tokenizer,
        image / audio adapter, model not yet loaded).

        The descriptor lets the worker-sidecar discover per-model
        capabilities at runtime; see
        ``packages/sie_server_sidecar/docs/architecture-guide.md``.

        Populates:

        * ``tokenizer_path`` — path to a sidecar-readable
          ``tokenizer.json`` materialised from
          ``preprocessor.backend_tokenizer.to_str(pretty=False)``. Stays
          ``None`` for slow tokenizers (no canonical JSON to ship) and
          for adapters that don't expose a ``TextPreprocessor`` (image /
          audio).
        * ``tokenizer_id`` — BLAKE3 (32 hex) of the canonical tokenizer
          JSON. Same value the sidecar will compute from the
          materialised file, so the two sides can verify byte-identity
          before enabling the Rust-tokenise fast path.
        * ``max_seq_len`` — ``tokenizer.model_max_length`` when sane
          (``< 10**6``; HF's ``VERY_LARGE_INTEGER`` sentinel reads as
          ``None`` here).
        * ``output_types`` — informational; left empty for now since
          per-request shape checks in ``_maybe_*_raw_output`` are the
          authoritative gate. Future work can populate from
          ``adapter.supported_output_types`` if/when adapters expose it.
        * ``supports_run_batch`` — every Python adapter does.

        Cached after the first successful build; the dispatcher
        re-handshakes on every batch and we don't want to re-stat the
        staging dir each time. Use
        :meth:`invalidate_model_descriptor` on unload / hot reload.
        """
        cached = self._descriptor_cache.get(model_id)
        if cached is not None:
            return cached

        # Gate on the model actually being loaded — calling this before
        # ``ensure_model_ready`` would race the registry. The IPC
        # server only invokes us on the ``ready`` branch so this is
        # belt-and-braces, but it also short-circuits MagicMock-based
        # tests where ``get_worker`` returns ``None`` for unknown ids.
        try:
            if self._registry.get_worker(model_id) is None:
                return None
        except (KeyError, AttributeError):
            return None

        tokenizer_path: str | None = None
        tokenizer_id: str | None = None
        max_seq_len: int | None = None

        try:
            preprocessor = self._registry.preprocessor_registry.get_preprocessor(model_id, "text")
        except Exception:  # noqa: BLE001
            preprocessor = None
        if preprocessor is not None and hasattr(preprocessor, "tokenizer_id"):
            try:
                candidate_id = preprocessor.tokenizer_id  # may be None for slow tokenizers
            except Exception:  # noqa: BLE001
                logger.debug("get_model_descriptor: tokenizer_id property raised for %s", model_id, exc_info=True)
                candidate_id = None
            # Strict type guard: only ``str`` survives. Test fixtures
            # often pass ``MagicMock``-based preprocessors whose
            # ``tokenizer_id`` would otherwise be a Mock; treating that
            # as a real id would crash msgpack later.
            if isinstance(candidate_id, str):
                tokenizer_id = candidate_id
            inner = getattr(preprocessor, "_tokenizer", None)
            if inner is not None:
                tokenizer_path = _materialise_tokenizer(model_id, inner)
                # ``model_max_length`` is set to a giant sentinel
                # (``int(1e30)``) when the tokenizer doesn't declare a
                # cap; treat anything implausibly large as "unset".
                raw_max = getattr(inner, "model_max_length", None)
                if (
                    isinstance(raw_max, int)
                    and not isinstance(raw_max, bool)
                    and 0 < raw_max < _TOKENIZER_MAX_SEQ_LEN_PLAUSIBILITY_CAP
                ):
                    max_seq_len = raw_max

        # Surface the adapter's model-default templates so the sidecar
        # can apply them before tokenising in Rust. All
        # text adapters store these as ``_query_template`` /
        # ``_doc_template`` (set in their ``__init__`` from the model
        # YAML). Image / audio adapters and adapters without text
        # templating leave them as ``None``, which keeps the sidecar
        # on the legacy "no Rust-side templating" path for that model.
        # ``MagicMock``-based test fixtures and adapters that don't
        # follow the convention silently degrade the same way.
        default_query_template: str | None = None
        default_doc_template: str | None = None
        try:
            worker = self._registry.get_worker(model_id)
            adapter = worker.adapter if worker is not None else None
        except Exception:  # noqa: BLE001
            adapter = None
        if adapter is not None:
            qt = getattr(adapter, "_query_template", None)
            if isinstance(qt, str):
                default_query_template = qt
            dt = getattr(adapter, "_doc_template", None)
            if isinstance(dt, str):
                default_doc_template = dt

        descriptor = ModelDescriptor(
            tokenizer_path=tokenizer_path,
            tokenizer_id=tokenizer_id,
            max_seq_len=max_seq_len,
            output_types=[],
            supports_run_batch=True,
            default_query_template=default_query_template,
            default_doc_template=default_doc_template,
        )
        self._descriptor_cache[model_id] = descriptor
        return descriptor

    def get_batch_budget(self, model_id: str) -> int | None:
        """Return the per-batch dispatch budget for a model, or ``None`` if
        unknown / model not loaded.

        Reads ``worker._batch_config.max_batch_requests`` when available. Queue
        consumers (Python or Rust) use this to cap how many messages for
        one model are processed per fetch batch so a hot model doesn't
        monopolise the GPU.
        """
        try:
            worker = self._registry.get_worker(model_id)
        except (KeyError, AttributeError):
            return None
        if worker is None:
            return None
        batch_config = getattr(worker, "_batch_config", None)
        if batch_config is None:
            return None
        budget = getattr(batch_config, "max_batch_requests", None)
        if isinstance(budget, int) and budget > 0:
            return budget
        return None

    @staticmethod
    def _options_key(options: dict[str, Any] | None) -> bytes:
        return msgpack.packb(options, use_bin_type=True) if options else b""

    @classmethod
    def _score_sub_group_sizes(cls, items: list[ScoreBatchItem]) -> list[int]:
        groups: dict[tuple[Any, ...], int] = {}
        for bi in items:
            key = (bi.instruction, cls._options_key(bi.options))
            groups[key] = groups.get(key, 0) + 1
        return list(groups.values())

    @staticmethod
    def _extract_lora(options: dict[str, Any] | None) -> str | None:
        if not options:
            return None
        raw = options.get("lora")
        if isinstance(raw, str) and raw:
            return raw
        return None

    @classmethod
    def _extract_sub_group_sizes(cls, items: list[ExtractBatchItem]) -> list[int]:
        groups: dict[tuple[Any, ...], int] = {}
        for bi in items:
            key = (
                cls._extract_lora(bi.options),
                tuple(bi.labels) if bi.labels else None,
                bi.instruction,
                cls._options_key(bi.options),
            )
            groups[key] = groups.get(key, 0) + 1
        return list(groups.values())

    # -- Encode ------------------------------------------------------------

    async def process_encode_batch(self, req: ProcessEncodeBatchRequest) -> BatchOutcome:
        """Run an encode batch through EncodePipeline and return per-item outcomes.

        Sub-grouping: items from different API requests may have different ``output_types``,
        ``instruction``, ``is_query``, or ``options`` — these cannot share a
        single ``EncodePipeline.run_encode()`` call.
        """
        from sie_server.core.encode_pipeline import EncodePipeline  # noqa: PLC0415

        model_id = req.model_id
        items = req.items

        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            # Model evicted mid-batch — NAK all items so another worker (or
            # this one after re-loading) processes them.
            return BatchOutcome(outcomes=[_nak_outcome(it) for it in items])

        outcomes: dict[str, ItemOutcome] = {}

        # Group key includes `profile_id` and `bundle_config_hash` — the
        # gateway uses those to select adapter variants / postprocessors,
        # and merging items with different values would run the wrong
        # pipeline. Keep the key in sync with whatever `EncodePipeline.run_encode`
        # actually reads; add more fields here if it grows.
        groups: dict[tuple, list[EncodeBatchItem]] = {}
        for bi in items:
            output_types = tuple(bi.output_types or ["dense"])
            options_key = msgpack.packb(bi.options, use_bin_type=True) if bi.options else b""
            key = (
                output_types,
                bi.instruction,
                bi.is_query,
                options_key,
                bi.profile_id,
                bi.bundle_config_hash,
            )
            groups.setdefault(key, []).append(bi)

        # Record IPC-batch shape so operators can see the fragmentation
        # ratio (items per IPC batch vs items per GPU forward pass).
        # Emitted once per batch *after* grouping but *before* running
        # any forward pass, so batches whose inference throws are still
        # accounted for in the histogram. Model-eviction NAKs above
        # are intentionally not recorded — no real work was attempted.
        record_ipc_batch_shape(
            model=model_id,
            endpoint="encode",
            total_items=len(items),
            sub_group_sizes=[len(g) for g in groups.values()],
        )

        for group_key, group in groups.items():
            (output_types_t, instruction, is_query, options_key, _profile_id, _bundle_hash) = group_key
            output_types = list(output_types_t)
            options = msgpack.unpackb(options_key, raw=False) if options_key else {}
            # Validate each item against the typed Item contract at the seam
            # (parity with the HTTP path). A per-item decode failure is isolated
            # as an INVALID_INPUT outcome so one malformed item cannot fail its
            # whole sub-group. See decode_item / issue #1537.
            good_group: list[EncodeBatchItem] = []
            server_items: list[Item] = []
            for bi in group:
                try:
                    server_items.append(decode_item(bi.item))
                except (msgspec.ValidationError, InvalidMediaError) as decode_exc:
                    outcomes[bi.work_item_id] = _inference_exception_outcome(bi, decode_exc)
                    continue
                good_group.append(bi)
            if not good_group:
                continue
            # Collect worker-sidecar prepared_tokens aligned with
            # ``server_items``. ``None`` per item is expected for the
            # v1 safety-rule skips (`is_query`, `instruction`,
            # non-text, empty text). The pipeline accepts the mix
            # per-item: items with usable Rust bytes skip Python
            # tokenisation, items with ``None`` are tokenised in
            # Python and spliced back — see
            # ``TextPreprocessor.try_prepare_from_prepared_tokens``
            # for the hybrid policy. Whole-batch fallback only fires
            # on correctness-critical drift (tokenizer_id mismatch,
            # malformed wire shape).
            prepared_tokens_per_item = [bi.prepared_tokens for bi in good_group]
            if not any(pt is not None for pt in prepared_tokens_per_item):
                # Not a single fast-path candidate — pass None so the
                # pipeline doesn't even bother looking up the
                # preprocessor's cached tokenizer_id.
                prepared_tokens_per_item = None

            try:
                # The Rust gateway publishes only the raw SDK options to the
                # queue, so — unlike the single-server HTTP path (api.encode) —
                # profile ``adapter_options.runtime`` defaults (query_template,
                # default_instruction, pooling, normalize, …) are not yet merged
                # in. Merge them here so the adapter sees the same effective
                # options regardless of ingress; without this, instruction-tuned
                # embedders silently lose their query template on the cluster
                # path (#1489). An unknown profile name raises ValueError, which
                # the surrounding except turns into per-item failures.
                options = merge_runtime_options(config, options)
                # Profile-default ``output_types`` parity with the OSS HTTP path
                # (api.encode resolves ``profile > request > default``). The
                # gateway forwards only the request-level output_types, so a
                # profile whose runtime declares an output_types default (e.g.
                # the ``bge-m3:sparse`` variant, whose promoted "default" profile
                # carries ``output_types: [sparse]``) would otherwise be served
                # dense-only here — the managed path silently dropped it. Reuse
                # the merged profile runtime (same resolver, no duplicated
                # precedence) and apply it exactly as api.encode does. The group
                # key already folded request→``["dense"]``, so this is precisely
                # ``profile or request or default``, keeping managed == OSS (P6.6).
                profile_output_types = options.get("output_types")
                if profile_output_types:
                    output_types = list(profile_output_types)
                formatted_outputs, timing = await EncodePipeline.run_encode(
                    registry=self._registry,
                    model=model_id,
                    items=server_items,
                    output_types=output_types,
                    instruction=instruction,
                    config=config,
                    is_query=is_query,
                    options=options,
                    prepared_tokens_per_item=prepared_tokens_per_item,
                    preformed_batch=True,
                )

                if len(formatted_outputs) != len(good_group):
                    # Output/input length mismatch means the adapter
                    # silently dropped items. Surface as a per-item error
                    # rather than publishing empty success results.
                    logger.warning(
                        "Encode sub-batch for %s returned %d outputs for %d items — emitting per-item errors",
                        model_id,
                        len(formatted_outputs),
                        len(good_group),
                    )
                    for bi in good_group:
                        outcomes[bi.work_item_id] = _error_outcome(
                            bi,
                            _INFERENCE_ERROR_CODE,
                            "adapter returned fewer outputs than items",
                        )
                else:
                    # Authoritative unit counts for metering: per-item real
                    # tokenizer counts recorded by the pipeline during
                    # tokenization (see ``RequestTiming.input_token_counts``).
                    # ``None`` (image path / char-count estimators) leaves
                    # ``ItemOutcome.units`` unset — metering edges fall back
                    # to their reserve estimate rather than bill an estimate
                    # as a count.
                    token_counts = timing.input_token_counts
                    if token_counts is not None and len(token_counts) != len(good_group):
                        token_counts = None  # misaligned — never mis-attribute counts
                    # Authoritative per-image counts for the §7 "$ per image"
                    # dimension: any vision adapter (CLIP/SigLIP) inherits the
                    # base ``count_input_images`` hook, so an image-input encode
                    # bills per image the same way a text encode bills per
                    # token. ``None`` (adapter evicted, or an all-text batch)
                    # leaves ``images`` unset. Aligned 1:1 with ``server_items``
                    # (== ``good_group`` order).
                    try:
                        encode_adapter = self._registry.get(model_id)
                    except KeyError:
                        encode_adapter = None
                    image_counts = _per_item_image_counts(encode_adapter, server_items, len(good_group))
                    for idx, bi in enumerate(good_group):
                        raw_output: RawOutput | None = None
                        result_msgpack: bytes | None = None
                        # Dispatch by the single declared output type —
                        # the v1 wire contract is exactly one variant
                        # per ``RawOutput``. Multi-output items (no
                        # single key matches) and adapter outputs that
                        # don't pass the per-helper safety rules drop
                        # to the legacy ``_wrap_encode_output`` path
                        # below.
                        if output_types == ["dense"]:
                            raw_output = _maybe_dense_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        elif output_types == ["sparse"]:
                            raw_output = _maybe_sparse_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        elif output_types == ["multivector"]:
                            raw_output = _maybe_multivector_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        if raw_output is None:
                            output = _wrap_encode_output(formatted_outputs[idx], config)
                            # Echo the caller's item id (G2b, P2.8 finding): the
                            # HTTP path stamps ``result["id"] = item.id`` in
                            # ``api.encode._build_response_items`` and the SDK
                            # copies it back (``parse_encode_results``), but the
                            # queue-path result blob dropped it. Bake it into the
                            # legacy blob here so it round-trips like SCORE's
                            # ``item_id``. The RawOutput fast paths carry the id
                            # at framing time instead (frame_raw_output on the
                            # lane); this branch is the multi-output / non-f32
                            # fallback.
                            item_id = server_items[idx].id
                            if item_id is not None:
                                output = {"id": item_id, **output}
                            result_msgpack = msgpack.packb(output, use_bin_type=True)
                        outcomes[bi.work_item_id] = ItemOutcome(
                            work_item_id=bi.work_item_id,
                            request_id=bi.request_id,
                            item_index=bi.item_index,
                            disposition="publish_and_ack",
                            result_msgpack=result_msgpack,
                            raw_output=raw_output,
                            inference_ms=timing.inference_ms,
                            tokenization_ms=timing.tokenization_ms if timing.tokenization_ms > 0 else None,
                            postprocessing_ms=timing.postprocessing_ms if timing.postprocessing_ms > 0 else None,
                            units=_encode_units(
                                token_counts[idx] if token_counts is not None else None,
                                image_counts[idx] if image_counts is not None else None,
                            ),
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning("Encode sub-batch failed for model %s: %s", model_id, e)
                for bi in good_group:
                    outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)

        return BatchOutcome(outcomes=[outcomes[bi.work_item_id] for bi in items])

    # -- Score -------------------------------------------------------------

    async def process_score_batch(self, req: ProcessScoreBatchRequest) -> BatchOutcome:
        """Run score work items from the sidecar IPC path.

        The worker-sidecar owns queue batching and scheduling. Python prepares
        each score work item and executes it through ModelWorker's pre-formed
        batch entrypoint so it does not re-enter the Python BatchFormer.
        """
        model_id = req.model_id
        if not req.items:
            return BatchOutcome(outcomes=[])

        record_ipc_batch_shape(
            model=model_id,
            endpoint="score",
            total_items=len(req.items),
            sub_group_sizes=self._score_sub_group_sizes(req.items),
        )

        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for score: %s — NAKing", model_id, e)
            return BatchOutcome(outcomes=[_nak_outcome(bi) for bi in req.items])

        outcomes: dict[str, ItemOutcome] = {}
        requests: list[PreformedScoreRequest] = []
        request_context: list[tuple[ScoreBatchItem, Item, list[Item]]] = []

        for bi in req.items:
            try:
                query_item = decode_item(bi.query_item)
                score_items = [decode_item(it) for it in bi.score_items]

                timing = RequestTiming()
                timing.start_tokenization()
                # ``score_pair_cost`` here is the char-count BATCHING proxy
                # only. Authoritative billable counts (§7.3) are the reranker's
                # real per-pair tokenizer lengths, surfaced on the resulting
                # ScoreOutput and summed into ``ItemOutcome.units`` in
                # ``_score_success_outcome`` — never this proxy.
                prepared_items = build_score_prepared_items(query_item, score_items)
                timing.end_tokenization()

                requests.append(
                    PreformedScoreRequest(
                        prepared_items=prepared_items,
                        query=query_item,
                        items=score_items,
                        instruction=bi.instruction,
                        options=bi.options or {},
                        request_id=bi.request_id,
                        timing=timing,
                    )
                )
                request_context.append((bi, query_item, score_items))
            except Exception as e:  # noqa: BLE001
                logger.warning("Score preparation failed for %s: %s", bi.work_item_id, e)
                outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)

        # Adapter for the metering backfill (§7.3). Read via the registry — the
        # same sync accessor the encode seam uses — so a reranker that owns its
        # tokenization can re-derive real per-pair counts. ``None`` (evicted
        # mid-batch) simply leaves the meter on its reserve estimate.
        try:
            score_adapter = self._registry.get(model_id)
        except KeyError:
            score_adapter = None

        if requests:
            try:
                futures = await worker.submit_score_preformed_batch(requests)
                results = await asyncio.gather(*futures, return_exceptions=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Score batch failed for model %s: %s", model_id, e)
                for bi, _query_item, _score_items in request_context:
                    outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)
            else:
                for (bi, query_item, score_items), result in zip(request_context, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.warning("Score failed for %s: %s", bi.work_item_id, result)
                        outcomes[bi.work_item_id] = _inference_exception_outcome(bi, result)
                    else:
                        _backfill_score_units(score_adapter, bi, query_item, score_items, result)
                        outcomes[bi.work_item_id] = _score_success_outcome(
                            bi,
                            score_items,
                            result,
                        )

        return BatchOutcome(outcomes=[outcomes[bi.work_item_id] for bi in req.items])

    # -- Extract -----------------------------------------------------------

    async def process_extract_batch(self, req: ProcessExtractBatchRequest) -> BatchOutcome:
        """Run extract work items from the sidecar IPC path."""
        model_id = req.model_id
        if not req.items:
            return BatchOutcome(outcomes=[])

        record_ipc_batch_shape(
            model=model_id,
            endpoint="extract",
            total_items=len(req.items),
            sub_group_sizes=self._extract_sub_group_sizes(req.items),
        )

        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for extract: %s — NAKing", model_id, e)
            return BatchOutcome(outcomes=[_nak_outcome(bi) for bi in req.items])

        outcomes: dict[str, ItemOutcome] = {}
        grouped_requests: dict[str | None, list[PreformedExtractRequest]] = {}
        # Carry the decoded ``Item`` alongside each ``ExtractBatchItem`` so the
        # success path can bill vision extract (Florence-2) per image via the
        # adapter's ``count_input_images`` hook (§7 "$ per image").
        grouped_context: dict[str | None, list[tuple[ExtractBatchItem, Item]]] = {}

        # Adapter for the per-image metering seam (§7), read via the registry —
        # the same sync accessor the encode/score seams use. ``None`` (evicted
        # mid-batch) simply leaves the images dimension unset.
        try:
            extract_adapter = self._registry.get(model_id)
        except KeyError:
            extract_adapter = None

        for bi in req.items:
            try:
                server_item = decode_item(bi.item)

                timing = RequestTiming()
                timing.start_tokenization()
                # ``char_count`` here is the BATCHING cost proxy only.
                # Authoritative billable counts (§7.3) are the extractor's real
                # per-doc tokenizer length, surfaced on the resulting
                # ExtractOutput and stamped into ``ItemOutcome.units`` in
                # ``_extract_success_outcome`` — never this proxy. (``pages``
                # for future docling/OCR inputs stays a separate seam.)
                char_count = len(server_item.text) if server_item.text else 0
                prepared_items = [ExtractPreparedItem(cost=char_count, original_index=0)]
                timing.end_tokenization()

                lora = self._extract_lora(bi.options)
                grouped_requests.setdefault(lora, []).append(
                    PreformedExtractRequest(
                        prepared_items=prepared_items,
                        items=[server_item],
                        labels=bi.labels,
                        output_schema=bi.output_schema,
                        instruction=bi.instruction,
                        options=bi.options or {},
                        request_id=bi.request_id,
                        timing=timing,
                    )
                )
                grouped_context.setdefault(lora, []).append((bi, server_item))
            except Exception as e:  # noqa: BLE001
                logger.warning("Extract preparation failed for %s: %s", bi.work_item_id, e)
                outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)

        for lora, requests_for_lora in grouped_requests.items():
            context = grouped_context[lora]
            try:
                futures = await worker.submit_extract_preformed_batch(requests_for_lora, lora=lora)
                results = await asyncio.gather(*futures, return_exceptions=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Extract batch failed for model %s: %s", model_id, e)
                for bi, _server_item in context:
                    outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)
            else:
                for (bi, server_item), result in zip(context, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.warning("Extract failed for %s: %s", bi.work_item_id, result)
                        outcomes[bi.work_item_id] = _inference_exception_outcome(bi, result)
                    else:
                        outcomes[bi.work_item_id] = _extract_success_outcome(extract_adapter, bi, server_item, result)

        return BatchOutcome(outcomes=[outcomes[bi.work_item_id] for bi in req.items])


def _per_item_image_counts(adapter: Any, items: list[Item], expected_len: int) -> list[int] | None:
    """Per-item authoritative input-image counts via the adapter's shared
    ``count_input_images`` hook (§7 "$ per image").

    Returns ``None`` — leaving the images dimension unset so the metering edge
    stays on its token/reserve basis — on a missing adapter or any misaligned /
    malformed list, so a per-image count is never mis-attributed. Never raises:
    metering must not fail inference.
    """
    if adapter is None:
        return None
    try:
        counts = adapter.count_input_images(items)
    except Exception:  # noqa: BLE001 — metering must never fail inference
        return None
    if (
        isinstance(counts, list)
        and len(counts) == expected_len
        and all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts)
    ):
        return counts
    return None


def _encode_units(token_count: int | None, image_count: int | None) -> UnitCounts | None:
    """Assemble one encode item's ``UnitCounts`` from its authoritative token
    and image counts.

    Each dimension is set only when present, and ``images`` only when ``> 0``
    (a text-only item never emits ``images=0``). An item with neither yields
    ``None`` so the metering edge falls back to its reserve estimate. Kept
    byte-identical to the prior ``UnitCounts(input_tokens=...)`` emitter when
    there are no images: ``images`` defaults to ``None`` on the struct.
    """
    images = image_count if (image_count is not None and image_count > 0) else None
    if token_count is None and images is None:
        return None
    return UnitCounts(input_tokens=token_count, images=images)


def _with_images(units: UnitCounts | None, image_count: int | None) -> UnitCounts | None:
    """Fold an authoritative image count into an existing ``UnitCounts`` (§7
    "$ per image"), minting one when only images are present.

    ``image_count`` of ``None`` / ``<= 0`` is a no-op (never emits ``images=0``)
    so text extract (GLiNER, …) is unchanged.
    """
    if image_count is None or image_count <= 0:
        return units
    if units is None:
        return UnitCounts(images=image_count)
    return UnitCounts(input_tokens=units.input_tokens, pages=units.pages, images=image_count)


def _with_pages(units: UnitCounts | None, page_count: int | None) -> UnitCounts | None:
    """Fold an authoritative page count into an existing ``UnitCounts`` — the §7
    canonical parse/OCR dimension ("$ per 1k pages") — minting one when only
    pages are present.

    ``page_count`` of ``None`` / ``<= 0`` is a no-op (never emits ``pages=0``)
    so token/vision extract is unchanged. Preserves the token and image
    dimensions so folds compose in any order.
    """
    if page_count is None or page_count <= 0:
        return units
    if units is None:
        return UnitCounts(pages=page_count)
    return UnitCounts(input_tokens=units.input_tokens, pages=page_count, images=units.images)


def _page_total(pages: Any, expected_len: int) -> int | None:
    """Sum an adapter-surfaced per-item page list (``ExtractOutput.pages``) into a
    single billable page count for the work item.

    Returns ``None`` — leaving the pages dimension unset so the meter falls back
    to its reserve estimate — unless the list is well-formed (aligned 1:1 with
    the item's outputs, non-negative ints) and sums to ``> 0``. A misaligned or
    malformed list is dropped rather than mis-attributed.
    """
    if not isinstance(pages, list) or len(pages) != expected_len:
        return None
    if not all(isinstance(p, int) and not isinstance(p, bool) and p >= 0 for p in pages):
        return None
    total = sum(pages)
    return total if total > 0 else None


def _units_from_token_counts(counts: Any, expected_len: int) -> UnitCounts | None:
    """Sum authoritative per-item token counts into a work item's ``UnitCounts``.

    Mirrors the encode metering contract (§7.3): billing counts, never
    estimates. Returns ``None`` — leaving ``ItemOutcome.units`` unset so the
    metering edge falls back to its reserve estimate — unless the adapter
    surfaced a well-formed list aligned 1:1 with the item's outputs. A
    misaligned or malformed list is dropped rather than mis-attributed.
    """
    if not isinstance(counts, list) or len(counts) != expected_len:
        return None
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in counts):
        return None
    return UnitCounts(input_tokens=sum(int(c) for c in counts))


def _backfill_score_units(
    adapter: Any,
    bi: ScoreBatchItem,
    query_item: Item,
    score_items: list[Item],
    worker_result: Any,
) -> None:
    """Shared metering seam: stamp authoritative per-pair token counts onto the
    ``ScoreOutput`` when the reranker did not surface them itself.

    Rerankers that own their tokenization but don't populate
    ``ScoreOutput.input_token_counts`` (every flash cross-encoder) otherwise
    leave the meter blind. The base ``count_pair_input_tokens`` hook recovers
    the real joint (query, doc) lengths with the adapter's own tokenizer — the
    §7.3 basis the in-tree ``cross_encoder`` already surfaces. Pure fallback:
    never overwrites counts an adapter already produced (so bge-m3 / cross_encoder
    keep their exact values), and a ``None`` recovery (server-backed adapters)
    leaves the meter on its reserve estimate.
    """
    if adapter is None:
        return
    output = getattr(worker_result, "output", None)
    if output is None or getattr(output, "input_token_counts", None) is not None:
        return
    try:
        counts = adapter.count_pair_input_tokens(query_item, score_items, instruction=bi.instruction)
    except Exception:  # noqa: BLE001 — metering must never fail inference
        return
    if (
        isinstance(counts, list)
        and len(counts) == output.batch_size
        and all(isinstance(count, int) and not isinstance(count, bool) for count in counts)
    ):
        output.input_token_counts = counts


def _score_success_outcome(
    bi: ScoreBatchItem,
    score_items: list[Item],
    worker_result: Any,
) -> ItemOutcome:
    score_output = worker_result.output
    raw_scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]
    item_ids: list[str] = [
        (sid if (sid := score_items[i].id) is not None else f"item-{i}") for i in range(score_output.batch_size)
    ]
    # Authoritative unit counts (§7.3): the reranker tokenizes each (query, doc)
    # pair, and the score handler carries those real per-pair counts on the
    # assembled ScoreOutput. Bill their sum — the total input tokens the model
    # processed for this query × N-docs work item — as $/1M input tokens
    # (§7.1). ``None`` (char-proxy rerankers) leaves units unset → reserve
    # fallback, never an estimate billed as a count.
    units = _units_from_token_counts(
        getattr(score_output, "input_token_counts", None),
        score_output.batch_size,
    )

    # Score output is always Rust-frameable: the Python and Rust
    # sort/rank paths produce byte-identical results (see the
    # parity test in ``test_queue_executor_stage1d.py``).
    # Rust-side framing is unconditional for score; the Python-framed fallback
    # sort+pack lives only as a doc-comment record of what
    # ``sie_server_sidecar::output::build_score_payload`` mirrors.
    raw_output: RawOutput | None = RawOutput(
        score=ScoreOutputRaw(scores=raw_scores, item_ids=item_ids),
    )
    result_msgpack: bytes | None = None

    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="publish_and_ack",
        result_msgpack=result_msgpack,
        raw_output=raw_output,
        inference_ms=worker_result.timing.inference_ms,
        tokenization_ms=worker_result.timing.tokenization_ms if worker_result.timing.tokenization_ms > 0 else None,
        postprocessing_ms=worker_result.timing.postprocessing_ms
        if worker_result.timing.postprocessing_ms > 0
        else None,
        units=units,
    )


def _extract_success_outcome(
    adapter: Any,
    bi: ExtractBatchItem,
    server_item: Item,
    worker_result: Any,
) -> ItemOutcome:
    extract_output = worker_result.output
    extraction_results = ExtractHandler.format_output(extract_output)
    if not extraction_results:
        # Adapter returned no results for a single-item request —
        # surface as an error instead of silently publishing an
        # empty object (which the client would parse as "success
        # with no fields").
        return _error_outcome(bi, "inference_error", "adapter returned no extraction results")
    result_msgpack = msgpack.packb(extraction_results[0], use_bin_type=True)

    # Authoritative unit counts (§7.3): the extractor tokenizes the document
    # and the extract handler carries that real per-doc count on the assembled
    # ExtractOutput. Extract work items are single-doc, so ``batch_size == 1``;
    # bill the count as $/1M input tokens (§7.1). ``None`` leaves units unset →
    # reserve fallback, never an estimate billed as a count.
    units = _units_from_token_counts(
        getattr(extract_output, "input_token_counts", None),
        extract_output.batch_size,
    )
    # Parse/OCR extract (docling) has no token count but parses document PAGES —
    # bill it per page (§7 "$ per 1k pages", the canonical parse dimension) from
    # the real page count the adapter surfaced on ``ExtractOutput.pages``. Folds
    # alongside any token/image count; token/vision extract (no pages) unchanged.
    units = _with_pages(units, _page_total(getattr(extract_output, "pages", None), extract_output.batch_size))
    # Vision extract (Florence-2 caption/OCR/extract) has no token count but
    # consumes image inputs — bill it per image (§7 "$ per image") via the
    # shared ``count_input_images`` hook. Folds alongside any token count;
    # text extract (GLiNER, single text doc, no images) is unchanged.
    image_counts = _per_item_image_counts(adapter, [server_item], 1)
    units = _with_images(units, image_counts[0] if image_counts is not None else None)

    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="publish_and_ack",
        result_msgpack=result_msgpack,
        inference_ms=worker_result.timing.inference_ms,
        tokenization_ms=worker_result.timing.tokenization_ms if worker_result.timing.tokenization_ms > 0 else None,
        postprocessing_ms=worker_result.timing.postprocessing_ms
        if worker_result.timing.postprocessing_ms > 0
        else None,
        units=units,
    )


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def _nak_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="nak_retry",
        nak_delay_ms=int(_default_nak_delay_s() * 1000),
    )


def _oom_nak_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="nak_retry",
        nak_delay_ms=int(_oom_nak_delay_s() * 1000),
    )


def _inference_exception_outcome(
    bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem,
    exc: BaseException,
) -> ItemOutcome:
    if is_oom_error(exc):
        return _oom_nak_outcome(bi)
    if isinstance(exc, (InvalidMediaError, msgspec.ValidationError)):
        # A typed-decode failure (decode_item) or a media contract violation;
        # both surface as INVALID_INPUT (HTTP 400), matching the HTTP path.
        return _error_outcome(bi, ErrorCode.INVALID_INPUT.value, str(exc))
    return _error_outcome(bi, _INFERENCE_ERROR_CODE, str(exc))


def _error_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem, code: str, message: str) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="publish_error_and_ack",
        error=message,
        error_code=code,
    )


def _default_nak_delay_s() -> float:
    """NAK delay (seconds) used when a model is evicted / not yet ready.

    Must agree with the worker-sidecar's ``SIE_NAK_DELAY_S`` default so
    redelivery behaviour is consistent across either side choosing the
    fallback.
    """
    return float(os.environ.get("SIE_NAK_DELAY_S", "5.0"))


def _oom_nak_delay_s() -> float:
    """NAK delay (seconds) for retryable OOM / RESOURCE_EXHAUSTED failures.

    This mirrors the deleted Python NATS worker contract: do not publish an
    error and ACK for OOM, because JetStream redelivery may land on a sibling
    worker after memory pressure clears.
    """
    return float(os.environ.get("SIE_OOM_NAK_DELAY_S", "10.0"))
