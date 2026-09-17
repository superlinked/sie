"""SGLang adapter for large LLM embedding models (4B+).

SGLang provides memory-efficient inference for LLM embedding models by
pre-allocating KV cache. This prevents OOM under concurrent load that
PyTorch-based adapters can experience with 4B+ models.

Target models:
- Qwen3-Embedding-4B, Qwen3-Embedding-8B
- GTE-Qwen2-7B-instruct
- E5-Mistral-7B-instruct, SFR-Embedding-Mistral
- LLaMA-Embed-Nemotron-8B, NV-Embed-v2

Implementation: Uses SGLang's HTTP server mode (subprocess) rather than the
Engine API to avoid event loop conflicts with uvicorn/uvloop.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters.sglang import _server
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.tokenizer import load_tokenizer

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Fan-out knob for the SGLang embedding HTTP client. A single blocking
# ``requests.post`` per ``encode()`` presents only ONE request to SGLang's
# continuous batcher at a time — on the single-threaded serving loop (local
# ``sie-server serve`` / ``ModelWorker`` runs one forward at a time) that
# starves the batcher and the served path lands far below the engine knee
# (measured Qwen3-Embedding-4B: A100 ~13k served vs 30.8k engine @ C=16;
# evidence: commit ``9ecae860a``). Sharding a
# batch into ``concurrency`` concurrent POSTs keeps that many requests in
# flight so SGLang's batcher fills. ``1`` restores the legacy single-post
# behavior (used to A/B the lift). Overridable per-deploy via the env var.
_EMBED_CONCURRENCY_ENV = "SIE_SGLANG_EMBED_CONCURRENCY"
_DEFAULT_EMBED_CONCURRENCY = 8


_EMBED_READ_TIMEOUT_S = 60.0
"""Read cap for one embedding POST.

A stalled collective produces no bytes and no error, so an unbounded read
would hold the request, and every accelerator in the group, open forever.
"""


def _resolve_embed_concurrency(configured: int | None) -> int:
    """Resolve the embedding POST concurrency (env override wins, then config)."""
    raw = os.environ.get(_EMBED_CONCURRENCY_ENV)
    if raw is not None and raw.strip():
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; expected an integer", _EMBED_CONCURRENCY_ENV, raw)
        else:
            if value >= 1:
                return value
            logger.warning("Ignoring %s=%r; must be >= 1", _EMBED_CONCURRENCY_ENV, raw)
    if configured is not None and configured >= 1:
        return configured
    return _DEFAULT_EMBED_CONCURRENCY


class SGLangEmbeddingAdapter(BaseAdapter):
    """Adapter for LLM embedding models using SGLang HTTP server backend.

    SGLang pre-allocates GPU memory for the KV cache, providing stable memory
    usage under concurrent load. This is critical for 4B+ LLM embeddings that
    would otherwise OOM with dynamic memory allocation.

    Key differences from PyTorchEmbeddingAdapter:
    - Memory is pre-allocated at load time (controlled by mem_fraction_static)
    - Uses SGLang's HTTP server (subprocess) for inference
    - Supports last-token pooling only (standard for LLM embeddings)

    Note: This adapter starts SGLang as a subprocess server during load().
    The child process owns its own main thread and signal handlers, so parent
    load can run in the registry's model-load executor.

    Example:
        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            mem_fraction_static=0.5,
        )
        adapter.load("cuda:0")
        results = adapter.encode([Item(text="hello")], ["dense"])
    """

    spec = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=("_process", "_server_url", "_dense_dim"))

    # The SGLang child process owns signal handling; parent load must not block
    # the uvicorn event loop while polling child readiness.
    requires_main_thread: bool = False
    manages_own_load_timeout: bool = True

    def _check_loaded(self) -> None:
        if self._server_url is None:
            raise RuntimeError(ERR_NOT_LOADED)
        process = self._process
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        # Without this the adapter never notices its engine exiting, and every
        # later request opens a connection to a port nobody is listening on,
        # waits out the connect timeout and returns a transport error, with no
        # reload and no readiness change.
        msg = (
            f"SGLang engine for {self._model_name_or_path!r} is not running "
            f"(process exited with code {exit_code}). The model must be reloaded."
        )
        raise RuntimeError(msg)

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        mem_fraction_static: float = 0.85,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        revision: str | None = None,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
        append_eos: bool = False,
        lora_paths: dict[str, str] | None = None,
        max_loras_per_batch: int = 8,
        dense_dim: int | None = None,
        startup_timeout_s: float | None = None,
        embed_concurrency: int | None = None,
        # Accepted for the same reason the generation adapter accepts it: the
        # width is a property of a load, not of a task. No embedding profile
        # needs it today, and one is the byte-identical single-device case.
        tensor_parallel_size: int = 1,
        disable_piecewise_cuda_graph: bool | None = None,
        request_read_timeout_s: float | None = None,
        watchdog_timeout_s: float | None = None,
        nccl_port: int | None = None,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        r"""Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize embeddings.
            max_seq_length: Maximum sequence length.
            mem_fraction_static: Fraction of GPU memory to pre-allocate (0.0-1.0).
                Lower values leave more headroom for other models. Default 0.85.
            compute_precision: Compute precision (bfloat16 recommended).
            trust_remote_code: Whether to trust remote code in model files.
            revision: Pinned HuggingFace revision (commit SHA / branch / tag) to serve.
                Forwarded to ``sglang.launch_server`` as ``--revision`` so the served
                weights match the YAML-pinned SHA. None serves the repo default.
            query_template: Template for formatting queries. Use {instruction} and
                {text} placeholders. Example: "Instruct: {instruction}\nQuery:{text}"
            doc_template: Template for formatting documents. Use {text} placeholder.
            default_instruction: Default instruction when query_template uses
                {instruction} but none is provided.
            append_eos: When True, append the model's EOS token to every text
                sent to SGLang (after template formatting). Required for
                last-token-pooled LLM embedders whose tokenizer does NOT add an
                EOS automatically (e.g. the Mistral-family embedders
                e5-mistral-7b-instruct / SFR-Embedding / Linq-Embed-Mistral):
                without the trailing EOS, last-token pooling reads the wrong
                position and retrieval collapses (#1489). The EOS string is read
                from the model's own tokenizer at load time (see ``load``), so it
                is never hardcoded. Leave False for models SGLang already handles
                (e.g. Qwen3-Embedding).
            lora_paths: LoRA adapters to load. Dict mapping adapter name to path.
                Example: {"legal": "org/legal-lora", "medical": "/path/to/medical"}.
                At request time, select via lora parameter in encode().
            max_loras_per_batch: Maximum LoRA adapters per batch. Default 8.
            dense_dim: Configured dense embedding dimension.
            startup_timeout_s: SGLang startup-health timeout in seconds.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._mem_fraction_static = mem_fraction_static
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._query_template = query_template
        self._doc_template = doc_template
        self._default_instruction = default_instruction
        self._append_eos = append_eos
        self._eos_token = ""  # Resolved from the model tokenizer in load() when append_eos
        self._lora_paths = lora_paths or {}
        self._max_loras_per_batch = max_loras_per_batch
        self._startup_timeout_s = _server.resolve_startup_timeout(startup_timeout_s)
        self._embed_concurrency = _resolve_embed_concurrency(embed_concurrency)
        self._tensor_parallel_size = _server.validate_tensor_parallel_size(tensor_parallel_size)
        self._disable_piecewise_cuda_graph = (
            self._tensor_parallel_size > 1 if disable_piecewise_cuda_graph is None else disable_piecewise_cuda_graph
        )
        self._request_read_timeout_s = (
            _server.validate_optional_positive(request_read_timeout_s, field="request_read_timeout_s")
            or _EMBED_READ_TIMEOUT_S
        )
        self._watchdog_timeout_s = _server.validate_optional_positive(watchdog_timeout_s, field="watchdog_timeout_s")
        self._declared_nccl_port = _server.validate_optional_port(nccl_port)
        if self._declared_nccl_port is not None and self._tensor_parallel_size == 1:
            msg = "nccl_port applies only above tensor_parallel_size=1; a single rank has no rendezvous"
            raise ValueError(msg)
        self._nccl_port: int | None = None
        # Declared or chosen, the rendezvous port is reserved in the shared set,
        # so teardown hands back exactly what this adapter holds.
        self._reserved_nccl_port: int | None = None

        self._process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._port: int | None = None
        # Shared HTTP client + thread pool for concurrent POSTs to SGLang. Only
        # populated (in ``load``) when concurrency > 1; a single-post encode
        # keeps using module-level ``requests.post`` so the concurrency=1 A/B
        # baseline is byte-identical to the legacy path.
        self._session: requests.Session | None = None
        self._post_executor: ThreadPoolExecutor | None = None
        self._device: str | None = None
        self._configured_dense_dim: int | None = dense_dim
        self._dense_dim: int | None = dense_dim
        self._active_lora: str | None = None  # Set by set_active_lora() before encode()
        self._output_file: tempfile._TemporaryFileWrapper | None = None
        # Lazy, weights-free tokenizer used ONLY for exact §7.3 input-token
        # metering (see ``_get_metering_tokenizer`` / ``_stamp_input_token_counts``).
        # Loaded on first ``encode`` and cached; a load failure degrades to the
        # meter's reserve estimate rather than failing inference.
        self._metering_tokenizer_obj: Any = None
        self._metering_tokenizer_loaded: bool = False

    @property
    def available_loras(self) -> list[str]:
        """Return list of available LoRA adapter names.

        These are the names that can be passed to encode(lora=...).
        """
        return list(self._lora_paths.keys())

    @property
    def lora_enabled(self) -> bool:
        """Return whether LoRA adapters are configured."""
        return bool(self._lora_paths)

    def load(self, device: str) -> None:
        """Load the model by starting SGLang HTTP server as subprocess.

        Args:
            device: Device string (e.g., "cuda:0", "cuda:1").
                    Note: SGLang primarily supports CUDA devices.

        Raises:
            RuntimeError: If server fails to start within timeout.
        """
        self._device = device

        if self._append_eos:
            self._eos_token = self._resolve_eos_token()

        # Before any accelerator is claimed: a width the model's own head counts
        # cannot be divided by fails here rather than minutes into the engine's
        # own assertion, with the whole group held in the meantime.
        _server.validate_width_against_model_config(
            self._tensor_parallel_size,
            model_name_or_path=str(self._model_name_or_path),
            revision=self._revision,
        )
        device_index = _server.parse_device_index(device)
        device_indices = _server.resolve_device_group(device_index, self._tensor_parallel_size)
        port = _server.find_free_port()
        self._port = port
        self._server_url = f"http://localhost:{port}"

        logger.info(
            "Starting SGLang server for %s on device=%s (gpu_ids=%s, tp=%d) at port %d",
            self._model_name_or_path,
            device,
            ",".join(str(index) for index in device_indices),
            self._tensor_parallel_size,
            port,
        )

        # Build server command. Use sys.executable so we get the same Python
        # interpreter that has sglang installed (important for uv envs).
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_name_or_path,
            "--is-embedding",
            "--port",
            str(port),
            "--dtype",
            self._compute_precision,
            "--context-length",
            str(self._max_seq_length),
            "--mem-fraction-static",
            str(self._mem_fraction_static),
            # Full flag name rather than the "--tp" abbreviation, which a
            # future upstream flag sharing that prefix would make ambiguous.
            "--tensor-parallel-size",
            str(self._tensor_parallel_size),
            "--log-level",
            "warning",
        ]

        if self._trust_remote_code:
            cmd.append("--trust-remote-code")

        # Overrides the engine's default watchdog, which decides how long a
        # forward batch may run before the engine crashes itself.
        if self._watchdog_timeout_s is not None:
            cmd.extend(["--watchdog-timeout", str(self._watchdog_timeout_s)])

        # SGLang's default piecewise capture path is measured to hang at width
        # two and to exhaust memory at width four. Disabled above width one.
        if self._disable_piecewise_cuda_graph:
            cmd.append("--disable-piecewise-cuda-graph")

        if self._revision is not None:
            cmd.extend(["--revision", self._revision])

        # LoRA configuration for the SGLang server.
        if self._lora_paths:
            cmd.append("--enable-lora")
            lora_path_str = ",".join(f"{name}={path}" for name, path in self._lora_paths.items())
            cmd.extend(["--lora-paths", lora_path_str])
            cmd.extend(["--max-loras-per-batch", str(self._max_loras_per_batch)])
            logger.info(
                "LoRA enabled with %d adapters: %s",
                len(self._lora_paths),
                list(self._lora_paths.keys()),
            )

        # The port (and, once opened, the log fd) belong to this adapter from
        # the reservation above onward, so every raise below hands them back
        # rather than leaking one slot of the 100-port span per attempt: a full
        # /tmp in open_output_log, a failed exec, or a cancelled load included.
        try:
            # Only a group rendezvouses, so width one reserves nothing and its
            # launch argv is unchanged. Left to the engine it picks a random
            # port, which is the collision NCCL_BASE_PORT exists to prevent.
            if self._tensor_parallel_size > 1:
                if self._declared_nccl_port is not None:
                    _server.reserve_port(self._declared_nccl_port)
                    self._reserved_nccl_port = self._declared_nccl_port
                else:
                    self._reserved_nccl_port = _server.find_free_port(_server.NCCL_BASE_PORT)
                self._nccl_port = self._reserved_nccl_port
                cmd.extend(["--nccl-port", str(self._nccl_port)])
            self._output_file = _server.open_output_log()
            self._process = _server.launch_sglang_server(
                cmd, device_indices=device_indices, output_file=self._output_file
            )

            if not _server.wait_for_server(
                self._server_url,
                self._process,
                output_file=self._output_file,
                timeout_s=self._startup_timeout_s,
            ):
                # Capture crash-vs-timeout before terminate makes poll() ambiguous.
                crash_exit_code = self._process.poll()
                _server.terminate_process(self._process)
                self._process = None
                # Build the error before the abort deletes the log it reads.
                raise _server.startup_failure_error(self._output_file, crash_exit_code=crash_exit_code)
        except BaseException:
            self._abort_failed_load()
            raise

        # Concurrent-POST fan-out: a keep-alive session with a connection pool
        # sized to the concurrency and a matching thread pool, so a sharded
        # ``encode`` keeps ``embed_concurrency`` requests in flight against
        # SGLang's continuous batcher (see ``_embed_texts``). Left as ``None``
        # for concurrency==1 (legacy single blocking post).
        if self._embed_concurrency > 1:
            self._session = requests.Session()
            pool_size = max(self._embed_concurrency, 10)
            http_adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            self._session.mount("http://", http_adapter)
            self._session.mount("https://", http_adapter)
            self._post_executor = ThreadPoolExecutor(
                max_workers=self._embed_concurrency,
                thread_name_prefix="sglang-embed-post",
            )

        logger.info(
            "SGLang server ready: %s at %s (embed_concurrency=%d)",
            self._model_name_or_path,
            self._server_url,
            self._embed_concurrency,
        )

    def _abort_failed_load(self) -> None:
        """Tear down a partially-started child + reset state after a failed load.

        The registry does not call unload() on a failed load, so this mirrors
        unload(): terminate the child, release the reserved port, drop the temp
        log, and clear _server_url/_device so the adapter does not look "loaded".
        Mirrors the generation siblings (SGLang + MLX).
        """
        _server.terminate_process(self._process)
        self._process = None
        _server.release_port(self._port)
        # Guarded rather than unconditional: width one never reserves a
        # collective port, and its teardown must stay exactly as it was.
        if self._reserved_nccl_port is not None:
            _server.release_port(self._reserved_nccl_port)
            self._reserved_nccl_port = None
        self._nccl_port = None
        self._port = None
        self._server_url = None
        self._device = None
        self._cleanup_output_log()

    def unload(self) -> None:
        """Unload the model by stopping SGLang server subprocess."""
        if self._post_executor is not None:
            self._post_executor.shutdown(wait=False)
            self._post_executor = None
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._process is not None:
            logger.info("Shutting down SGLang server for %s", self._model_name_or_path)
            _server.terminate_process(self._process)
            self._process = None

        # Release only after the child is down so a concurrent load can't be
        # handed a port the dying child still holds bound.
        _server.release_port(self._port)
        # Guarded rather than unconditional: width one never reserves a
        # collective port, and its teardown must stay exactly as it was.
        if self._reserved_nccl_port is not None:
            _server.release_port(self._reserved_nccl_port)
            self._reserved_nccl_port = None
        self._nccl_port = None
        self._port = None
        self._server_url = None
        self._device = None
        self._dense_dim = self._configured_dense_dim
        self._cleanup_output_log()

    def _cleanup_output_log(self) -> None:
        """Close + delete the subprocess stdout/stderr log (best-effort, idempotent).

        Without this, one ``sglang_*.log`` leaks into /tmp per load — unbounded
        under LRU-eviction / hot-reload churn — and the open fd is held for the
        subprocess lifetime. Mirrors the MLX sibling's cleanup.
        """
        if self._output_file is not None:
            with contextlib.suppress(OSError):
                self._output_file.close()
            with contextlib.suppress(OSError):
                Path(self._output_file.name).unlink()
            self._output_file = None

    def memory_footprint(self) -> int:
        """Return the GPU memory usage in bytes.

        SGLang pre-allocates memory in the subprocess. Return 0 and let the
        registry use actual GPU memory monitoring instead.
        """
        return 0

    def load_required_memory_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        """Return SGLang's startup reservation requirement for load staging."""
        return _server.estimate_load_required_memory_bytes(
            device_type=device_type,
            device_total_bytes=device_total_bytes,
            mem_fraction_static=self._mem_fraction_static,
        )

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Which outputs to return (only "dense" supported).
            instruction: Optional instruction for queries.
            is_query: Whether items are queries (True) or documents (False).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.

        Raises:
            ValueError: If active LoRA is not loaded.

        Note:
            LoRA is set via set_active_lora() called by the worker before encode().
        """
        self._check_loaded()

        # Validate active LoRA if specified
        lora = self._active_lora
        if lora is not None and lora not in self._lora_paths:
            available = list(self._lora_paths.keys()) if self._lora_paths else []
            msg = f"LoRA '{lora}' not loaded. Available: {available}"
            raise ValueError(msg)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        # Note: pooling is not overridable. The engine decides it from the
        # checkpoint, and neither pinned version accepts a flag for it.
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)
        default_instruction = opts.get("default_instruction", self._default_instruction)
        normalize = opts.get("normalize", self._normalize)

        texts = self._format_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            default_instruction=default_instruction,
        )

        # SGLang rejects empty/whitespace-only inputs, so we need to:
        # 1. Track which indices have empty text
        # 2. Only send non-empty texts to SGLang
        # 3. Insert zero vectors for empty items in the result
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # If all texts are empty, return zero vectors
        if not non_empty_texts:
            if self._dense_dim is None:
                msg = "SGLang embedding dimension is unknown; configure tasks.encode.dense.dim"
                raise RuntimeError(msg)
            dim = self._dense_dim
            embeddings = np.zeros((len(items), dim), dtype=np.float32)
            empty_output = EncodeOutput(
                dense=embeddings,
                batch_size=len(items),
                is_query=is_query,
                dense_dim=dim,
            )
            # Nothing was posted to sglang, so every item bills zero input
            # tokens — exact, since the zero-vector fallback runs no inference.
            empty_output.extra["input_token_counts"] = [0] * len(items)
            return empty_output

        # Call SGLang HTTP API (OpenAI-compatible embeddings endpoint).
        # When LoRA is specified, use it as the model name; otherwise "default".
        # ``_embed_texts`` fans the batch out into ``embed_concurrency``
        # concurrent POSTs so SGLang's continuous batcher stays fed (the serving
        # loop is single-threaded, so a lone blocking POST under-drives it).
        model_name = lora if lora is not None else "default"
        non_empty_embeddings = self._embed_texts(non_empty_texts, model_name)

        dense_dim = self._validate_or_set_dense_dim(non_empty_embeddings)

        # Normalize if configured
        if normalize:
            non_empty_embeddings = self._normalize_embeddings(non_empty_embeddings)

        # Reconstruct full result array with zero vectors for empty inputs
        embeddings = np.zeros((len(items), dense_dim), dtype=np.float32)
        for result_idx, original_idx in enumerate(non_empty_indices):
            embeddings[original_idx] = non_empty_embeddings[result_idx]

        output = EncodeOutput(
            dense=embeddings,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=dense_dim,
        )
        # Unit-meter seam (§7.3): stamp exact per-item input-token counts for the
        # texts we actually POSTed to sglang. Server-backed, so this is the only
        # place the real counts exist (see ``_stamp_input_token_counts``).
        self._stamp_input_token_counts(output, non_empty_texts, non_empty_indices, len(items))
        return output

    def _get_metering_tokenizer(self) -> Any | None:
        """Lazily load a weights-free HF tokenizer for exact §7.3 metering.

        SGLang runs the model in a subprocess, so this adapter has no
        in-process tokenizer for the base ``count_input_tokens`` seam and
        ``units.input_tokens`` would otherwise stay 0 (the meter's reserve
        fallback) for this promoted dense-SMARTEST tier. Loading the served
        model's own tokenizer is tokenizer-only (no weights) and cheap; it is
        cached after the first call. Best-effort: a load failure is logged once
        and degrades metering to the reserve estimate rather than failing
        inference (mirrors ``_resolve_eos_token``).
        """
        if self._metering_tokenizer_loaded:
            return self._metering_tokenizer_obj
        self._metering_tokenizer_loaded = True
        try:
            self._metering_tokenizer_obj = load_tokenizer(
                self._model_name_or_path, trust_remote_code=self._trust_remote_code
            )
        except Exception:  # noqa: BLE001 — metering must never take the model down
            logger.warning(
                "metering: could not load tokenizer for %s; units.input_tokens will "
                "fall back to the meter's reserve estimate",
                self._model_name_or_path,
                exc_info=True,
            )
            self._metering_tokenizer_obj = None
        return self._metering_tokenizer_obj

    def _stamp_input_token_counts(
        self,
        output: EncodeOutput,
        non_empty_texts: list[str],
        non_empty_indices: list[int],
        total: int,
    ) -> None:
        """Stamp exact per-item input-token counts (§7.3) onto ``output.extra``.

        SGLang is server-backed, so the base ``count_input_tokens`` seam has no
        in-process tokenizer and ``units.input_tokens`` would fall to the
        meter's reserve estimate — never exact for this promoted dense tier. We
        count the EXACT strings POSTed to sglang (``non_empty_texts`` — already
        query/doc-template- and (when enabled) EOS-formatted by
        ``_format_texts``, and truncated at ``max_seq_length`` the same way
        sglang's ``--context-length`` truncates), then scatter them back to item
        positions with ``0`` for the empty items that took the zero-vector
        fallback and were never sent. The summed counts equal sglang's own
        ``usage.prompt_tokens`` for the request. Mirrors the ``bert_flash`` /
        ``bge_m3_flash`` seam; ``_token_counts_or_none`` is the shared base
        helper. Best-effort: any tokenizer quirk (or a load failure) leaves
        ``extra`` unstamped so the meter falls back to its reserve estimate
        rather than billing an approximation.
        """
        tokenizer = self._get_metering_tokenizer()
        if tokenizer is None:
            return
        counts = self._token_counts_or_none(tokenizer, non_empty_texts, expected_len=len(non_empty_texts))
        if counts is None:
            return
        per_item = [0] * total
        for result_idx, original_idx in enumerate(non_empty_indices):
            per_item[original_idx] = counts[result_idx]
        output.extra["input_token_counts"] = per_item

    def _embed_texts(self, texts: list[str], model_name: str) -> np.ndarray:
        """Embed ``texts`` via SGLang, fanning out concurrent POSTs.

        A single blocking POST presents one request to SGLang's continuous
        batcher; on the single-threaded serving loop that means the batcher
        rarely sees more than one request at a time and the served throughput
        lands far below the engine knee. Sharding the batch into up to
        ``embed_concurrency`` contiguous chunks and POSTing them concurrently
        keeps that many requests in flight, so the batcher fills.

        Correctness is preserved: each shard's embeddings are re-ordered by the
        response ``index`` inside :meth:`_extract_embeddings`, and the shards are
        concatenated back in input order, so the returned rows align 1:1 with
        ``texts`` exactly as the single-POST path did.
        """
        concurrency = self._embed_concurrency
        if concurrency <= 1 or len(texts) <= 1:
            return self._post_embedding_batch(texts, model_name, session=None)

        n_shards = min(concurrency, len(texts))
        shard_size = math.ceil(len(texts) / n_shards)
        shards = [texts[i : i + shard_size] for i in range(0, len(texts), shard_size)]

        executor = self._post_executor
        session = self._session
        if executor is None or session is None:
            # Defensive: load() populates both when concurrency > 1. Fall back
            # to a serial pass rather than fail if they are somehow missing.
            return np.vstack([self._post_embedding_batch(shard, model_name, session=session) for shard in shards])

        futures = [executor.submit(self._post_embedding_batch, shard, model_name, session) for shard in shards]
        # Iterating in submission order preserves shard order; ``.result()``
        # re-raises any shard error just like the single-POST path did.
        parts = [future.result() for future in futures]
        return np.vstack(parts)

    def _post_embedding_batch(
        self,
        texts: list[str],
        model_name: str,
        session: requests.Session | None,
    ) -> np.ndarray:
        """POST one (possibly sharded) batch of texts and return its embeddings."""
        poster = session.post if session is not None else requests.post
        response = poster(
            f"{self._server_url}/v1/embeddings",
            json={
                "model": model_name,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=self._request_read_timeout_s,
        )
        if response.status_code != 200:
            logger.error(
                "SGLang error %d for %d texts: %s",
                response.status_code,
                len(texts),
                response.text[:500],
            )
        response.raise_for_status()
        result = response.json()

        # Response format: {"data": [{"embedding": [...], "index": 0}, ...]}
        return self._extract_embeddings(result, len(texts))

    def _validate_or_set_dense_dim(self, embeddings: np.ndarray) -> int:
        """Validate observed SGLang embedding width against configured dense_dim."""
        if embeddings.ndim != 2:
            msg = f"SGLang server returned embeddings with invalid shape {embeddings.shape}"
            raise ValueError(msg)
        observed_dim = embeddings.shape[1]
        if self._configured_dense_dim is not None and observed_dim != self._configured_dense_dim:
            msg = (
                "SGLang embedding dimension mismatch: "
                f"configured dense_dim={self._configured_dense_dim}, observed={observed_dim}"
            )
            raise ValueError(msg)
        if self._dense_dim is None:
            self._dense_dim = observed_dim
            logger.info("Detected embedding dimension: %d", self._dense_dim)
        elif observed_dim != self._dense_dim:
            msg = f"SGLang embedding dimension changed: expected {self._dense_dim}, observed {observed_dim}"
            raise ValueError(msg)
        assert self._dense_dim is not None
        return self._dense_dim

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. SGLang adapter only supports 'dense'."
            raise ValueError(msg)

    def _resolve_eos_token(self) -> str:
        """Read the model's EOS token string from its own tokenizer.

        Reuses the shared ``core.tokenizer.load_tokenizer`` helper (the same path
        ``PyTorchEmbeddingAdapter`` uses) rather than hardcoding an EOS string, so
        the value always matches the served model. Only invoked when
        ``append_eos`` is set, so models that don't opt in never load a tokenizer
        in-process. A resolution failure degrades to "no EOS appended" (logged)
        rather than failing the whole model load.
        """
        try:
            tokenizer = load_tokenizer(self._model_name_or_path, trust_remote_code=self._trust_remote_code)
        except Exception:  # noqa: BLE001 - tokenizer load must never take the model down
            logger.warning(
                "append_eos: could not load tokenizer for %s to resolve EOS; last-token pooling may be degraded",
                self._model_name_or_path,
                exc_info=True,
            )
            return ""

        eos = tokenizer.eos_token
        if not eos and tokenizer.eos_token_id is not None:
            eos = tokenizer.decode([tokenizer.eos_token_id]).strip()
        if not eos:
            logger.warning(
                "append_eos: model %s exposes no EOS token; last-token pooling may be degraded",
                self._model_name_or_path,
            )
            return ""
        logger.info("append_eos enabled for %s: appending EOS %r", self._model_name_or_path, eos)
        return eos

    def _format_texts(
        self,
        items: list[Item],
        instruction: str | None,
        *,
        is_query: bool,
        query_template: str | None = None,
        doc_template: str | None = None,
        default_instruction: str | None = None,
    ) -> list[str]:
        r"""Format texts using configured templates.

        For queries with query_template, formats using the template.
        For documents with doc_template, formats using the template.
        Otherwise returns text as-is.
        """
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        default_instruction = default_instruction if default_instruction is not None else self._default_instruction
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="SGLangEmbeddingAdapter"))

            text = item.text

            if is_query and query_template:
                # Use provided instruction or default
                instr = instruction or default_instruction or ""
                text = query_template.format(instruction=instr, text=text)
            elif not is_query and doc_template:
                text = doc_template.format(text=text)
            elif instruction:
                # Fallback: prepend instruction if provided but no template
                text = f"{instruction} {text}"

            # Append the model's EOS so last-token pooling reads the trained
            # summary position (queries and documents alike). Skip empty /
            # whitespace text so it still takes the zero-vector fallback in
            # encode() instead of becoming an EOS-only input.
            if self._append_eos and self._eos_token and text.strip():
                text = f"{text}{self._eos_token}"

            texts.append(text)
        return texts

    def _extract_embeddings(self, result: dict[str, Any], num_items: int) -> np.ndarray:
        """Extract embeddings from SGLang OpenAI-compatible HTTP response.

        SGLang returns OpenAI-format response:
        {"data": [{"embedding": [...], "index": 0}, ...], "model": "...", "usage": {...}}
        """
        data = result.get("data")
        if not data:
            msg = "SGLang server returned empty response"
            raise RuntimeError(msg)

        if len(data) != num_items:
            msg = f"Expected {num_items} embeddings, got {len(data)}"
            raise RuntimeError(msg)

        # Sort by index to ensure correct order
        data_sorted = sorted(data, key=lambda x: x.get("index", 0))

        # Extract embeddings from each result object
        embeddings_list = []
        for i, item in enumerate(data_sorted):
            embedding = item.get("embedding")
            if embedding is None:
                msg = f"SGLang response item {i} missing 'embedding' key"
                raise RuntimeError(msg)
            embeddings_list.append(embedding)

        # Convert to numpy array [batch, dim]
        embeddings_np = np.array(embeddings_list, dtype=np.float32)

        return embeddings_np

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    # -------------------------------------------------------------------------
    # LoRA Support
    # -------------------------------------------------------------------------

    def supports_lora(self) -> bool:
        """Return True if LoRA adapters are configured."""
        return bool(self._lora_paths)

    def supports_hot_lora_reload(self) -> bool:
        """Return False - SGLang blocks during LoRA loading.

        SGLang's /load_lora_adapter endpoint blocks the server until loading
        completes. This is not true hot-reload like PEFT provides.
        """
        return False

    def set_active_lora(self, lora_name: str | None) -> None:
        """Set the active LoRA for the next encode() call.

        For SGLang, we store the active LoRA and use it as the model name
        in the HTTP request to the SGLang server.

        Args:
            lora_name: LoRA adapter name, or None for base model.
        """
        self._active_lora = lora_name
