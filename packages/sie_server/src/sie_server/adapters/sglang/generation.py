"""SGLang adapter for autoregressive text generation.

Sibling of :class:`SGLangEmbeddingAdapter` — same subprocess machinery
(via :mod:`sie_server.adapters.sglang._server`), but launches
``sglang.launch_server`` *without* ``--is-embedding`` and exposes
:meth:`generate` against SGLang's ``/generate`` HTTP endpoint.

Streaming async-iterator contract. The adapter opens an HTTP
streaming connection (``stream: true``) to SGLang's ``/generate`` endpoint,
parses SSE chunks, and yields :class:`GenerationChunk` objects. The terminal
chunk carries ``finish_reason`` plus prompt/completion token counts.

Cancellation: dropping the async iterator (``aclose()``) closes the upstream
HTTP connection, which SGLang treats as a cancel signal. A best-effort
``/abort_request`` POST is also issued when the request carries an
``rid`` so SGLang can free GPU work promptly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationChunk,
    GenerationResult,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.sglang import _server
from sie_server.observability.generation_diagnostics import GenerationStreamTimer
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput, media_bytes

logger = logging.getLogger(__name__)

# HTTP timeout knobs for /generate. The worker-side admission/cancel layer is
# the source of truth for total request lifetime, so the streaming read timeout
# is disabled by default (``read=None`` — keep reading as long as bytes arrive).
# Connect/write/pool stay bounded so we fail fast on a wedged subprocess. Set
# ``SIE_SGLANG_GENERATE_READ_TIMEOUT_S`` (float) to re-enable a wall-clock cap.
_GENERATE_CONNECT_TIMEOUT_S = float(os.environ.get("SIE_SGLANG_GENERATE_CONNECT_TIMEOUT_S", "10"))
_GENERATE_WRITE_TIMEOUT_S = float(os.environ.get("SIE_SGLANG_GENERATE_WRITE_TIMEOUT_S", "10"))
_GENERATE_POOL_TIMEOUT_S = float(os.environ.get("SIE_SGLANG_GENERATE_POOL_TIMEOUT_S", "10"))

# Wall-clock cap for the best-effort ``/abort_request`` POST issued when a
# generation is cancelled. Deliberately shorter than the streaming
# processor's 2s ``aclose()`` teardown cap (see ``_abort_tasks`` docstring
# in :class:`SGLangGenerationAdapter`) so the abort runs to completion as
# an independent background task instead of being cut off by the teardown
# wait_for. Override via ``SIE_SGLANG_ABORT_REQUEST_TIMEOUT_S``.
_ABORT_REQUEST_TIMEOUT_S = float(os.environ.get("SIE_SGLANG_ABORT_REQUEST_TIMEOUT_S", "1.5"))

# How many leading token POSITIONS to scan for the guard's Yes/No verdict
# distribution. A guard's verdict usually sits at position 0, but a leading
# whitespace/punctuation/preamble token can push it back a slot — so we scan
# the first few positions for the first one that carries a Yes/No top_logprobs
# distribution. Mirrors the eval runner's ``content[:3]`` scan
# (``sie_bench.eval.generation_runner._p_unsafe_from_logprobs``) so serving and
# eval agree on which position the verdict is read from.
_GUARD_VERDICT_SCAN_POSITIONS = 3


def _resolve_read_timeout() -> float | None:
    raw = os.environ.get("SIE_SGLANG_GENERATE_READ_TIMEOUT_S")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


_GENERATE_READ_TIMEOUT_S: float | None = _resolve_read_timeout()


def _mamba_scheduler_strategy_value(extra_launch_args: list[str]) -> str | None:
    """Return the value passed to ``--mamba-scheduler-strategy``, or ``None``.

    Parses both the two-token form (``["--mamba-scheduler-strategy",
    "extra_buffer"]``) and the ``--mamba-scheduler-strategy=value`` form; the
    last occurrence wins (argparse semantics). Returns ``None`` when the flag is
    absent. The speculative guard uses this to require the value be *exactly*
    ``extra_buffer`` — substring-matching the joined args would let a wrong value
    (e.g. ``--mamba-scheduler-strategy default``) slip past if the token
    ``extra_buffer`` happened to appear elsewhere in the args.
    """
    flag = "--mamba-scheduler-strategy"
    prefix = f"{flag}="
    value: str | None = None
    for i, arg in enumerate(extra_launch_args):
        if arg == flag:
            value = extra_launch_args[i + 1] if i + 1 < len(extra_launch_args) else None
        elif arg.startswith(prefix):
            value = arg[len(prefix) :]
    return value


# Format hints we re-embed into the SGLang ``image_data`` MIME type. Anything
# else falls back to ``jpeg`` (the engine sniffs the real format from bytes).
_ALLOWED_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg", "webp", "gif"})


def _encode_image_data(images: list[ImageInput] | None) -> list[str] | None:
    """Translate wire ``ImageInput`` entries into SGLang ``image_data`` URIs.

    SGLang's ``/generate`` accepts a top-level ``image_data`` field — a list of
    images, each as a base64 string, an ``http(s)`` URL, or a local file path.
    We emit ``data:image/<fmt>;base64,<...>`` data URIs so the format hint
    travels with the bytes and SGLang's image loader can decode without
    sniffing. Bytes are validated through :func:`media_bytes`, the single
    enforcement point for the wire contract (raises :class:`InvalidMediaError`
    on a non-bytes ``data``, e.g. an un-decoded base64 JSON string).

    Returns ``None`` when there are no images so the request body stays
    byte-identical to the text-only path — vision plumbing is inert for the
    text-only models that share this adapter.
    """
    if not images:
        return None
    encoded: list[str] = []
    for image in images:
        raw = media_bytes(image, kind="image")
        fmt = (image.get("format") or "jpeg").strip().lower() or "jpeg"
        # Clamp the client-controlled format hint to a known set before
        # re-embedding it in the data-URI MIME type — an arbitrary subtype
        # would produce a malformed URI for SGLang's loader. The engine
        # sniffs the real format from the bytes regardless, so an unknown
        # hint safely falls back to jpeg.
        if fmt not in _ALLOWED_IMAGE_FORMATS:
            fmt = "jpeg"
        elif fmt == "jpg":
            # ``image/jpg`` is not a registered MIME type; normalise to jpeg.
            fmt = "jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        encoded.append(f"data:image/{fmt};base64,{b64}")
    return encoded


def _tail_file(path: str, *, max_lines: int = 200) -> str:
    """Return the final lines from a startup log for diagnostics."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"
    return "".join(lines[-max_lines:])


# Kwargs the MLX adapter understands. The non-CUDA swap (``create_for_device``)
# projects the SGLang constructor kwargs onto this set and DROPS everything else
# — the CUDA/SGLang-only flags (mem_fraction_static, speculative, attention_backend,
# grammar_backend, reasoning_parser, tool_call_parser, lora_paths, disable_cuda_graph,
# extra_launch_args, extra_env, guard, compute_precision, …) that the MLX backend
# has no concept of.
_MLX_PASSTHROUGH_KWARGS = frozenset(
    {
        "model_name_or_path",
        "mlx_repo",
        "max_seq_length",
        "default_sampling",
        "stop_tokens",
        "served_model_name",
        "trust_remote_code",
        "startup_timeout_s",
    }
)


def _translate_to_mlx_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Project SGLang adapter kwargs onto the MLX adapter's constructor.

    Keeps only the engine-agnostic knobs the MLX backend uses; drops the
    CUDA/SGLang-only flags (which the MLX adapter would otherwise silently
    swallow via ``**kwargs``). ``mlx_repo`` comes from the model YAML's
    ``adapter_options.loadtime`` and selects the MLX-quantized repo to serve.
    """
    return {k: v for k, v in kwargs.items() if k in _MLX_PASSTHROUGH_KWARGS}


class SGLangGenerationAdapter(GenerationAdapter):
    """Streaming SGLang generation adapter.

    Lifecycle is identical to :class:`SGLangEmbeddingAdapter`: one subprocess
    per model, started in :meth:`load`, terminated in :meth:`unload`.

    Inference path: HTTP POST to ``{server_url}/generate`` with
    ``stream: true``. The response is consumed as a line stream; each
    non-empty ``data:`` line carries the cumulative decoded ``text`` plus
    ``meta_info`` (token counts, finish reason on terminal chunk).
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("tokens",),
        unload_fields=("_process", "_server_url"),
    )

    # The SGLang child process owns signal handling; parent load must not block
    # the uvicorn event loop while polling child readiness.
    requires_main_thread: bool = False
    manages_own_load_timeout: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int = 32768,
        mem_fraction_static: float = 0.85,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        # Pinned HuggingFace revision (commit SHA / branch / tag). Forwarded to
        # ``sglang.launch_server`` as ``--revision`` so the served weights match
        # the YAML-pinned SHA; None serves the repo default.
        revision: str | None = None,
        served_model_name: str | None = None,
        default_sampling: dict[str, Any] | None = None,
        stop_tokens: list[str] | None = None,
        disable_cuda_graph: bool = False,
        attention_backend: str | None = None,
        grammar_backend: str | None = "outlines",
        reasoning_parser: str | None = None,
        tool_call_parser: str | None = None,
        # Generative guard models (CHECK POLICY) emit a one-token Yes/No verdict.
        # When set (``{"threshold": 0.8, "positive": "Yes"}``) the adapter reads
        # the verdict-token logprobs, computes P(unsafe) over the Yes/No tokens,
        # and returns "Yes" iff P(unsafe) >= threshold — a precision/recall dial
        # that lifts the argmax 0.16 precision (measured: F1 0.27->0.38, prec
        # 0.16->0.29 at 0.8; see the guard baseline). Only guard model YAMLs set
        # this, so it is inert for every other model.
        guard: dict[str, Any] | None = None,
        speculative: dict[str, Any] | None = None,
        # When ``speculative.enabled``, the adapter normally REQUIRES
        # ``--mamba-scheduler-strategy extra_buffer`` in ``extra_launch_args``
        # — the Qwen3.x Gated-DeltaNet NEXTN + radix-cache pairing. Models whose
        # speculative path doesn't need it (e.g. Gemma 4 MTP — a standard
        # hybrid-attention model using an external ``-it-assistant`` NEXTN draft)
        # set this ``False`` so the guard doesn't reject an otherwise-valid
        # launch. Default ``True`` keeps the Qwen3.x contract unchanged.
        speculative_needs_extra_buffer: bool = True,
        extra_launch_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        startup_timeout_s: float | None = None,
        # Multi-LoRA: served-name → HF-id/path map. When non-empty the server
        # launches with ``--enable-lora`` and per-request ``lora_path``
        # selection is available. ``max_loras_per_batch`` caps concurrent
        # adapters in one batch (SGLang ``--max-loras-per-batch``).
        lora_paths: dict[str, str] | None = None,
        max_loras_per_batch: int = 4,
        **kwargs: Any,  # accept extra args from loader for compatibility
    ) -> None:
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._mem_fraction_static = mem_fraction_static
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._served_model_name = served_model_name or model_name_or_path
        self._default_sampling = default_sampling or {}
        self._stop_tokens = stop_tokens or []
        # Escape hatches for environments where the default SGLang launch
        # flags don't work — most commonly Modal sandboxes that lack
        # ``nvcc`` (so flashinfer JIT compilation fails) and need
        # ``--disable-cuda-graph`` plus ``--attention-backend triton``.
        # ``extra_launch_args`` is the raw passthrough for any flag we
        # haven't surfaced as a typed kwarg yet.
        self._disable_cuda_graph = disable_cuda_graph
        self._attention_backend = attention_backend
        # SGLang structured-output backend. Defaults to Outlines. A prior
        # revision forced Qwen3.5 to XGrammar because the worker-side
        # Outlines preflight crashed on the transformers>=5 tokenizer
        # ("'TokenizersBackend' object has no attribute 'vocabulary'");
        # ``compile_outlines`` now wraps the tokenizer in Outlines'
        # ``TransformerTokenizer`` adapter, so Outlines works again. Model
        # configs advertising grammar support should set this explicitly,
        # e.g. "outlines" or "xgrammar". Pass ``None`` to omit the flag
        # entirely and let SGLang pick its own default.
        self._grammar_backend = grammar_backend
        # Qwen3.5 and similar reasoning-capable models need
        # ``--reasoning-parser qwen3`` so SGLang strips ``<think>...</think>``
        # blocks from the visible stream. Set ``None`` to omit the flag.
        self._reasoning_parser = reasoning_parser
        # OpenAI-compatible tool-call streaming requires SGLang's
        # ``--tool-call-parser`` (e.g. ``qwen3_coder`` for Qwen3.5).
        # Set ``None`` to omit the flag.
        self._tool_call_parser = tool_call_parser
        # Guard verdict thresholding (see the kwarg docstring). Empty for every
        # non-guard model, which makes the verdict rewrite in ``generate`` a
        # no-op for them.
        self._guard = guard or {}
        # Generic speculative-decoding surface accepts
        # ``{enabled, algorithm, num_steps?, eagle_topk?,
        # num_draft_tokens?, draft_model?}``; translated by
        # ``_speculative_launch_args`` into SGLang CLI flags. Default-on
        # for Qwen3.5-4B (NEXTN/MTP trained in); intentionally off for
        # Qwen3-4B-Instruct-2507 (no NEXTN heads — see the
        # speculative-decoding investigation result).
        self._speculative = speculative
        self._speculative_needs_extra_buffer = speculative_needs_extra_buffer
        self._extra_launch_args = list(extra_launch_args or [])
        self._extra_env = dict(extra_env or {})
        self._startup_timeout_s = _server.resolve_startup_timeout(startup_timeout_s)
        self._lora_paths = dict(lora_paths or {})
        self._max_loras_per_batch = max_loras_per_batch

        self._process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._device: str | None = None
        self._output_file: tempfile._TemporaryFileWrapper | None = None
        # Shared httpx client. Opened lazily on the first ``generate()`` call
        # so we don't pay the event-loop-bound construction cost during
        # ``load()`` (which may run on a sync code path). Closed in
        # ``unload()``. A single shared client lets concurrent requests
        # multiplex over a connection pool instead of paying per-request
        # TCP setup. Measured impact at c=16 on A100: ~20% TTFT improvement
        # vs the previous "new AsyncClient per request" implementation.
        self._http_client: httpx.AsyncClient | None = None
        # Two locks for the lazy ``_http_client``:
        #
        # * ``_http_client_init_lock`` is a sync ``threading.Lock`` used
        #   *only* to guard the one-shot construction of
        #   ``_http_client_lock``. We can't create the asyncio.Lock
        #   directly in ``__init__`` because ``__init__`` may run on a
        #   thread with no running loop (registry boot path), and
        #   ``asyncio.Lock()`` binds to whatever loop happens to be
        #   current then. The sync lock is held only across two
        #   attribute assignments — no awaits inside — so the GIL
        #   already makes it cheap, and it isn't held when the actual
        #   client construction happens below.
        # * ``_http_client_lock`` is the asyncio.Lock that guards the
        #   actual ``httpx.AsyncClient`` construction. Without it, two
        #   concurrent first callers can both pass the ``is None``
        #   check, each build a client, and the loser's client
        #   overwrites the winner's in the attribute — the winner's
        #   connections then live in the pool untracked until GC, and
        #   ``unload()`` only closes one of the two.
        self._http_client_init_lock: threading.Lock = threading.Lock()
        self._http_client_lock: asyncio.Lock | None = None
        # Tracks the in-flight aclose task scheduled by ``unload()`` when
        # called from a running event loop. Keeping a strong ref here
        # prevents the loop from silently dropping the task and lets a
        # future shutdown coordinator await close completion before tear-
        # down. Cleared by the done-callback.
        self._pending_aclose: asyncio.Task[None] | None = None
        # Strong references to in-flight ``/abort_request`` POSTs spawned
        # from the per-request generator's ``GeneratorExit`` handler. The
        # abort MUST NOT be awaited inside ``GeneratorExit`` — the
        # streaming processor tears the iterator down via
        # ``asyncio.wait_for(chunks_iter.aclose(), timeout=2.0)`` and
        # ``aclose()`` drives the ``GeneratorExit`` handler, so an awaited
        # abort would be cancelled by that 2s cap exactly when SGLang is
        # slow, orphaning the generation (it keeps holding KV/GPU). We
        # instead spawn the abort as a fire-and-forget task on the
        # adapter's long-lived event loop (the adapter outlives any single
        # request) and keep a strong ref here so the loop doesn't GC the
        # task mid-flight. The done-callback discards the ref. The abort's
        # own timeout (``_ABORT_REQUEST_TIMEOUT_S``) is deliberately
        # shorter than the 2s teardown cap so a normally-responsive SGLang
        # completes the abort even though the generator no longer waits.
        self._abort_tasks: set[asyncio.Task[Any]] = set()

    # -- Lifecycle -----------------------------------------------------------

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> GenerationAdapter:
        """Device-aware factory: SGLang on CUDA, MLX (Apple Silicon) elsewhere.

        SGLang is CUDA-only, so on a non-CUDA device (a Mac's ``mps``/``cpu``)
        the generation path is served by the MLX subprocess adapter instead.
        Unsuffixed generation models therefore "just work" on a Mac without
        ``:mac`` profile variants. Importing this module on a Mac is safe —
        ``sglang`` is only ever invoked as a subprocess, never imported at
        module top level — so this swap runs without sglang installed.
        """
        if device.startswith("cuda"):
            return cls(**kwargs)
        from sie_server.adapters.mlx.generation import MLXGenerationAdapter

        logger.info(
            "Non-CUDA device %r: serving generation via the MLX (Apple-Silicon) backend instead of SGLang",
            device,
        )
        return MLXGenerationAdapter(**_translate_to_mlx_kwargs(kwargs))

    def load(self, device: str) -> None:
        self._device = device
        device_index = _server.parse_device_index(device)
        port = _server.find_free_port()
        self._server_url = f"http://localhost:{port}"

        logger.info(
            "Starting SGLang generation server for %s on device=%s (gpu_id=%d) at port %d",
            self._model_name_or_path,
            device,
            device_index,
            port,
        )

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_name_or_path,
            # Deliberately NO --is-embedding flag here — this is generation.
            "--port",
            str(port),
            "--dtype",
            self._compute_precision,
            "--context-length",
            str(self._max_seq_length),
            "--mem-fraction-static",
            str(self._mem_fraction_static),
            "--tp",
            "1",
            "--log-level",
            "warning",
            "--served-model-name",
            self._served_model_name,
        ]
        if self._trust_remote_code:
            cmd.append("--trust-remote-code")
        if self._revision is not None:
            cmd.extend(["--revision", self._revision])
        if self._disable_cuda_graph:
            cmd.append("--disable-cuda-graph")
        if self._attention_backend:
            cmd.extend(["--attention-backend", self._attention_backend])
        if self._grammar_backend:
            cmd.extend(["--grammar-backend", self._grammar_backend])
        if self._reasoning_parser:
            cmd.extend(["--reasoning-parser", self._reasoning_parser])
        if self._tool_call_parser:
            cmd.extend(["--tool-call-parser", self._tool_call_parser])
        cmd.extend(self._speculative_launch_args(self._speculative))
        # Multi-LoRA serving: one base, N adapters, per-request selection via
        # ``sampling_params["lora_path"]``. Only emitted when the profile
        # declared adapters, so non-LoRA deployments are byte-identical.
        if self._lora_paths:
            cmd.append("--enable-lora")
            cmd.extend(["--max-loras-per-batch", str(self._max_loras_per_batch)])
            cmd.append("--lora-paths")
            cmd.extend(f"{served_name}={path}" for served_name, path in self._lora_paths.items())
        cmd.extend(self._extra_launch_args)

        self._output_file = _server.open_output_log()
        # SGLang 0.5.10+ requires ``SGLANG_ENABLE_SPEC_V2=1`` for NEXTN spec
        # decoding to coexist with the radix cache on hybrid-architecture
        # models (Qwen3.5 family — Gated DeltaNet + Gated Attention). The
        # ``--mamba-scheduler-strategy extra_buffer`` flag, set via the model
        # YAML's ``extra_launch_args``, is the matching CLI side; both are
        # required as a pair. Empirically validated 2026-05-18 on L4 +
        # A100-40GB. Set unconditionally when speculative is on — the env var
        # is a no-op when sglang doesn't see speculative args.
        extra_env: dict[str, str] = dict(self._extra_env)
        if self._speculative and self._speculative.get("enabled"):
            extra_env["SGLANG_ENABLE_SPEC_V2"] = "1"
            # The pair is required: setting ``SGLANG_ENABLE_SPEC_V2=1``
            # without ``--mamba-scheduler-strategy extra_buffer`` (or
            # equivalent) crashes the radix cache mid-run on Qwen3.5
            # hybrid models with a confusing "spec_v2 requires extra
            # buffer" trace. Refuse to launch when the YAML omits the
            # flag so the misconfiguration surfaces as a startup error
            # instead of a runtime OOM that pages oncall.
            # Require the flag AND its exact ``extra_buffer`` value — a present
            # flag with a wrong value (e.g. ``--mamba-scheduler-strategy default``)
            # must not bypass the guard. Parse the flag's value precisely (both
            # the two-token and ``flag=value`` forms; last occurrence wins) rather
            # than substring-matching the joined args, which a stray
            # ``extra_buffer`` token elsewhere could satisfy.
            if (
                self._speculative_needs_extra_buffer
                and _mamba_scheduler_strategy_value(self._extra_launch_args) != "extra_buffer"
            ):
                raise RuntimeError(
                    "speculative decoding requires '--mamba-scheduler-strategy extra_buffer' "
                    "in extra_launch_args (see Qwen3.5-4B model YAML). Add the flag, set "
                    "speculative_needs_extra_buffer=false (non-DeltaNet models e.g. Gemma 4 "
                    "MTP), or disable speculative.enabled in the model config."
                )
        logger.warning(
            "Resolved SGLang generation command: %s",
            " ".join(shlex.quote(str(arg)) for arg in cmd),
        )
        logger.warning(
            "Resolved SGLang generation extra_env: %s",
            {key: extra_env[key] for key in sorted(extra_env)},
        )
        self._process = _server.launch_sglang_server(
            cmd,
            device_index=device_index,
            output_file=self._output_file,
            extra_env=extra_env or None,
        )

        if not _server.wait_for_server(
            self._server_url,
            self._process,
            output_file=self._output_file,
            timeout_s=self._startup_timeout_s,
        ):
            log_path = getattr(self._output_file, "name", None)
            if log_path:
                logger.error("SGLang failed to reach health. log_path=%s", log_path)
                logger.error("SGLang startup log tail:\n%s", _tail_file(str(log_path)))
            _server.terminate_process(self._process)
            self._process = None
            raise _server.startup_failure_error(self._output_file)

        # Best-effort runtime verification that the
        # ``--grammar-backend`` flag actually took effect. SGLang's
        # exact startup-log token varies across versions; we scan for
        # the backend name + the substring ``grammar`` in the same
        # line. Failure to find the line is logged as a warning but
        # does NOT fail startup — a future SGLang release that changes
        # the log format shouldn't take the whole worker down. The
        # gateway's grammar requests still flow through and SGLang
        # will surface its own 500 if the flag is actually broken.
        if self._grammar_backend:
            self._verify_grammar_backend_log()

        logger.info(
            "SGLang generation server ready: %s at %s",
            self._model_name_or_path,
            self._server_url,
        )

    def _verify_grammar_backend_log(self) -> None:
        """Scan ``self._output_file`` for evidence SGLang accepted the
        ``--grammar-backend`` flag. Best-effort; never raises.
        """
        backend = self._grammar_backend
        if backend is None or self._output_file is None:
            return
        try:
            log_path = self._output_file.name
        except AttributeError:
            return
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            logger.debug("could not re-read SGLang log to verify grammar backend")
            return
        # Look for any line that mentions both the backend name and the
        # word ``grammar``. SGLang has historically emitted lines like
        # ``Grammar backend: outlines`` or ``grammar_backend=outlines``;
        # neither is contractual but both fit this pattern.
        matched = False
        for line in text.splitlines():
            low = line.lower()
            if backend in low and "grammar" in low:
                matched = True
                logger.info(
                    "SGLang grammar backend confirmed (%s): %s",
                    backend,
                    line.strip()[:200],
                )
                break
        if not matched:
            # The matcher is intentionally loose — SGLang's exact
            # startup log line wording is undocumented and version-
            # dependent. If this warning fires on a cluster where
            # grammar requests are visibly working, the log format
            # likely changed; check SGLang's recent release notes for
            # ``--grammar-backend`` and update the matcher above.
            logger.warning(
                "Could not confirm --grammar-backend=%s from SGLang startup log; "
                "Outlines may still be active but the launch-time evidence is missing. "
                "Check SGLang's startup output near `--grammar-backend` for the canonical line.",
                backend,
            )

    @staticmethod
    def _speculative_launch_args(spec: dict[str, Any] | None) -> list[str]:
        """Translate a config-shape ``speculative`` block into SGLang CLI flags.

        Generic surface (algorithm-agnostic) so the same code path serves
        all currently-supported algorithms without per-algorithm
        branches in the adapter constructor:

        - ``NEXTN`` (a.k.a. MTP in SGLang 0.5.x — multi-token prediction
          via trained-in draft heads; required for Qwen3.5-4B's
          ``MTP: trained with multi-steps`` capability).
        - ``EAGLE3`` (requires an external draft model whose
          architecture is registered with transformers; blocked for
          Qwen3-4B-Instruct-2507 on the pinned stack — see the
          speculative-decoding investigation note in
          ``qwen3-4b-speculative-side.yaml``).
        - ``NGRAM`` (model-independent; the documented fallback that
          works for any base model — the speculative-decoding investigation
          chose this for the Qwen3-4B-Instruct-2507 side-cell).

        Recognised keys:

        ``enabled`` (bool, default False)
            Top-level gate. When False (or block missing), returns
            ``[]`` — caller still gets the unchanged baseline command.
        ``algorithm`` (str, required when enabled)
            One of ``nextn``, ``eagle3``, ``ngram`` (case-insensitive).
            Surfaced verbatim to SGLang's ``--speculative-algo``.
        ``num_steps`` (int, optional)
            ``--speculative-num-steps``. Qwen3.5-4B's documented
            NEXTN recipe uses 3.
        ``eagle_topk`` (int, optional)
            ``--speculative-eagle-topk``. Qwen3.5-4B's NEXTN recipe
            uses 1.
        ``num_draft_tokens`` (int, optional)
            ``--speculative-num-draft-tokens``. Qwen3.5-4B's NEXTN
            recipe uses 4. NGRAM doesn't require this (the n-gram
            window is set by separate SGLang knobs we leave at
            defaults — see ``qwen3-4b-speculative-side.yaml`` header).
        ``draft_model`` (str, optional)
            ``--speculative-draft-model-path``. EAGLE3 needs this;
            NEXTN/NGRAM do not.

        Unknown keys are intentionally ignored (with a debug log) so a
        future SGLang flag can be wired here without forcing a
        config-schema migration. Validation is light — SGLang's own
        ``launch_server`` is the ultimate source of truth and will
        reject malformed combinations at startup.
        """
        if not spec or not spec.get("enabled"):
            return []
        algorithm_raw = spec.get("algorithm")
        if not isinstance(algorithm_raw, str) or not algorithm_raw.strip():
            logger.warning(
                "speculative.enabled=true but algorithm missing/invalid (%r); "
                "skipping --speculative-* flags. Set algorithm to one of "
                "'nextn' | 'eagle3' | 'ngram'.",
                algorithm_raw,
            )
            return []
        algorithm = algorithm_raw.strip().upper()
        if algorithm not in {"NEXTN", "EAGLE3", "NGRAM"}:
            logger.warning(
                "speculative.algorithm=%r is not in the recognised set "
                "{NEXTN, EAGLE3, NGRAM}; forwarding to SGLang anyway "
                "(it will reject if unsupported).",
                algorithm_raw,
            )
        args: list[str] = ["--speculative-algo", algorithm]

        num_steps = spec.get("num_steps")
        if isinstance(num_steps, int) and num_steps > 0:
            args.extend(["--speculative-num-steps", str(num_steps)])
        elif num_steps is not None:
            logger.debug("speculative.num_steps=%r ignored (need positive int)", num_steps)

        eagle_topk = spec.get("eagle_topk")
        if isinstance(eagle_topk, int) and eagle_topk > 0:
            args.extend(["--speculative-eagle-topk", str(eagle_topk)])
        elif eagle_topk is not None:
            logger.debug("speculative.eagle_topk=%r ignored (need positive int)", eagle_topk)

        num_draft_tokens = spec.get("num_draft_tokens")
        if isinstance(num_draft_tokens, int) and num_draft_tokens > 0:
            args.extend(["--speculative-num-draft-tokens", str(num_draft_tokens)])
        elif num_draft_tokens is not None:
            logger.debug(
                "speculative.num_draft_tokens=%r ignored (need positive int)",
                num_draft_tokens,
            )

        draft_model = spec.get("draft_model")
        if isinstance(draft_model, str) and draft_model.strip():
            args.extend(["--speculative-draft-model-path", draft_model.strip()])
        elif algorithm == "EAGLE3":
            logger.warning(
                "speculative.algorithm=EAGLE3 without speculative.draft_model — "
                "SGLang will fail to start. Set speculative.draft_model to the "
                "EAGLE3 draft checkpoint path."
            )

        logger.info("speculative decoding enabled: %s (args=%s)", algorithm, args)
        return args

    async def _abort_request(self, client: httpx.AsyncClient, server_url: str, rid: str) -> None:
        """Best-effort POST to SGLang's ``/abort_request`` for ``rid``.

        Bounded by ``_ABORT_REQUEST_TIMEOUT_S`` and swallows all errors —
        cancellation cleanup must never raise. Runs as an independent
        background task (see :meth:`_spawn_abort_request`) so it is not
        subject to the streaming processor's iterator-teardown cap.
        """
        with contextlib.suppress(Exception):
            await client.post(
                f"{server_url}/abort_request",
                json={"rid": rid},
                timeout=_ABORT_REQUEST_TIMEOUT_S,
            )

    def _spawn_abort_request(self, client: httpx.AsyncClient, server_url: str, rid: str) -> None:
        """Schedule an independent ``/abort_request`` POST and track it.

        Called from the generator's ``GeneratorExit`` handler. We must NOT
        ``await`` the abort there: ``aclose()`` drives ``GeneratorExit``
        under the streaming processor's 2s ``wait_for`` cap, so an awaited
        abort would be cancelled mid-flight when SGLang is slow, leaving an
        orphaned generation holding the GPU slot. Instead we create a task
        on the running loop (the adapter is the long-lived worker adapter,
        so the loop persists well beyond this request) and keep a strong
        reference in ``_abort_tasks`` until it finishes — without the ref
        the loop may GC the task before it runs.

        If there is no running loop (e.g. a sync teardown path) the abort
        is skipped — there is nothing to drive it and raising here would
        corrupt the ``GeneratorExit`` cleanup.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("no running loop to spawn /abort_request for rid=%s; skipping", rid)
            return
        task = loop.create_task(self._abort_request(client, server_url, rid))
        self._abort_tasks.add(task)
        task.add_done_callback(self._abort_tasks.discard)

    def _clear_pending_aclose(self, task: asyncio.Task[None]) -> None:
        # Only clear the slot if it still references *this* task. A second
        # ``unload()`` call between the first task finishing and its
        # callback firing would otherwise have the first callback wipe
        # the slot the second call just populated, leaving the second
        # close task untracked.
        if self._pending_aclose is task:
            self._pending_aclose = None

    async def aclose_client(self) -> None:
        """Await a clean close of the shared HTTP client and pending aborts.

        Awaitable counterpart to the close that ``unload()`` otherwise
        does fire-and-forget. The worker shutdown path (registry
        ``_do_unload``) calls this BEFORE ``unload()`` terminates the
        SGLang subprocess so the client's connections are drained against
        a still-live server instead of being abandoned (which leaked file
        descriptors and could wedge on a half-open socket). Safe to call
        when no client was ever opened (no-op). Idempotent: a subsequent
        ``unload()`` sees ``_http_client is None`` and skips its own close.

        Also drains any in-flight ``/abort_request`` tasks and the
        ``unload()``-scheduled ``_pending_aclose`` so cancellation cleanup
        completes against the live subprocess too.
        """
        # Drain in-flight abort POSTs first — they target the still-live
        # subprocess and must complete before we tear it down. Snapshot
        # the set: each task's done-callback mutates ``_abort_tasks``.
        if self._abort_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*tuple(self._abort_tasks), return_exceptions=True)

        # If a prior ``unload()`` scheduled an aclose, await it rather than
        # racing it with our own close of the same/new client.
        pending = self._pending_aclose
        if pending is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(pending), timeout=2.0)

        client = self._http_client
        if client is None:
            return
        self._http_client = None
        try:
            await asyncio.wait_for(client.aclose(), timeout=2.0)
        except Exception as exc:  # noqa: BLE001 - close is best-effort
            logger.warning("aclose_client() failed: %s", exc)

    def unload(self) -> None:
        if self._http_client is not None:
            client = self._http_client
            self._http_client = None
            # Two cases:
            # (1) Called from inside a running loop (e.g. registry hot
            #     reload from an async context): schedule the aclose and
            #     stash the task on ``self`` so the loop retains a strong
            #     ref AND a future shutdown coordinator can await the
            #     pending close before tearing the loop down. The bare
            #     ``loop.create_task(...)`` pattern with a dropped ref
            #     was prone to silent skips at loop teardown.
            # (2) Called outside a running loop (process shutdown): skip
            #     the async close entirely. ``httpx.AsyncClient`` is bound
            #     to the loop it was created on (the adapter's long-lived
            #     request loop); driving ``aclose()`` from a *new* loop
            #     here can raise or leak the connection pool because the
            #     pool's transports belong to the original loop. This
            #     branch only runs at process exit, so abandoning the
            #     sockets is harmless — the OS reclaims the fds when the
            #     interpreter exits. The hot-unload path closes on the
            #     correct loop via ``aclose_client()`` before reaching
            #     here, so a genuinely-leaked client is not expected.
            # Prefer ``get_running_loop`` (3.10+ idiom; the older
            # ``get_event_loop`` is deprecated in 3.12 and raises on
            # threads with no loop). When there's a running loop,
            # schedule the close on it and stash the task. Otherwise
            # skip the async close (best-effort) and fall through to
            # subprocess teardown.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(client.aclose())
                self._pending_aclose = task
                task.add_done_callback(self._clear_pending_aclose)
            else:
                logger.debug(
                    "unload(): no running loop; skipping httpx aclose() "
                    "(client is loop-bound, cannot close from a new loop) "
                    "and proceeding to terminate the SGLang subprocess"
                )
        if self._process is not None:
            logger.info("Shutting down SGLang generation server for %s", self._model_name_or_path)
            _server.terminate_process(self._process)
            self._process = None
        self._server_url = None
        self._device = None

    def memory_footprint(self) -> int:
        # SGLang pre-allocates in the subprocess; let the registry measure GPU.
        return 0

    def load_required_memory_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        """Return SGLang's startup reservation requirement for load staging."""
        return _server.estimate_load_required_memory_bytes(
            device_type=device_type,
            device_total_bytes=device_total_bytes,
            mem_fraction_static=self._mem_fraction_static,
        )

    # -- Inference -----------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._server_url is None:
            raise RuntimeError(ERR_NOT_LOADED)

    async def _get_or_create_http_client(self) -> httpx.AsyncClient:
        """Return the shared client, opening it if this is the first call.

        Must be called from within a running event loop. The first caller
        builds the client under an ``asyncio.Lock``; subsequent callers
        return the cached instance without acquiring the lock again on
        the fast path. The fast-path check is intentionally lock-free
        because it's a pure read of an attribute set under the lock —
        Python's GIL makes that read atomic, and once set the attribute
        never goes back to ``None`` (``unload()`` swaps to a new
        ``None``-then-replace cycle, but a request mid-flight already
        holds its own reference to the previous client).
        """
        client = self._http_client
        if client is not None:
            return client
        # Guard *creation* of the asyncio.Lock with a sync threading.Lock.
        # Without this, two concurrent first callers can both see
        # ``_http_client_lock is None``, each create their own Lock(),
        # assign it (last write wins), and one of them ends up holding
        # a lock that nothing else acquires — silently bypassing the
        # double-check. The threading.Lock is held only across two
        # attribute reads/writes (no awaits inside) so it does not
        # block the event loop in any meaningful way.
        if self._http_client_lock is None:
            with self._http_client_init_lock:
                if self._http_client_lock is None:
                    self._http_client_lock = asyncio.Lock()
        async with self._http_client_lock:
            # Double-checked after acquiring the lock — another caller
            # may have populated the client while we waited.
            if self._http_client is None:
                # ``http2=False``: SGLang's /generate endpoint is HTTP/1.1 only
                # (Python http.server / starlette under the hood). HTTP/2
                # negotiation costs handshake time and gains nothing here.
                # ``max_connections``: cap at 256, enough for the SGLang server
                # worker count (~max_running_requests=48 on Qwen3.5-4B A100)
                # without runaway pool growth.
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=_GENERATE_CONNECT_TIMEOUT_S,
                        read=_GENERATE_READ_TIMEOUT_S,
                        write=_GENERATE_WRITE_TIMEOUT_S,
                        pool=_GENERATE_POOL_TIMEOUT_S,
                    ),
                    http2=False,
                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                )
            return self._http_client

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stream: bool = False,
        lora_path: str | None = None,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self._check_loaded()

        # Vision input: encode any images into SGLang's top-level ``image_data``
        # field once, then attach to whichever request body we build below. The
        # ``prompt`` is expected to already carry the model's image placeholder
        # tokens (the chat template renders them worker-side). ``None`` when
        # there are no images, keeping the text-only request body unchanged.
        image_data = _encode_image_data(images)

        # Guard verdict thresholding only runs on the single-candidate (n=1)
        # path, so reject multi-candidate sampling up front — otherwise a guard
        # request with n>1 / best_of>1 would silently return an UN-thresholded
        # verdict from the multi-candidate path. Inert for non-guard models.
        if self._guard and ((n is not None and n > 1) or (best_of is not None and best_of > 1)):
            raise ValueError("guard models support single-candidate generation only (n=1, best_of<=1)")
        # Whether the CLIENT asked for logprobs, captured before the guard
        # forcing below. Guard models force logprobs on internally to compute
        # the verdict threshold; those forced logprobs are an implementation
        # detail and MUST NOT leak to a client that did not request them
        # (GenerationChunk.logprobs contract). The streaming guard intercept
        # uses this to decide whether to strip the forced logprobs.
        client_requested_logprobs = logprobs
        # Thresholding needs the verdict-token distribution — force logprobs on
        # even if the caller didn't ask. Only affects the n=1 path below.
        if self._guard:
            logprobs = True
            top_logprobs = max(top_logprobs or 0, 20)

        sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # SGLang accepts ``top_k`` (int) and ``repetition_penalty`` (float)
        # natively under ``sampling_params``. Forward only when the gateway
        # provided a value so model defaults stay in effect otherwise.
        if top_k is not None:
            sampling_params["top_k"] = top_k
        if repetition_penalty is not None:
            sampling_params["repetition_penalty"] = repetition_penalty
        # SGLang ``sampling_params["min_new_tokens"]`` — minimum tokens
        # before stop. Plumbed end-to-end from the gateway's chat
        # ``min_tokens`` knob (workaround for Qwen3.6's first-token-EOS
        # bug under greedy decode).
        if min_new_tokens is not None:
            sampling_params["min_new_tokens"] = min_new_tokens
        # SGLang accepts both penalty knobs natively under
        # ``sampling_params`` with the same names OpenAI uses. Pass them
        # through only when the gateway provided a value so model
        # defaults stay in effect for the typical no-penalty request.
        if frequency_penalty is not None:
            sampling_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            sampling_params["presence_penalty"] = presence_penalty
        # OpenAI ``seed`` → SGLang ``sampling_params["sampling_seed"]``.
        # SGLang only applies this value when the server is launched with
        # ``--enable-deterministic-inference``; otherwise it is accepted but
        # has no sampling effect.
        if seed is not None:
            sampling_params["sampling_seed"] = seed
        # OpenAI ``logit_bias`` → SGLang ``sampling_params["logit_bias"]``.
        # SGLang accepts the same ``{token_id_str: float}`` shape OpenAI
        # uses, so we forward verbatim.
        if logit_bias:
            sampling_params["logit_bias"] = logit_bias
        # Multi-LoRA: ``lora_path`` is a **top-level** ``/generate`` field in
        # SGLang 0.5.10 (NOT a sampling param — SamplingParams rejects it),
        # alongside ``return_logprob``. Added to the request bodies below.
        # Merge default sampling from model config (request fields win).
        for k, v in self._default_sampling.items():
            sampling_params.setdefault(k, v)

        stop_list = list(stop or [])
        stop_list.extend(s for s in self._stop_tokens if s not in stop_list)
        if stop_list:
            sampling_params["stop"] = stop_list

        # Forward the structured-output spec to SGLang.
        # SGLang's ``/generate`` accepts ``json_schema``, ``regex``, and
        # ``ebnf`` as ``sampling_params`` fields when launched with
        # ``--grammar-backend outlines``. The worker has already
        # cache-validated the compile via :func:`compile_outlines`, so
        # any failure here is genuinely a backend bug — SGLang surfaces
        # it as HTTP 500 which becomes a ``finish_reason: "error"``
        # chunk via the existing exception handler.
        if grammar is not None:
            if grammar.kind == "json_schema":
                # SGLang's Outlines backend expects the schema either
                # as a JSON-encoded string or as a dict — newer versions
                # accept both. JSON-encoding is safer across the wire
                # (no surprises around msgpack ↔ dict identity).
                sampling_params["json_schema"] = json.dumps(grammar.value)
            elif grammar.kind == "regex":
                sampling_params["regex"] = grammar.value
            elif grammar.kind == "ebnf":
                # SGLang accepts EBNF only on EBNF-capable backends
                # (``xgrammar``/``llguidance``) — the default ``outlines``
                # backend skips/fails it. Models must therefore only
                # advertise ``"ebnf"`` in their grammar capabilities when
                # their profile pins such a backend (gated by the gateway
                # capability check + the profile-backend consistency test).
                # The gateway has size-capped the source via ``MAX_EBNF_LEN``;
                # further compile-time validation happens inside the backend.
                sampling_params["ebnf"] = grammar.value

        # Multi-candidate (``n > 1`` and/or ``best_of``): ask SGLang for all
        # candidates in one non-streaming call (``sampling_params["n"]`` → native
        # parallel sampling) and emit them as one terminal chunk's ``candidates``
        # array. For ``best_of`` we generate ``best_of`` candidates, rank by
        # cumulative logprob, and return the top ``n`` (re-indexed). Reuses
        # SGLang's batched sampling and avoids interleaving N streams through the
        # gateway aggregator. ``best_of`` is non-streaming only; ``n>1`` is
        # supported on both paths (streaming branch directly below).
        return_count = n if (n is not None and n >= 1) else 1

        # Streaming multi-candidate (``n>1 && stream``): drive SGLang's streaming
        # ``/generate`` with ``sampling_params["n"]`` and fan the per-index events
        # out as per-candidate delta chunks tagged with ``choice_index``. SGLang
        # emits cumulative ``text`` per ``index`` on each event; we diff to a
        # delta. Per-choice terminals carry their own ``finish_reason`` /
        # ``logprobs`` (done=False so they ride the regular delta path); a final
        # ``done=True`` global terminal closes the stream and carries aggregate
        # usage. (``best_of`` is non-streaming — the gateway rejects
        # ``best_of && stream`` — so it never reaches here.)
        if stream and return_count > 1:
            sp = dict(sampling_params)
            sp["n"] = return_count
            sbody: dict[str, Any] = {
                "text": prompt,
                "sampling_params": sp,
                "stream": True,
                "rid": uuid.uuid4().hex,
            }
            if lora_path:
                sbody["lora_path"] = lora_path
            if image_data:
                sbody["image_data"] = image_data
            if logprobs:
                sbody["return_logprob"] = True
                # Without this SGLang omits the decoded token TEXT from
                # output_(top_)logprobs (entries are [logprob, token_id, None]),
                # so the OpenAI ``token`` field comes back empty and the guard
                # verdict thresholding cannot match ``yes``/``no``.
                sbody["return_text_in_logprobs"] = True
                if top_logprobs is not None and top_logprobs > 0:
                    sbody["top_logprobs_num"] = top_logprobs
            sclient = await self._get_or_create_http_client()
            last_text: dict[int, str] = {}
            # Per-candidate logprob watermark: SGLang's
            # ``meta_info.output_token_logprobs`` is a per-candidate cumulative
            # list growing across events for that index. Slicing
            # ``[surfaced[idx]:]`` per event yields exactly the new entries
            # introduced on this event for that candidate. Mirrors the
            # single-candidate cursor at line ~1013.
            logprobs_surfaced: dict[int, int] = {}
            prompt_tokens: int | None = None
            total_completion = 0
            emitted_first = False
            async with sclient.stream("POST", f"{self._server_url}/generate", json=sbody) as sresp:
                sresp.raise_for_status()
                async for raw_line in sresp.aiter_lines():
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = int(event.get("index", 0))
                    cumulative = event.get("text", "")
                    if not isinstance(cumulative, str):
                        cumulative = last_text.get(idx, "")
                    delta = cumulative[len(last_text.get(idx, "")) :]
                    last_text[idx] = cumulative
                    meta = event.get("meta_info") or {}
                    if prompt_tokens is None and isinstance(meta.get("prompt_tokens"), int):
                        prompt_tokens = meta["prompt_tokens"]
                    fr = meta.get("finish_reason")
                    fr_type = fr.get("type") if isinstance(fr, dict) else fr
                    candidate_done = fr_type is not None
                    if candidate_done and isinstance(meta.get("completion_tokens"), int):
                        total_completion += meta["completion_tokens"]
                    # Per-candidate logprob slice — same shape conversion as the
                    # single-candidate path (`_chunk_from_sglang_event`), but
                    # the cursor is keyed by candidate index so each candidate
                    # gets its own monotonic slice.
                    chunk_logprobs: tuple[dict[str, Any], ...] | None = None
                    if logprobs and isinstance(meta, dict):
                        all_token_lp = meta.get("output_token_logprobs")
                        all_top_lp = meta.get("output_top_logprobs")
                        prior = logprobs_surfaced.get(idx, 0)
                        if isinstance(all_token_lp, list) and len(all_token_lp) > prior:
                            new_slice = all_token_lp[prior:]
                            new_top_slice = (
                                all_top_lp[prior:]
                                if isinstance(all_top_lp, list) and len(all_top_lp) >= len(all_token_lp)
                                else [None] * len(new_slice)
                            )
                            built: list[dict[str, Any]] = []
                            for token_entry, top_entry in zip(new_slice, new_top_slice, strict=False):
                                tok_lp, _tok_id, tok_text = _unpack_sglang_token_logprob(token_entry)
                                if tok_lp is None:
                                    continue
                                top_list: list[dict[str, Any]] = []
                                if isinstance(top_entry, list):
                                    for top_token in top_entry:
                                        t_lp, _t_id, t_text = _unpack_sglang_token_logprob(top_token)
                                        if t_lp is None:
                                            continue
                                        top_list.append(
                                            {
                                                "token": t_text or "",
                                                "logprob": float(t_lp),
                                                "bytes": list((t_text or "").encode("utf-8")),
                                            }
                                        )
                                built.append(
                                    {
                                        "token": tok_text or "",
                                        "logprob": float(tok_lp),
                                        "bytes": list((tok_text or "").encode("utf-8")),
                                        "top_logprobs": top_list,
                                    }
                                )
                            if built:
                                chunk_logprobs = tuple(built)
                            # Advance to the cumulative reported length rather
                            # than incrementing by ``len(built)`` so a skipped
                            # malformed entry does not re-surface next event.
                            logprobs_surfaced[idx] = len(all_token_lp)
                    if not delta and not candidate_done and not chunk_logprobs:
                        continue
                    is_first = bool(delta) and not emitted_first
                    emitted_first = emitted_first or bool(delta)
                    yield GenerationChunk(
                        text_delta=delta,
                        done=False,
                        is_first=is_first,
                        finish_reason=cast("FinishReason | None", fr_type if candidate_done else None),
                        choice_index=idx,
                        logprobs=chunk_logprobs,
                    )
            # Single global terminal closes the multi-candidate stream (carries
            # aggregate usage). Each candidate already received its own
            # ``finish_reason`` on the per-choice completion chunk above; this
            # terminal is the stream-level "all candidates done" signal that
            # drives the processor's loop break and the gateway's [DONE].
            yield GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
            )
            return

        gen_count = best_of if (best_of is not None and best_of > 1) else return_count
        if gen_count > 1:
            # Rank only when over-generating (best_of > the returned count);
            # ranking needs per-candidate logprobs from SGLang.
            rank = best_of is not None and best_of > return_count
            sp = dict(sampling_params)
            sp["n"] = gen_count
            nbody: dict[str, Any] = {"text": prompt, "sampling_params": sp, "stream": False}
            if lora_path:
                nbody["lora_path"] = lora_path
            if image_data:
                nbody["image_data"] = image_data
            if logprobs or rank:
                nbody["return_logprob"] = True
                # Surface decoded token text (see streaming body below) so the
                # OpenAI ``token`` field is populated and guard thresholding works.
                nbody["return_text_in_logprobs"] = True
                if top_logprobs is not None and top_logprobs > 0:
                    nbody["top_logprobs_num"] = top_logprobs
            nclient = await self._get_or_create_http_client()
            nresp = await nclient.post(f"{self._server_url}/generate", json=nbody)
            nresp.raise_for_status()
            results = nresp.json()
            # SGLang returns a list of ``n`` result objects for ``n > 1``;
            # tolerate a single dict defensively.
            if isinstance(results, dict):
                results = [results]
            if rank:
                # Highest cumulative token-logprob first; keep the top return_count.
                results = sorted(results, key=_cumulative_logprob, reverse=True)[:return_count]
            candidates: list[dict[str, Any]] = []
            total_completion = 0
            prompt_tokens: int | None = None
            for r in results:
                meta = r.get("meta_info", {}) if isinstance(r, dict) else {}
                fr = meta.get("finish_reason")
                fr_type = fr.get("type") if isinstance(fr, dict) else fr
                if prompt_tokens is None and isinstance(meta.get("prompt_tokens"), int):
                    prompt_tokens = meta["prompt_tokens"]
                if isinstance(meta.get("completion_tokens"), int):
                    total_completion += meta["completion_tokens"]
                # Per-candidate logprobs: only emit when the request
                # asked for them. ``return_logprob`` is also set for
                # ``best_of`` ranking; do NOT surface the ranking-only
                # logprobs to the client — that would make ``logprobs:
                # false`` requests sprout a ``logprobs`` payload.
                cand_logprobs: list[dict[str, Any]] | None = None
                if logprobs and isinstance(meta, dict):
                    all_token_lp = meta.get("output_token_logprobs")
                    all_top_lp = meta.get("output_top_logprobs")
                    if isinstance(all_token_lp, list) and all_token_lp:
                        top_iter: list[Any]
                        if isinstance(all_top_lp, list) and len(all_top_lp) >= len(all_token_lp):
                            top_iter = list(all_top_lp)
                        else:
                            top_iter = [None] * len(all_token_lp)
                        built: list[dict[str, Any]] = []
                        for token_entry, top_entry in zip(all_token_lp, top_iter, strict=False):
                            tok_lp, _tok_id, tok_text = _unpack_sglang_token_logprob(token_entry)
                            if tok_lp is None:
                                continue
                            top_list: list[dict[str, Any]] = []
                            if isinstance(top_entry, list):
                                for top_token in top_entry:
                                    t_lp, _t_id, t_text = _unpack_sglang_token_logprob(top_token)
                                    if t_lp is None:
                                        continue
                                    top_list.append(
                                        {
                                            "token": t_text or "",
                                            "logprob": float(t_lp),
                                            "bytes": list((t_text or "").encode("utf-8")),
                                        }
                                    )
                            built.append(
                                {
                                    "token": tok_text or "",
                                    "logprob": float(tok_lp),
                                    "bytes": list((tok_text or "").encode("utf-8")),
                                    "top_logprobs": top_list,
                                }
                            )
                        if built:
                            cand_logprobs = built
                candidates.append(
                    {
                        "text": r.get("text", "") if isinstance(r, dict) else "",
                        "finish_reason": fr_type if isinstance(fr_type, str) else "stop",
                        "logprobs": cand_logprobs,
                    }
                )
            yield GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                candidates=tuple(candidates),
            )
            return

        # rid lets us best-effort cancel mid-stream via /abort_request.
        rid = uuid.uuid4().hex
        body: dict[str, Any] = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
            "rid": rid,
        }
        # Multi-LoRA: select the adapter by served-name. Top-level field (NOT a
        # sampling param — SGLang 0.5.10's SamplingParams rejects ``lora_path``;
        # verified on L4). Empirically applies the adapter in-batch per request.
        if lora_path:
            body["lora_path"] = lora_path
        if image_data:
            body["image_data"] = image_data
        # OpenAI ``logprobs`` → SGLang ``return_logprob`` (top-level body
        # flag, not under sampling_params). ``top_logprobs`` →
        # ``top_logprobs_num``. SGLang surfaces them under
        # ``meta_info.output_token_logprobs`` /
        # ``meta_info.output_top_logprobs`` on each stream event; the
        # chunk translator below converts to the OpenAI ``choices[i]
        # .logprobs.content`` shape.
        if logprobs:
            body["return_logprob"] = True
            # SGLang only includes the decoded token TEXT in
            # output_(top_)logprobs when this is set; without it the entries are
            # [logprob, token_id, None] and the OpenAI ``token`` field (and the
            # guard ``yes``/``no`` verdict match) sees only empty strings.
            body["return_text_in_logprobs"] = True
            if top_logprobs is not None and top_logprobs > 0:
                body["top_logprobs_num"] = top_logprobs

        # Adapter-level TTFT/TPOT timer. Started here (just
        # before the upstream HTTP request) so the worker-side TTFT
        # measures the same window the gateway labels "publish → first
        # chunk", minus the NATS hop. ``finalize`` is invoked in the
        # finally block so cancellation paths still emit TPOT if at
        # least one non-empty chunk was produced. The ``grammar`` label
        # mirrors the gateway side so the overhead-attribution panel can
        # subtract worker latency from gateway latency per mode.
        grammar_label = "none" if grammar is None else grammar.kind
        stream_timer = GenerationStreamTimer(self._served_model_name, grammar=grammar_label)
        terminal_prompt_tokens: int | None = None
        terminal_completion_tokens: int | None = None

        # Use the shared client. The httpx client outlives the generator
        # (it's process-lifetime once opened) so we do NOT enter it via
        # ``async with`` — we just borrow a reference. Cancellation /
        # GeneratorExit still cleans up the stream context below.
        client = await self._get_or_create_http_client()
        try:
            async with client.stream("POST", f"{self._server_url}/generate", json=body) as response:
                if response.status_code != 200:
                    # Drain a bit of the body for diagnostics, then raise.
                    body_preview = await response.aread()
                    logger.error(
                        "SGLang /generate stream error %d: %s",
                        response.status_code,
                        body_preview[:500],
                    )
                    response.raise_for_status()

                last_cumulative_text = ""
                first_yield_done = False
                terminal_yielded = False
                # Number of token-logprob entries already surfaced
                # on prior chunks. SGLang accumulates them on
                # ``meta_info.output_token_logprobs`` (a flat list,
                # one entry per output token, growing each event),
                # so we slice off the tail-since-last-event each
                # round to build per-chunk OpenAI-shape logprobs.
                logprobs_surfaced = 0
                # Guard verdict buffering (CHECK POLICY) — inert for non-guard
                # models. A guard's Yes/No verdict can sit a few token positions
                # in, behind a leading whitespace/punctuation/preamble token that
                # SGLang spreads across streaming chunks. We accumulate the
                # leading chunks' per-token logprob entries (``guard_lp_buffer``)
                # and SUPPRESS their text (the guard consumer wants just the
                # verdict, not the preamble) until a verdict resolves within the
                # first ``_GUARD_VERDICT_SCAN_POSITIONS`` positions — or the
                # stream terminates first, in which case we flush the raw buffered
                # chunks unchanged (fallback, never drop output).
                guard_active = bool(self._guard)
                guard_resolved = False
                guard_lp_buffer: list[dict[str, Any]] = []
                guard_pending: list[GenerationChunk] = []

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("SGLang stream: skipping non-JSON line: %s", line[:200])
                        continue

                    chunk = _chunk_from_sglang_event(
                        event,
                        previous_cumulative_text=last_cumulative_text,
                        first_yield_done=first_yield_done,
                        logprobs_enabled=logprobs,
                        logprobs_surfaced=logprobs_surfaced,
                    )
                    if chunk is None:
                        continue
                    last_cumulative_text = (
                        event.get("text", last_cumulative_text)
                        if isinstance(event.get("text"), str)
                        else last_cumulative_text
                    )
                    if chunk.text_delta and not first_yield_done:
                        first_yield_done = True
                    if chunk.done:
                        terminal_yielded = True
                        terminal_prompt_tokens = chunk.prompt_tokens
                        terminal_completion_tokens = chunk.completion_tokens
                    # Advance to SGLang's cumulative reported length rather
                    # than incrementing by ``len(chunk.logprobs)``: the
                    # translator may skip malformed entries, and that
                    # delta would silently re-emit them on the next event.
                    if logprobs:
                        event_meta = event.get("meta_info") if isinstance(event, dict) else None
                        if isinstance(event_meta, dict):
                            cumulative_lp = event_meta.get("output_token_logprobs")
                            if isinstance(cumulative_lp, list):
                                logprobs_surfaced = max(logprobs_surfaced, len(cumulative_lp))
                    # Guard verdict thresholding (CHECK POLICY): resolve the
                    # P(unsafe)>=threshold verdict from the first parseable
                    # position within ``_GUARD_VERDICT_SCAN_POSITIONS`` and emit a
                    # single verdict chunk. Inert for non-guard models, which take
                    # the byte-for-byte unchanged ``else`` path below.
                    if guard_active and not guard_resolved:
                        # Accumulate this chunk's forced logprob entries so a
                        # verdict that lands on a later token position is visible.
                        if chunk.logprobs:
                            guard_lp_buffer.extend(chunk.logprobs)
                        guard_pending.append(chunk)
                        verdict = _thresholded_verdict(tuple(guard_lp_buffer), self._guard)
                        if verdict is not None:
                            guard_resolved = True
                            # Carry through terminal state if the verdict resolved
                            # on (or only by) the terminal chunk, so done /
                            # finish_reason / completion_tokens are preserved.
                            last = guard_pending[-1]
                            # Strip the internally-forced logprobs when the client
                            # did not ask for them (implementation detail). When
                            # the client did ask, drop the single verdict entry
                            # that was consumed/rewritten (it described the raw
                            # sampled token, not the served threshold verdict) and
                            # keep the rest of the buffered token metadata.
                            if client_requested_logprobs:
                                v_idx = _verdict_position(tuple(guard_lp_buffer))
                                kept = [e for i, e in enumerate(guard_lp_buffer) if i != v_idx]
                                remaining = tuple(kept) or None
                            else:
                                remaining = None
                            verdict_chunk = dataclasses.replace(
                                last,
                                text_delta=verdict,
                                is_first=True,
                                logprobs=remaining,
                            )
                            stream_timer.mark_yield(has_text=True)
                            yield verdict_chunk
                            if verdict_chunk.done:
                                break
                        elif chunk.done:
                            # Terminal reached without a parseable verdict in the
                            # first N positions: flush the raw buffered chunks
                            # unchanged so the response is never dropped. Strip the
                            # forced logprobs only when the client didn't ask.
                            guard_resolved = True
                            for buffered in guard_pending:
                                if not client_requested_logprobs:
                                    buffered = dataclasses.replace(buffered, logprobs=None)
                                stream_timer.mark_yield(has_text=bool(buffered.text_delta))
                                yield buffered
                            break
                        # else: keep buffering (suppress this leading chunk's text).
                    else:
                        # Guard tail chunks (after the verdict resolved) must still
                        # honour the M4 logprobs contract: strip the internally
                        # forced logprobs when the client didn't request them.
                        if guard_active and not client_requested_logprobs and chunk.logprobs is not None:
                            chunk = dataclasses.replace(chunk, logprobs=None)
                        stream_timer.mark_yield(has_text=bool(chunk.text_delta))
                        yield chunk
                        if chunk.done:
                            break

                if not terminal_yielded:
                    raise RuntimeError("SGLang stream terminated without terminal event")
        except GeneratorExit:
            # Caller dropped the iterator (cancellation / aclose). Issue a
            # best-effort POST to /abort_request so SGLang frees the slot
            # promptly. CRITICAL: do NOT ``await`` the abort here. The
            # streaming processor tears this iterator down via
            # ``asyncio.wait_for(chunks_iter.aclose(), timeout=2.0)`` and
            # ``aclose()`` drives this very handler — so awaiting a 5s POST
            # would be cancelled by the 2s teardown cap exactly when SGLang
            # is slow, orphaning the generation (it keeps holding KV/GPU).
            # Awaiting here would also risk ``RuntimeError: async generator
            # ignored GeneratorExit`` if the await is interrupted. Instead
            # we spawn the abort as an independent, tracked background task
            # (bounded by ``_ABORT_REQUEST_TIMEOUT_S`` < the 2s cap) on the
            # adapter's long-lived loop and re-raise GeneratorExit cleanly.
            # The shared client is reused so the abort piggybacks on an
            # existing connection rather than paying a fresh TCP handshake.
            # If a concurrent ``unload()`` / ``aclose_client()`` already
            # closed the shared client, posting through it raises (suppressed
            # in ``_abort_request``) and the abort silently no-ops, leaking
            # the SGLang GPU slot until SGLang's own timeout. Skip the abort
            # in that case — there is no live client to drive it.
            server_url = self._server_url
            if server_url is not None and not client.is_closed:
                self._spawn_abort_request(client, server_url, rid)
            elif server_url is not None:
                logger.debug(
                    "skipping /abort_request for rid=%s: shared HTTP client already closed",
                    rid,
                )
            raise
        finally:
            # Emit TPOT regardless of normal completion vs cancellation.
            # ``finalize`` is a no-op if no non-empty chunks were observed.
            stream_timer.finalize(
                prompt_tokens=terminal_prompt_tokens,
                completion_tokens=terminal_completion_tokens,
            )


def _p_unsafe_from_entry(entry: Any) -> float | None:
    """``P(unsafe)`` from one OpenAI-shape content token's ``top_logprobs``.

    Renormalises ``exp(lp_yes)/(exp(lp_yes)+exp(lp_no))`` over the ``yes``/``no``
    verdict tokens in this single position. ``None`` when neither appears.
    """
    if not isinstance(entry, dict):
        return None
    lp_yes: float | None = None
    lp_no: float | None = None
    for top in entry.get("top_logprobs") or []:
        if not isinstance(top, dict):
            continue
        tok = str(top.get("token") or "").strip().lower()
        val = top.get("logprob")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if tok == "yes":
            lp_yes = val if lp_yes is None else max(lp_yes, val)
        elif tok == "no":
            lp_no = val if lp_no is None else max(lp_no, val)
    if lp_yes is None and lp_no is None:
        return None
    ey = math.exp(lp_yes) if lp_yes is not None else 0.0
    en = math.exp(lp_no) if lp_no is not None else 0.0
    return ey / (ey + en) if (ey + en) > 0 else None


def _verdict_position(chunk_logprobs: Any, scan_positions: int = _GUARD_VERDICT_SCAN_POSITIONS) -> int | None:
    """Index of the first position (within ``scan_positions``) carrying a verdict
    distribution, or ``None``. The consumed/rewritten verdict entry the streaming
    intercept drops from client-requested logprobs.
    """
    if not chunk_logprobs:
        return None
    for idx, entry in enumerate(chunk_logprobs[:scan_positions]):
        if _p_unsafe_from_entry(entry) is not None:
            return idx
    return None


def _p_unsafe_from_verdict_logprobs(
    chunk_logprobs: Any, scan_positions: int = _GUARD_VERDICT_SCAN_POSITIONS
) -> float | None:
    """``P(unsafe)`` from a guard verdict chunk's logprobs, or ``None``.

    ``chunk_logprobs`` is the OpenAI ``content`` shape this adapter builds —
    ``({"token", "logprob", "top_logprobs": [{"token", "logprob"}, ...]}, ...)``.
    Scans the first up-to ``scan_positions`` content tokens for the first whose
    ``top_logprobs`` carries a ``yes``/``no`` verdict distribution, then
    renormalises ``exp(lp_yes)/(exp(lp_yes)+exp(lp_no))`` over those two tokens.
    Scanning past position 0 keeps a leading whitespace/punctuation/preamble
    token from hiding the verdict, matching the eval runner's ``content[:3]``
    scan. ``None`` when no verdict token appears in range (caller keeps raw).
    """
    if not chunk_logprobs:
        return None
    for entry in chunk_logprobs[:scan_positions]:
        p_unsafe = _p_unsafe_from_entry(entry)
        if p_unsafe is not None:
            return p_unsafe
    return None


def _thresholded_verdict(
    chunk_logprobs: Any,
    guard: dict[str, Any],
    scan_positions: int = _GUARD_VERDICT_SCAN_POSITIONS,
) -> str | None:
    """The guard's thresholded verdict token, or ``None`` to leave output as-is.

    ``guard`` is ``{"threshold": float, "positive": "Yes", "negative": "No"}``
    (positive/negative default to Yes/No). Returns the ``positive`` label iff
    ``P(unsafe) >= threshold``, else ``negative``; ``None`` when P(unsafe) can't
    be computed (no verdict logprobs within ``scan_positions``) so the raw model
    token is preserved.
    """
    p_unsafe = _p_unsafe_from_verdict_logprobs(chunk_logprobs, scan_positions)
    if p_unsafe is None:
        return None
    threshold = guard.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return None
    positive = str(guard.get("positive") or "Yes")
    negative = str(guard.get("negative") or "No")
    return positive if p_unsafe >= float(threshold) else negative


def _cumulative_logprob(result: Any) -> float:
    """Sum a candidate's per-token output logprobs (for ``best_of`` ranking).

    SGLang's ``meta_info.output_token_logprobs`` is a list of
    ``[logprob, token_id, ...]`` entries; we sum the leading logprob. A candidate
    with no/garbled logprobs scores ``-inf`` so it ranks last rather than
    crashing the sort (graceful fallback if SGLang returned no logprobs).
    """
    if not isinstance(result, dict):
        return float("-inf")
    meta = result.get("meta_info")
    if not isinstance(meta, dict):
        return float("-inf")
    lps = meta.get("output_token_logprobs")
    if not isinstance(lps, list) or not lps:
        return float("-inf")
    total = 0.0
    for entry in lps:
        if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (int, float)):
            total += float(entry[0])
    return total


def _chunk_from_sglang_event(
    event: Any,
    *,
    previous_cumulative_text: str,
    first_yield_done: bool,
    logprobs_enabled: bool = False,
    logprobs_surfaced: int = 0,
) -> GenerationChunk | None:
    """Translate one SGLang stream event into a :class:`GenerationChunk`.

    SGLang emits cumulative ``text`` on every event (not per-token deltas),
    so we diff against the last observed cumulative text to compute the
    delta. The terminal event has ``meta_info.finish_reason`` populated.

    When ``logprobs_enabled`` is True, the translator slices SGLang's
    ``meta_info.output_token_logprobs`` (and ``output_top_logprobs``,
    when ``top_logprobs_num`` was requested) from
    ``logprobs_surfaced`` to the current event length, producing the
    OpenAI ``ChatCompletionTokenLogprob`` shape for the tokens
    introduced by *this* event.
    """
    if not isinstance(event, dict):
        return None
    cumulative = event.get("text", "")
    if not isinstance(cumulative, str):
        cumulative = ""

    delta = ""
    if cumulative.startswith(previous_cumulative_text):
        # Normal monotonic growth: SGLang's cumulative buffer extends the
        # text we last saw, so the delta is just the new suffix.
        delta = cumulative[len(previous_cumulative_text) :]
    else:
        # Non-monotonic cumulative text: SGLang reported a buffer that is
        # NOT an extension of what we already streamed (it diverged or is
        # shorter). Emitting ``cumulative`` whole here would duplicate the
        # previously-streamed prefix onto the wire. There is no safe delta
        # to emit, so skip this event's text entirely (``delta`` stays
        # ""). A terminal event still produces its terminal chunk below
        # (with an empty text delta); a non-terminal event is dropped by
        # the ``not delta and not chunk_logprobs`` guard further down.
        logger.warning(
            "SGLang stream: non-monotonic cumulative text (len prev=%d, len cur=%d, "
            "current is not a prefix-extension); skipping delta to avoid duplicate output",
            len(previous_cumulative_text),
            len(cumulative),
        )

    meta = event.get("meta_info") or {}
    raw_finish = meta.get("finish_reason") if isinstance(meta, dict) else None
    if isinstance(raw_finish, dict):
        raw_finish = raw_finish.get("type")
    is_terminal = raw_finish is not None or bool(event.get("finished"))

    is_first = (not first_yield_done) and bool(delta)

    # OpenAI-shape per-token logprobs for the slice of tokens this event
    # added. SGLang's flat shape is ``[(logprob, token_id, token_text),
    # ...]`` for the bottom list and ``[[(lp, tid, tt), ...], ...]`` for
    # the top-k list. We surface whatever SGLang gave us and tolerate a
    # different inner shape on newer versions (skip the entry rather
    # than raise — partial logprobs are a UX papercut, not a bug).
    chunk_logprobs: tuple[dict[str, Any], ...] | None = None
    if logprobs_enabled and isinstance(meta, dict):
        all_token_lp = meta.get("output_token_logprobs")
        all_top_lp = meta.get("output_top_logprobs")
        if isinstance(all_token_lp, list) and len(all_token_lp) > logprobs_surfaced:
            new_slice = all_token_lp[logprobs_surfaced:]
            new_top_slice = (
                all_top_lp[logprobs_surfaced:]
                if isinstance(all_top_lp, list) and len(all_top_lp) >= len(all_token_lp)
                else [None] * len(new_slice)
            )
            built: list[dict[str, Any]] = []
            for token_entry, top_entry in zip(new_slice, new_top_slice, strict=False):
                tok_lp, _tok_id, tok_text = _unpack_sglang_token_logprob(token_entry)
                if tok_lp is None:
                    continue
                top_list: list[dict[str, Any]] = []
                if isinstance(top_entry, list):
                    for top_token in top_entry:
                        t_lp, _t_id, t_text = _unpack_sglang_token_logprob(top_token)
                        if t_lp is None:
                            continue
                        top_list.append(
                            {
                                "token": t_text or "",
                                "logprob": float(t_lp),
                                "bytes": list((t_text or "").encode("utf-8")),
                            }
                        )
                built.append(
                    {
                        "token": tok_text or "",
                        "logprob": float(tok_lp),
                        "bytes": list((tok_text or "").encode("utf-8")),
                        "top_logprobs": top_list,
                    }
                )
            if built:
                chunk_logprobs = tuple(built)

    if is_terminal:
        finish_reason: FinishReason
        if raw_finish in ("stop", "length", "cancelled", "error"):
            finish_reason = raw_finish  # type: ignore[assignment]
        else:
            finish_reason = "stop"
        prompt_tokens = meta.get("prompt_tokens") if isinstance(meta, dict) else None
        completion_tokens = meta.get("completion_tokens") if isinstance(meta, dict) else None
        return GenerationChunk(
            text_delta=delta,
            done=True,
            is_first=is_first,
            finish_reason=finish_reason,
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else None,
            logprobs=chunk_logprobs,
        )

    if not delta and not chunk_logprobs:
        # No new text or logprobs and not terminal — skip
        # (keeps NATS message rate down).
        return None

    return GenerationChunk(
        text_delta=delta,
        done=False,
        is_first=is_first,
        logprobs=chunk_logprobs,
    )


def _unpack_sglang_token_logprob(entry: Any) -> tuple[float | None, int | None, str | None]:
    """Unpack a single SGLang token-logprob entry.

    SGLang has shipped two shapes across recent versions: a 3-tuple
    ``(logprob, token_id, token_text)`` and a 2-tuple
    ``(logprob, token_id)`` (with the token text reconstructable from
    the tokenizer). We support both and silently fall back to ``None``
    fields when SGLang ships a shape we don't recognise (logprobs
    surface for diagnostics; a partial result is still useful).
    """
    if isinstance(entry, list | tuple):
        if len(entry) >= 3:
            lp, tid, text = entry[0], entry[1], entry[2]
            return (
                float(lp) if isinstance(lp, int | float) else None,
                int(tid) if isinstance(tid, int) else None,
                str(text) if isinstance(text, str) else None,
            )
        if len(entry) == 2:
            lp, tid = entry
            return (
                float(lp) if isinstance(lp, int | float) else None,
                int(tid) if isinstance(tid, int) else None,
                None,
            )
    return (None, None, None)


# Backwards-compatibility shim for the walking-skeleton non-streaming response shape
# (the local-dev ``/v1/generate`` route consumed this directly). Kept so any
# external caller / test that imports it still resolves; new code uses
# ``collect_generation`` from the base module.
def _parse_sglang_generate_response(result: Any) -> GenerationResult:
    """Map a walking-skeleton-shape SGLang ``/generate`` (non-streaming) response.

    Retained for the test suite that exercised the blocking shape — the
    streaming path no longer produces this envelope on its own; tests
    that need the aggregate value drain the iterator via
    :func:`collect_generation`.
    """
    if isinstance(result, list):
        if not result:
            msg = "SGLang /generate returned an empty list"
            raise RuntimeError(msg)
        result = result[0]
    if not isinstance(result, dict):
        msg = f"SGLang /generate returned unexpected shape: {type(result).__name__}"
        raise RuntimeError(msg)

    text = result.get("text", "")
    if not isinstance(text, str):
        msg = "SGLang /generate response missing 'text'"
        raise RuntimeError(msg)

    meta = result.get("meta_info") or {}
    prompt_tokens = int(meta.get("prompt_tokens", 0))
    completion_tokens = int(meta.get("completion_tokens", 0))

    raw_finish = meta.get("finish_reason")
    if isinstance(raw_finish, dict):
        raw_finish = raw_finish.get("type")
    finish_reason = raw_finish if raw_finish in ("stop", "length") else "stop"

    return GenerationResult(
        text=text,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
