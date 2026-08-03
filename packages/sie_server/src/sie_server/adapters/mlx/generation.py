"""MLX adapter for autoregressive text generation on Apple Silicon.

The Mac-native generation backend. Generation runs in a managed
``mlx_lm.server`` subprocess (Apple's MLX framework, Metal-native) that this
adapter spawns and proxies — launched from the server's own environment (the
device-agnostic ``sglang`` bundle installs ``mlx-lm`` on Apple Silicon; see
:mod:`sie_server.adapters.mlx._server`). A subprocess rather than in-process
because ``mlx_lm.server`` already exposes the OpenAI ``/v1/completions`` server
this adapter proxies, and its own process gives clean single-tenant lifecycle +
Metal-memory isolation.

Selection is automatic: :meth:`SGLangGenerationAdapter.create_for_device`
swaps to this adapter on any non-CUDA device, so unsuffixed generation models
"just work" on a Mac (served via ``mise run serve -b sglang``).

Streaming async-iterator contract (same as the SGLang adapter). The adapter
opens an HTTP streaming connection to the child's OpenAI-compatible
``/v1/completions`` endpoint (``stream: true`` + ``stream_options.include_usage``),
parses the SSE, and yields :class:`GenerationChunk` objects. The terminal
chunk carries ``finish_reason`` plus prompt/completion token counts.

Cancellation: dropping the async iterator (``aclose()``) exits the
``httpx.stream`` context, closing the upstream connection — which the
single-tenant ``mlx_lm.server`` treats as a stop signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationChunk,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.mlx import _server
from sie_server.types.inputs import ImageInput

logger = logging.getLogger(__name__)

# Inter-chunk read timeout. Unlike the SGLang path (which sits behind the worker /
# gateway with its own deadline + abort), the MLX ``generate()`` is invoked DIRECTLY
# by the /v1/generate and /v1/chat/completions routes, so it must bound itself — a
# wedged child (Metal stall, paused decode) must not hang the request forever. The
# default is generous (300s between tokens is "truly stuck", not "slow"); override
# via ``SIE_MLX_READ_TIMEOUT_S``.
_CONNECT_TIMEOUT_S = 10.0
_WRITE_TIMEOUT_S = 10.0
_POOL_TIMEOUT_S = 10.0
try:
    _READ_TIMEOUT_S = float(os.environ.get("SIE_MLX_READ_TIMEOUT_S", "300"))
except ValueError:
    # Invalid override must degrade gracefully, not crash adapter import
    # (mirrors _server.resolve_startup_timeout).
    logger.warning(
        "Invalid SIE_MLX_READ_TIMEOUT_S=%r; falling back to 300s",
        os.environ.get("SIE_MLX_READ_TIMEOUT_S"),
    )
    _READ_TIMEOUT_S = 300.0

# Finish reasons mlx_lm.server emits, mapped to the SIE contract. Anything else
# falls back to "stop".
_FINISH_REASONS: frozenset[str] = frozenset({"stop", "length"})

# default_sampling keys mlx_lm.server's /v1/completions actually understands. The
# curated generation profiles also set SGLang/OpenAI-only knobs (presence_penalty,
# frequency_penalty) for the CUDA path; those are filtered out here rather than
# forwarded to the MLX child, which does not use them (and may reject unknown fields
# on some versions) — keeping this adapter's "those are unsupported" contract honest.
_MLX_SAMPLING_KEYS: frozenset[str] = frozenset(
    {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens", "seed", "stop"}
)


def normalize_mlx_seed(seed: int) -> int:
    """Convert a signed i64 seed to MLX's equivalent uint64 bit pattern."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(1 << 63) <= seed <= (1 << 63) - 1:
        raise ValueError("MLX seed must be a signed 64-bit integer")
    return seed & ((1 << 64) - 1)


class MLXGenerationAdapter(GenerationAdapter):
    """Streaming MLX generation adapter (Apple-Silicon, ``mlx_lm.server`` subprocess).

    Lifecycle mirrors :class:`SGLangGenerationAdapter`: one subprocess per
    model, started in :meth:`load`, terminated in :meth:`unload`. Inference is
    an HTTP POST to ``{server_url}/v1/completions`` with ``stream: true``.
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("tokens",),
        unload_fields=("_process", "_server_url"),
    )

    # The mlx_lm.server child owns its own signal handling; the parent load must
    # not block the uvicorn event loop while polling child readiness.
    requires_main_thread: bool = False
    manages_own_load_timeout: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        *,
        mlx_repo: str | None = None,
        max_seq_length: int = 32768,
        default_sampling: dict[str, Any] | None = None,
        stop_tokens: list[str] | None = None,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        # Pinned HuggingFace revision. Accepted for constructor parity with the
        # SGLang adapters, but NOT forwarded: this backend serves the pre-pinned
        # ``mlx_repo`` (not ``model_name_or_path``) via an ``mlx_lm.server``
        # subprocess whose argv is built in ``_server.build_launch_command``,
        # which exposes no ``--revision`` seam. No pinned YAML routes here today.
        revision: str | None = None,
        startup_timeout_s: float | None = None,
        **kwargs: Any,  # accept (and drop) CUDA/SGLang-only kwargs from the swap
    ) -> None:
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        # The MLX-quantized repo to serve (e.g. ``mlx-community/Qwen3.5-4B-4bit``), from
        # the model YAML's ``adapter_options.loadtime.mlx_repo``. REQUIRED for the Mac
        # generation path — load() fails fast if it is missing rather than letting
        # mlx_lm download a multi-GB full-precision HF model that then OOMs on Metal
        # (e.g. a non-curated generation model matched into the `sglang` bundle).
        self._mlx_repo = mlx_repo
        self._max_seq_length = max_seq_length
        self._default_sampling = default_sampling or {}
        self._stop_tokens = stop_tokens or []
        self._served_model_name = served_model_name or self._model_name_or_path
        self._trust_remote_code = trust_remote_code
        self._revision = revision  # stored for parity; mlx_lm.server exposes no revision seam (see __init__)
        self._startup_timeout_s = startup_timeout_s

        self._process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._device: str | None = None
        self._output_file: tempfile._TemporaryFileWrapper | None = None
        # Single-flight lock: mlx_lm.server is single-tenant and the cancellation
        # contract assumes one in-flight generation. Bound to the running loop on
        # first acquire (Python 3.10+), so safe to construct off-loop here.
        self._gen_lock = asyncio.Semaphore(1)

    # -- Lifecycle -----------------------------------------------------------

    @property
    def server_url(self) -> str | None:
        """The child ``mlx_lm.server`` base URL (used by the local chat-completions proxy)."""
        return self._server_url

    @property
    def mlx_repo(self) -> str | None:
        """The MLX repo / model id the child serves (the OpenAI ``model`` field).

        ``None`` until set by config; a loaded adapter always has it (load() guards).
        """
        return self._mlx_repo

    def generation_slot(self) -> asyncio.Semaphore:
        """Single-flight lock for this model's MLX child.

        ``generate()`` holds it for its whole stream; the chat-completions proxy
        acquires the same lock so the single-tenant child only ever serves one
        generation at a time.
        """
        return self._gen_lock

    def load(self, device: str) -> None:
        # Fail fast (before any download/subprocess) if the model has no MLX repo. This
        # guards non-curated generation models that match the `sglang` bundle by adapter
        # module but lack adapter_options.loadtime.mlx_repo — without this, mlx_lm would
        # download the full-precision HF weights (e.g. a 27B model) and OOM on Metal.
        if not self._mlx_repo:
            raise RuntimeError(
                f"Generation model '{self._model_name_or_path}' has no mlx_repo set; the Mac MLX "
                "backend requires an MLX-quantized repo. Add adapter_options.loadtime.mlx_repo "
                "to the model YAML (e.g. 'mlx-community/<model>-4bit'), or serve a curated Mac model "
                "(Qwen/Qwen3.5-4B)."
            )
        mlx_repo = self._mlx_repo  # narrowed to str for the calls below
        # mlx-lm is installed by the device-agnostic ``sglang`` bundle on Apple Silicon.
        # If it is missing, the child launch would fail with an opaque error — surface
        # the actionable fix instead.
        if not _server.mlx_lm_available():
            raise RuntimeError(
                "mlx-lm is not installed in this environment; the Apple-Silicon generation "
                "backend requires it. Serve generation with the 'sglang' bundle "
                "(`mise run serve -b sglang`), which installs mlx-lm on macOS."
            )
        # Defensive: a prior load that left a live subprocess would be orphaned (it
        # survives in its own session, holding its port + Metal memory). Tear it down
        # first so a double-load (e.g. a retry) is idempotent rather than leaking.
        if self._process is not None:
            self.unload()
        self._device = device
        port = _server.find_free_port()
        self._server_url = f"http://127.0.0.1:{port}"

        cmd = _server.build_launch_command(mlx_repo=mlx_repo, port=port)
        logger.info(
            "Starting MLX generation server for %s (repo=%s) on device=%s at port %d",
            self._model_name_or_path,
            self._mlx_repo,
            device,
            port,
        )
        logger.info("Loading model %s via MLX (first run downloads weights — this can take minutes)…", self._mlx_repo)

        self._output_file = _server.open_output_log()
        self._process = _server.launch_mlx_server(cmd, output_file=self._output_file)

        # One budget covers BOTH health-readiness and the model-load warmup below, so a
        # slow health bind can't let warmup tack on a second full timeout.
        budget = _server.resolve_startup_timeout(self._startup_timeout_s)
        started = time.monotonic()
        if not _server.wait_for_server(
            self._server_url,
            self._process,
            output_file=self._output_file,
            timeout_s=budget,
        ):
            self._abort_failed_load()
            raise RuntimeError(_server.ERR_SERVER_STARTUP)

        # mlx_lm.server answers /health as soon as httpd is up, possibly before the model
        # finishes loading. Warm it up with a 1-token request so the cold model load
        # completes inside load() (deterministic readiness) and the user's first real request
        # is fast. Use the REMAINING budget so wait+warmup stay within one timeout. A failed
        # warmup means the child bound the port but cannot serve the model, so treat it as a
        # load failure rather than reporting "ready" and handing the error to the first request.
        warmup_budget = max(30.0, budget - (time.monotonic() - started))
        if not _server.warmup_model(self._server_url, mlx_repo, timeout_s=warmup_budget):
            self._abort_failed_load()
            raise RuntimeError(_server.ERR_WARMUP_FAILED)

        logger.info("MLX generation server ready: %s at %s", self._model_name_or_path, self._server_url)

    def _abort_failed_load(self) -> None:
        """Tear down a partially-started child + reset state after a failed load.

        The registry does not call unload() on a failed load, so this mirrors unload():
        terminate the child, drop the temp log, and clear _server_url/_device so the adapter
        does not look "loaded" (_check_loaded() gates only on _server_url).
        """
        _server.terminate_process(self._process)
        self._process = None
        self._server_url = None
        self._device = None
        self._cleanup_output_log()

    def unload(self) -> None:
        if self._process is not None:
            logger.info("Shutting down MLX generation server for %s", self._model_name_or_path)
            _server.terminate_process(self._process)
            self._process = None
        self._server_url = None
        self._device = None
        self._cleanup_output_log()

    def _cleanup_output_log(self) -> None:
        """Close + delete the subprocess stdout/stderr log (best-effort, idempotent).

        Without this, one ``mlx_*.log`` (holding the child's HF download URLs) leaks
        into /tmp per load — unbounded under idle-eviction / hot-reload churn — and the
        open fd is held for the subprocess lifetime.
        """
        if self._output_file is not None:
            with contextlib.suppress(OSError):
                self._output_file.close()
            with contextlib.suppress(OSError):
                Path(self._output_file.name).unlink()
            self._output_file = None

    def memory_footprint(self) -> int:
        # The model lives in the MLX subprocess (Metal-direct), invisible to the
        # parent torch-MPS tracker; let the registry treat it as 0 like SGLang.
        return 0

    # -- Inference -----------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._server_url is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def _build_sampling_body(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        stop: list[str] | None,
        seed: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._mlx_repo,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if top_k is not None:
            body["top_k"] = top_k
        if seed is not None:
            # The public/worker contract preserves signed i64 values, while
            # ``mlx.core.random.seed`` binds a uint64. Preserve the same 64-bit
            # seed pattern at this backend boundary (e.g. ``-1`` → u64::MAX).
            body["seed"] = seed
        stop_list = list(stop or [])
        stop_list.extend(s for s in self._stop_tokens if s not in stop_list)
        if stop_list:
            body["stop"] = stop_list
        # Merge model-config default sampling — but only keys mlx_lm.server supports
        # (drop presence_penalty / frequency_penalty etc. that the curated profile sets
        # for the CUDA path). Explicit request fields set above win (setdefault only
        # fills omitted keys).
        for key, value in self._default_sampling.items():
            if key in _MLX_SAMPLING_KEYS:
                body.setdefault(key, value)
        if "seed" in body:
            # Normalize after defaults so model-config and request seeds obey
            # the same MLX uint64 boundary contract.
            body["seed"] = normalize_mlx_seed(body["seed"])
        return body

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
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
        **kwargs: Any,  # tolerate SGLang-only kwargs (grammar, n, best_of, stream, …)
    ) -> AsyncIterator[GenerationChunk]:
        self._check_loaded()
        # Vision is deferred on the Mac MLX path (mlx-vlm would need its own
        # transformers>=5.5 process — see the implementation plan §9). Fail loud
        # rather than silently dropping the images and returning a wrong answer.
        if images:
            raise ValueError("vision input is not supported on the Mac MLX generation path yet (see plan §9)")
        # Unused on the MLX path today (kept to satisfy the streaming contract):
        # mlx_lm.server's /v1/completions does not expose these knobs uniformly.
        if min_new_tokens is not None:
            raise ValueError("min_new_tokens is not supported on the MLX generation path")
        _ = (frequency_penalty, presence_penalty, repetition_penalty, logit_bias, logprobs, top_logprobs, kwargs)

        body = self._build_sampling_body(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=seed,
        )

        finish_reason: FinishReason = "stop"
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        counted_completion = 0
        first_yield_done = False
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S, write=_WRITE_TIMEOUT_S, pool=_POOL_TIMEOUT_S
        )

        # Serialize generations: mlx_lm.server is single-tenant, and the "drop the
        # iterator to stop" cancellation contract is only safe with one generation in
        # flight. The lock is held for the whole stream lifetime (released on
        # completion, cancellation, or error). The chat-completions proxy acquires the
        # same lock via generation_slot().
        async with self._gen_lock:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", f"{self._server_url}/v1/completions", json=body) as response,
            ):
                if response.status_code != 200:
                    preview = await response.aread()
                    logger.error("MLX /v1/completions error %d: %s", response.status_code, preview[:500])
                    response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    # Skip SSE comments (": keepalive N/M") and blanks.
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("MLX stream: skipping non-JSON line: %s", line[:200])
                        continue

                    # Usage-only event (choices empty) carries the token counts; it
                    # arrives after the terminal choice event.
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        if isinstance(usage.get("prompt_tokens"), int):
                            prompt_tokens = usage["prompt_tokens"]
                        if isinstance(usage.get("completion_tokens"), int):
                            completion_tokens = usage["completion_tokens"]

                    choices = event.get("choices")
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("text") or ""
                    fr = choice.get("finish_reason")
                    if fr is not None:
                        finish_reason = cast("FinishReason", fr if fr in _FINISH_REASONS else "stop")
                    if delta:
                        counted_completion += 1
                        is_first = not first_yield_done
                        first_yield_done = True
                        yield GenerationChunk(text_delta=delta, done=False, is_first=is_first)

            # Single terminal chunk carrying finish_reason + usage. mlx_lm.server emits a
            # ``stream_options.include_usage`` usage event (the authoritative token count).
            # The fallback counts SSE delta CHUNKS, which approximates — but may undercount —
            # completion tokens if a future/older mlx-lm omits the usage event or batches
            # multiple tokens per chunk.
            yield GenerationChunk(
                text_delta="",
                done=True,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if completion_tokens is not None else counted_completion,
            )
