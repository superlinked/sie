"""Generic TensorRT-LLM encoder-decoder generation adapter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
from transformers import AutoTokenizer

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationChunk,
    GenerationDrainingError,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
    GenerationPreflightResult,
    GenerationUnsupportedFieldError,
    aclose_with_error_precedence,
    consume_generation_preflight,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.tensorrt_llm import _server
from sie_server.adapters.tensorrt_llm.compat import install_transformers_5_t5_tokenizer_compatibility
from sie_server.config.model import is_immutable_revision
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 10.0
_READ_TIMEOUT_S = 300.0
_WRITE_TIMEOUT_S = 10.0
_POOL_TIMEOUT_S = 10.0
_CLIENT_CLOSE_TIMEOUT_S = 2.0


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _batch_sizes(name: str, values: list[int] | tuple[int, ...], *, maximum: int) -> list[int]:
    result = [_positive_int(name, value) for value in values]
    if not result or result != sorted(set(result)) or result[-1] > maximum:
        raise ValueError(f"{name} must be sorted unique positive integers <= {maximum}")
    return result


def _fraction(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number in (0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    return result


def _boolean(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _completion_logprobs(payload: Any) -> tuple[dict[str, Any], ...] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise RuntimeError("TensorRT-LLM completion logprobs must be an object")
    tokens = payload.get("tokens")
    token_logprobs = payload.get("token_logprobs")
    top_logprobs = payload.get("top_logprobs")
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list) or len(tokens) != len(token_logprobs):
        raise RuntimeError("TensorRT-LLM completion logprobs have misaligned token arrays")
    if top_logprobs is None:
        top_logprobs = [{} for _ in tokens]
    if not isinstance(top_logprobs, list) or len(top_logprobs) != len(tokens):
        raise RuntimeError("TensorRT-LLM completion logprobs have misaligned top-logprob arrays")

    built: list[dict[str, Any]] = []
    for token, token_logprob, alternatives in zip(tokens, token_logprobs, top_logprobs, strict=True):
        if not isinstance(token, str) or isinstance(token_logprob, bool) or not isinstance(token_logprob, int | float):
            raise RuntimeError("TensorRT-LLM completion logprobs contain an invalid token")
        if alternatives is None:
            alternatives = {}
        if not isinstance(alternatives, Mapping):
            raise RuntimeError("TensorRT-LLM top logprobs must be token-to-logprob objects")
        top: list[dict[str, Any]] = []
        for alternative, logprob in alternatives.items():
            if not isinstance(alternative, str) or isinstance(logprob, bool) or not isinstance(logprob, int | float):
                raise RuntimeError("TensorRT-LLM top logprobs contain an invalid entry")
            top.append(
                {
                    "token": alternative,
                    "logprob": float(logprob),
                    "bytes": list(alternative.encode("utf-8")),
                }
            )
        built.append(
            {
                "token": token,
                "logprob": float(token_logprob),
                "bytes": list(token.encode("utf-8")),
                "top_logprobs": top,
            }
        )
    return tuple(built) or None


class TensorRTLLMGenerationAdapter(GenerationAdapter):
    """Serve a direct-HF encoder-decoder model through one local rc24 child."""

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("tokens",),
        unload_fields=("_process", "_server_url", "_tokenizer"),
    )
    requires_main_thread = False
    manages_own_load_timeout = True
    context_length_accounting = "independent"
    prompt_tokenization_add_special_tokens = True

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        served_model_name: str | None = None,
        max_seq_length: int = 4096,
        max_input_len: int = 2048,
        max_batch_size: int = 8,
        max_num_tokens: int = 8192,
        encoder_max_num_tokens: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        decoder_cuda_graph_batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        encoder_cuda_graph_batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        encoder_cuda_graph_num_tokens: list[int] | tuple[int, ...] = (64, 256, 1024, 2048),
        encoder_cuda_graph_seq_lens: list[int] | tuple[int, ...] = (64, 256, 1024, 2048),
        enable_decoder_cuda_graphs: bool = True,
        enable_encoder_cuda_graphs: bool = True,
        decoder_cuda_graph_enable_padding: bool = True,
        encoder_cuda_graph_enable_padding: bool = True,
        enable_encoder_decoder_mixed_cuda_graph: bool = True,
        disable_overlap_scheduler: bool = False,
        kv_cache_free_gpu_memory_fraction: float = 0.85,
        cross_kv_cache_fraction: float = 0.5,
        startup_timeout_s: float = _server.DEFAULT_STARTUP_TIMEOUT_S,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("TensorRT-LLM requires a Hugging Face model id")
        if not is_immutable_revision(revision):
            raise ValueError("TensorRT-LLM requires revision to be an immutable 40-character commit SHA")
        if not isinstance(compute_precision, str) or compute_precision not in {"bfloat16", "float32"}:
            raise ValueError("TensorRT-LLM rc24 requires compute_precision=bfloat16 or compute_precision=float32")

        self._model_name_or_path = model_name_or_path
        self._revision = revision
        self._served_model_name = served_model_name or model_name_or_path
        self._max_seq_length = _positive_int("max_seq_length", max_seq_length)
        self._max_input_len = _positive_int("max_input_len", max_input_len)
        self._max_batch_size = _positive_int("max_batch_size", max_batch_size)
        self._max_num_tokens = _positive_int("max_num_tokens", max_num_tokens)
        self._encoder_max_num_tokens = _positive_int("encoder_max_num_tokens", encoder_max_num_tokens)
        if self._max_input_len > self._max_seq_length:
            raise ValueError("max_input_len must not exceed max_seq_length")
        if self._max_num_tokens < self._max_batch_size:
            raise ValueError("max_num_tokens must be at least max_batch_size")
        if self._encoder_max_num_tokens < self._max_input_len:
            raise ValueError("encoder_max_num_tokens must be at least max_input_len")

        self._decoder_graph_batches = _batch_sizes(
            "decoder_cuda_graph_batch_sizes", decoder_cuda_graph_batch_sizes, maximum=self._max_batch_size
        )
        self._encoder_graph_batches = _batch_sizes(
            "encoder_cuda_graph_batch_sizes", encoder_cuda_graph_batch_sizes, maximum=self._max_batch_size
        )
        self._encoder_graph_tokens = _batch_sizes(
            "encoder_cuda_graph_num_tokens", encoder_cuda_graph_num_tokens, maximum=self._encoder_max_num_tokens
        )
        self._encoder_graph_seq_lens = _batch_sizes(
            "encoder_cuda_graph_seq_lens", encoder_cuda_graph_seq_lens, maximum=self._max_input_len
        )
        self._enable_decoder_cuda_graphs = _boolean("enable_decoder_cuda_graphs", enable_decoder_cuda_graphs)
        self._enable_encoder_cuda_graphs = _boolean("enable_encoder_cuda_graphs", enable_encoder_cuda_graphs)
        self._decoder_cuda_graph_enable_padding = _boolean(
            "decoder_cuda_graph_enable_padding", decoder_cuda_graph_enable_padding
        )
        self._encoder_cuda_graph_enable_padding = _boolean(
            "encoder_cuda_graph_enable_padding", encoder_cuda_graph_enable_padding
        )
        self._enable_encoder_decoder_mixed_cuda_graph = _boolean(
            "enable_encoder_decoder_mixed_cuda_graph", enable_encoder_decoder_mixed_cuda_graph
        )
        self._disable_overlap_scheduler = _boolean("disable_overlap_scheduler", disable_overlap_scheduler)
        if compute_precision == "float32" and self._enable_decoder_cuda_graphs:
            raise ValueError("compute_precision=float32 requires enable_decoder_cuda_graphs=false")
        if compute_precision == "float32" and self._enable_encoder_decoder_mixed_cuda_graph:
            raise ValueError("compute_precision=float32 requires enable_encoder_decoder_mixed_cuda_graph=false")
        if self._enable_encoder_decoder_mixed_cuda_graph and (
            not self._enable_decoder_cuda_graphs or not self._enable_encoder_cuda_graphs
        ):
            raise ValueError("enable_encoder_decoder_mixed_cuda_graph requires both CUDA graph families")
        self._compute_precision = compute_precision
        self._kv_cache_fraction = _fraction("kv_cache_free_gpu_memory_fraction", kv_cache_free_gpu_memory_fraction)
        self._cross_kv_fraction = _fraction("cross_kv_cache_fraction", cross_kv_cache_fraction)
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and > 0")
        self._startup_timeout_s = float(startup_timeout_s)

        self._process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._tokenizer: Any | None = None
        self._port: int | None = None
        self._device: str | None = None
        self._output_file: tempfile._TemporaryFileWrapper | None = None
        self._config_path: Path | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_lock: asyncio.Lock | None = None
        self._http_client_init_lock = threading.Lock()
        self._pending_aclose: asyncio.Task[None] | None = None
        self._generation_draining = False
        self._active_generations = 0
        self._generation_idle: asyncio.Event | None = None

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> GenerationAdapter:
        _server.parse_cuda_device_index(device)
        return cls(**kwargs)

    @property
    def runtime_config(self) -> dict[str, Any]:
        """Return the checked TensorRT-LLM LLM-API profile written for the child."""
        decoder_graph_config = (
            {
                "batch_sizes": self._decoder_graph_batches,
                "enable_padding": self._decoder_cuda_graph_enable_padding,
            }
            if self._enable_decoder_cuda_graphs
            else None
        )
        encoder_graph_config = (
            {
                "batch_sizes": self._encoder_graph_batches,
                "num_tokens": self._encoder_graph_tokens,
                "seq_lens": self._encoder_graph_seq_lens,
                "enable_padding": self._encoder_cuda_graph_enable_padding,
            }
            if self._enable_encoder_cuda_graphs
            else None
        )
        return {
            "attn_backend": "TRTLLM",
            "dtype": self._compute_precision,
            # rc24's encoder-decoder KV dtype inference also consults the HF
            # model kwargs.  The top-level dtype alone leaves
            # ``model_engine.dtype`` unset; both fields are required by the
            # qualified direct-HF path.
            "model_kwargs": {"torch_dtype": self._compute_precision},
            "enable_chunked_prefill": False,
            "max_batch_size": self._max_batch_size,
            "max_beam_width": 1,
            "max_input_len": self._max_input_len,
            "max_num_tokens": self._max_num_tokens,
            "max_seq_len": self._max_seq_length,
            "encoder_max_batch_size": self._max_batch_size,
            "encoder_max_num_tokens": self._encoder_max_num_tokens,
            # rc24 default-constructs decoder graphs when this key is absent;
            # explicit null is the only supported way to disable them.
            "cuda_graph_config": decoder_graph_config,
            "encoder_cuda_graph_config": encoder_graph_config,
            "enable_encoder_decoder_mixed_cuda_graph": self._enable_encoder_decoder_mixed_cuda_graph,
            "disable_overlap_scheduler": self._disable_overlap_scheduler,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": self._kv_cache_fraction,
                "cross_kv_cache_fraction": self._cross_kv_fraction,
                "use_kv_cache_manager_v2": False,
            },
            "scheduler_config": {"use_python_scheduler": True},
        }

    def load(self, device: str) -> None:
        device_index = _server.parse_cuda_device_index(device)
        self._device = device
        try:
            self._port = _server.reserve_port()
            self._tokenizer = self._load_tokenizer()
            self._server_url = f"http://127.0.0.1:{self._port}"
            with tempfile.NamedTemporaryFile(
                mode="w", prefix="trtllm_", suffix=".json", encoding="utf-8", delete=False
            ) as config_file:
                json.dump(self.runtime_config, config_file, sort_keys=True)
                config_file.write("\n")
                self._config_path = Path(config_file.name)
            self._output_file = _server.open_output_log()
            command = [
                "trtllm-serve",
                self._model_name_or_path,
                "--backend",
                "pytorch",
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
                "--config",
                str(self._config_path),
                "--hf_revision",
                self._revision,
                "--served_model_name",
                self._served_model_name,
                "--no-telemetry",
            ]
            compat_dir = Path(__file__).with_name("_compat")
            inherited_pythonpath = os.environ.get("PYTHONPATH")
            pythonpath = os.pathsep.join(str(path) for path in (compat_dir, inherited_pythonpath) if path)
            self._process = _server.launch(
                command,
                device_index=device_index,
                output_file=self._output_file,
                environment={
                    "PYTHONPATH": pythonpath,
                    "SIE_TRTLLM_RC24_COMPAT": "1",
                },
            )
            _server.wait_until_ready(
                self._process,
                self._server_url,
                served_model_name=self._served_model_name,
                output_file=self._output_file,
                timeout_s=self._startup_timeout_s,
            )
            self._generation_draining = False
        except BaseException:
            self._teardown_child()
            raise

    def _load_tokenizer(self) -> Any:
        install_transformers_5_t5_tokenizer_compatibility()
        return AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            revision=self._revision,
            trust_remote_code=False,
        )

    def _check_loaded(self) -> str:
        if self._server_url is None or self._process is None or self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._server_url

    def _count_prompt_tokens(self, prompt: str) -> int:
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        try:
            token_ids = tokenizer.encode(
                prompt,
                add_special_tokens=self.prompt_tokenization_add_special_tokens,
            )
        except Exception as exc:
            raise RuntimeError("TensorRT-LLM encoder prompt tokenization failed") from exc
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            )
        ):
            raise RuntimeError("TensorRT-LLM encoder prompt tokenization returned invalid token ids")
        return len(token_ids)

    def _ensure_generation_accepting(self) -> None:
        if self._generation_draining:
            raise GenerationDrainingError("TensorRT-LLM generation is draining")

    @staticmethod
    def _normalize_generate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "stop": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "top_k": None,
            "repetition_penalty": None,
            "min_new_tokens": None,
            "grammar": None,
            "seed": None,
            "logit_bias": None,
            "logprobs": False,
            "top_logprobs": None,
            "n": None,
            "best_of": None,
            "stream": False,
            "lora_path": None,
            "images": None,
        }
        normalized.update(parameters)
        return normalized

    def _validate_generate_parameters(self, parameters: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        normalized = self._normalize_generate_parameters(parameters)
        prompt = normalized.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise GenerationInvalidRequestError("prompt", "prompt must be a non-empty string")
        max_new_tokens = normalized.get("max_new_tokens")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise GenerationInvalidRequestError(
                "max_new_tokens",
                "max_new_tokens must be an integer > 0",
            )
        if normalized.get("grammar") is not None:
            raise GenerationUnsupportedFieldError("grammar")
        if normalized.get("images"):
            raise GenerationUnsupportedFieldError("images")
        if normalized.get("lora_path"):
            raise GenerationUnsupportedFieldError(
                "lora_path",
                "'lora_adapter' is not supported by this generation backend",
            )
        if normalized.get("n") not in (None, 1):
            raise GenerationUnsupportedFieldError("n", "TensorRT-LLM completions support n=1 only")
        if normalized.get("best_of") not in (None, 1):
            raise GenerationUnsupportedFieldError(
                "best_of",
                "TensorRT-LLM completions support best_of=1 only",
            )
        if normalized.get("top_logprobs") is not None and not normalized.get("logprobs", False):
            raise GenerationInvalidRequestError("top_logprobs", "top_logprobs requires logprobs=true")
        min_new_tokens = normalized.get("min_new_tokens")
        if min_new_tokens is not None and min_new_tokens > max_new_tokens:
            raise GenerationInvalidRequestError(
                "min_new_tokens",
                "min_new_tokens must not exceed max_new_tokens",
            )
        prompt_token_count = self._count_prompt_tokens(prompt)
        if prompt_token_count > self._max_input_len:
            raise GenerationInputTooLongError(
                f"prompt token count ({prompt_token_count}) exceeds max_input_len ({self._max_input_len})"
            )
        return normalized, prompt_token_count

    def preflight_generate(
        self,
        parameters: Mapping[str, Any],
        *,
        stream: bool,
    ) -> GenerationPreflightResult:
        """Reject unsupported or oversized requests before shared admission."""
        _ = stream  # TensorRT streams internally for buffered and SSE callers.
        self._ensure_generation_accepting()
        self._check_loaded()
        normalized, prompt_token_count = self._validate_generate_parameters(parameters)
        return GenerationPreflightResult(self, normalized, prompt_token_count)

    def _begin_generation(self) -> None:
        self._ensure_generation_accepting()
        self._active_generations += 1
        if self._generation_idle is not None:
            self._generation_idle.clear()

    def _finish_generation(self) -> None:
        self._active_generations -= 1
        if self._active_generations == 0 and self._generation_idle is not None:
            self._generation_idle.set()

    async def drain_generation(self) -> None:
        """Close admission, await in-flight streams, then close the client."""
        self._generation_draining = True
        if self._active_generations:
            if self._generation_idle is None:
                self._generation_idle = asyncio.Event()
            self._generation_idle.clear()
            await self._generation_idle.wait()
        await self.aclose_client()

    async def _get_or_create_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        if self._http_client_lock is None:
            with self._http_client_init_lock:
                if self._http_client_lock is None:
                    self._http_client_lock = asyncio.Lock()
        async with self._http_client_lock:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=_CONNECT_TIMEOUT_S,
                        read=_READ_TIMEOUT_S,
                        write=_WRITE_TIMEOUT_S,
                        pool=_POOL_TIMEOUT_S,
                    ),
                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                    http2=False,
                )
            return self._http_client

    async def aclose_client(self) -> None:
        pending = self._pending_aclose
        if pending is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(pending), timeout=_CLIENT_CLOSE_TIMEOUT_S)
        client = self._http_client
        if client is None:
            return
        self._http_client = None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.aclose(), timeout=_CLIENT_CLOSE_TIMEOUT_S)

    def _teardown_child(self) -> None:
        self._generation_draining = True
        try:
            _server.terminate(self._process)
        finally:
            self._process = None
            self._tokenizer = None
            _server.release_port(self._port)
            self._port = None
            self._server_url = None
            self._device = None
            if self._output_file is not None:
                with contextlib.suppress(OSError):
                    self._output_file.close()
                with contextlib.suppress(OSError):
                    Path(self._output_file.name).unlink()
                self._output_file = None
            if self._config_path is not None:
                with contextlib.suppress(OSError):
                    self._config_path.unlink()
                self._config_path = None

    def _clear_pending_aclose(self, task: asyncio.Task[None]) -> None:
        if self._pending_aclose is task:
            self._pending_aclose = None

    def unload(self) -> None:
        self._generation_draining = True
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(client.aclose())
                self._pending_aclose = task
                task.add_done_callback(self._clear_pending_aclose)
            else:
                with contextlib.suppress(Exception):
                    asyncio.run(asyncio.wait_for(client.aclose(), timeout=_CLIENT_CLOSE_TIMEOUT_S))
        self._teardown_child()

    def memory_footprint(self) -> int:
        return 0

    def load_required_memory_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        if device_type != "cuda" or device_total_bytes <= 0:
            return None
        return min(int(device_total_bytes * self._kv_cache_fraction) + 1024**3, device_total_bytes)

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
        parameters: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "min_new_tokens": min_new_tokens,
            "grammar": grammar,
            "seed": seed,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "n": n,
            "best_of": best_of,
            "stream": stream,
            "lora_path": lora_path,
            "images": images,
        }
        self._ensure_generation_accepting()
        server_url = self._check_loaded()
        normalized = self._normalize_generate_parameters(parameters)
        preflight_prompt_token_count = consume_generation_preflight(self, normalized)
        if isinstance(preflight_prompt_token_count, int):
            prompt_token_count = preflight_prompt_token_count
        else:
            _, prompt_token_count = self._validate_generate_parameters(parameters)
        self._begin_generation()

        request: dict[str, Any] = {
            "model": self._served_model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
            "n": 1,
        }
        optional = {
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "min_tokens": min_new_tokens,
            "seed": seed,
            "logit_bias": logit_bias,
        }
        request.update({name: value for name, value in optional.items() if value is not None})
        if logprobs:
            request["logprobs"] = top_logprobs if top_logprobs is not None else 1

        completion: AsyncIterator[GenerationChunk] | None = None
        terminal_outcome_selected = False
        try:
            completion = self._stream_completion(
                server_url,
                request,
                logprobs_requested=logprobs,
                prompt_token_count=prompt_token_count,
            )
            async for chunk in completion:
                if chunk.done:
                    terminal_outcome_selected = True
                yield chunk
        finally:
            try:
                if completion is not None:
                    await aclose_with_error_precedence(
                        completion,
                        outcome_selected=terminal_outcome_selected,
                        context="TensorRT-LLM completion iterator",
                    )
            finally:
                self._finish_generation()

    async def _stream_completion(
        self,
        server_url: str,
        request: dict[str, Any],
        *,
        logprobs_requested: bool,
        prompt_token_count: int,
    ) -> AsyncGenerator[GenerationChunk]:
        client = await self._get_or_create_http_client()
        finish_reason: FinishReason | None = None
        usage_completion_tokens: int | None = None
        first_text_emitted = False
        saw_done = False

        async with client.stream("POST", f"{server_url}/v1/completions", json=request) as response:
            if response.status_code != 200:
                await response.aread()
                response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    raise RuntimeError("TensorRT-LLM completion stream emitted a non-SSE line")
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    if finish_reason is None or usage_completion_tokens is None:
                        raise RuntimeError("TensorRT-LLM completion ended without finish reason and usage")
                    saw_done = True
                    yield GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_token_count,
                        completion_tokens=usage_completion_tokens,
                    )
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("TensorRT-LLM completion stream emitted invalid JSON") from exc
                if not isinstance(event, Mapping):
                    raise RuntimeError("TensorRT-LLM completion stream event must be an object")
                if "error" in event:
                    error = event["error"]
                    message = error.get("message") if isinstance(error, Mapping) else error
                    raise RuntimeError(f"TensorRT-LLM completion error: {str(message)[:500]}")

                choices = event.get("choices")
                if not isinstance(choices, list):
                    raise RuntimeError("TensorRT-LLM completion event is missing choices")
                if choices:
                    if len(choices) != 1 or not isinstance(choices[0], Mapping) or choices[0].get("index", 0) != 0:
                        raise RuntimeError("TensorRT-LLM completion emitted unsupported multiple choices")
                    choice = choices[0]
                    text = choice.get("text", "")
                    if not isinstance(text, str):
                        raise RuntimeError("TensorRT-LLM completion text must be a string")
                    event_finish = choice.get("finish_reason")
                    if event_finish is not None:
                        if event_finish not in {"stop", "length"}:
                            raise RuntimeError(f"unsupported TensorRT-LLM finish reason {event_finish!r}")
                        if finish_reason is not None:
                            raise RuntimeError("TensorRT-LLM completion emitted multiple finish reasons")
                        finish_reason = cast("FinishReason", event_finish)
                    if text:
                        if finish_reason is not None and event_finish is None:
                            raise RuntimeError("TensorRT-LLM emitted text after its terminal choice")
                        choice_logprobs = _completion_logprobs(choice.get("logprobs"))
                        if logprobs_requested and choice_logprobs is None:
                            raise RuntimeError("TensorRT-LLM omitted requested completion logprobs")
                        yield GenerationChunk(
                            text_delta=text,
                            is_first=not first_text_emitted,
                            logprobs=choice_logprobs,
                        )
                        first_text_emitted = True

                event_usage = event.get("usage")
                if event_usage is not None:
                    if usage_completion_tokens is not None or not isinstance(event_usage, Mapping):
                        raise RuntimeError("TensorRT-LLM completion emitted invalid duplicate usage")
                    child_prompt_tokens = event_usage.get("prompt_tokens")
                    child_completion_tokens = event_usage.get("completion_tokens")
                    if (
                        isinstance(child_prompt_tokens, bool)
                        or not isinstance(child_prompt_tokens, int)
                        or child_prompt_tokens < 0
                        or isinstance(child_completion_tokens, bool)
                        or not isinstance(child_completion_tokens, int)
                        or child_completion_tokens < 0
                    ):
                        raise RuntimeError("TensorRT-LLM completion usage is invalid")
                    # rc24 reports its internal decoder prefix here for
                    # encoder-decoder models. Validate the wire shape, but only
                    # the local encoder count is prompt-usage authority.
                    usage_completion_tokens = child_completion_tokens

        if not saw_done:
            raise RuntimeError("TensorRT-LLM completion stream ended before [DONE]")
