"""In-process CTranslate2 sequence-to-sequence generation adapter."""

from __future__ import annotations

import asyncio
import importlib
import logging
import math
import threading
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationCapacityError,
    GenerationChunk,
    GenerationDrainingError,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
    GenerationPreflightResult,
    GenerationUnsupportedFieldError,
    consume_generation_preflight,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

CTranslate2ComputeType = Literal[
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]

_COMPUTE_TYPES = frozenset(
    {
        "default",
        "auto",
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    }
)
_AUTOMATIC_COMPUTE_TYPES = frozenset({"default", "auto"})
_EXPECTED_CTRANSLATE2_VERSION = "4.8.1"

logger = logging.getLogger(__name__)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _non_negative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number >= 0")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return result


def _parse_device(device: str) -> tuple[str, int]:
    if device == "cpu":
        return "cpu", 0
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:"):
        raw_index = device.removeprefix("cuda:")
        if raw_index.isdigit():
            return "cuda", int(raw_index)
    raise ValueError("CTranslate2 requires device='cpu', 'cuda', or 'cuda:<index>'")


@dataclass(frozen=True, slots=True)
class _DecodeOptions:
    max_new_tokens: int
    min_new_tokens: int
    repetition_penalty: float
    seed: int


@dataclass(frozen=True, slots=True)
class _PreparedGeneration:
    source_tokens: tuple[str, ...]
    prompt_tokens: int
    options: _DecodeOptions


@dataclass(slots=True)
class _Request:
    prepared: _PreparedGeneration
    loop: asyncio.AbstractEventLoop
    enqueued_at: float
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    final_text: str | None = None
    completion_tokens: int | None = None
    finish_reason: Literal["stop", "length"] | None = None
    error: BaseException | None = None
    done: bool = False

    @property
    def cost(self) -> int:
        return len(self.prepared.source_tokens)

    def complete(self, text: str, completion_tokens: int, finish_reason: Literal["stop", "length"]) -> None:
        if self.cancelled.is_set() or self.done:
            return
        self.final_text = text
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.done = True
        self.wake.set()

    def fail(self, error: BaseException) -> None:
        if self.cancelled.is_set() or self.done:
            return
        self.error = error
        self.done = True
        self.wake.set()


class CTranslate2GenerationAdapter(GenerationAdapter):
    """Serve a locally converted sequence-to-sequence artifact in process."""

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("tokens",),
        unload_fields=("_translator", "_tokenizer", "_ctranslate2"),
    )
    requires_main_thread = False
    context_length_accounting = "independent"
    prompt_tokenization_add_special_tokens = True

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        ct2_compute_type: CTranslate2ComputeType,
        max_seq_length: int = 4096,
        compute_precision: ComputePrecision = "float16",
        native_batch_max_size: int = 8,
        native_batch_max_tokens: int = 8192,
        native_batch_wait_ms: float = 2.0,
        max_pending_requests: int = 64,
    ) -> None:
        artifact = Path(artifact_path)
        if not str(artifact_path):
            raise ValueError("artifact_path must be a non-empty local path")
        if not isinstance(ct2_compute_type, str) or ct2_compute_type not in _COMPUTE_TYPES:
            choices = ", ".join(sorted(_COMPUTE_TYPES))
            raise ValueError(f"ct2_compute_type must be one of: {choices}")
        self._artifact_path = artifact
        self._ct2_compute_type = ct2_compute_type
        self._max_seq_length = _positive_int("max_seq_length", max_seq_length)
        self._compute_precision = compute_precision
        self._native_batch_max_size = _positive_int("native_batch_max_size", native_batch_max_size)
        self._native_batch_max_tokens = _positive_int("native_batch_max_tokens", native_batch_max_tokens)
        self._native_batch_wait_s = _non_negative_float("native_batch_wait_ms", native_batch_wait_ms) / 1000
        self._max_pending_requests = _positive_int("max_pending_requests", max_pending_requests)

        self._translator: Any | None = None
        self._tokenizer: Any | None = None
        self._ctranslate2: Any | None = None
        self._device: str | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._pending: deque[_Request] = deque()
        self._pending_wake: asyncio.Event | None = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._active_batch: tuple[_Request, ...] = ()
        self._draining = False
        self._active_batches = 0
        self._idle: asyncio.Event | None = None
        self._force_stop = threading.Event()
        self._scheduler_stopped = threading.Event()
        self._scheduler_stopped.set()

    def load(self, device: str) -> None:
        if not self._scheduler_stopped.is_set():
            raise RuntimeError("CTranslate2 generation scheduler is still stopping")
        device_type, device_index = _parse_device(device)
        if not self._artifact_path.is_dir():
            raise ValueError(f"artifact_path is not a local directory: {self._artifact_path}")

        ctranslate2 = importlib.import_module("ctranslate2")
        version = getattr(ctranslate2, "__version__", None)
        if version != _EXPECTED_CTRANSLATE2_VERSION:
            raise RuntimeError(f"CTranslate2 {_EXPECTED_CTRANSLATE2_VERSION} is required, found {version or 'unknown'}")
        supported = set(ctranslate2.get_supported_compute_types(device_type, device_index))
        if self._ct2_compute_type not in _AUTOMATIC_COMPUTE_TYPES and self._ct2_compute_type not in supported:
            available = ", ".join(sorted(supported)) or "none"
            raise ValueError(
                f"ct2_compute_type={self._ct2_compute_type!r} is not supported on {device}; supported: {available}"
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self._artifact_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            translator = ctranslate2.Translator(
                str(self._artifact_path),
                device=device_type,
                device_index=device_index,
                compute_type=self._ct2_compute_type,
                inter_threads=1,
                max_queued_batches=1,
            )
        except BaseException:
            self._translator = None
            self._tokenizer = None
            self._ctranslate2 = None
            self._device = None
            raise

        self._ctranslate2 = ctranslate2
        self._tokenizer = tokenizer
        self._translator = translator
        self._device = device
        self._force_stop.clear()
        self._draining = False

    def _check_loaded(self) -> tuple[Any, Any, Any]:
        if self._translator is None or self._tokenizer is None or self._ctranslate2 is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._translator, self._tokenizer, self._ctranslate2

    def _tokenize_prompt(self, prompt: str) -> tuple[tuple[str, ...], int]:
        _, tokenizer, _ = self._check_loaded()
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=self.prompt_tokenization_add_special_tokens)
            source_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        except Exception as exc:
            raise RuntimeError("CTranslate2 prompt tokenization failed") from exc
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            )
            or not isinstance(source_tokens, list)
            or len(source_tokens) != len(token_ids)
            or any(not isinstance(token, str) or not token for token in source_tokens)
        ):
            raise RuntimeError("CTranslate2 prompt tokenization returned invalid tokens")
        return tuple(source_tokens), len(token_ids)

    def _decode_tokens(self, tokens: Sequence[str]) -> str:
        _, tokenizer, _ = self._check_loaded()
        try:
            token_ids = tokenizer.convert_tokens_to_ids(list(tokens))
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )
        except Exception as exc:
            raise RuntimeError("CTranslate2 output detokenization failed") from exc
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != len(tokens)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            )
            or not isinstance(text, str)
        ):
            raise RuntimeError("CTranslate2 output detokenization returned invalid data")
        return text

    def _end_tokens(self) -> frozenset[str]:
        _, tokenizer, _ = self._check_loaded()
        token_ids = getattr(tokenizer, "eos_token_id", None)
        if isinstance(token_ids, int) and not isinstance(token_ids, bool):
            token_ids = [token_ids]
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            )
        ):
            raise RuntimeError("CTranslate2 tokenizer has invalid end token IDs")
        try:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        except Exception as exc:
            raise RuntimeError("CTranslate2 end token conversion failed") from exc
        if not isinstance(tokens, list):
            raise RuntimeError("CTranslate2 tokenizer returned invalid end tokens")
        end_tokens = [token for token in tokens if isinstance(token, str) and token]
        if len(end_tokens) != len(token_ids) or len(end_tokens) != len(tokens):
            raise RuntimeError("CTranslate2 tokenizer returned invalid end tokens")
        return frozenset(end_tokens)

    @staticmethod
    def _normalize_generate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "temperature": 0.0,
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

    def _prepare_generation(self, parameters: Mapping[str, Any]) -> _PreparedGeneration:
        prompt = parameters.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise GenerationInvalidRequestError("prompt", "prompt must be a non-empty string")
        max_new_tokens = parameters.get("max_new_tokens")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise GenerationInvalidRequestError("max_new_tokens", "max_new_tokens must be an integer > 0")

        if parameters.get("stop") not in (None, []):
            raise GenerationUnsupportedFieldError("stop")
        for name in ("frequency_penalty", "presence_penalty", "grammar", "logit_bias"):
            if parameters.get(name) is not None:
                raise GenerationUnsupportedFieldError(name)
        if parameters.get("images") not in (None, []):
            raise GenerationUnsupportedFieldError("images")
        if parameters.get("lora_path") is not None:
            raise GenerationUnsupportedFieldError(
                "lora_path",
                "'lora_adapter' is not supported by this generation backend",
            )
        if parameters.get("logprobs"):
            raise GenerationUnsupportedFieldError("logprobs")
        if parameters.get("top_logprobs") not in (None, 0):
            raise GenerationUnsupportedFieldError("top_logprobs")
        n = parameters.get("n")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n != 1):
            raise GenerationUnsupportedFieldError("n")
        best_of = parameters.get("best_of")
        if best_of is not None and (isinstance(best_of, bool) or not isinstance(best_of, int) or best_of != 1):
            raise GenerationUnsupportedFieldError("best_of")

        temperature = parameters.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, int | float) or float(temperature) != 0.0:
            raise GenerationInvalidRequestError("temperature", "CTranslate2 greedy generation requires temperature=0")
        top_p = parameters.get("top_p", 1.0)
        if isinstance(top_p, bool) or not isinstance(top_p, int | float) or float(top_p) != 1.0:
            raise GenerationInvalidRequestError("top_p", "CTranslate2 greedy generation requires top_p=1")
        top_k = parameters.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 1):
            raise GenerationInvalidRequestError("top_k", "CTranslate2 greedy generation requires top_k=1")
        seed = parameters.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed != 0):
            raise GenerationInvalidRequestError("seed", "CTranslate2 greedy generation requires seed=0")

        min_new_tokens = parameters.get("min_new_tokens")
        if min_new_tokens is None:
            min_new_tokens = 1
        if (
            isinstance(min_new_tokens, bool)
            or not isinstance(min_new_tokens, int)
            or min_new_tokens < 0
            or min_new_tokens > max_new_tokens
        ):
            raise GenerationInvalidRequestError(
                "min_new_tokens",
                "min_new_tokens must be an integer between 0 and max_new_tokens",
            )
        repetition_penalty = parameters.get("repetition_penalty")
        if repetition_penalty is None:
            repetition_penalty = 1.0
        if (
            isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, int | float)
            or not math.isfinite(float(repetition_penalty))
            or not 0.0 < float(repetition_penalty) <= 2.0
        ):
            raise GenerationInvalidRequestError(
                "repetition_penalty",
                "repetition_penalty must be a finite number in (0, 2]",
            )

        source_tokens, prompt_tokens = self._tokenize_prompt(prompt)
        if prompt_tokens > self._max_seq_length:
            raise GenerationInputTooLongError(
                f"prompt token count {prompt_tokens} exceeds max_seq_length={self._max_seq_length}"
            )
        if prompt_tokens > self._native_batch_max_tokens:
            raise GenerationInputTooLongError(
                f"prompt token count {prompt_tokens} exceeds native_batch_max_tokens={self._native_batch_max_tokens}"
            )
        return _PreparedGeneration(
            source_tokens=source_tokens,
            prompt_tokens=prompt_tokens,
            options=_DecodeOptions(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=float(repetition_penalty),
                seed=0,
            ),
        )

    def preflight_generate(
        self,
        parameters: Mapping[str, Any],
        *,
        stream: bool,
    ) -> GenerationPreflightResult:
        del stream
        self._check_loaded()
        if self._draining:
            raise GenerationDrainingError("CTranslate2 generation is draining")
        self._discard_cancelled_pending()
        if len(self._pending) >= self._max_pending_requests:
            raise GenerationCapacityError("CTranslate2 generation scheduler is at capacity")
        normalized = self._normalize_generate_parameters(parameters)
        prepared = self._prepare_generation(normalized)
        return GenerationPreflightResult(self, normalized, prepared)

    def _bind_owner_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            self._pending_wake = asyncio.Event()
            self._idle = asyncio.Event()
            self._idle.set()
        elif self._owner_loop is not loop:
            raise RuntimeError("CTranslate2 generation scheduler is bound to another event loop")
        return loop

    def _enqueue(self, prepared: _PreparedGeneration) -> _Request:
        loop = self._bind_owner_loop()
        if self._draining:
            raise GenerationDrainingError("CTranslate2 generation is draining")
        self._discard_cancelled_pending()
        if len(self._pending) >= self._max_pending_requests:
            raise GenerationCapacityError("CTranslate2 generation scheduler is at capacity")
        request = _Request(prepared=prepared, loop=loop, enqueued_at=loop.time())
        self._pending.append(request)
        if self._idle is not None:
            self._idle.clear()
        if self._pending_wake is not None:
            self._pending_wake.set()
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_stopped.clear()
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        return request

    def _discard_cancelled_pending(self) -> None:
        if self._pending:
            self._pending = deque(request for request in self._pending if not request.cancelled.is_set())

    def _take_batch(self) -> list[_Request]:
        while self._pending and self._pending[0].cancelled.is_set():
            self._pending.popleft()
        if not self._pending:
            return []
        first = self._pending.popleft()
        options = first.prepared.options
        batch = [first]
        cost = first.cost
        deferred: deque[_Request] = deque()
        while self._pending:
            request = self._pending.popleft()
            if request.cancelled.is_set():
                continue
            compatible = request.prepared.options == options
            fits = len(batch) < self._native_batch_max_size and cost + request.cost <= self._native_batch_max_tokens
            if compatible and fits:
                batch.append(request)
                cost += request.cost
            else:
                deferred.append(request)
        self._pending = deferred
        return batch

    async def _scheduler_loop(self) -> None:
        try:
            while True:
                if not self._pending:
                    if self._draining:
                        return
                    if self._pending_wake is None:
                        return
                    self._pending_wake.clear()
                    if not self._pending:
                        await self._pending_wake.wait()
                    continue

                deadline = self._pending[0].enqueued_at + self._native_batch_wait_s
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining > 0 and len(self._pending) < self._native_batch_max_size:
                    if self._pending_wake is not None:
                        self._pending_wake.clear()
                        try:
                            await asyncio.wait_for(self._pending_wake.wait(), timeout=remaining)
                        except TimeoutError:
                            # Reaching the batching deadline is the expected path when no peer request arrives.
                            pass
                        continue

                batch = self._take_batch()
                if not batch:
                    continue
                self._active_batch = tuple(batch)
                self._active_batches += 1
                try:
                    await asyncio.to_thread(self._execute_native_batch, batch)
                except Exception as exc:  # noqa: BLE001 - isolate a failed native batch from the scheduler.
                    for request in batch:
                        request.fail(exc)
                finally:
                    self._active_batches -= 1
                    self._active_batch = ()
        finally:
            if self._draining and not self._pending and self._active_batches == 0 and self._idle is not None:
                self._idle.set()
            self._scheduler_stopped.set()

    def _execute_native_batch(self, batch: list[_Request]) -> None:
        translator, _, ctranslate2 = self._check_loaded()
        end_tokens = self._end_tokens()
        options = batch[0].prepared.options
        ctranslate2.set_random_seed(options.seed)

        def callback(step: Any) -> bool:
            return self._force_stop.is_set() or batch[step.batch_id].cancelled.is_set()

        results = translator.translate_batch(
            [list(request.prepared.source_tokens) for request in batch],
            max_batch_size=self._native_batch_max_tokens,
            batch_type="tokens",
            beam_size=1,
            num_hypotheses=1,
            max_input_length=0,
            max_decoding_length=options.max_new_tokens,
            min_decoding_length=options.min_new_tokens,
            sampling_topk=1,
            sampling_topp=1.0,
            sampling_temperature=1.0,
            repetition_penalty=options.repetition_penalty,
            return_end_token=True,
            callback=callback,
        )
        if not isinstance(results, list) or len(results) != len(batch):
            raise RuntimeError("CTranslate2 returned an invalid native batch result")
        for request, result in zip(batch, results, strict=True):
            if self._force_stop.is_set() or request.cancelled.is_set():
                continue
            hypotheses = getattr(result, "hypotheses", None)
            if (
                not isinstance(hypotheses, list)
                or len(hypotheses) != 1
                or not isinstance(hypotheses[0], list)
                or any(not isinstance(token, str) for token in hypotheses[0])
            ):
                request.loop.call_soon_threadsafe(
                    request.fail,
                    RuntimeError("CTranslate2 returned an invalid translation hypothesis"),
                )
                continue
            result_tokens = hypotheses[0]
            ended = bool(result_tokens and result_tokens[-1] in end_tokens)
            output_tokens = result_tokens[:-1] if ended else result_tokens
            text = self._decode_tokens(output_tokens)
            completion_tokens = len(output_tokens)
            finish_reason: Literal["stop", "length"] = "stop" if ended else "length"
            request.loop.call_soon_threadsafe(request.complete, text, completion_tokens, finish_reason)

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
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
        normalized = self._normalize_generate_parameters(parameters)
        prepared = consume_generation_preflight(self, normalized)
        if not isinstance(prepared, _PreparedGeneration):
            prepared = self._prepare_generation(normalized)
        request = self._enqueue(prepared)
        terminal_yielded = False
        try:
            while True:
                await request.wake.wait()
                request.wake.clear()
                if request.error is not None:
                    raise request.error
                if not request.done:
                    continue
                if request.final_text is None or request.completion_tokens is None or request.finish_reason is None:
                    raise RuntimeError("CTranslate2 generation completed without a terminal result")
                if request.final_text:
                    yield GenerationChunk(
                        text_delta=request.final_text,
                        is_first=True,
                    )
                terminal_yielded = True
                yield GenerationChunk(
                    text_delta="",
                    done=True,
                    finish_reason=request.finish_reason,
                    prompt_tokens=prepared.prompt_tokens,
                    completion_tokens=request.completion_tokens,
                )
                return
        finally:
            if not terminal_yielded:
                request.cancelled.set()
                request.wake.set()

    async def drain_generation(self) -> None:
        self._draining = True
        if self._owner_loop is None:
            return
        if self._owner_loop is not asyncio.get_running_loop():
            raise RuntimeError("CTranslate2 generation drain must run on the scheduler event loop")
        draining_error = GenerationDrainingError("CTranslate2 generation is draining")
        while self._pending:
            self._pending.popleft().fail(draining_error)
        if self._pending_wake is not None:
            self._pending_wake.set()
        if not self._pending and self._active_batches == 0:
            if self._idle is not None:
                self._idle.set()
        if self._idle is not None:
            await self._idle.wait()
        task = self._scheduler_task
        if task is not None:
            await task

    def _fail_undrained_generation(self) -> None:
        error = GenerationDrainingError("CTranslate2 generation was cancelled during unload")
        while self._pending:
            self._pending.popleft().fail(error)
        for request in self._active_batch:
            request.fail(error)
            request.cancelled.set()
        if self._pending_wake is not None:
            self._pending_wake.set()

    def unload(self) -> None:
        task = self._scheduler_task
        undrained = bool(self._pending) or self._active_batches > 0 or (task is not None and not task.done())
        if undrained:
            owner_loop = self._owner_loop
            if owner_loop is None or not owner_loop.is_running():
                raise RuntimeError("CTranslate2 generation cannot force-drain without its scheduler event loop")
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is owner_loop:
                raise RuntimeError("CTranslate2 generation unload must run off its scheduler event loop")
            logger.warning(
                "CTranslate2 generation did not drain before unload; cancelling %d pending and %d active requests",
                len(self._pending),
                len(self._active_batch),
            )
            self._draining = True
            self._force_stop.set()
            owner_loop.call_soon_threadsafe(self._fail_undrained_generation)
            self._scheduler_stopped.wait()
        translator = self._translator
        if translator is not None:
            translator.unload_model()
        self._translator = None
        self._tokenizer = None
        self._ctranslate2 = None
        self._device = None
        self._owner_loop = None
        self._pending_wake = None
        self._scheduler_task = None
        self._active_batch = ()
        self._active_batches = 0
        self._idle = None
