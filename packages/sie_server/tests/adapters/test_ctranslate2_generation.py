from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sie_server.adapters._generation_base import (
    GenerationCapacityError,
    GenerationDrainingError,
    GenerationError,
    GenerationInvalidRequestError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters.ctranslate2.generation import CTranslate2GenerationAdapter


class _FakeTokenizer:
    def __init__(self, *, clean_up_tokenization_spaces: bool = False) -> None:
        self.encode_calls = 0
        self.decode_cleanup_calls: list[bool | None] = []
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self._tokens = ["<s>", "</s>", "hello", "world", "bonjour", " monde", " !", "!"]
        self.eos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls += 1
        ids = [self._tokens.index(token) for token in text.split()]
        return [0, *ids, 1] if add_special_tokens else ids

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._tokens[token_id] for token_id in token_ids]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._tokens.index(token) for token in tokens]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> str:
        assert skip_special_tokens is True
        self.decode_cleanup_calls.append(clean_up_tokenization_spaces)
        clean_up = (
            self.clean_up_tokenization_spaces if clean_up_tokenization_spaces is None else clean_up_tokenization_spaces
        )
        text = "".join(self._tokens[token_id] for token_id in token_ids if token_id not in (0, 1))
        return text.replace(" !", "!") if clean_up else text


@dataclass
class _FakeResult:
    hypotheses: list[list[str]]


class _FakeTranslator:
    def __init__(self, *, block: threading.Event | None = None, append_end_token: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.block = block
        self.append_end_token = append_end_token
        self.started = threading.Event()
        self.callback_stops: list[bool] = []
        self.unload_calls = 0

    def translate_batch(self, source: list[list[str]], **kwargs: Any) -> list[_FakeResult]:
        self.calls.append({"source": source, **kwargs})
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=2)
        max_length = kwargs["max_decoding_length"]
        outputs = ["bonjour", " monde"][:max_length]
        callback = kwargs["callback"]
        for batch_id in range(len(source)):
            for token in outputs:
                should_stop = callback(SimpleNamespace(batch_id=batch_id, hypothesis_id=0, token=token))
                self.callback_stops.append(should_stop)
                if should_stop:
                    break
        result_tokens = [*outputs, "</s>"] if self.append_end_token else outputs
        return [_FakeResult(hypotheses=[list(result_tokens)]) for _ in source]

    def unload_model(self) -> None:
        self.unload_calls += 1


class _FakeCTranslate2:
    __version__ = "4.8.1"

    def __init__(self, translator: _FakeTranslator | None = None) -> None:
        self.translator = translator or _FakeTranslator()
        self.translator_kwargs: dict[str, Any] | None = None
        self.seeds: list[int] = []

    @staticmethod
    def get_supported_compute_types(device: str, device_index: int) -> set[str]:
        assert device_index >= 0
        return {"float16", "int8_float16"} if device == "cuda" else {"float32", "int8_float32"}

    def Translator(self, _path: str, **kwargs: Any) -> _FakeTranslator:  # noqa: N802
        self.translator_kwargs = kwargs
        return self.translator

    def set_random_seed(self, seed: int) -> None:
        self.seeds.append(seed)


def _adapter(
    tmp_path: Path,
    *,
    translator: _FakeTranslator | None = None,
    native_batch_max_size: int = 8,
    native_batch_max_tokens: int = 64,
    native_batch_wait_ms: float = 5,
    max_pending_requests: int = 64,
) -> tuple[CTranslate2GenerationAdapter, _FakeTranslator, _FakeCTranslate2, _FakeTokenizer]:
    fake_translator = translator or _FakeTranslator()
    fake_module = _FakeCTranslate2(fake_translator)
    tokenizer = _FakeTokenizer()
    adapter = CTranslate2GenerationAdapter(
        artifact_path=tmp_path,
        ct2_compute_type="float16",
        max_seq_length=32,
        native_batch_max_size=native_batch_max_size,
        native_batch_max_tokens=native_batch_max_tokens,
        native_batch_wait_ms=native_batch_wait_ms,
        max_pending_requests=max_pending_requests,
    )
    adapter._translator = fake_translator
    adapter._tokenizer = tokenizer
    adapter._ctranslate2 = fake_module
    adapter._device = "cuda:0"
    return adapter, fake_translator, fake_module, tokenizer


async def _collect(iterator: Any) -> list[Any]:
    return [chunk async for chunk in iterator]


def test_context_length_accounting_is_independent() -> None:
    assert CTranslate2GenerationAdapter.context_length_accounting == "independent"


def test_load_uses_verified_local_artifact_and_validates_compute_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = CTranslate2GenerationAdapter(
        artifact_path=tmp_path,
        ct2_compute_type="int8_float16",
    )
    module = _FakeCTranslate2()
    tokenizer = _FakeTokenizer()
    tokenizer_load: dict[str, Any] = {}

    monkeypatch.setattr(
        "sie_server.adapters.ctranslate2.generation.importlib.import_module",
        lambda name: module if name == "ctranslate2" else pytest.fail(name),
    )

    def from_pretrained(path: str, **kwargs: Any) -> _FakeTokenizer:
        tokenizer_load.update({"path": path, **kwargs})
        return tokenizer

    monkeypatch.setattr(
        "sie_server.adapters.ctranslate2.generation.AutoTokenizer.from_pretrained",
        from_pretrained,
    )

    adapter.load("cuda:2")

    assert tokenizer_load == {
        "path": str(tmp_path),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert module.translator_kwargs == {
        "device": "cuda",
        "device_index": 2,
        "compute_type": "int8_float16",
        "inter_threads": 1,
        "max_queued_batches": 1,
    }


def test_load_rejects_unsupported_compute_type_before_translator_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = CTranslate2GenerationAdapter(
        artifact_path=tmp_path,
        ct2_compute_type="int8_bfloat16",
    )
    module = _FakeCTranslate2()
    monkeypatch.setattr(
        "sie_server.adapters.ctranslate2.generation.importlib.import_module",
        lambda _name: module,
    )

    with pytest.raises(ValueError, match="not supported on cuda:0"):
        adapter.load("cuda:0")

    assert module.translator_kwargs is None


def test_preflight_evidence_streams_and_reports_usage_with_exact_greedy_kwargs(tmp_path: Path) -> None:
    adapter, translator, module, tokenizer = _adapter(tmp_path)
    parameters = {
        "prompt": "hello world",
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    preflight = adapter.preflight_generate(parameters, stream=True)

    chunks = asyncio.run(_collect(adapter.generate_with_preflight(parameters, preflight)))

    assert tokenizer.encode_calls == 1
    assert "".join(chunk.text_delta for chunk in chunks) == "bonjour monde"
    assert chunks[0].is_first is True
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "length"
    assert (chunks[-1].prompt_tokens, chunks[-1].completion_tokens) == (4, 2)
    assert module.seeds == [0]
    call = translator.calls[0]
    assert call["source"] == [["<s>", "hello", "world", "</s>"]]
    assert {
        key: call[key]
        for key in (
            "max_batch_size",
            "batch_type",
            "beam_size",
            "num_hypotheses",
            "max_input_length",
            "max_decoding_length",
            "min_decoding_length",
            "sampling_topk",
            "sampling_topp",
            "sampling_temperature",
            "repetition_penalty",
            "return_end_token",
        )
    } == {
        "max_batch_size": 64,
        "batch_type": "tokens",
        "beam_size": 1,
        "num_hypotheses": 1,
        "max_input_length": 0,
        "max_decoding_length": 2,
        "min_decoding_length": 1,
        "sampling_topk": 1,
        "sampling_topp": 1.0,
        "sampling_temperature": 1.0,
        "repetition_penalty": 1.0,
        "return_end_token": True,
    }


@pytest.mark.parametrize(
    ("clean_up_tokenization_spaces", "expected"),
    [(False, "bonjour monde !"), (True, "bonjour monde!")],
)
def test_output_detokenization_honors_tokenizer_cleanup_config(
    tmp_path: Path,
    clean_up_tokenization_spaces: bool,
    expected: str,
) -> None:
    adapter, _, _, tokenizer = _adapter(tmp_path)
    tokenizer.clean_up_tokenization_spaces = clean_up_tokenization_spaces

    assert adapter._decode_tokens(["bonjour", " monde", " !"]) == expected
    assert tokenizer.decode_cleanup_calls == [None]


def test_natural_end_token_at_max_length_reports_stop_without_counting_or_decoding_it(tmp_path: Path) -> None:
    adapter, _, _, _ = _adapter(tmp_path, translator=_FakeTranslator(append_end_token=True))

    chunks = asyncio.run(_collect(adapter.generate("hello", max_new_tokens=2, seed=0)))

    assert "".join(chunk.text_delta for chunk in chunks) == "bonjour monde"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].completion_tokens == 2


def test_scheduler_batches_only_exact_compatible_requests(tmp_path: Path) -> None:
    async def run() -> tuple[list[Any], list[Any], list[Any]]:
        adapter, translator, _, _ = _adapter(tmp_path, native_batch_wait_ms=20)
        first, second = await asyncio.gather(
            _collect(adapter.generate("hello", max_new_tokens=2, seed=0)),
            _collect(adapter.generate("world", max_new_tokens=2, seed=0)),
        )
        third = await _collect(adapter.generate("hello", max_new_tokens=1, seed=0))
        await adapter.drain_generation()
        assert [len(call["source"]) for call in translator.calls] == [2, 1]
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first[-1].completion_tokens == second[-1].completion_tokens == 2
    assert third[-1].completion_tokens == 1


def test_scheduler_enforces_native_source_token_bound(tmp_path: Path) -> None:
    async def run() -> None:
        adapter, translator, _, _ = _adapter(
            tmp_path,
            native_batch_max_tokens=5,
            native_batch_wait_ms=20,
        )
        await asyncio.gather(
            _collect(adapter.generate("hello", max_new_tokens=2, seed=0)),
            _collect(adapter.generate("world", max_new_tokens=2, seed=0)),
        )
        await adapter.drain_generation()
        assert [len(call["source"]) for call in translator.calls] == [1, 1]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "error_type", "param"),
    [
        ({"temperature": 0.5}, GenerationInvalidRequestError, "temperature"),
        ({"top_p": 0.9}, GenerationInvalidRequestError, "top_p"),
        ({"seed": 1}, GenerationInvalidRequestError, "seed"),
        ({"stop": ["!"]}, GenerationUnsupportedFieldError, "stop"),
        ({"frequency_penalty": 0.0}, GenerationUnsupportedFieldError, "frequency_penalty"),
        ({"n": 2}, GenerationUnsupportedFieldError, "n"),
        ({"n": True}, GenerationUnsupportedFieldError, "n"),
        ({"best_of": True}, GenerationUnsupportedFieldError, "best_of"),
        ({"top_k": True}, GenerationInvalidRequestError, "top_k"),
        ({"seed": False}, GenerationInvalidRequestError, "seed"),
        ({"logprobs": True}, GenerationUnsupportedFieldError, "logprobs"),
    ],
)
def test_preflight_rejects_non_greedy_or_unsupported_controls(
    tmp_path: Path,
    kwargs: dict[str, Any],
    error_type: type[Exception],
    param: str,
) -> None:
    adapter, translator, _, _ = _adapter(tmp_path)
    parameters = {
        "prompt": "hello",
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        **kwargs,
    }

    with pytest.raises(error_type) as raised:
        adapter.preflight_generate(parameters, stream=False)

    assert isinstance(raised.value, GenerationError)
    assert raised.value.param == param
    assert translator.calls == []


def test_capacity_is_bounded_while_native_batch_is_running(tmp_path: Path) -> None:
    async def run() -> None:
        release = threading.Event()
        adapter, translator, _, _ = _adapter(
            tmp_path,
            translator=_FakeTranslator(block=release),
            native_batch_max_size=1,
            native_batch_wait_ms=0,
            max_pending_requests=1,
        )
        active = asyncio.create_task(_collect(adapter.generate("hello", max_new_tokens=2, seed=0)))
        assert await asyncio.to_thread(translator.started.wait, 1)
        queued = asyncio.create_task(_collect(adapter.generate("world", max_new_tokens=2, seed=0)))
        await asyncio.sleep(0)
        assert len(adapter._pending) == 1
        with pytest.raises(GenerationCapacityError):
            adapter.preflight_generate(
                {"prompt": "hello", "max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0},
                stream=False,
            )
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        chunks = await active
        assert chunks[-1].done is True
        await adapter.drain_generation()

    asyncio.run(run())


def test_cancelling_running_member_discards_it_without_stopping_neighbor(tmp_path: Path) -> None:
    async def run() -> None:
        release = threading.Event()
        translator = _FakeTranslator(block=release)
        adapter, _, _, _ = _adapter(
            tmp_path,
            translator=translator,
            native_batch_max_size=2,
            native_batch_wait_ms=20,
        )
        cancelled, neighbor = (
            asyncio.create_task(_collect(adapter.generate("hello", max_new_tokens=2, seed=0))),
            asyncio.create_task(_collect(adapter.generate("world", max_new_tokens=2, seed=0))),
        )
        assert await asyncio.to_thread(translator.started.wait, 1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        chunks = await neighbor
        await adapter.drain_generation()
        assert chunks[-1].done is True
        assert translator.callback_stops
        assert any(translator.callback_stops)
        assert not all(translator.callback_stops)

    asyncio.run(run())


def test_cancelling_all_running_members_stops_native_batch(tmp_path: Path) -> None:
    async def run() -> None:
        release = threading.Event()
        translator = _FakeTranslator(block=release)
        adapter, _, _, _ = _adapter(
            tmp_path,
            translator=translator,
            native_batch_max_size=1,
            native_batch_wait_ms=0,
        )
        request = asyncio.create_task(_collect(adapter.generate("hello", max_new_tokens=2, seed=0)))
        assert await asyncio.to_thread(translator.started.wait, 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        await adapter.drain_generation()
        assert translator.callback_stops
        assert translator.callback_stops[0] is True

    asyncio.run(run())


def test_drain_fails_queued_request_and_is_idempotent(tmp_path: Path) -> None:
    async def run() -> None:
        release = threading.Event()
        translator = _FakeTranslator(block=release)
        adapter, _, _, _ = _adapter(
            tmp_path,
            translator=translator,
            native_batch_max_size=1,
            native_batch_wait_ms=0,
        )
        active = asyncio.create_task(_collect(adapter.generate("hello", max_new_tokens=2, seed=0)))
        assert await asyncio.to_thread(translator.started.wait, 1)
        queued = asyncio.create_task(_collect(adapter.generate("world", max_new_tokens=2, seed=0)))
        await asyncio.sleep(0)
        assert len(adapter._pending) == 1
        drain = asyncio.create_task(adapter.drain_generation())
        with pytest.raises(GenerationDrainingError):
            await queued
        release.set()
        chunks = await active
        assert chunks[-1].done is True
        await drain
        await adapter.drain_generation()
        adapter.unload()
        adapter.unload()
        assert translator.unload_calls == 1

    asyncio.run(run())


def test_unload_force_stops_undrained_native_batch_before_releasing_model(tmp_path: Path) -> None:
    async def run() -> None:
        release = threading.Event()
        translator = _FakeTranslator(block=release)
        adapter, _, _, _ = _adapter(
            tmp_path,
            translator=translator,
            native_batch_max_size=1,
            native_batch_wait_ms=0,
        )
        active = asyncio.create_task(_collect(adapter.generate("hello", max_new_tokens=2, seed=0)))
        assert await asyncio.to_thread(translator.started.wait, 1)
        queued = asyncio.create_task(_collect(adapter.generate("world", max_new_tokens=2, seed=0)))
        await asyncio.sleep(0)
        assert len(adapter._pending) == 1

        unloading = asyncio.create_task(asyncio.to_thread(adapter.unload))
        assert await asyncio.to_thread(adapter._force_stop.wait, 1)
        assert not unloading.done()
        assert translator.unload_calls == 0

        release.set()
        with pytest.raises(GenerationDrainingError):
            await active
        with pytest.raises(GenerationDrainingError):
            await queued
        await unloading

        assert translator.unload_calls == 1
        assert adapter._translator is None
        assert adapter._scheduler_task is None
        assert adapter._active_batch == ()
        assert adapter._active_batches == 0
        assert adapter._scheduler_stopped.is_set()

    asyncio.run(run())
