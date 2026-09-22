"""Tests for SGLangGenerationAdapter (streaming async iterator).

Mocks the subprocess + httpx layer to exercise:

- ``load()`` launches ``sglang.launch_server`` **without** ``--is-embedding``.
- ``generate()`` POSTs ``stream: true`` to ``/generate`` and yields chunks
  parsed from SSE ``data:`` lines (cumulative ``text`` diffed into deltas).
- Caller-cancellation (``aclose()``) issues a best-effort ``/abort_request``.
- ``unload()`` terminates the subprocess.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sie_server.adapters._generation_base import (
    GenerationChunk,
    GenerationError,
    GenerationInvalidRequestError,
    collect_generation,
    suppress_thinking_blocks,
)
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.sglang import _server
from sie_server.adapters.sglang import generation as generation_module
from sie_server.adapters.sglang.cuda13 import SGLangCuda13Adapter, SGLangStrictThinkingAdapter
from sie_server.adapters.sglang.gemma import SGLangGemmaAdapter
from sie_server.adapters.sglang.generation import (
    SGLangGenerationAdapter,
    _chunk_from_sglang_event,
    _encode_image_data,
    _encode_video_data,
    _mamba_strategy_value,
    _p_unsafe_from_verdict_logprobs,
    _parse_sglang_generate_response,
    _raise_for_sglang_event_error,
    _raise_for_sglang_http_error,
    _thresholded_verdict,
)
from sie_server.types.grammar import OUTLINES_JSON_SCHEMA_TYPE_MESSAGE, GrammarSpec
from sie_server.types.inputs import InvalidMediaError


@pytest.fixture
def adapter():
    return SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3-4B-Instruct",
        max_seq_length=32768,
        mem_fraction_static=0.85,
        served_model_name="Qwen/Qwen3-4B-Instruct",
    )


def test_capabilities_declare_tokens(adapter) -> None:
    caps = adapter.capabilities
    assert caps.inputs == ["text"]
    assert caps.outputs == ["tokens"]


def test_load_contract_flags() -> None:
    assert SGLangGenerationAdapter.requires_main_thread is False
    assert SGLangGenerationAdapter.manages_own_load_timeout is True


def test_cuda13_strict_thinking_adapter_adds_launch_guard_once() -> None:
    added = SGLangStrictThinkingAdapter(
        "test-model",
        extra_launch_args=["--quantization", "fp8"],
    )
    retained = SGLangStrictThinkingAdapter(
        "test-model",
        extra_launch_args=["--quantization", "fp8", "--enable-strict-thinking"],
    )

    expected = [
        "--quantization",
        "fp8",
        "--enable-strict-thinking",
    ]
    assert added._extra_launch_args == expected
    assert retained._extra_launch_args == expected


def test_load_required_memory_bytes_uses_mem_fraction_static() -> None:
    gb = 1024**3
    adapter = SGLangGenerationAdapter("test-model", mem_fraction_static=0.8)

    assert adapter.load_required_memory_bytes(device_type="cuda", device_total_bytes=10 * gb) == 9 * gb
    assert adapter.load_required_memory_bytes(device_type="cpu", device_total_bytes=10 * gb) is None


@pytest.mark.parametrize("flag", ["--mamba-scheduler-strategy", "--mamba-radix-cache-strategy"])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # two-token form (Qwen3.5-4B YAML shape) → exact value
        (["FLAG", "extra_buffer"], "extra_buffer"),
        (["FLAG", "default"], "default"),
        # flag=value form
        (["FLAG=extra_buffer"], "extra_buffer"),
        (["FLAG=default"], "default"),
        # absent → None
        (["--disable-overlap-schedule"], None),
        ([], None),
        # trailing flag with no value → None (no crash)
        (["FLAG"], None),
        # a stray ``extra_buffer`` token elsewhere must NOT be read as the value
        (["--some-other-flag", "extra_buffer", "FLAG", "default"], "default"),
        # last occurrence wins (argparse semantics)
        (["FLAG", "default", "FLAG", "extra_buffer"], "extra_buffer"),
    ],
)
def test_mamba_strategy_value(flag: str, args: list[str], expected: str | None) -> None:
    assert _mamba_strategy_value([arg.replace("FLAG", flag) for arg in args], flag) == expected


def test_mamba_strategy_value_ignores_the_other_engines_spelling() -> None:
    assert _mamba_strategy_value(["--mamba-scheduler-strategy", "extra_buffer"], "--mamba-radix-cache-strategy") is None


def test_speculative_launch_args_support_qwen_eagle_and_gemma_assistant() -> None:
    assert SGLangGenerationAdapter._speculative_launch_args(
        {
            "enabled": True,
            "algorithm": "eagle",
            "num_steps": 3,
            "eagle_topk": 1,
            "num_draft_tokens": 4,
        }
    ) == [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    assert SGLangGenerationAdapter._speculative_launch_args(
        {
            "enabled": True,
            "algorithm": "nextn",
            "num_steps": 3,
            "eagle_topk": 1,
            "num_draft_tokens": 4,
            "draft_model": "google/gemma-4-31B-it-assistant",
            "draft_model_revision": "6" * 40,
        }
    )[-4:] == [
        "--speculative-draft-model-path",
        "google/gemma-4-31B-it-assistant",
        "--speculative-draft-model-revision",
        "6" * 40,
    ]


def test_unloaded_generate_raises(adapter) -> None:
    # ``generate`` is now an async generator function; the loaded-check
    # fires when we first try to drive the iterator, not at call time.
    async def _run() -> None:
        gen = adapter.generate(prompt="hi", max_new_tokens=8)
        await gen.__anext__()

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_drops_is_embedding(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    adapter,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter.load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "Qwen/Qwen3-4B-Instruct" in cmd
    assert "--is-embedding" not in cmd
    assert "--grammar-backend" in cmd
    assert cmd[cmd.index("--grammar-backend") + 1] == "outlines"
    assert "--served-model-name" in cmd
    assert adapter._server_url == "http://localhost:30005"
    child_env = mock_popen.call_args.kwargs["env"]
    assert child_env["SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT"] == "1"
    expected_compat_dir = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang/_compat"
    assert child_env["PYTHONPATH"].split(os.pathsep)[0] == str(expected_compat_dir)


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_ignores_configured_pythonpath(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_requests_get.return_value = MagicMock(status_code=200)
    inherited = os.pathsep.join(("/operator/one", "/operator/two"))
    monkeypatch.setenv("PYTHONPATH", inherited)
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3-4B-Instruct",
        extra_env={"PYTHONPATH": "/checkpoint-controlled", "SAFE_FLAG": "1"},
    )

    adapter.load("cuda:0")

    child_env = mock_popen.call_args.kwargs["env"]
    expected_compat_dir = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang/_compat"
    assert child_env["PYTHONPATH"] == os.pathsep.join((str(expected_compat_dir), inherited))
    assert "/checkpoint-controlled" not in child_env["PYTHONPATH"]
    assert child_env["SAFE_FLAG"] == "1"


@patch("sie_server.adapters.sglang._server.wait_for_server", return_value=True)
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_passes_profile_startup_timeout_to_health_wait(
    mock_find_port: MagicMock,
    mock_popen: MagicMock,
    mock_wait_for_server: MagicMock,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.6-27B",
        served_model_name="Qwen/Qwen3.6-27B",
        startup_timeout_s=1800,
    )

    adapter.load("cuda:0")

    assert mock_wait_for_server.call_args.kwargs["timeout_s"] == 1800


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_can_opt_into_xgrammar_backend(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_requests_get.return_value = MagicMock(status_code=200)
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.5-4B",
        served_model_name="Qwen/Qwen3.5-4B",
        grammar_backend="xgrammar",
    )

    adapter.load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert "--grammar-backend" in cmd
    assert cmd[cmd.index("--grammar-backend") + 1] == "xgrammar"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_emits_lora_launch_args(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_requests_get.return_value = MagicMock(status_code=200)
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
        lora_paths={"acme-support": "acme/support-lora", "acme-legal": "acme/legal-lora"},
        max_loras_per_batch=2,
    )

    adapter.load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert "--enable-lora" in cmd
    assert "--max-loras-per-batch" in cmd
    assert cmd[cmd.index("--max-loras-per-batch") + 1] == "2"
    assert "--lora-paths" in cmd
    # served-name=path pairs follow --lora-paths.
    assert "acme-support=acme/support-lora" in cmd
    assert "acme-legal=acme/legal-lora" in cmd


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_load_omits_lora_args_when_no_adapters(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    adapter,
) -> None:
    mock_find_port.return_value = 30005
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter.load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert "--enable-lora" not in cmd
    assert "--lora-paths" not in cmd


class _FakeStreamingResponse:
    """Minimal async-context-manager stand-in for ``httpx.AsyncClient.stream``."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"error body"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


def _make_client_with_stream(stream: _FakeStreamingResponse) -> MagicMock:
    client_instance = MagicMock()
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=None)
    client_instance.stream = MagicMock(return_value=stream)
    client_instance.post = AsyncMock()
    # httpx exposes ``is_closed``; the GeneratorExit abort path checks it
    # before spawning /abort_request. Default to open (a bare MagicMock
    # attribute would be a truthy mock and wrongly read as "closed").
    client_instance.is_closed = False
    return client_instance


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streams_sse_into_chunks(mock_async_client: MagicMock, adapter) -> None:
    # SGLang emits cumulative ``text`` per event; we expect deltas.
    sse_lines = [
        'data: {"text": "Hello", "meta_info": {"prompt_tokens": 5}}',
        'data: {"text": "Hello, world", "meta_info": {"prompt_tokens": 5}}',
        'data: {"text": "Hello, world!", "meta_info": {"prompt_tokens": 5, "completion_tokens": 3, "finish_reason": {"type": "stop"}}}',
        "data: [DONE]",
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="Hi", max_new_tokens=64, temperature=0.7, top_p=0.9):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())

    # Three text events → 2 deltas + 1 terminal (which also carries the final delta).
    assert [c.text_delta for c in chunks] == ["Hello", ", world", "!"]
    assert chunks[0].is_first is True
    assert chunks[1].is_first is False
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].prompt_tokens == 5
    assert chunks[-1].completion_tokens == 3


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_surfaces_in_band_sglang_error(mock_async_client: MagicMock, adapter) -> None:
    stream = _FakeStreamingResponse(
        [
            'data: {"error": {"message": "vision processor rejected image"}}',
            "data: [DONE]",
        ]
    )
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=8):
            pass

    with pytest.raises(
        RuntimeError,
        match="SGLang /generate returned an in-band error",
    ):
        asyncio.run(_collect())


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_collect_into_result(mock_async_client: MagicMock, adapter) -> None:
    sse_lines = [
        'data: {"text": "abc", "meta_info": {"prompt_tokens": 1}}',
        'data: {"text": "abcdef", "meta_info": {"prompt_tokens": 1, "completion_tokens": 2, "finish_reason": {"type": "length"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    client_instance = _make_client_with_stream(stream)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    result = asyncio.run(
        collect_generation(adapter.generate(prompt="Hi", max_new_tokens=64, temperature=0.7, top_p=0.9))
    )
    assert result.text == "abcdef"
    assert result.finish_reason == "length"
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 2
    client_instance.post.assert_not_awaited()


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_queue_child_task_wrapper_close_does_not_abort_terminal_request(
    mock_async_client: MagicMock,
    adapter,
) -> None:
    stream = _FakeStreamingResponse(
        [
            'data: {"text": "done", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
        ]
    )
    client_instance = _make_client_with_stream(stream)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        chunks = suppress_thinking_blocks(adapter.generate(prompt="Hi", max_new_tokens=8))
        while True:
            chunk = await asyncio.create_task(anext(chunks))
            if chunk.done:
                break
        await chunks.aclose()

    asyncio.run(_run())

    client_instance.post.assert_not_awaited()


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_n_gt_one_fans_out_into_candidates(mock_async_client: MagicMock, adapter) -> None:
    """`n>1` → one non-streaming SGLang call → one terminal chunk carrying all
    candidates. The mocked response uses the shape confirmed on SGLang 0.5.10
    (L4): a list of `n` objects, each with `meta_info.finish_reason={type,...}`,
    `completion_tokens`, `prompt_tokens`.
    """
    sglang_results = [
        {
            "text": " red, blue",
            "meta_info": {
                "finish_reason": {"type": "length", "length": 16},
                "completion_tokens": 16,
                "prompt_tokens": 4,
            },
        },
        {
            "text": " one, two",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 9,
                "prompt_tokens": 4,
            },
        },
    ]
    resp = MagicMock()
    resp.json = MagicMock(return_value=sglang_results)
    resp.raise_for_status = MagicMock()
    client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
    resp.aread = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    client_instance.stream.return_value = resp
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="List colors", max_new_tokens=16, n=2):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    # Exactly one terminal chunk carrying both candidates.
    assert len(chunks) == 1
    term = chunks[0]
    assert term.done is True
    assert term.candidates is not None
    assert len(term.candidates) == 2
    assert term.candidates[0]["text"] == " red, blue"
    assert term.candidates[0]["finish_reason"] == "length"
    assert term.candidates[1]["finish_reason"] == "stop"
    # Aggregate usage: prompt counted once, completion summed across candidates.
    assert term.prompt_tokens == 4
    assert term.completion_tokens == 25
    # The request asked SGLang for n candidates, non-streaming.
    body = client_instance.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["n"] == 2
    assert body["stream"] is False


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_forwards_lora_path(mock_async_client: MagicMock, adapter) -> None:
    """`lora_path` (the served-name) is forwarded as SGLang
    sampling_params.lora_path for per-request adapter selection.
    """
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    asyncio.run(collect_generation(adapter.generate(prompt="Hi", max_new_tokens=8, lora_path="acme-support")))
    body = client_instance_stream_body(mock_async_client)
    # lora_path is a TOP-LEVEL /generate field in SGLang 0.5.10 (verified on
    # L4), not a sampling param.
    assert body["lora_path"] == "acme-support"
    assert "lora_path" not in body["sampling_params"]


def client_instance_stream_body(mock_async_client: MagicMock) -> dict:
    return mock_async_client.return_value.stream.call_args.kwargs["json"]


def test_encode_image_data_builds_data_uris() -> None:
    # Bytes → base64 data URI carrying the format hint; order preserved.
    out = _encode_image_data(
        [
            {"data": b"\xff\xd8\xff\xe0jpegbytes", "format": "jpeg"},
            {"data": b"\x89PNGpngbytes", "format": "PNG"},
        ]
    )
    assert out is not None
    assert len(out) == 2
    assert out[0].startswith("data:image/jpeg;base64,")
    # Format hint is lower-cased.
    assert out[1].startswith("data:image/png;base64,")
    # Round-trips back to the original bytes.
    payload = out[0].split(",", 1)[1]
    assert base64.b64decode(payload) == b"\xff\xd8\xff\xe0jpegbytes"


def test_encode_image_data_defaults_format_and_handles_empty() -> None:
    # Missing/blank format defaults to jpeg.
    out = _encode_image_data([{"data": b"raw"}])
    assert out is not None
    assert out[0].startswith("data:image/jpeg;base64,")
    # No images → None so the request body is byte-identical to text-only.
    assert _encode_image_data(None) is None
    assert _encode_image_data([]) is None


def test_encode_image_data_rejects_non_bytes() -> None:
    # The wire contract: ``data`` must be bytes (un-decoded base64 strings raise).
    with pytest.raises(InvalidMediaError):
        _encode_image_data([{"data": "not-bytes", "format": "jpeg"}])


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_forwards_image_data(mock_async_client: MagicMock, adapter) -> None:
    """Images are forwarded as SGLang's top-level ``image_data`` field
    (list of data URIs), leaving ``sampling_params`` untouched.
    """
    sse_lines = [
        'data: {"text": "a cat", "meta_info": {"prompt_tokens": 5, "completion_tokens": 2, "finish_reason": {"type": "stop"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    images = [{"data": b"\xff\xd8\xff\xe0fakejpeg", "format": "jpeg"}]
    asyncio.run(collect_generation(adapter.generate(prompt="<image>describe this", max_new_tokens=8, images=images)))

    body = client_instance_stream_body(mock_async_client)
    assert isinstance(body["image_data"], list)
    assert body["image_data"][0].startswith("data:image/jpeg;base64,")
    # image_data is a TOP-LEVEL /generate field, not a sampling param.
    assert "image_data" not in body["sampling_params"]


def test_encode_video_data_builds_data_uris_and_clamps_format() -> None:
    out = _encode_video_data([{"data": b"\x00\x00\x00\x18ftypisom", "format": "mp4"}])
    assert out is not None
    assert out[0].startswith("data:video/mp4;base64,")
    assert base64.b64decode(out[0].split(",", 1)[1]) == b"\x00\x00\x00\x18ftypisom"
    clamped = _encode_video_data([{"data": b"x", "format": "x-mpegurl"}])
    assert clamped is not None
    assert clamped[0].startswith("data:video/mp4;base64,")
    assert _encode_video_data(None) is None
    assert _encode_video_data([]) is None
    with pytest.raises(InvalidMediaError):
        _encode_video_data([{"data": "not-bytes", "format": "mp4"}])


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_forwards_video_data(mock_async_client: MagicMock, adapter) -> None:
    sse_lines = [
        'data: {"text": "red", "meta_info": {"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    adapter._server_url = "http://localhost:30005"

    videos = [{"data": b"\x00\x00\x00\x18ftypisom", "format": "mp4"}]
    asyncio.run(collect_generation(adapter.generate(prompt="<video>what happens", max_new_tokens=8, videos=videos)))

    body = client_instance_stream_body(mock_async_client)
    assert body["video_data"][0].startswith("data:video/mp4;base64,")
    assert "video_data" not in body["sampling_params"]
    assert "image_data" not in body


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_omits_image_data_without_images(mock_async_client: MagicMock, adapter) -> None:
    """Text-only generation never sets ``image_data`` — the body stays
    byte-identical for the models that share this adapter without vision.
    """
    sse_lines = [
        'data: {"text": "hi", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    asyncio.run(collect_generation(adapter.generate(prompt="Hi", max_new_tokens=8)))
    body = client_instance_stream_body(mock_async_client)
    assert "image_data" not in body


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_grammar_and_images_coexist(mock_async_client: MagicMock, adapter) -> None:
    """Structured output (Outlines) still applies on a vision request: a
    request carrying BOTH images and a json_schema grammar forwards
    ``image_data`` (top-level) AND ``json_schema`` (sampling_params). The two
    are orthogonal on the ``/generate`` body, so grammar-constrained decoding
    is unaffected by the presence of an image.
    """
    sse_lines = [
        'data: {"text": "{\\"color\\":\\"red\\"}", "meta_info": {"prompt_tokens": 5, "completion_tokens": 4, "finish_reason": {"type": "stop"}}}',
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    adapter._server_url = "http://localhost:30005"

    images = [{"data": b"\xff\xd8\xff\xe0fakejpeg", "format": "jpeg"}]
    grammar = GrammarSpec(
        kind="json_schema",
        value={"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]},
    )
    asyncio.run(
        collect_generation(
            adapter.generate(prompt="<image>what colour?", max_new_tokens=16, images=images, grammar=grammar)
        )
    )

    body = client_instance_stream_body(mock_async_client)
    # Vision payload (top-level) AND grammar constraint (sampling_params) both present.
    assert body["image_data"][0].startswith("data:image/jpeg;base64,")
    assert "json_schema" in body["sampling_params"]


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_best_of_ranks_by_logprob_and_trims(mock_async_client: MagicMock, adapter) -> None:
    """`best_of=3, n=1`: generate 3 candidates, return the single highest by
    cumulative output logprob. Confirms over-generate + rank + trim, and that
    the request asked SGLang for `best_of` candidates with return_logprob.
    """

    def cand(text: str, lp: float) -> dict:
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 2,
                "prompt_tokens": 4,
                "output_token_logprobs": [[lp, 1, "a"], [lp, 2, "b"]],
            },
        }

    sglang_results = [cand(" low", -2.0), cand(" best", -0.1), cand(" mid", -1.0)]
    resp = MagicMock()
    resp.json = MagicMock(return_value=sglang_results)
    resp.raise_for_status = MagicMock()
    client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
    resp.aread = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    client_instance.stream.return_value = resp
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="x", max_new_tokens=8, n=1, best_of=3):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    term = chunks[0]
    assert term.candidates is not None
    assert len(term.candidates) == 1  # trimmed to n
    assert term.candidates[0]["text"] == " best"  # highest cumulative logprob (-0.2)
    body = client_instance.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["n"] == 3  # over-generated best_of
    assert body["return_logprob"] is True  # ranking needs logprobs


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streaming_n_gt_one_fans_out_choice_index(mock_async_client: MagicMock, adapter) -> None:
    """`n>1 && stream`: SGLang's per-index streaming events are demuxed into
    per-candidate delta chunks tagged with choice_index, plus a single terminal
    that aggregates usage.
    """
    sse_lines = [
        'data: {"index": 0, "text": "A", "meta_info": {"prompt_tokens": 3}}',
        'data: {"index": 1, "text": "X", "meta_info": {"prompt_tokens": 3}}',
        'data: {"index": 0, "text": "Alpha", "meta_info": {"prompt_tokens": 3, "completion_tokens": 2, "finish_reason": {"type": "stop"}}}',
        'data: {"index": 1, "text": "Xray", "meta_info": {"prompt_tokens": 3, "completion_tokens": 2, "finish_reason": {"type": "length"}}}',
        "data: [DONE]",
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="hi", max_new_tokens=8, n=2, stream=True):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    deltas = [c for c in chunks if not c.done]
    assert {c.choice_index for c in deltas} == {0, 1}
    c0 = [c for c in deltas if c.choice_index == 0]
    c1 = [c for c in deltas if c.choice_index == 1]
    assert "".join(c.text_delta for c in c0) == "Alpha"  # "A" + "lpha" (diffed)
    assert "".join(c.text_delta for c in c1) == "Xray"
    assert any(c.finish_reason == "stop" for c in c0)
    assert any(c.finish_reason == "length" for c in c1)
    term = chunks[-1]
    assert term.done is True
    assert term.prompt_tokens == 3
    assert term.completion_tokens == 4  # summed across candidates
    # The request asked SGLang for n candidates, streaming.
    body = mock_async_client.return_value.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["n"] == 2
    assert body["stream"] is True


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streaming_n_gt_one_aclose_aborts_incomplete_request(
    mock_async_client: MagicMock,
    adapter,
) -> None:
    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"index": 0, "text": "first", "meta_info": {"prompt_tokens": 1}}'
            await asyncio.Event().wait()  # pragma: no cover - closed by the test

    client_instance = _make_client_with_stream(_NeverEnding(lines=[]))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        chunks = adapter.generate(prompt="Hi", max_new_tokens=8, n=2, stream=True)
        first = await anext(chunks)
        assert first.text_delta == "first"
        await chunks.aclose()
        if adapter._abort_tasks:
            await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    request_rid = client_instance.stream.call_args.kwargs["json"]["rid"]
    client_instance.post.assert_awaited_once()
    args, kwargs = client_instance.post.await_args
    assert args[0].endswith("/abort_request")
    assert kwargs["json"] == {"rid": request_rid}


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streaming_n_gt_one_cancellation_aborts_incomplete_request(
    mock_async_client: MagicMock,
    adapter,
) -> None:
    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"index": 0, "text": "first", "meta_info": {"prompt_tokens": 1}}'
            await asyncio.Event().wait()  # pragma: no cover - cancelled by the test

    client_instance = _make_client_with_stream(_NeverEnding(lines=[]))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        chunks = adapter.generate(prompt="Hi", max_new_tokens=8, n=2, stream=True)
        assert (await anext(chunks)).text_delta == "first"
        pending = asyncio.create_task(anext(chunks))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        if adapter._abort_tasks:
            await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    request_rid = client_instance.stream.call_args.kwargs["json"]["rid"]
    client_instance.post.assert_awaited_once()
    args, kwargs = client_instance.post.await_args
    assert args[0].endswith("/abort_request")
    assert kwargs["json"] == {"rid": request_rid}


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streaming_n_gt_one_terminal_close_does_not_abort(
    mock_async_client: MagicMock,
    adapter,
) -> None:
    stream = _FakeStreamingResponse(
        [
            'data: {"index": 0, "text": "A", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
            'data: {"index": 1, "text": "B", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
            "data: [DONE]",
        ]
    )
    client_instance = _make_client_with_stream(stream)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        chunks = adapter.generate(prompt="Hi", max_new_tokens=8, n=2, stream=True)
        while not (await anext(chunks)).done:
            pass
        await chunks.aclose()
        if adapter._abort_tasks:
            await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    client_instance.post.assert_not_awaited()


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_request_body_uses_stream_true(mock_async_client: MagicMock, adapter) -> None:
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    client_instance = _make_client_with_stream(stream)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=64, temperature=0.7, top_p=0.9):
            pass

    asyncio.run(_drain())

    # AsyncClient.stream("POST", url, json=body)
    args, kwargs = client_instance.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "http://localhost:30005/generate"
    body = kwargs["json"]
    assert body["text"] == "Hi"
    assert body["stream"] is True
    assert body["sampling_params"]["max_new_tokens"] == 64
    assert body["sampling_params"]["temperature"] == pytest.approx(0.7)
    assert "skip_special_tokens" not in body["sampling_params"]
    assert "rid" in body  # cancellation handle


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_gemma_generate_preserves_reasoning_markers_for_sie_suppression(mock_async_client: MagicMock) -> None:
    sse_lines = [
        'data: {"text": "<|channel>thought\\nprivate<channel|>answer", "meta_info": {"prompt_tokens": 1, "completion_tokens": 4, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter = SGLangGenerationAdapter(
        model_name_or_path="google/gemma-4-31B-it",
        served_model_name="google/gemma-4-31B-it",
        reasoning_parser="gemma4",
    )
    adapter._server_url = "http://localhost:30005"

    result = asyncio.run(
        collect_generation(
            suppress_thinking_blocks(
                adapter.generate(prompt="Hi", max_new_tokens=8),
                reasoning_format="gemma4",
            )
        )
    )

    body = client_instance.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["skip_special_tokens"] is False
    assert result.text == "answer"


def _posted_generate_body(
    mock_async_client: MagicMock,
    prompt: str,
    *,
    reasoning_parser: str | None,
    n: int | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    finish = '"meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}'
    if n is not None and not stream:
        response = MagicMock()
        response.json = MagicMock(
            return_value=[
                {
                    "text": "x",
                    "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 1, "prompt_tokens": 1},
                }
            ]
            * n
        )
        response.raise_for_status = MagicMock()
        response.aread = AsyncMock()
        response.__aenter__ = AsyncMock(return_value=response)
        client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
        client_instance.stream.return_value = response
    elif n is not None:
        lines = [f'data: {{"index": {index}, "text": "x", {finish}}}' for index in range(n)] + ["data: [DONE]"]
        client_instance = _make_client_with_stream(_FakeStreamingResponse(lines))
    else:
        client_instance = _make_client_with_stream(_FakeStreamingResponse([f'data: {{"text": "x", {finish}}}']))
    mock_async_client.return_value = client_instance
    adapter = SGLangGenerationAdapter(
        model_name_or_path="zai-org/GLM-5.3-Flash",
        served_model_name="zai-org/GLM-5.3-Flash",
        reasoning_parser=reasoning_parser,
    )
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt=prompt, max_new_tokens=8, n=n, stream=stream):
            pass

    asyncio.run(_drain())
    return client_instance.stream.call_args.kwargs["json"]


@pytest.mark.parametrize(("n", "stream"), [(None, False), (2, True), (2, False)])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_requires_reasoning_when_the_prompt_leaves_thinking_open(
    mock_async_client: MagicMock, n: int | None, stream: bool
) -> None:
    body = _posted_generate_body(
        mock_async_client, "[gMASK]<sop><|user|>Hi<|assistant|><think>", reasoning_parser="glm45", n=n, stream=stream
    )

    assert body["require_reasoning"] is True


@pytest.mark.parametrize(
    ("prompt", "reasoning_parser"),
    [
        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", "qwen3"),
        ("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "qwen3"),
        ("[gMASK]<sop><|user|>Hi<|assistant|><think>", None),
    ],
)
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_leaves_reasoning_to_the_engine_default_otherwise(
    mock_async_client: MagicMock, prompt: str, reasoning_parser: str | None
) -> None:
    body = _posted_generate_body(mock_async_client, prompt, reasoning_parser=reasoning_parser)

    assert "require_reasoning" not in body


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_requires_reasoning_after_a_seeded_gemma_channel(mock_async_client: MagicMock) -> None:
    body = _posted_generate_body(mock_async_client, "<|turn>model\n<|channel>", reasoning_parser="gemma4")

    assert body["require_reasoning"] is True


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_logprobs_request_sets_return_text_in_logprobs(mock_async_client: MagicMock, adapter) -> None:
    """A logprobs request must ask SGLang for decoded token TEXT.

    Without ``return_text_in_logprobs`` SGLang returns
    ``[logprob, token_id, None]`` and the OpenAI ``token`` field (and the guard
    ``yes``/``no`` verdict match) sees only empty strings — confirmed on a real
    Granite Guardian L4 run.
    """
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=8, logprobs=True, top_logprobs=20):
            pass

    asyncio.run(_drain())

    body = client_instance.stream.call_args.kwargs["json"]
    assert body["return_logprob"] is True
    assert body["return_text_in_logprobs"] is True
    assert body["top_logprobs_num"] == 20


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_forwards_top_k_and_repetition_penalty(mock_async_client: MagicMock, adapter) -> None:
    """``top_k`` / ``repetition_penalty`` reach SGLang's sampling_params
    when provided, and are omitted otherwise so model defaults hold.
    """
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=64, top_k=10, repetition_penalty=1.1):
            pass

    asyncio.run(_drain())

    body = client_instance.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["top_k"] == 10
    assert body["sampling_params"]["repetition_penalty"] == pytest.approx(1.1)


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_caps_profile_default_min_new_tokens_to_request_max(mock_async_client: MagicMock) -> None:
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "length"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.6-27B",
        served_model_name="Qwen/Qwen3.6-27B",
        default_sampling={"min_new_tokens": 10},
    )
    adapter._server_url = "http://localhost:30005"

    asyncio.run(collect_generation(adapter.generate(prompt="Hi", max_new_tokens=1)))

    sampling_params = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert sampling_params["max_new_tokens"] == 1
    assert sampling_params["min_new_tokens"] == 1


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_preserves_profile_default_min_new_tokens_below_request_max(mock_async_client: MagicMock) -> None:
    sse_lines = [
        'data: {"text": "ok", "meta_info": {"prompt_tokens": 1, "completion_tokens": 2, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.6-27B",
        served_model_name="Qwen/Qwen3.6-27B",
        default_sampling={"min_new_tokens": 10},
    )
    adapter._server_url = "http://localhost:30005"

    asyncio.run(collect_generation(adapter.generate(prompt="Hi", max_new_tokens=64)))

    sampling_params = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert sampling_params["max_new_tokens"] == 64
    assert sampling_params["min_new_tokens"] == 10


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_rejects_explicit_min_new_tokens_above_max_before_sglang(
    mock_async_client: MagicMock,
) -> None:
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.6-27B",
        served_model_name="Qwen/Qwen3.6-27B",
        default_sampling={"min_new_tokens": 10},
    )
    adapter._server_url = "http://localhost:30005"

    with pytest.raises(
        ValueError,
        match=r"min_new_tokens \(10\) must not exceed max_new_tokens \(1\)",
    ):
        asyncio.run(
            collect_generation(
                adapter.generate(
                    prompt="Hi",
                    max_new_tokens=1,
                    min_new_tokens=10,
                )
            )
        )

    mock_async_client.assert_not_called()


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_explicit_min_new_tokens_overrides_profile_default_when_valid(
    mock_async_client: MagicMock,
) -> None:
    sse_lines = [
        'data: {"text": "ok", "meta_info": {"prompt_tokens": 1, "completion_tokens": 2, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.6-27B",
        served_model_name="Qwen/Qwen3.6-27B",
        default_sampling={"min_new_tokens": 10},
    )
    adapter._server_url = "http://localhost:30005"

    asyncio.run(
        collect_generation(
            adapter.generate(
                prompt="Hi",
                max_new_tokens=8,
                min_new_tokens=2,
            )
        )
    )

    sampling_params = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert sampling_params["max_new_tokens"] == 8
    assert sampling_params["min_new_tokens"] == 2


@pytest.fixture(
    params=[
        pytest.param(GrammarSpec(kind="json_schema", value={"type": "string"}), id="json_schema"),
        pytest.param(GrammarSpec(kind="regex", value="[a-z]+"), id="regex"),
        pytest.param(GrammarSpec(kind="ebnf", value='root ::= "ok"'), id="ebnf"),
    ]
)
def grammar_default_sampling_spec(request: pytest.FixtureRequest) -> GrammarSpec:
    return request.param


@pytest.mark.parametrize(
    ("min_new_tokens", "top_k", "repetition_penalty"),
    [
        pytest.param(None, None, None, id="inherited"),
        pytest.param(0, None, None, id="explicit-zero"),
        pytest.param(2, 5, 1.2, id="explicit-overrides"),
    ],
)
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_grammar_default_sampling(
    mock_async_client: MagicMock,
    grammar_default_sampling_spec: GrammarSpec,
    min_new_tokens: int | None,
    top_k: int | None,
    repetition_penalty: float | None,
) -> None:
    sse_lines = [
        'data: {"text": "ok", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    default_sampling = {"min_new_tokens": 10, "top_k": 17, "repetition_penalty": 1.1}
    expected_defaults = default_sampling.copy()
    adapter = SGLangGenerationAdapter("test-model", default_sampling=default_sampling)
    adapter._server_url = "http://localhost:30005"
    grammar = grammar_default_sampling_spec

    asyncio.run(
        collect_generation(
            adapter.generate(
                prompt="Hi",
                max_new_tokens=3,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                grammar=grammar,
            )
        )
    )

    sampling_params = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert sampling_params["max_new_tokens"] == 3
    if min_new_tokens is None:
        assert "min_new_tokens" not in sampling_params
    else:
        assert sampling_params["min_new_tokens"] == min_new_tokens
    assert sampling_params["top_k"] == (17 if top_k is None else top_k)
    assert sampling_params["repetition_penalty"] == pytest.approx(
        1.1 if repetition_penalty is None else repetition_penalty
    )
    expected_grammar = json.dumps(grammar.value) if grammar.kind == "json_schema" else grammar.value
    assert {key: sampling_params[key] for key in ("json_schema", "regex", "ebnf") if key in sampling_params} == {
        grammar.kind: expected_grammar
    }
    assert adapter._default_sampling == expected_defaults
    assert default_sampling == expected_defaults

    for max_new_tokens in (1, 64):
        asyncio.run(collect_generation(adapter.generate(prompt="Hi", max_new_tokens=max_new_tokens)))
        unconstrained_sampling = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
        assert unconstrained_sampling["min_new_tokens"] == min(10, max_new_tokens)
        assert unconstrained_sampling["top_k"] == 17
        assert unconstrained_sampling["repetition_penalty"] == pytest.approx(1.1)
        assert not {"json_schema", "regex", "ebnf"}.intersection(unconstrained_sampling)

    assert adapter._default_sampling == expected_defaults
    assert default_sampling == expected_defaults


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_grammar_default_sampling_rejects_explicit_min_above_max(
    mock_async_client: MagicMock,
    grammar_default_sampling_spec: GrammarSpec,
) -> None:
    adapter = SGLangGenerationAdapter(
        "test-model",
        default_sampling={"min_new_tokens": 10, "top_k": 17, "repetition_penalty": 1.1},
    )
    adapter._server_url = "http://localhost:30005"

    with pytest.raises(ValueError, match=r"min_new_tokens \(10\) must not exceed max_new_tokens \(1\)"):
        asyncio.run(
            collect_generation(
                adapter.generate(
                    prompt="Hi",
                    max_new_tokens=1,
                    min_new_tokens=10,
                    grammar=grammar_default_sampling_spec,
                )
            )
        )

    mock_async_client.assert_not_called()


@pytest.mark.parametrize("seed", [-1, 0, 1])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_maps_seed_to_sglang_sampling_seed(mock_async_client: MagicMock, adapter, seed: int) -> None:
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=64, seed=seed):
            pass

    asyncio.run(_drain())

    sampling_params = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert sampling_params["sampling_seed"] == seed
    assert "seed" not in sampling_params


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_omits_top_k_and_repetition_penalty_when_unset(mock_async_client: MagicMock, adapter) -> None:
    sse_lines = [
        'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}',
    ]
    client_instance = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt="Hi", max_new_tokens=64):
            pass

    asyncio.run(_drain())

    sp = client_instance.stream.call_args.kwargs["json"]["sampling_params"]
    assert "top_k" not in sp
    assert "repetition_penalty" not in sp


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_aclose_triggers_abort_request(mock_async_client: MagicMock, adapter) -> None:
    # An empty stream that hangs — caller will aclose() mid-stream.
    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            # Yield one chunk, then suspend forever.
            yield 'data: {"text": "first"}'
            await asyncio.Event().wait()  # pragma: no cover

    stream = _NeverEnding(lines=[])
    client_instance = _make_client_with_stream(stream)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        gen = adapter.generate(prompt="Hi", max_new_tokens=64)
        first = await gen.__anext__()
        assert first.text_delta == "first"
        await gen.aclose()

    asyncio.run(_run())

    # /abort_request was POSTed best-effort with the rid carried in the body.
    client_instance.post.assert_awaited()
    args, _ = client_instance.post.await_args
    assert args[0].endswith("/abort_request")


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_task_cancellation_triggers_abort_request(mock_async_client: MagicMock, adapter) -> None:
    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"text": "first"}'
            await asyncio.Event().wait()  # pragma: no cover - cancelled by the test

    client_instance = _make_client_with_stream(_NeverEnding(lines=[]))
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        chunks = adapter.generate(prompt="Hi", max_new_tokens=64)
        assert (await anext(chunks)).text_delta == "first"
        pending = asyncio.create_task(anext(chunks))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        if adapter._abort_tasks:
            await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    client_instance.post.assert_awaited_once()
    args, _ = client_instance.post.await_args
    assert args[0].endswith("/abort_request")


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_abort_completes_even_when_aclose_capped(mock_async_client: MagicMock, adapter) -> None:
    """B2 regression: the /abort_request POST must be issued and complete
    even when the iterator is closed under a SHORT ``wait_for`` cap.

    The streaming processor tears the iterator down via
    ``asyncio.wait_for(gen.aclose(), timeout=2.0)``. Because the abort is
    spawned as an independent background task (NOT awaited inside the
    GeneratorExit handler), a tiny aclose cap does not cancel it: aclose
    returns promptly and the abort runs to completion afterwards.
    """

    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"text": "first"}'
            await asyncio.Event().wait()  # pragma: no cover — held until aclose

    stream = _NeverEnding(lines=[])
    client_instance = _make_client_with_stream(stream)

    # Make the abort POST slower than the aclose cap so that, if the abort
    # were (incorrectly) awaited inside GeneratorExit, the short cap would
    # cancel it. With the fix it runs as an independent task and completes.
    post_completed = asyncio.Event()

    async def _slow_post(*args, **kwargs):
        await asyncio.sleep(0.05)
        post_completed.set()
        return MagicMock(status_code=200)

    client_instance.post = AsyncMock(side_effect=_slow_post)
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        gen = adapter.generate(prompt="Hi", max_new_tokens=64)
        first = await gen.__anext__()
        assert first.text_delta == "first"
        # Tear down with a SHORT cap — much shorter than the abort's own
        # 1.5s timeout — mirroring the streaming processor's wait_for.
        await asyncio.wait_for(gen.aclose(), timeout=0.01)
        # aclose returned promptly; the abort task is tracked and still
        # running. It must NOT have been cancelled by the cap.
        assert adapter._abort_tasks, "abort task was not spawned/tracked"
        await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    assert post_completed.is_set(), "abort POST did not complete"
    client_instance.post.assert_awaited()
    args, _ = client_instance.post.await_args
    assert args[0].endswith("/abort_request")


@patch("sie_server.adapters.sglang._server.os.getpgid")
@patch("sie_server.adapters.sglang._server.os.killpg")
def test_unload_terminates_process(mock_killpg: MagicMock, mock_getpgid: MagicMock, adapter) -> None:
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.wait.return_value = None
    mock_getpgid.return_value = 12345

    adapter._process = mock_process
    adapter._server_url = "http://localhost:30005"
    adapter._device = "cuda:0"

    adapter.unload()

    mock_killpg.assert_called()
    assert adapter._process is None
    assert adapter._server_url is None


@patch("sie_server.adapters.sglang._server.os.getpgid")
@patch("sie_server.adapters.sglang._server.os.killpg")
def test_unload_releases_port_and_cleans_output_log(
    mock_killpg: MagicMock,
    mock_getpgid: MagicMock,
    adapter,
) -> None:
    """Unload returns the reserved port to the pool and removes the temp log."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.wait.return_value = None
    mock_getpgid.return_value = 12345

    adapter._process = mock_process
    adapter._server_url = "http://localhost:30005"
    adapter._device = "cuda:0"
    adapter._port = 30005
    adapter._output_file = _server.open_output_log(prefix="sie_test_sglang_")
    log_path = Path(adapter._output_file.name)

    with patch("sie_server.adapters.sglang._server.release_port") as mock_release:
        adapter.unload()

    mock_release.assert_called_once_with(30005)
    assert adapter._port is None
    assert adapter._output_file is None
    assert not log_path.exists()


@patch("sie_server.adapters.sglang._server.os.getpgid")
@patch("sie_server.adapters.sglang._server.os.killpg")
@patch("sie_server.adapters.sglang._server.wait_for_server", return_value=False)
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_failed_startup_releases_port_and_cleans_output_log(
    mock_find_port: MagicMock,
    mock_popen: MagicMock,
    mock_wait: MagicMock,
    mock_killpg: MagicMock,
    mock_getpgid: MagicMock,
    adapter,
) -> None:
    """A startup-health failure must not leak the reserved port or the temp log."""
    mock_find_port.return_value = 30006
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.poll.return_value = None  # Process running but not healthy (timeout path)
    mock_popen.return_value = mock_process
    mock_getpgid.return_value = 12345

    with (
        patch("sie_server.adapters.sglang._server.release_port") as mock_release,
        pytest.raises(RuntimeError, match="failed to start"),
    ):
        adapter.load("cuda:0")

    mock_release.assert_called_once_with(30006)
    assert adapter._port is None
    assert adapter._server_url is None
    assert adapter._output_file is None


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_speculative_guard_abort_releases_port_and_cleans_output_log(
    mock_find_port: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """The pre-launch extra_buffer validation abort must not leak the port or log.

    The guard raises after the port was reserved and the temp log opened but
    before the subprocess launches — a distinct abort seam from the
    startup-health failure above.
    """
    mock_find_port.return_value = 30007
    adapter = SGLangGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.5-4B",
        served_model_name="Qwen/Qwen3.5-4B",
        speculative={"enabled": True, "algorithm": "nextn"},
    )

    with (
        patch("sie_server.adapters.sglang._server.release_port") as mock_release,
        pytest.raises(RuntimeError, match="extra_buffer"),
    ):
        adapter.load("cuda:0")

    mock_release.assert_called_once_with(30007)
    assert adapter._port is None
    assert adapter._server_url is None
    assert adapter._output_file is None
    mock_popen.assert_not_called()


@patch("sie_server.adapters.sglang.generation.asyncio.new_event_loop")
@patch("sie_server.adapters.sglang._server.os.getpgid")
@patch("sie_server.adapters.sglang._server.os.killpg")
def test_unload_no_running_loop_skips_aclose_and_terminates(
    mock_killpg: MagicMock,
    mock_getpgid: MagicMock,
    mock_new_event_loop: MagicMock,
    adapter,
) -> None:
    """Fix #1: with an open http client but NO running event loop (process
    exit path), ``unload()`` must NOT build a new loop to drive ``aclose()``
    (the httpx client is bound to its original loop — closing it from a
    fresh loop can raise/leak the pool). It skips the async close and still
    terminates the SGLang subprocess.
    """
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_getpgid.return_value = 12345

    client = MagicMock()
    client.aclose = AsyncMock()
    adapter._http_client = client
    adapter._process = mock_process
    adapter._server_url = "http://localhost:30005"
    adapter._device = "cuda:0"

    # No event loop is running in this synchronous test → the no-loop branch.
    adapter.unload()

    # Did NOT spin up a dedicated loop, and did NOT drive aclose() on one.
    mock_new_event_loop.assert_not_called()
    client.aclose.assert_not_called()
    # The shared-client ref was still cleared and the subprocess terminated.
    assert adapter._http_client is None
    mock_killpg.assert_called()
    assert adapter._process is None
    assert adapter._server_url is None


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_aclose_during_generatorexit_skipped_when_client_closed(mock_async_client: MagicMock, adapter) -> None:
    """Fix #2: if the shared HTTP client was already closed (e.g. by a
    concurrent ``aclose_client()`` during unload) when the generator is torn
    down, the GeneratorExit handler must NOT spawn an /abort_request through
    the dead client — it skips and logs instead, so no orphaned task fails
    silently against a closed pool.
    """

    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"text": "first"}'
            await asyncio.Event().wait()  # pragma: no cover — held until aclose

    stream = _NeverEnding(lines=[])
    client_instance = _make_client_with_stream(stream)
    # Simulate a concurrent unload having already closed the shared client.
    client_instance.is_closed = True
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        gen = adapter.generate(prompt="Hi", max_new_tokens=64)
        first = await gen.__anext__()
        assert first.text_delta == "first"
        await gen.aclose()

    asyncio.run(_run())

    # No abort POST was attempted (client was closed) and nothing was tracked.
    client_instance.post.assert_not_awaited()
    assert not adapter._abort_tasks


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_aclose_during_generatorexit_aborts_when_client_open(mock_async_client: MagicMock, adapter) -> None:
    """Companion to the closed-client case: when the client is still open
    (``is_closed`` False), the GeneratorExit handler DOES spawn the abort.
    """

    class _NeverEnding(_FakeStreamingResponse):
        async def aiter_lines(self):
            yield 'data: {"text": "first"}'
            await asyncio.Event().wait()  # pragma: no cover — held until aclose

    stream = _NeverEnding(lines=[])
    client_instance = _make_client_with_stream(stream)
    client_instance.is_closed = False
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _run() -> None:
        gen = adapter.generate(prompt="Hi", max_new_tokens=64)
        await gen.__anext__()
        await gen.aclose()
        if adapter._abort_tasks:
            await asyncio.gather(*tuple(adapter._abort_tasks))

    asyncio.run(_run())

    client_instance.post.assert_awaited()
    args, _ = client_instance.post.await_args
    assert args[0].endswith("/abort_request")


def test_aclose_client_awaits_client_close(adapter) -> None:
    """H5: ``aclose_client`` awaits the shared HTTP client's ``aclose`` and
    clears the reference, so the worker shutdown path can close it BEFORE
    terminating the subprocess (rather than fire-and-forget racing it).
    """
    client = MagicMock()
    client.aclose = AsyncMock()
    adapter._http_client = client

    asyncio.run(adapter.aclose_client())

    client.aclose.assert_awaited_once()
    assert adapter._http_client is None


def test_aclose_client_drains_pending_abort_tasks(adapter) -> None:
    """``aclose_client`` drains in-flight /abort_request tasks before
    closing — they target the still-live subprocess and must finish first.
    """
    client = MagicMock()
    client.aclose = AsyncMock()
    adapter._http_client = client

    abort_done = asyncio.Event()

    async def _run() -> None:
        async def _abort() -> None:
            await asyncio.sleep(0.02)
            abort_done.set()

        task = asyncio.ensure_future(_abort())
        adapter._abort_tasks.add(task)
        task.add_done_callback(adapter._abort_tasks.discard)

        await adapter.aclose_client()
        # The abort completed before aclose_client returned.
        assert abort_done.is_set()

    asyncio.run(_run())
    client.aclose.assert_awaited_once()


def test_aclose_client_noop_when_no_client(adapter) -> None:
    """No client ever opened → ``aclose_client`` is a clean no-op."""
    assert adapter._http_client is None
    asyncio.run(adapter.aclose_client())  # must not raise


# -- Legacy non-streaming parser kept for back-compat — direct tests --------


def test_parse_response_with_list_shape() -> None:
    result = _parse_sglang_generate_response(
        [
            {
                "text": "abc",
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "finish_reason": "length",
                },
            }
        ]
    )
    assert result.text == "abc"
    assert result.finish_reason == "length"
    assert result.completion_tokens == 1


def test_parse_response_missing_meta_is_an_error() -> None:
    with pytest.raises(GenerationError, match="without a finish reason"):
        _parse_sglang_generate_response({"text": "xyz"})


def test_chunk_translator_surfaces_logprobs_3tuple() -> None:
    """SGLang's 3-tuple shape ``(logprob, token_id, token_text)`` translates
    to OpenAI-shape ``ChatCompletionTokenLogprob`` entries on the chunk.
    """
    event = {
        "text": "hi",
        "meta_info": {
            "output_token_logprobs": [
                [-0.1, 100, "h"],
                [-0.2, 200, "i"],
            ],
            "output_top_logprobs": [
                [[-0.1, 100, "h"], [-2.0, 999, "H"]],
                [[-0.2, 200, "i"], [-3.0, 998, "I"]],
            ],
        },
    }
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="",
        first_yield_done=False,
        logprobs_enabled=True,
        logprobs_surfaced=0,
    )
    assert chunk is not None
    assert chunk.logprobs is not None
    assert len(chunk.logprobs) == 2
    first = chunk.logprobs[0]
    assert first["token"] == "h"  # noqa: S105 — generation token, not a secret
    assert first["logprob"] == -0.1
    assert len(first["top_logprobs"]) == 2
    assert first["top_logprobs"][1]["token"] == "H"  # noqa: S105 — generation token, not a secret


def test_chunk_translator_slices_against_surfaced_count() -> None:
    """On subsequent events SGLang's output_token_logprobs is cumulative;
    the translator slices off the tail-since-last-event using
    ``logprobs_surfaced``.
    """
    event = {
        "text": "hi there",
        "meta_info": {
            "output_token_logprobs": [
                [-0.1, 100, "h"],
                [-0.2, 200, "i"],
                [-0.3, 300, " there"],
            ],
        },
    }
    # We've already surfaced the first two; this event should only
    # produce one logprob entry for the new token.
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="hi",
        first_yield_done=True,
        logprobs_enabled=True,
        logprobs_surfaced=2,
    )
    assert chunk is not None
    assert chunk.logprobs is not None
    assert len(chunk.logprobs) == 1
    assert chunk.logprobs[0]["token"] == " there"  # noqa: S105 — generation token, not a secret


def test_chunk_translator_disabled_returns_no_logprobs() -> None:
    """Default path: ``logprobs_enabled=False`` → ``chunk.logprobs is None``
    even if SGLang sent the field (defensive).
    """
    event = {
        "text": "hi",
        "meta_info": {"output_token_logprobs": [[-0.1, 100, "h"]]},
    }
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="",
        first_yield_done=False,
    )
    assert chunk is not None
    assert chunk.logprobs is None


def test_chunk_translator_skips_non_monotonic_cumulative_text() -> None:
    """Non-monotonic cumulative text (current is NOT a prefix-extension of
    the previous) must NOT re-emit the whole buffer — that duplicates
    already-streamed output. The delta is skipped (empty) and the
    non-terminal event is dropped.
    """
    # previous cumulative = "Hello, world"; current diverges (does not start
    # with the previous text) — e.g. SGLang reset/regenerate.
    event = {"text": "Goodbye"}
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="Hello, world",
        first_yield_done=True,
    )
    # Non-terminal divergent event → dropped entirely (no duplicate emit).
    assert chunk is None


def test_chunk_translator_skips_shorter_cumulative_text() -> None:
    """A shorter cumulative buffer (truncation) is also non-monotonic and
    must be skipped rather than re-emitted.
    """
    event = {"text": "Hel"}  # shorter than previous, and a prefix OF previous
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="Hello",
        first_yield_done=True,
    )
    # "Hello".startswith("Hel") is True but "Hel".startswith("Hello") is
    # False → treated as non-monotonic → dropped.
    assert chunk is None


def test_chunk_translator_terminal_divergent_still_terminates_with_empty_delta() -> None:
    """A terminal event whose cumulative text diverged still produces a
    terminal chunk (so the stream ends) but with an empty text delta (no
    duplicate output).
    """
    event = {
        "text": "Goodbye",
        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 3},
    }
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="Hello, world",
        first_yield_done=True,
    )
    assert chunk is not None
    assert chunk.done is True
    assert chunk.text_delta == ""
    assert chunk.finish_reason == "stop"


def test_chunk_translator_tolerates_2tuple_shape() -> None:
    """Older SGLang versions ship a 2-tuple ``(logprob, token_id)``.
    The translator should accept and produce empty ``token`` strings
    rather than raising.
    """
    event = {
        "text": "x",
        "meta_info": {"output_token_logprobs": [[-0.5, 42]]},
    }
    chunk = _chunk_from_sglang_event(
        event,
        previous_cumulative_text="",
        first_yield_done=False,
        logprobs_enabled=True,
        logprobs_surfaced=0,
    )
    assert chunk is not None
    assert chunk.logprobs is not None
    assert chunk.logprobs[0]["logprob"] == -0.5
    assert chunk.logprobs[0]["token"] == ""


# ── Multi-candidate logprobs ─────────────────────────────────────


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_streaming_n_gt_one_attaches_per_candidate_logprobs(mock_async_client: MagicMock, adapter) -> None:
    """H4: ``n>1 && stream`` with ``logprobs=True``: each per-candidate
    delta carries the logprob slice introduced for that candidate on
    this event. The watermark is per-index, so candidate 0 and candidate
    1 each get their own monotonic slice (no cross-candidate leakage).
    """
    sse_lines = [
        # Candidate 0, first event: text "A", one token logprob.
        (
            'data: {"index": 0, "text": "A", "meta_info": {"prompt_tokens": 3, '
            '"output_token_logprobs": [[-0.10, 1, "A"]]}}'
        ),
        # Candidate 1, first event: text "X", one token logprob.
        (
            'data: {"index": 1, "text": "X", "meta_info": {"prompt_tokens": 3, '
            '"output_token_logprobs": [[-0.20, 2, "X"]]}}'
        ),
        # Candidate 0 finish, +1 new logprob entry (cumulative len 2).
        (
            'data: {"index": 0, "text": "Ab", "meta_info": {"prompt_tokens": 3, '
            '"completion_tokens": 2, "finish_reason": {"type": "stop"}, '
            '"output_token_logprobs": [[-0.10, 1, "A"], [-0.30, 3, "b"]]}}'
        ),
        # Candidate 1 finish, +1 new logprob entry.
        (
            'data: {"index": 1, "text": "Xy", "meta_info": {"prompt_tokens": 3, '
            '"completion_tokens": 2, "finish_reason": {"type": "length"}, '
            '"output_token_logprobs": [[-0.20, 2, "X"], [-0.40, 4, "y"]]}}'
        ),
        "data: [DONE]",
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="hi", max_new_tokens=8, n=2, stream=True, logprobs=True):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    deltas = [c for c in chunks if not c.done]
    # Group logprobs by choice_index and verify per-choice slicing.
    lp_by_choice: dict[int, list] = {}
    for c in deltas:
        if c.logprobs:
            lp_by_choice.setdefault(c.choice_index, []).extend(c.logprobs)
    # Each candidate yields 2 logprob entries across the 2 events for that index.
    assert len(lp_by_choice.get(0, [])) == 2
    assert len(lp_by_choice.get(1, [])) == 2
    # No cross-candidate leakage: candidate 0's tokens are A/b, candidate 1's are X/y.
    tokens_0 = [e["token"] for e in lp_by_choice[0]]
    tokens_1 = [e["token"] for e in lp_by_choice[1]]
    assert tokens_0 == ["A", "b"]
    assert tokens_1 == ["X", "y"]
    # Request asked SGLang for logprobs.
    body = mock_async_client.return_value.stream.call_args.kwargs["json"]
    assert body["return_logprob"] is True


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_n_gt_one_non_streaming_emits_per_candidate_logprobs(mock_async_client: MagicMock, adapter) -> None:
    """Non-streaming ``n>1`` with ``logprobs=True`` populates each
    candidate's ``logprobs`` field from SGLang's
    ``meta_info.output_token_logprobs`` (was ``None`` pre-fix).
    """
    sglang_results = [
        {
            "text": "alpha",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
                "prompt_tokens": 3,
                "output_token_logprobs": [[-0.5, 1, "alpha"]],
            },
        },
        {
            "text": "beta",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
                "prompt_tokens": 3,
                "output_token_logprobs": [[-0.9, 2, "beta"]],
            },
        },
    ]
    resp = MagicMock()
    resp.json = MagicMock(return_value=sglang_results)
    resp.raise_for_status = MagicMock()
    client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
    resp.aread = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    client_instance.stream.return_value = resp
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="x", max_new_tokens=4, n=2, logprobs=True):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    term = chunks[0]
    assert term.candidates is not None
    cands = list(term.candidates)
    assert cands[0]["logprobs"] is not None
    assert cands[0]["logprobs"][0]["token"] == "alpha"  # noqa: S105
    assert cands[0]["logprobs"][0]["logprob"] == -0.5
    assert cands[1]["logprobs"] is not None
    assert cands[1]["logprobs"][0]["token"] == "beta"  # noqa: S105


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_n_gt_one_non_streaming_omits_logprobs_when_not_requested(
    mock_async_client: MagicMock, adapter
) -> None:
    """``logprobs=False`` (default): the worker does NOT surface ranking-
    only logprobs (used internally for best_of) on the candidate body —
    that would make ``logprobs: false`` requests sprout a logprobs payload.
    """
    sglang_results = [
        {
            "text": "a",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
                "prompt_tokens": 3,
                "output_token_logprobs": [[-0.5, 1, "a"]],
            },
        },
    ]
    sglang_results.append(
        {
            "text": "b",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
                "prompt_tokens": 3,
                "output_token_logprobs": [[-1.0, 2, "b"]],
            },
        }
    )
    resp = MagicMock()
    resp.json = MagicMock(return_value=sglang_results)
    resp.raise_for_status = MagicMock()
    client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
    resp.aread = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    client_instance.stream.return_value = resp
    mock_async_client.return_value = client_instance
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        # logprobs=False (default), but best_of>1 forces SGLang return_logprob.
        async for chunk in adapter.generate(prompt="x", max_new_tokens=4, n=1, best_of=2, logprobs=False):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    term = chunks[0]
    assert term.candidates is not None
    # Ranking is enabled (best_of>n triggers return_logprob), but the
    # candidate's surfaced logprobs field must remain None.
    assert term.candidates[0]["logprobs"] is None


def _lp(token: str, logprob: float) -> dict[str, Any]:
    return {"token": token, "logprob": logprob}


class TestGuardVerdictThreshold:
    """P(unsafe) threshold applied to a guard model's verdict (CHECK POLICY)."""

    def _chunk_logprobs(self, lp_yes: float, lp_no: float) -> tuple[dict[str, Any], ...]:
        # One content token with a top_logprobs distribution over Yes/No.
        return ({"token": "Yes", "logprob": lp_yes, "top_logprobs": [_lp("Yes", lp_yes), _lp("No", lp_no)]},)

    def test_p_unsafe_renormalises_over_yes_no(self) -> None:
        # Equal logprobs -> 0.5; Yes dominant -> >0.5.
        assert _p_unsafe_from_verdict_logprobs(self._chunk_logprobs(-0.5, -0.5)) == pytest.approx(0.5)
        assert _p_unsafe_from_verdict_logprobs(self._chunk_logprobs(-0.1, -2.3)) > 0.8

    def test_p_unsafe_none_without_verdict_tokens(self) -> None:
        lp = ({"token": "Maybe", "logprob": -0.1, "top_logprobs": [_lp("Maybe", -0.1), _lp("Perhaps", -1.0)]},)
        assert _p_unsafe_from_verdict_logprobs(lp) is None
        assert _p_unsafe_from_verdict_logprobs(()) is None

    def test_threshold_dial(self) -> None:
        # P(unsafe)=~0.5 (equal logprobs). threshold 0.5 -> Yes; 0.8 -> No.
        lp = self._chunk_logprobs(-0.5, -0.5)
        assert _thresholded_verdict(lp, {"threshold": 0.5}) == "Yes"
        assert _thresholded_verdict(lp, {"threshold": 0.8}) == "No"

    def test_threshold_high_recall_vs_precision(self) -> None:
        # A borderline-unsafe row (P(unsafe)=0.6): caught at 0.5, missed at 0.8.
        lp = self._chunk_logprobs(-0.51, -0.92)  # exp ratio ~0.6
        assert _thresholded_verdict(lp, {"threshold": 0.5}) == "Yes"
        assert _thresholded_verdict(lp, {"threshold": 0.8}) == "No"

    def test_no_threshold_or_no_logprobs_is_invalid(self) -> None:
        assert _thresholded_verdict(self._chunk_logprobs(-0.1, -2.0), {}) is None
        assert _thresholded_verdict((), {"threshold": 0.8}) is None

    def test_custom_labels(self) -> None:
        lp = self._chunk_logprobs(-0.1, -3.0)  # P(unsafe) high
        assert _thresholded_verdict(lp, {"threshold": 0.5, "positive": "UNSAFE", "negative": "SAFE"}) == "UNSAFE"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_generate_applies_guard_threshold_in_serving_path(mock_async_client: MagicMock) -> None:
    """End-to-end: the served verdict is the thresholded one, not the raw argmax.

    The model argmax-emits "Yes" (top_logprobs Yes -0.1 / No -2.0 → P(unsafe)≈0.87),
    but a high-precision guard threshold (0.95) flips the served verdict to "No".
    Proves the SGLang adapter applies the threshold in the real streaming path.
    """
    import json as _json

    guard_adapter = SGLangGenerationAdapter(
        model_name_or_path="ibm-granite/granite-guardian-3.0-2b",
        served_model_name="ibm-granite/granite-guardian-3.0-2b",
        guard={"threshold": 0.95},
    )
    lp = {
        "output_token_logprobs": [[-0.1, 100, "Yes"]],
        "output_top_logprobs": [[[-0.1, 100, "Yes"], [-2.0, 200, "No"]]],
    }
    sse_lines = [
        "data: " + _json.dumps({"text": "Yes", "meta_info": {"prompt_tokens": 5, **lp}}),
        "data: "
        + _json.dumps(
            {
                "text": "Yes",
                "meta_info": {"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": {"type": "stop"}, **lp},
            }
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    guard_adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in guard_adapter.generate(prompt="is this unsafe?", max_new_tokens=8):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    # The raw model token was "Yes"; the 0.95 threshold rewrites it to "No".
    assert chunks[0].text_delta == "No"
    assert chunks[0].is_first is True
    # The stale leading logprob entry (described the original "Yes") is dropped
    # so callers aren't handed token metadata inconsistent with the served verdict.
    assert chunks[0].logprobs is None


def test_generate_rejects_multi_candidate_for_guard_models() -> None:
    """A guard request with n>1 / best_of>1 must fail fast, not silently skip the
    threshold via the multi-candidate path.
    """
    guard_adapter = SGLangGenerationAdapter(
        model_name_or_path="ibm-granite/granite-guardian-3.0-2b",
        served_model_name="ibm-granite/granite-guardian-3.0-2b",
        guard={"threshold": 0.5},
    )
    guard_adapter._server_url = "http://localhost:30005"

    async def _drive(**kw: Any) -> None:
        await guard_adapter.generate(prompt="x", max_new_tokens=8, **kw).__anext__()

    for kw in ({"n": 2}, {"best_of": 2}):
        with pytest.raises(ValueError, match="single-candidate"):
            asyncio.run(_drive(**kw))


# -- Guard streaming: verdict-position scan + logprobs leakage (M4/H2) -------


def _sglang_token(token: str, logprob: float, tid: int = 0) -> list[Any]:
    """One SGLang flat token-logprob entry: ``[logprob, token_id, token_text]``."""
    return [logprob, tid, token]


def _guard_top(yes: float | None = None, no: float | None = None, *, filler: str = "x") -> list[list[Any]]:
    """Top-logprobs distribution for one position. With ``yes``/``no`` it carries
    a verdict; otherwise just a non-verdict filler token (no Yes/No).
    """
    if yes is None and no is None:
        return [_sglang_token(filler, -0.1, 1)]
    out: list[list[Any]] = []
    if yes is not None:
        out.append(_sglang_token("Yes", yes, 100))
    if no is not None:
        out.append(_sglang_token("No", no, 200))
    return out


def _guard_event(
    text: str,
    token_lp: list[list[Any]],
    top_lp: list[list[list[Any]]],
    *,
    terminal: bool = False,
) -> str:
    """A cumulative SGLang SSE event line carrying flat + top logprobs."""
    import json as _json

    meta: dict[str, Any] = {
        "prompt_tokens": 5,
        "output_token_logprobs": token_lp,
        "output_top_logprobs": top_lp,
    }
    if terminal:
        meta["completion_tokens"] = len(token_lp)
        meta["finish_reason"] = {"type": "stop"}
    return "data: " + _json.dumps({"text": text, "meta_info": meta})


def _guard_adapter(**guard_overrides: Any) -> SGLangGenerationAdapter:
    guard = {"threshold": 0.5, **guard_overrides}
    a = SGLangGenerationAdapter(
        model_name_or_path="ibm-granite/granite-guardian-3.0-2b",
        served_model_name="ibm-granite/granite-guardian-3.0-2b",
        guard=guard,
    )
    a._server_url = "http://localhost:30005"
    return a


def _drive_guard(adapter: SGLangGenerationAdapter, sse_lines: list[str], **gen_kwargs: Any) -> list[GenerationChunk]:
    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="is this unsafe?", max_new_tokens=8, **gen_kwargs):
            out.append(chunk)
        return out

    return asyncio.run(_collect())


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_leading_whitespace_then_verdict_applies_threshold(mock_async_client: MagicMock) -> None:
    """H2: a leading WHITESPACE token (no Yes/No at position 0) must not skip the
    threshold — the scan finds the verdict at position 1 and applies it.
    """
    # Position 0: whitespace, no verdict. Position 1: "Yes" with P(unsafe)≈0.5
    # over Yes/No (equal logprobs) → 0.5 threshold yields "Yes".
    sse_lines = [
        _guard_event(" ", [_sglang_token(" ", -0.2)], [_guard_top(filler=" ")]),
        _guard_event(
            " Yes",
            [_sglang_token(" ", -0.2), _sglang_token("Yes", -0.3, 100)],
            [_guard_top(filler=" "), _guard_top(yes=-0.5, no=-0.5)],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines)
    verdicts = [c for c in chunks if c.text_delta]
    assert verdicts, "expected at least one verdict chunk"
    assert verdicts[0].text_delta == "Yes"
    assert verdicts[0].is_first is True
    # No leading whitespace chunk leaked.
    assert all(c.text_delta in ("Yes", "") for c in chunks)


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_leading_punctuation_then_verdict_applies_threshold(mock_async_client: MagicMock) -> None:
    """H2: a leading PUNCTUATION/preamble token before the verdict is suppressed,
    and the threshold is applied to the verdict at the next position.
    """
    sse_lines = [
        _guard_event(":", [_sglang_token(":", -0.2)], [_guard_top(filler=":")]),
        _guard_event(
            ": Yes",
            [_sglang_token(":", -0.2), _sglang_token("Yes", -0.1, 100)],
            # P(unsafe) high (Yes -0.1 / No -2.0 ≈ 0.87). 0.95 threshold → "No".
            [_guard_top(filler=":"), _guard_top(yes=-0.1, no=-2.0)],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(threshold=0.95), sse_lines)
    verdicts = [c for c in chunks if c.text_delta]
    assert verdicts[0].text_delta == "No"  # threshold flips argmax "Yes" → "No"
    assert verdicts[0].is_first is True
    assert all(c.text_delta in ("No", "") for c in chunks)


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_verdict_in_second_position_applies_threshold(mock_async_client: MagicMock) -> None:
    """H2: verdict logprobs accumulated across chunks resolve from the SECOND
    position even when both leading tokens arrive in separate stream events.
    """
    sse_lines = [
        _guard_event("Pre", [_sglang_token("Pre", -0.2)], [_guard_top(filler="Pre")]),
        _guard_event(
            "PreYes",
            [_sglang_token("Pre", -0.2), _sglang_token("Yes", -0.3, 100)],
            [_guard_top(filler="Pre"), _guard_top(yes=-0.5, no=-0.5)],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines)
    verdicts = [c for c in chunks if c.text_delta]
    assert verdicts[0].text_delta == "Yes"
    assert verdicts[0].is_first is True


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_no_verdict_in_scan_window_fails_closed(mock_async_client: MagicMock) -> None:
    """A missing verdict must never be returned as a successful guard result."""
    sse_lines = [
        _guard_event("a", [_sglang_token("a", -0.1)], [_guard_top(filler="a")]),
        _guard_event(
            "ab",
            [_sglang_token("a", -0.1), _sglang_token("b", -0.1, 2)],
            [_guard_top(filler="a"), _guard_top(filler="b")],
        ),
        _guard_event(
            "abc",
            [_sglang_token("a", -0.1), _sglang_token("b", -0.1, 2), _sglang_token("c", -0.1, 3)],
            [_guard_top(filler="a"), _guard_top(filler="b"), _guard_top(filler="c")],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines)
    assert all(c.text_delta == "" for c in chunks)
    assert chunks[-1].done
    assert chunks[-1].finish_reason == "error"
    assert chunks[-1].error_code == "invalid_guard_verdict"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_no_client_logprobs_strips_forced_logprobs_on_success(mock_async_client: MagicMock) -> None:
    """M4: client did NOT request logprobs → every yielded chunk has
    logprobs=None even though the guard forced logprobs internally.
    """
    sse_lines = [
        _guard_event(" ", [_sglang_token(" ", -0.2)], [_guard_top(filler=" ")]),
        _guard_event(
            " Yes",
            [_sglang_token(" ", -0.2), _sglang_token("Yes", -0.3, 100)],
            [_guard_top(filler=" "), _guard_top(yes=-0.5, no=-0.5)],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines)  # logprobs not requested
    assert all(c.logprobs is None for c in chunks)
    assert next(c for c in chunks if c.text_delta).text_delta == "Yes"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_no_client_logprobs_strips_forced_logprobs_on_error(mock_async_client: MagicMock) -> None:
    """Unusable guard output exposes neither raw text nor forced logprobs."""
    sse_lines = [
        _guard_event("a", [_sglang_token("a", -0.1)], [_guard_top(filler="a")]),
        _guard_event(
            "ab",
            [_sglang_token("a", -0.1), _sglang_token("b", -0.1, 2)],
            [_guard_top(filler="a"), _guard_top(filler="b")],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines)
    assert all(c.logprobs is None for c in chunks)
    assert "".join(c.text_delta for c in chunks) == ""
    assert chunks[-1].error_code == "invalid_guard_verdict"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_client_logprobs_omitted_for_rewritten_verdict(mock_async_client: MagicMock) -> None:
    """Rewritten verdicts cannot expose logprobs for discarded tokens."""
    # Position 0: leading whitespace (no verdict). Position 1: the "Yes" verdict.
    # Both arrive in one event so the buffer holds two entries at resolution; the
    # consumed verdict entry (position 1) is dropped, the whitespace entry kept.
    sse_lines = [
        _guard_event(
            " Yes",
            [_sglang_token(" ", -0.2), _sglang_token("Yes", -0.3, 100)],
            [_guard_top(filler=" "), _guard_top(yes=-0.5, no=-0.5)],
            terminal=True,
        ),
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    chunks = _drive_guard(_guard_adapter(), sse_lines, logprobs=True, top_logprobs=20)
    verdict_chunk = next(c for c in chunks if c.text_delta)
    assert verdict_chunk.text_delta == "Yes"
    assert verdict_chunk.logprobs is None


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_non_guard_model_streaming_unaffected(mock_async_client: MagicMock, adapter) -> None:
    """A non-guard model takes the byte-for-byte unchanged path: no suppression,
    no stripping, logprobs only present when the client asked.
    """
    sse_lines = [
        'data: {"text": "Hello", "meta_info": {"prompt_tokens": 5}}',
        'data: {"text": "Hello!", "meta_info": {"prompt_tokens": 5, "completion_tokens": 2, "finish_reason": {"type": "stop"}}}',
        "data: [DONE]",
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(sse_lines))
    adapter._server_url = "http://localhost:30005"

    async def _collect() -> list[GenerationChunk]:
        out: list[GenerationChunk] = []
        async for chunk in adapter.generate(prompt="hi", max_new_tokens=8):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    # Leading "Hello" chunk is NOT suppressed; deltas reproduce the output.
    assert "".join(c.text_delta for c in chunks) == "Hello!"
    assert chunks[0].text_delta == "Hello"
    assert chunks[0].is_first is True
    # Client didn't request logprobs and the model isn't a guard → all None.
    assert all(c.logprobs is None for c in chunks)
    assert any(c.done for c in chunks)


def test_mm_process_config_compat_forwards_image_kwargs(tmp_path: Path) -> None:
    package_root = tmp_path / "fake-package"
    processors = package_root / "sglang" / "srt" / "multimodal" / "processors"
    processors.mkdir(parents=True)
    package_dirs = (
        package_root / "sglang",
        package_root / "sglang" / "srt",
        package_root / "sglang" / "srt" / "multimodal",
        processors,
    )
    for package in package_dirs:
        (package / "__init__.py").write_text("", encoding="utf-8")
    (processors / "base_processor.py").write_text(
        """class BaseMultimodalProcessor:
    def __init__(self):
        self.image_config = {"min_pixels": 65536, "max_pixels": 1003520}

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        return kwargs
""",
        encoding="utf-8",
    )

    compat_dir = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang/_compat"
    script = """import sitecustomize

sitecustomize._install_mm_process_config_compat()
sitecustomize._install_mm_process_config_compat()

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

sitecustomize._patch_base_processor_module(__import__(
    "sglang.srt.multimodal.processors.base_processor",
    fromlist=["BaseMultimodalProcessor"],
))
processor = BaseMultimodalProcessor()
assert processor.process_mm_data("x", images=[b"image"]) == {
    "images_kwargs": {"min_pixels": 65536, "max_pixels": 1003520}
}
assert processor.process_mm_data("x", images=None) == {}
assert not hasattr(BaseMultimodalProcessor.process_mm_data.__wrapped__, "__wrapped__")
print("mm-process-config-ready")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(compat_dir), str(package_root)))
    env["SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT"] = "1"

    completed = subprocess.run(  # noqa: S603 - executes the fixed local interpreter
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Error in sitecustomize" not in completed.stderr
    assert completed.stdout.strip() == "mm-process-config-ready"


def test_mm_process_config_compat_redacts_media_load_failures(tmp_path: Path) -> None:
    package_root = tmp_path / "fake-package"
    processors = package_root / "sglang" / "srt" / "multimodal" / "processors"
    processors.mkdir(parents=True)
    for package in (
        package_root / "sglang",
        package_root / "sglang" / "srt",
        package_root / "sglang" / "srt" / "multimodal",
        processors,
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (processors / "base_processor.py").write_text(
        """import enum


class Modality(enum.Enum):
    IMAGE = 1
    VIDEO = 2


class BaseMultimodalProcessor:
    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        return kwargs

    @classmethod
    def _load_single_item(cls, data, modality, frame_count_limit=None, audio_sample_rate=None, discard=True):
        try:
            if data == "ok":
                return "loaded"
            raise ValueError(f"cannot decode {data}")
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")
""",
        encoding="utf-8",
    )

    compat_dir = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang/_compat"
    script = """import sitecustomize
import traceback

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor, Modality

secret = "data:video/mp4;base64,PRIVATEPAYLOADPRIVATEPAYLOAD"
assert BaseMultimodalProcessor._load_single_item("ok", Modality.VIDEO) == "loaded"
try:
    BaseMultimodalProcessor._load_single_item(secret, Modality.VIDEO, None, None, True)
except ValueError as exc:
    rendered = "".join(traceback.format_exception(exc))
    assert "PRIVATEPAYLOAD" not in rendered, rendered
    assert str(exc) == "Error while loading VIDEO data (ValueError)", str(exc)
else:
    raise AssertionError("load failure was swallowed")
print("media-load-redaction-ready")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(compat_dir), str(package_root)))
    env["SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT"] = "1"

    completed = subprocess.run(  # noqa: S603 - executes the fixed local interpreter
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PRIVATEPAYLOAD" not in completed.stderr
    assert completed.stdout.strip() == "media-load-redaction-ready"


@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, True, None), (2, True, None), (2, False, None), (1, False, 2)]
)
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_backend_grammar_abort_never_emits_placeholder_or_usage(
    mock_async_client: MagicMock, adapter, n: int, stream: bool, best_of: int | None
) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": ["string", "null"]}},
        "required": ["value"],
        "additionalProperties": False,
    }
    abort = {
        "text": "[]",
        "meta_info": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "finish_reason": {
                "type": "abort",
                "status_code": 400,
                "message": "Invalid grammar request: private-schema-content",
            },
        },
    }
    client = _make_client_with_stream(_FakeStreamingResponse(["data: " + json.dumps(abort)]))
    response = MagicMock()
    response.json.return_value = [
        {"text": "valid", "meta_info": {"finish_reason": {"type": "stop"}}},
        abort,
    ]
    response.aread = AsyncMock()
    response.__aenter__ = AsyncMock(return_value=response)
    if not stream:
        client.stream.return_value = response
    mock_async_client.return_value = client
    adapter._server_url = "http://localhost:30005"
    chunks = []

    async def consume() -> None:
        async for chunk in adapter.generate(
            prompt="Extract the optional value.",
            max_new_tokens=64,
            n=n,
            stream=stream,
            best_of=best_of,
            grammar=GrammarSpec(kind="json_schema", value=schema),
        ):
            chunks.append(chunk)

    with pytest.raises(GenerationInvalidRequestError, match="rejected the requested json_schema grammar") as exc:
        asyncio.run(consume())
    assert exc.value.code == "invalid_request"
    assert exc.value.param == "grammar"
    assert "private-schema-content" not in str(exc.value)
    assert chunks == []


@pytest.mark.parametrize(
    "finish", ["abort", "error", "cancelled", {"type": "abort", "status_code": 500}, {"type": "unknown"}]
)
def test_chunk_translator_rejects_failed_or_unknown_terminal(finish: Any) -> None:
    with pytest.raises(GenerationError):
        _chunk_from_sglang_event(
            {"text": "[]", "meta_info": {"finish_reason": finish, "prompt_tokens": 1, "completion_tokens": 1}},
            previous_cumulative_text="",
            first_yield_done=False,
        )


@pytest.mark.parametrize("meta", [None, {}, []])
def test_chunk_translator_rejects_terminal_without_reason(meta: Any) -> None:
    with pytest.raises(GenerationError, match="without a finish reason"):
        _chunk_from_sglang_event(
            {"text": "[]", "finished": True, "meta_info": meta},
            previous_cumulative_text="",
            first_yield_done=False,
        )


def test_legacy_parser_rejects_backend_abort() -> None:
    with pytest.raises(GenerationError, match="aborted"):
        _parse_sglang_generate_response({"text": "[]", "meta_info": {"finish_reason": {"type": "abort"}}})


@pytest.mark.parametrize("finish", [{}, {"type": None}])
def test_chunk_translator_rejects_malformed_nonnull_finish_metadata(finish: Any) -> None:
    with pytest.raises(GenerationError, match="malformed finish reason"):
        _chunk_from_sglang_event(
            {"text": "[]", "meta_info": {"finish_reason": finish, "prompt_tokens": 1, "completion_tokens": 1}},
            previous_cumulative_text="",
            first_yield_done=False,
        )


@pytest.mark.parametrize("indexes", [[], [0], [0, 0], [0, 2], [0, True], [0, "1"], [1, None]])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_streaming_candidates_require_exact_distinct_terminals(mock_async_client: MagicMock, adapter, indexes) -> None:
    events = [
        {
            **({"index": index} if index is not None else {}),
            "text": "value",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 4,
                "completion_tokens": 1,
            },
        }
        for index in indexes
    ]
    mock_async_client.return_value = _make_client_with_stream(
        _FakeStreamingResponse([*("data: " + json.dumps(event) for event in events), "data: [DONE]"])
    )
    adapter._server_url = "http://localhost:30005"
    chunks = []

    async def consume() -> None:
        async for chunk in adapter.generate(prompt="Two values", max_new_tokens=8, n=2, stream=True):
            chunks.append(chunk)

    with pytest.raises(GenerationError):
        asyncio.run(consume())
    assert not any(chunk.done for chunk in chunks)
    assert all(chunk.prompt_tokens is None and chunk.completion_tokens is None for chunk in chunks)


@pytest.mark.parametrize(("count", "n", "best_of"), [(0, 2, None), (1, 2, None), (3, 2, None), (2, 1, 3)])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_buffered_candidates_require_exact_count_before_ranking(
    mock_async_client: MagicMock, adapter, count: int, n: int, best_of: int | None
) -> None:
    response = MagicMock()
    response.json.return_value = [
        {"text": "value", "meta_info": {"finish_reason": {"type": "stop"}}} for _ in range(count)
    ]
    client = _make_client_with_stream(_FakeStreamingResponse([]))
    response.aread = AsyncMock()
    response.__aenter__ = AsyncMock(return_value=response)
    client.stream.return_value = response
    mock_async_client.return_value = client
    adapter._server_url = "http://localhost:30005"
    chunks = []

    async def consume() -> None:
        async for chunk in adapter.generate(prompt="Two values", max_new_tokens=8, n=n, best_of=best_of):
            chunks.append(chunk)

    with pytest.raises(GenerationError, match="incorrect candidate count"):
        asyncio.run(consume())
    assert chunks == []


@pytest.mark.parametrize("text", ["", "No", "Maybe"])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_missing_verdict_distribution_is_typed_error(mock_async_client: MagicMock, text: str) -> None:
    lines = [_guard_event(text, [_sglang_token(text, -0.1)], [], terminal=True), "data: [DONE]"]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines, logprobs=True)
    assert len(chunks) == 1
    terminal = chunks[0]
    assert terminal.text_delta == ""
    assert terminal.logprobs is None
    assert terminal.finish_reason == "error"
    assert terminal.error_code == "invalid_guard_verdict"
    assert terminal.prompt_tokens == 5
    assert terminal.completion_tokens == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), 0.1, True, "-0.1"])
def test_guard_invalid_verdict_probability_cannot_be_safe(invalid: Any) -> None:
    logprobs = ({"token": "No", "logprob": -0.1, "top_logprobs": [_lp("Yes", invalid), _lp("No", -0.1)]},)
    assert _thresholded_verdict(logprobs, {"threshold": 0.5}) is None


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_guard_invalid_threshold_cannot_be_safe(threshold: Any) -> None:
    logprobs = ({"token": "Yes", "logprob": -0.5, "top_logprobs": [_lp("Yes", -0.5), _lp("No", -0.5)]},)
    assert _thresholded_verdict(logprobs, {"threshold": threshold}) is None


def test_guard_very_small_probabilities_are_normalized_without_underflow() -> None:
    logprobs = ({"token": "Yes", "logprob": -1000, "top_logprobs": [_lp("Yes", -1000), _lp("No", -1001)]},)
    assert _thresholded_verdict(logprobs, {"threshold": 0.5}) == "Yes"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_eos_with_verdict_alternatives_is_not_a_verdict(mock_async_client: MagicMock) -> None:
    lines = [_guard_event("", [_sglang_token("<|end_of_text|>", -0.1)], [_guard_top(yes=-4, no=-3)], terminal=True)]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines)
    assert chunks[-1].error_code == "invalid_guard_verdict"
    assert chunks[-1].text_delta == ""


@pytest.mark.parametrize("combined", [False, True])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_tail_cannot_change_thresholded_verdict(mock_async_client: MagicMock, combined: bool) -> None:
    lines = [
        _guard_event("Yes", [_sglang_token("Yes", -0.1)], [_guard_top(yes=-0.1, no=-3)]),
        _guard_event(
            "Yes because",
            [_sglang_token("Yes", -0.1), _sglang_token(" because", -0.2)],
            [_guard_top(yes=-0.1, no=-3), _guard_top(filler=" because")],
            terminal=True,
        ),
    ]
    if combined:
        lines = lines[-1:]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines, logprobs=True, top_logprobs=20)
    assert all(chunk.logprobs is None for chunk in chunks)
    assert "".join(c.text_delta for c in chunks) == "Yes"
    assert chunks[-1].done
    assert chunks[-1].completion_tokens == 2


@pytest.mark.parametrize("invalid", [None, "-0.1", True, False, float("nan"), float("inf"), float("-inf"), 0.1])
@pytest.mark.parametrize("position", ["sampled", "alternative"])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_malformed_wire_probability_fails_closed(
    mock_async_client: MagicMock, invalid: Any, position: str
) -> None:
    sampled = invalid if position == "sampled" else -0.1
    alternative = invalid if position == "alternative" else -0.1
    lines = [
        _guard_event(
            "No",
            [_sglang_token("No", sampled)],
            [[_sglang_token("Yes", alternative), _sglang_token("No", -0.1)]],
            terminal=True,
        )
    ]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines)
    assert chunks[-1].error_code == "invalid_guard_verdict"
    assert chunks[-1].finish_reason == "error"
    assert not any(c.text_delta for c in chunks)


@pytest.mark.parametrize("malformed_tail", ["sampled", "alternative", "missing_top", "malformed_top"])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_valid_verdict_ignores_malformed_trailing_metadata(
    mock_async_client: MagicMock, malformed_tail: str
) -> None:
    tokens = [_sglang_token("Yes", -0.1), _sglang_token(" because", -0.2)]
    top: list[Any] = [_guard_top(yes=-0.1, no=-3), _guard_top(filler=" because")]
    if malformed_tail == "sampled":
        tokens[1][0] = True
    elif malformed_tail == "alternative":
        top[1][0][0] = "-0.1"
    elif malformed_tail == "missing_top":
        top.pop()
    else:
        top[1] = None
    lines = [_guard_event("Yes because", tokens, top, terminal=True)]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines, logprobs=True)
    assert "".join(chunk.text_delta for chunk in chunks) == "Yes"
    assert chunks[-1].error_code is None
    assert all(chunk.logprobs is None for chunk in chunks)


@pytest.mark.parametrize(("sampled", "opposing"), [("Yes", "No"), ("No", "Yes")])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_uses_sampled_probability_when_absent_from_top_alternatives(
    mock_async_client: MagicMock, sampled: str, opposing: str
) -> None:
    lines = [_guard_event(sampled, [_sglang_token(sampled, -0.1)], [[_sglang_token(opposing, -3)]], terminal=True)]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines)
    assert "".join(chunk.text_delta for chunk in chunks) == sampled
    assert chunks[-1].error_code is None


@pytest.mark.parametrize("sampled", ["Yes", "No"])
@pytest.mark.parametrize("later_verdict", [False, True])
@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_missing_opposing_probability_fails_closed(
    mock_async_client: MagicMock, sampled: str, later_verdict: bool
) -> None:
    tokens = [_sglang_token(sampled, -0.1)]
    top = [[_sglang_token(sampled, -0.1)]]
    if later_verdict:
        tokens.append(_sglang_token("No", -0.1))
        top.append(_guard_top(yes=-3, no=-0.1))
    lines = [_guard_event(sampled, tokens, top, terminal=True)]
    mock_async_client.return_value = _make_client_with_stream(_FakeStreamingResponse(lines))
    chunks = _drive_guard(_guard_adapter(), lines)
    assert chunks[-1].error_code == "invalid_guard_verdict"
    assert not any(chunk.text_delta for chunk in chunks)


def test_guard_first_sampled_verdict_requires_complete_evidence() -> None:
    later_safe = {"token": "No", "logprob": -0.1, "top_logprobs": [_lp("Yes", -3), _lp("No", -0.1)]}
    incomplete_yes = {"token": "Yes", "logprob": -0.1, "top_logprobs": [_lp("Yes", -0.1)]}
    assert _p_unsafe_from_verdict_logprobs((incomplete_yes, later_safe)) is None
    missing_sampled = {"token": "Yes", "top_logprobs": [_lp("Yes", -0.1), _lp("No", -3)]}
    assert _p_unsafe_from_verdict_logprobs((missing_sampled, later_safe)) is None


_TYPE_DIAGNOSTIC = "Failed to compile json grammar: 'type' must be a string"
_TYPE_GRAMMAR = GrammarSpec(kind="json_schema", value={"type": ["string", "null"]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, False, None), (1, True, None), (2, True, None), (2, False, None), (1, False, 2)]
)
@pytest.mark.parametrize("transport", ["http", "abort"])
async def test_outlines_type_refusal_across_generate_paths(adapter, n, stream, best_of, transport) -> None:
    abort = {
        "text": "[]",
        "meta_info": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "finish_reason": {"type": "abort", "status_code": 400, "message": _TYPE_DIAGNOSTIC},
        },
    }

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if transport == "http":
            return httpx.Response(400, json={"error": {"message": _TYPE_DIAGNOSTIC}})
        if body["stream"]:
            return httpx.Response(200, text="data: " + json.dumps(abort) + "\n\n")
        return httpx.Response(200, json=[abort] * body["sampling_params"]["n"])

    chunks = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter._http_client = client
        adapter._server_url = "http://localhost:30005"
        with pytest.raises(GenerationInvalidRequestError) as error:
            async for chunk in adapter.generate(
                prompt="Optional value", max_new_tokens=8, n=n, stream=stream, best_of=best_of, grammar=_TYPE_GRAMMAR
            ):
                chunks.append(chunk)
    assert error.value.code == "invalid_request"
    assert error.value.param == "grammar"
    assert str(error.value) == OUTLINES_JSON_SCHEMA_TYPE_MESSAGE
    assert chunks == []


@pytest.mark.parametrize(
    ("backend", "grammar", "status", "message"),
    [
        ("xgrammar", _TYPE_GRAMMAR, 400, _TYPE_DIAGNOSTIC),
        (None, _TYPE_GRAMMAR, 400, _TYPE_DIAGNOSTIC),
        ("outlines", None, 400, _TYPE_DIAGNOSTIC),
        ("outlines", GrammarSpec(kind="regex", value=".*"), 400, _TYPE_DIAGNOSTIC),
        ("outlines", _TYPE_GRAMMAR, 500, _TYPE_DIAGNOSTIC),
        ("outlines", _TYPE_GRAMMAR, 400.0, _TYPE_DIAGNOSTIC),
        ("outlines", _TYPE_GRAMMAR, "400", _TYPE_DIAGNOSTIC),
        ("outlines", _TYPE_GRAMMAR, 400, _TYPE_DIAGNOSTIC + " secret"),
        ("outlines", _TYPE_GRAMMAR, 400, "secret " + _TYPE_DIAGNOSTIC),
        ("outlines", _TYPE_GRAMMAR, 400, [_TYPE_DIAGNOSTIC]),
        ("outlines", _TYPE_GRAMMAR, 400, "x" * 8192),
    ],
)
def test_abort_type_refusal_requires_exact_diagnostic_and_context(backend, grammar, status, message) -> None:
    event = {"meta_info": {"finish_reason": {"type": "abort", "status_code": status, "message": message}}}
    with pytest.raises(GenerationError) as error:
        _raise_for_sglang_event_error(event, grammar=grammar, grammar_backend=backend)
    assert str(error.value) != OUTLINES_JSON_SCHEMA_TYPE_MESSAGE
    assert "secret" not in str(error.value)
    assert _TYPE_DIAGNOSTIC not in str(error.value)


def test_unqualified_in_band_error_is_not_a_type_refusal() -> None:
    with pytest.raises(GenerationError) as error:
        _raise_for_sglang_event_error(
            {"error": {"message": _TYPE_DIAGNOSTIC}}, grammar=_TYPE_GRAMMAR, grammar_backend="outlines"
        )
    assert error.value.code == "inference_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"detail":"Failed to compile json grammar: \'type\' must be a string"}',
        b'{"error":{"message":"Failed to compile json grammar: \'type\' must be a string","message":"secret"}}',
        b'{"error":{"message":"secret","message":"Failed to compile json grammar: \'type\' must be a string"}}',
        json.dumps({"error": {"message": _TYPE_DIAGNOSTIC, "schema": "secret"}}).encode(),
        json.dumps({"error": {"message": _TYPE_DIAGNOSTIC + " secret"}}).encode(),
        json.dumps({"error": {"message": [_TYPE_DIAGNOSTIC]}}).encode(),
        json.dumps({"error": {"message": _TYPE_DIAGNOSTIC}}).encode() + b" " * 4096,
        b"[" * 2000 + b"]" * 2000,
    ],
)
async def test_http_type_refusal_rejects_unknown_malformed_or_oversized_body(body) -> None:
    response = httpx.Response(400, content=body, request=httpx.Request("POST", "http://localhost/generate"))
    with pytest.raises(httpx.HTTPStatusError) as error:
        await _raise_for_sglang_http_error(response, grammar=_TYPE_GRAMMAR, grammar_backend="outlines")
    assert "secret" not in str(error.value)
    assert _TYPE_DIAGNOSTIC not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "backend", "grammar"),
    [
        (500, "outlines", _TYPE_GRAMMAR),
        (502, "outlines", _TYPE_GRAMMAR),
        (400, "xgrammar", _TYPE_GRAMMAR),
        (400, "outlines", None),
        (400, "outlines", GrammarSpec(kind="regex", value=".*")),
    ],
)
async def test_http_type_refusal_requires_status_backend_and_grammar(status, backend, grammar) -> None:
    response = httpx.Response(
        status,
        json={"error": {"message": _TYPE_DIAGNOSTIC}},
        request=httpx.Request("POST", "http://localhost/generate"),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await _raise_for_sglang_http_error(response, grammar=grammar, grammar_backend=backend)


@pytest.mark.asyncio
@pytest.mark.parametrize(("modality", "param"), [("VIDEO", "videos"), ("IMAGE", "images")])
async def test_http_media_load_failure_is_invalid_request(modality: str, param: str) -> None:
    response = httpx.Response(
        400,
        json={"error": {"message": f"Error while loading {modality} data (ValueError)"}},
        request=httpx.Request("POST", "http://localhost/generate"),
    )
    with pytest.raises(GenerationInvalidRequestError) as error:
        await _raise_for_sglang_http_error(response, grammar=None, grammar_backend=None)
    assert error.value.param == param


@pytest.mark.parametrize(("modality", "param"), [("VIDEO", "videos"), ("IMAGE", "images")])
def test_stream_media_load_failure_is_invalid_request(modality: str, param: str) -> None:
    event = {"error": {"message": f"Error while loading {modality} data (RuntimeError)"}}
    with pytest.raises(GenerationInvalidRequestError) as error:
        _raise_for_sglang_event_error(event)
    assert error.value.param == param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Error while loading data data:video/mp4;base64,AAAA: boom"),
        (400, "Error while loading VIDEO data (ValueError) extra"),
        (400, "Error while loading AUDIO data (ValueError)"),
        (500, "Error while loading VIDEO data (ValueError)"),
    ],
)
async def test_media_load_mapping_requires_the_exact_hook_message(status: int, message: str) -> None:
    response = httpx.Response(
        status, json={"error": {"message": message}}, request=httpx.Request("POST", "http://localhost/generate")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await _raise_for_sglang_http_error(response, grammar=None, grammar_backend=None)
    if status == 400:
        with pytest.raises(GenerationError) as error:
            _raise_for_sglang_event_error({"error": {"message": message}})
        assert not isinstance(error.value, GenerationInvalidRequestError)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["oversized", "timeout", "cancel"])
@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, False, None), (2, True, None), (2, False, None), (1, False, 2)]
)
async def test_http_error_read_is_bounded_and_closes_stream(adapter, mode, n, stream, best_of) -> None:
    closed = asyncio.Event()
    started = asyncio.Event()
    reads = []

    class ErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            if mode == "oversized":
                reads.append(1)
                yield b" " * 4097
                raise AssertionError("oversized body must not be drained")
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(400, stream=ErrorStream()))
    ) as client:
        adapter._http_client = client
        adapter._server_url = "http://localhost:30005"
        iterator = adapter.generate(
            prompt="Optional value", max_new_tokens=8, grammar=_TYPE_GRAMMAR, n=n, stream=stream, best_of=best_of
        )
        task = asyncio.create_task(anext(iterator))
        await started.wait()
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await asyncio.wait_for(task, timeout=3)
        await iterator.aclose()
    assert closed.is_set()
    assert reads == ([1] if mode == "oversized" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding", ["gzip", "br", "GZIP", "IDENTITY", "identity, gzip", "identity, identity", "unknown", ""]
)
@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, False, None), (2, True, None), (2, False, None), (1, False, 2)]
)
async def test_encoded_http_error_is_not_read_or_decompressed(adapter, encoding, n, stream, best_of) -> None:
    closed = asyncio.Event()

    class EncodedErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("encoded error must not be read or decompressed")
            yield b""

        async def aclose(self):
            closed.set()

    def respond(_):
        return httpx.Response(400, headers={"Content-Encoding": encoding}, stream=EncodedErrorStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter._http_client = client
        adapter._server_url = "http://localhost:30005"
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in adapter.generate(
                prompt="Optional value", max_new_tokens=8, grammar=_TYPE_GRAMMAR, n=n, stream=stream, best_of=best_of
            ):
                raise AssertionError("error response must not yield a generation chunk")
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, False, None), (2, True, None), (2, False, None), (1, False, 2)]
)
async def test_duplicate_event_keys_raise_typed_error_before_output(adapter, n, stream, best_of) -> None:
    payload = '{"text":"secret","text":"[]","meta_info":{"finish_reason":{"type":"stop"},"prompt_tokens":1,"completion_tokens":1}}'

    def respond(request):
        body = json.loads(request.content)
        content = "data: " + payload + "\n\n" if body["stream"] else "[" + payload + "]"
        return httpx.Response(200, text=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter._http_client = client
        adapter._server_url = "http://localhost:30005"
        with pytest.raises(GenerationError, match="duplicate JSON keys") as error:
            async for _ in adapter.generate(
                prompt="Optional value", max_new_tokens=8, grammar=_TYPE_GRAMMAR, n=n, stream=stream, best_of=best_of
            ):
                raise AssertionError("duplicate keys must not yield text or usage")
    assert error.value.code == "inference_error"
    assert "secret" not in str(error.value)


def _tp_adapter(
    adapter_class: type[SGLangGenerationAdapter] = SGLangGenerationAdapter, **overrides: Any
) -> SGLangGenerationAdapter:
    kwargs: dict[str, Any] = {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct",
        "max_seq_length": 32768,
        "mem_fraction_static": 0.85,
        "served_model_name": "Qwen/Qwen3-4B-Instruct",
    }
    # A width above one requires a finite streaming read cap and a declared
    # startup budget, so supply both by default here and let the tests that are
    # about those rules set them explicitly.
    declared = overrides.get("tensor_parallel_size", 1)
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 1:
        kwargs["request_read_timeout_s"] = 120.0
        kwargs["startup_timeout_s"] = 600.0
    kwargs.update(overrides)
    return adapter_class(**kwargs)


def _launch(adapter: SGLangGenerationAdapter, mock_popen: MagicMock, device: str = "cuda:0") -> tuple[list[str], dict]:
    adapter.load(device)
    return mock_popen.call_args[0][0], mock_popen.call_args.kwargs["env"]


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_default_width_is_one_and_leaves_the_launch_unchanged(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """Every profile that does not ask for a width must launch exactly as before."""
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, env = _launch(_tp_adapter(), mock_popen)

    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert "--disable-piecewise-cuda-graph" not in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_declared_width_masks_the_whole_group_and_disables_piecewise_capture(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """Width four claims four ordered devices and turns off the capture path.

    SGLang's default piecewise capture is measured to hang at width two and to
    exhaust memory at width four on a memory fraction that serves at width one.
    """
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, env = _launch(_tp_adapter(tensor_parallel_size=4), mock_popen)

    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert "--disable-piecewise-cuda-graph" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_group_is_anchored_on_the_placement_device(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    _cmd, env = _launch(_tp_adapter(tensor_parallel_size=2), mock_popen, device="cuda:2")

    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_profile_environment_cannot_move_or_widen_the_device_claim(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """The mask is the registry's decision, not the profile's.

    Before the width was typed, a profile could set CUDA_VISIBLE_DEVICES in
    ``extra_env`` and silently serve on devices the registry had not reserved
    and was not accounting for. The mask is now written last.
    """
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter = _tp_adapter(
        tensor_parallel_size=2,
        extra_env={"CUDA_VISIBLE_DEVICES": "4,5,6,7", "SIE_UNRELATED": "kept"},
    )
    _cmd, env = _launch(adapter, mock_popen)

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["SIE_UNRELATED"] == "kept"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_capture_path_can_be_re_enabled_explicitly(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """A profile that has measured its own engine build may opt back in."""
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, _env = _launch(
        _tp_adapter(tensor_parallel_size=4, disable_piecewise_cuda_graph=False),
        mock_popen,
    )

    assert "--disable-piecewise-cuda-graph" not in cmd


@pytest.mark.parametrize("adapter_class", [SGLangCuda13Adapter, SGLangStrictThinkingAdapter, SGLangGemmaAdapter])
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_cuda13_adapters_turn_off_the_prefill_capture_path_by_its_current_name(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    adapter_class: type[SGLangGenerationAdapter],
) -> None:
    """The CUDA 13 engine removed the piecewise flag; its prefill-phase flag replaces it."""
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, _env = _launch(_tp_adapter(adapter_class, tensor_parallel_size=4), mock_popen)

    assert "--disable-prefill-cuda-graph" in cmd
    assert "--disable-piecewise-cuda-graph" not in cmd


@pytest.mark.parametrize("adapter_class", [SGLangCuda13Adapter, SGLangStrictThinkingAdapter, SGLangGemmaAdapter])
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"disable_piecewise_cuda_graph": False}, False),
        ({"disable_cuda_graph": True}, False),
    ],
)
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_cuda13_adapters_keep_the_prefill_graph_off_at_width_one(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    overrides: dict[str, Any],
    expected: bool,
    adapter_class: type[SGLangGenerationAdapter],
) -> None:
    """Every CUDA 13 profile was sized on an engine that never captured this graph.

    A profile may opt back in, and turning all CUDA graphs off needs no second flag.
    """
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, _env = _launch(_tp_adapter(adapter_class, **overrides), mock_popen)

    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert ("--disable-prefill-cuda-graph" in cmd) is expected


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_cuda13_speculative_guard_accepts_the_renamed_mamba_flag(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, env = _launch(
        _tp_adapter(
            SGLangCuda13Adapter,
            speculative={"enabled": True, "algorithm": "eagle"},
            extra_launch_args=["--mamba-radix-cache-strategy", "extra_buffer"],
        ),
        mock_popen,
    )

    assert cmd[cmd.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer"
    assert env["SGLANG_ENABLE_SPEC_V2"] == "1"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_cuda13_speculative_guard_rejects_the_retired_mamba_flag(
    mock_find_port: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """The CUDA 13 engine no longer declares the old spelling, so it cannot satisfy the guard."""
    mock_find_port.return_value = 30007
    adapter = _tp_adapter(
        SGLangCuda13Adapter,
        speculative={"enabled": True, "algorithm": "eagle"},
        extra_launch_args=["--mamba-scheduler-strategy", "extra_buffer"],
    )

    with pytest.raises(RuntimeError, match="'--mamba-radix-cache-strategy extra_buffer'"):
        adapter.load("cuda:0")

    mock_popen.assert_not_called()


@pytest.mark.parametrize("bad", [0, -1, 9, 2.0, "4", True, None])
def test_invalid_width_is_refused_at_construction(bad: Any) -> None:
    """A bad width fails where it is declared, not as an engine crash later."""
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        _tp_adapter(tensor_parallel_size=bad)


def test_width_one_is_the_only_width_that_needs_no_extra_devices() -> None:
    assert _server.resolve_device_group(0, 1) == [0]
    assert _server.resolve_device_group(3, 1) == [3]


def test_launcher_refuses_a_repeated_device() -> None:
    """Two ranks on one card deadlock rather than fail, so refuse it early."""
    with pytest.raises(ValueError, match="distinct"):
        _server.launch_sglang_server(["true"], device_indices=[0, 0], output_file=MagicMock())


def test_launcher_refuses_an_empty_group() -> None:
    with pytest.raises(ValueError, match="at least one device"):
        _server.launch_sglang_server(["true"], device_indices=[], output_file=MagicMock())


def test_width_above_one_requires_a_finite_streaming_read_cap() -> None:
    """A stalled collective emits no bytes and raises nothing.

    An unbounded read would hold the request open forever, and with it every
    accelerator in the group.
    """
    with pytest.raises(ValueError, match="request_read_timeout_s"):
        SGLangGenerationAdapter(
            model_name_or_path="Qwen/Qwen3-4B-Instruct",
            served_model_name="Qwen/Qwen3-4B-Instruct",
            tensor_parallel_size=4,
        )


def test_width_one_keeps_the_unbounded_default() -> None:
    """Single-device serving is unchanged: the worker owns request lifetime."""
    adapter = _tp_adapter()

    assert adapter._request_read_timeout_s is None


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "30"])
def test_invalid_read_cap_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="request_read_timeout_s"):
        _tp_adapter(request_read_timeout_s=bad)


@pytest.mark.parametrize("raw", ["inf", "+inf", "Infinity", "nan"])
def test_a_non_finite_inherited_read_cap_leaves_a_wide_profile_without_one(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """An infinite cap from the environment is no cap, so width two must still refuse it."""
    monkeypatch.setenv("SIE_SGLANG_GENERATE_READ_TIMEOUT_S", raw)
    monkeypatch.setattr(generation_module, "_GENERATE_READ_TIMEOUT_S", generation_module._resolve_read_timeout())

    assert generation_module._GENERATE_READ_TIMEOUT_S is None
    with pytest.raises(ValueError, match="must also declare a finite"):
        _tp_adapter(tensor_parallel_size=2, request_read_timeout_s=None)


def test_declared_read_cap_reaches_the_http_client() -> None:
    adapter = _tp_adapter(tensor_parallel_size=2, request_read_timeout_s=45.5)

    assert adapter._request_read_timeout_s == 45.5


def _clear_startup_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_server.STARTUP_TIMEOUT_ENV_VARS, _server.LIVENESS_BUDGET_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


def test_width_above_one_requires_a_declared_startup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup at a width is graph work per rank, so no inherited default describes it."""
    _clear_startup_environment(monkeypatch)

    with pytest.raises(ValueError, match="must also declare startup_timeout_s"):
        _tp_adapter(tensor_parallel_size=2, startup_timeout_s=None)


def test_an_environment_startup_budget_does_not_satisfy_a_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process-wide default is set for every model a worker loads, not for this profile."""
    _clear_startup_environment(monkeypatch)
    for name in _server.STARTUP_TIMEOUT_ENV_VARS:
        monkeypatch.setenv(name, "1200")

    with pytest.raises(ValueError, match="must also declare startup_timeout_s"):
        _tp_adapter(tensor_parallel_size=4, startup_timeout_s=None)


def test_width_one_keeps_inheriting_the_startup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("SIE_SGLANG_STARTUP_TIMEOUT_S", "1234")

    assert _tp_adapter()._startup_timeout_s == 1234
    assert _tp_adapter(tensor_parallel_size=1)._startup_timeout_s == 1234


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "900"])
def test_invalid_startup_budget_is_refused_at_construction(
    monkeypatch: pytest.MonkeyPatch, width: int, bad: Any
) -> None:
    """An unusable declared budget fails where it is declared instead of being replaced by a fallback."""
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("SIE_SGLANG_STARTUP_TIMEOUT_S", "1200")

    with pytest.raises(ValueError, match="startup_timeout_s"):
        _tp_adapter(tensor_parallel_size=width, startup_timeout_s=bad)


@patch("sie_server.adapters.sglang._server.wait_for_server", return_value=True)
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_declared_startup_budget_bounds_a_group_launch(
    mock_find_port: MagicMock,
    mock_popen: MagicMock,
    mock_wait_for_server: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("SIE_MODEL_READY_TIMEOUT_S", "300")
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))

    _launch(_tp_adapter(tensor_parallel_size=2, startup_timeout_s=1500), mock_popen)

    assert mock_wait_for_server.call_args.kwargs["timeout_s"] == 1500


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_a_dead_engine_fails_immediately_instead_of_timing_out_per_request(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """A crashed engine is a terminal answer, not a connection error forever.

    Measured on a four-rank group: a killed rank refused health in about 5 seconds while its
    process tree took about 67 seconds to exit. Before this check the adapter
    never noticed either, and kept accepting requests it could not serve.
    """
    mock_find_port.return_value = 30005
    process = MagicMock()
    process.poll.return_value = None
    mock_popen.return_value = process
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter = _tp_adapter(tensor_parallel_size=2)
    adapter.load("cuda:0")
    adapter._check_loaded()  # alive: no raise

    process.poll.return_value = -9

    with pytest.raises(RuntimeError, match="exited with code -9"):
        adapter._check_loaded()


def test_an_unloaded_adapter_still_reports_not_loaded_first() -> None:
    """The dead-engine check must not mask the ordinary not-loaded error."""
    adapter = _tp_adapter()

    with pytest.raises(RuntimeError, match=re.escape(ERR_NOT_LOADED)):
        adapter._check_loaded()


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_width_one_reserves_no_collective_port(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """Single-device serving does not rendezvous, so nothing extra is taken."""
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter = _tp_adapter()
    cmd, _env = _launch(adapter, mock_popen)

    assert adapter._nccl_port is None
    assert "--nccl-port" not in cmd
    assert "--watchdog-timeout" not in cmd


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_a_group_reserves_its_own_collective_port(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """The engine otherwise picks a random port and two groups can collide.

    A collision does not fail, it hangs in rendezvous, which is the failure
    shape hardest to attribute.
    """
    mock_find_port.side_effect = [30005, 30207]
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    adapter = _tp_adapter(tensor_parallel_size=2)
    cmd, _env = _launch(adapter, mock_popen)

    assert adapter._nccl_port == 30207
    assert cmd[cmd.index("--nccl-port") + 1] == "30207"
    # Reserved from a span kept clear of the HTTP ports.
    assert mock_find_port.call_args_list[-1].args[0] == _server.NCCL_BASE_PORT


@patch("sie_server.adapters.sglang._server.os.getpgid")
@patch("sie_server.adapters.sglang._server.os.killpg")
def test_unload_returns_the_collective_port(
    mock_killpg: MagicMock,
    mock_getpgid: MagicMock,
) -> None:
    """Both spans exhaust under reload churn if either is leaked."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.wait.return_value = None
    mock_getpgid.return_value = 12345

    adapter = _tp_adapter(tensor_parallel_size=2)
    adapter._process = mock_process
    adapter._server_url = "http://localhost:30005"
    adapter._device = "cuda:0"
    adapter._port = 30005
    # What ``load`` sets when it takes the rendezvous port from the span.
    adapter._nccl_port = 30207
    adapter._reserved_nccl_port = 30207
    adapter._output_file = _server.open_output_log(prefix="sie_test_sglang_")

    with patch("sie_server.adapters.sglang._server.release_port") as mock_release:
        adapter.unload()

    released = [call.args[0] for call in mock_release.call_args_list]
    assert released == [30005, 30207]
    assert adapter._nccl_port is None
    assert adapter._reserved_nccl_port is None


_DECLARED_NCCL_PORT = 30499


def _reserving_http_port(port: int = 30005) -> Any:
    """A ``find_free_port`` that reserves what it hands out, as the real one does."""

    def reserve(start_port: int = _server.BASE_PORT) -> int:
        _server._RESERVED_PORTS.add(port)
        return port

    return reserve


@patch("sie_server.adapters.sglang._server.os.getpgid", return_value=12345)
@patch("sie_server.adapters.sglang._server.os.killpg")
@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
def test_a_declared_collective_port_is_reserved_while_loaded_and_returned_on_unload(
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
    mock_killpg: MagicMock,
    mock_getpgid: MagicMock,
) -> None:
    mock_popen.return_value = MagicMock(pid=12345, poll=MagicMock(return_value=None), wait=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)
    adapter = _tp_adapter(tensor_parallel_size=2, nccl_port=_DECLARED_NCCL_PORT)

    try:
        with patch("sie_server.adapters.sglang._server.find_free_port", side_effect=_reserving_http_port()):
            cmd, _env = _launch(adapter, mock_popen)

        assert cmd[cmd.index("--nccl-port") + 1] == str(_DECLARED_NCCL_PORT)
        assert _DECLARED_NCCL_PORT in _server._RESERVED_PORTS

        adapter.unload()

        assert _DECLARED_NCCL_PORT not in _server._RESERVED_PORTS
        assert 30005 not in _server._RESERVED_PORTS
    finally:
        _server.release_port(_DECLARED_NCCL_PORT)
        _server.release_port(30005)


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
def test_a_declared_port_another_model_holds_is_refused_without_leaking(
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """Two groups sharing a rendezvous port hang rather than fail."""
    mock_requests_get.return_value = MagicMock(status_code=200)
    adapter = _tp_adapter(tensor_parallel_size=2, nccl_port=_DECLARED_NCCL_PORT)
    _server._RESERVED_PORTS.add(_DECLARED_NCCL_PORT)

    try:
        with (
            patch("sie_server.adapters.sglang._server.find_free_port", side_effect=_reserving_http_port()),
            pytest.raises(RuntimeError, match="already reserved by another model"),
        ):
            adapter.load("cuda:0")

        mock_popen.assert_not_called()
        assert 30005 not in _server._RESERVED_PORTS
        assert _DECLARED_NCCL_PORT in _server._RESERVED_PORTS, "the other model's reservation must survive"
    finally:
        _server.release_port(_DECLARED_NCCL_PORT)
        _server.release_port(30005)


def test_a_declared_collective_port_at_width_one_is_refused() -> None:
    """A single rank never rendezvouses, so the port would be silently ignored."""
    with pytest.raises(ValueError, match="nccl_port applies only above"):
        _tp_adapter(nccl_port=_DECLARED_NCCL_PORT)


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_a_declared_watchdog_bound_reaches_the_engine(
    mock_find_port: MagicMock,
    mock_requests_get: MagicMock,
    mock_popen: MagicMock,
) -> None:
    """A wedged forward batch must crash the engine inside a budget SIE owns."""
    mock_find_port.side_effect = [30005, 30207]
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    cmd, _env = _launch(_tp_adapter(tensor_parallel_size=2, watchdog_timeout_s=90.0), mock_popen)

    assert cmd[cmd.index("--watchdog-timeout") + 1] == "90.0"


@pytest.mark.parametrize("bad", [0, -1, float("inf"), True, "90"])
def test_an_invalid_watchdog_bound_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="watchdog_timeout_s"):
        _tp_adapter(watchdog_timeout_s=bad)


@pytest.mark.parametrize("bad", [0, 80, 70000, True, "30207"])
def test_an_invalid_collective_port_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="nccl_port must be"):
        _tp_adapter(tensor_parallel_size=2, nccl_port=bad)
