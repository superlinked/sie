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
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.adapters._generation_base import GenerationChunk, collect_generation
from sie_server.adapters.sglang.generation import (
    SGLangGenerationAdapter,
    _chunk_from_sglang_event,
    _encode_image_data,
    _mamba_scheduler_strategy_value,
    _p_unsafe_from_verdict_logprobs,
    _parse_sglang_generate_response,
    _thresholded_verdict,
)
from sie_server.types.grammar import GrammarSpec
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


def test_load_required_memory_bytes_uses_mem_fraction_static() -> None:
    gb = 1024**3
    adapter = SGLangGenerationAdapter("test-model", mem_fraction_static=0.8)

    assert adapter.load_required_memory_bytes(device_type="cuda", device_total_bytes=10 * gb) == 9 * gb
    assert adapter.load_required_memory_bytes(device_type="cpu", device_total_bytes=10 * gb) is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # two-token form (Qwen3.5-4B YAML shape) → exact value
        (["--mamba-scheduler-strategy", "extra_buffer"], "extra_buffer"),
        (["--mamba-scheduler-strategy", "default"], "default"),
        # flag=value form
        (["--mamba-scheduler-strategy=extra_buffer"], "extra_buffer"),
        (["--mamba-scheduler-strategy=default"], "default"),
        # absent → None
        (["--disable-overlap-schedule"], None),
        ([], None),
        # trailing flag with no value → None (no crash)
        (["--mamba-scheduler-strategy"], None),
        # a stray ``extra_buffer`` token elsewhere must NOT be read as the value
        (["--some-other-flag", "extra_buffer", "--mamba-scheduler-strategy", "default"], "default"),
        # last occurrence wins (argparse semantics)
        (
            ["--mamba-scheduler-strategy", "default", "--mamba-scheduler-strategy", "extra_buffer"],
            "extra_buffer",
        ),
    ],
)
def test_mamba_scheduler_strategy_value(args: list[str], expected: str | None) -> None:
    assert _mamba_scheduler_strategy_value(args) == expected


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
def test_generate_collect_into_result(mock_async_client: MagicMock, adapter) -> None:
    sse_lines = [
        'data: {"text": "abc", "meta_info": {"prompt_tokens": 1}}',
        'data: {"text": "abcdef", "meta_info": {"prompt_tokens": 1, "completion_tokens": 2, "finish_reason": {"type": "length"}}}',
    ]
    stream = _FakeStreamingResponse(sse_lines)
    mock_async_client.return_value = _make_client_with_stream(stream)
    adapter._server_url = "http://localhost:30005"

    result = asyncio.run(
        collect_generation(adapter.generate(prompt="Hi", max_new_tokens=64, temperature=0.7, top_p=0.9))
    )
    assert result.text == "abcdef"
    assert result.finish_reason == "length"
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 2


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
    client_instance.post = AsyncMock(return_value=resp)
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
    body = client_instance.post.call_args.kwargs["json"]
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
    client_instance.post = AsyncMock(return_value=resp)
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
    body = client_instance.post.call_args.kwargs["json"]
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
    assert "rid" in body  # cancellation handle


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


def test_parse_response_missing_meta_defaults_to_stop() -> None:
    result = _parse_sglang_generate_response({"text": "xyz"})
    assert result.text == "xyz"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


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
    client_instance.post = AsyncMock(return_value=resp)
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
    resp = MagicMock()
    resp.json = MagicMock(return_value=sglang_results)
    resp.raise_for_status = MagicMock()
    client_instance = _make_client_with_stream(_FakeStreamingResponse([]))
    client_instance.post = AsyncMock(return_value=resp)
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

    def test_no_threshold_or_no_logprobs_leaves_output(self) -> None:
        # Missing/invalid threshold or absent verdict logprobs -> None (raw kept).
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
def test_guard_no_verdict_in_scan_window_falls_back_to_raw(mock_async_client: MagicMock) -> None:
    """H2 fallback: no Yes/No anywhere in the first N positions → the raw buffered
    output is preserved (never dropped or hung).
    """
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
    # Raw output preserved: the concatenated deltas reproduce the model output.
    assert "".join(c.text_delta for c in chunks) == "abc"
    assert any(c.done for c in chunks)


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
def test_guard_no_client_logprobs_strips_forced_logprobs_on_fallback(mock_async_client: MagicMock) -> None:
    """M4: on the fallback path (no verdict parsed) the forced logprobs are still
    stripped when the client did not request them.
    """
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
    assert "".join(c.text_delta for c in chunks) == "ab"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_guard_client_logprobs_preserved_minus_verdict_entry(mock_async_client: MagicMock) -> None:
    """M4: client DID request logprobs → logprobs preserved, minus the consumed
    verdict entry (the position that produced the verdict).
    """
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
    # The consumed verdict entry (position 1) is dropped; the leading whitespace
    # entry remains so callers still get the surrounding token metadata.
    assert verdict_chunk.logprobs is not None
    assert [e["token"] for e in verdict_chunk.logprobs] == [" "]


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
