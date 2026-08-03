"""Tests for MLXGenerationAdapter (Apple-Silicon generation via mlx_lm.server).

These run on any platform (Linux CI included) — they never spawn a real
subprocess. They mock the httpx layer to exercise the OpenAI ``/v1/completions``
SSE parsing, and cover the device-swap factory, the kwarg translation, and the
``mlx_repo``-required guard.
"""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import patch

import pytest
from sie_server.adapters._generation_base import collect_generation
from sie_server.adapters.mlx.generation import MLXGenerationAdapter
from sie_server.adapters.sglang.generation import (
    SGLangGenerationAdapter,
    _translate_to_mlx_kwargs,
)


@pytest.fixture
def adapter() -> MLXGenerationAdapter:
    a = MLXGenerationAdapter(model_name_or_path="Qwen/Qwen3.5-4B", mlx_repo="mlx-community/Qwen3.5-4B-4bit")
    # Pretend it loaded (skip the real subprocess) so generate()'s loaded-check passes.
    a._server_url = "http://127.0.0.1:30200"
    return a


# -- Contract -----------------------------------------------------------------


def test_contract_flags_and_spec() -> None:
    assert MLXGenerationAdapter.requires_main_thread is False
    assert MLXGenerationAdapter.manages_own_load_timeout is True
    assert "tokens" in MLXGenerationAdapter.spec.outputs


def test_capabilities(adapter: MLXGenerationAdapter) -> None:
    caps = adapter.capabilities
    assert caps.inputs == ["text"]
    assert caps.outputs == ["tokens"]


# -- Device swap + kwarg translation -----------------------------------------


def test_create_for_device_cuda_keeps_sglang() -> None:
    a = SGLangGenerationAdapter.create_for_device("cuda:0", model_name_or_path="Qwen/Qwen3.5-4B")
    assert isinstance(a, SGLangGenerationAdapter)


def test_create_for_device_mps_swaps_to_mlx() -> None:
    a = SGLangGenerationAdapter.create_for_device(
        "mps",
        model_name_or_path="Qwen/Qwen3.5-4B",
        mlx_repo="mlx-community/Qwen3.5-4B-4bit",
        mem_fraction_static=0.85,
        speculative={"enabled": True},
    )
    assert isinstance(a, MLXGenerationAdapter)
    assert a.mlx_repo == "mlx-community/Qwen3.5-4B-4bit"


def test_translate_drops_cuda_only_and_keeps_mlx_kwargs() -> None:
    out = _translate_to_mlx_kwargs(
        {
            "model_name_or_path": "Qwen/Qwen3.5-4B",
            "mlx_repo": "mlx-community/Qwen3.5-4B-4bit",
            "max_seq_length": 8192,
            "default_sampling": {"temperature": 0.7},
            "stop_tokens": ["<|im_end|>"],
            "served_model_name": "Qwen/Qwen3.5-4B",
            # CUDA/SGLang-only — must be dropped:
            "mem_fraction_static": 0.9,
            "speculative": {"enabled": True},
            "attention_backend": "triton",
            "grammar_backend": "outlines",
            "tool_call_parser": "qwen3_coder",
            "lora_paths": {"a": "b"},
            "compute_precision": "bfloat16",
        }
    )
    assert out["mlx_repo"] == "mlx-community/Qwen3.5-4B-4bit"
    assert out["max_seq_length"] == 8192
    assert out["default_sampling"] == {"temperature": 0.7}
    for dropped in (
        "mem_fraction_static",
        "speculative",
        "attention_backend",
        "grammar_backend",
        "tool_call_parser",
        "lora_paths",
        "compute_precision",
    ):
        assert dropped not in out


# -- mlx_repo guard -----------------------------------------------------------


def test_load_without_mlx_repo_fails_fast() -> None:
    a = MLXGenerationAdapter(model_name_or_path="Qwen/Qwen3.6-27B")  # no mlx_repo
    assert a.mlx_repo is None
    with pytest.raises(RuntimeError, match="mlx_repo"):
        a.load("mps")


def test_load_aborts_when_warmup_fails(adapter: MLXGenerationAdapter) -> None:
    # Health passes but the warmup completion fails → load() must treat it as a load failure
    # (deterministic readiness): raise and reset state instead of reporting "ready".
    adapter._process = None  # not loaded yet (the fixture only pre-set _server_url)
    fake_log = type("L", (), {"name": "/tmp/mlx_warmup_test_does_not_exist.log", "close": lambda self: None})()  # noqa: S108 — fake path; unlink is suppressed
    with (
        patch("sie_server.adapters.mlx.generation._server.mlx_lm_available", return_value=True),
        patch("sie_server.adapters.mlx.generation._server.find_free_port", return_value=30210),
        patch("sie_server.adapters.mlx.generation._server.open_output_log", return_value=fake_log),
        patch("sie_server.adapters.mlx.generation._server.launch_mlx_server", return_value=object()),
        patch("sie_server.adapters.mlx.generation._server.wait_for_server", return_value=True),
        patch("sie_server.adapters.mlx.generation._server.warmup_model", return_value=False),
        patch("sie_server.adapters.mlx.generation._server.terminate_process") as term,
        pytest.raises(RuntimeError, match="warm up"),
    ):
        adapter.load("mps")
    assert adapter._server_url is None
    assert adapter._process is None
    term.assert_called_once()


# -- generate(): OpenAI /v1/completions SSE parsing ---------------------------


class _FakeResponse:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self.status_code = status
        self._lines = lines

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"error body"

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self._lines = lines
        self._status = status

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False

    def stream(self, _method: str, _url: str, **_kw: Any) -> _FakeResponse:
        return _FakeResponse(self._lines, self._status)


# Mirrors the real mlx_lm.server /v1/completions stream (verified live): a
# ``: keepalive`` SSE comment, incremental ``choices[0].text`` deltas, a terminal
# choice with finish_reason + empty text, then a usage-only event, then [DONE].
_SSE_LINES = [
    ": keepalive 1/1",
    "",
    'data: {"object": "text_completion", "choices": [{"index": 0, "finish_reason": null, "text": "Hello"}]}',
    "",
    'data: {"object": "text_completion", "choices": [{"index": 0, "finish_reason": null, "text": " world"}]}',
    "",
    'data: {"object": "text_completion", "choices": [{"index": 0, "finish_reason": "length", "text": ""}]}',
    "",
    'data: {"object": "chat.completion", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}',
    "",
    "data: [DONE]",
    "",
]


async def test_generate_parses_sse(adapter: MLXGenerationAdapter) -> None:
    with patch(
        "sie_server.adapters.mlx.generation.httpx.AsyncClient",
        return_value=_FakeClient(_SSE_LINES),
    ):
        chunks = [c async for c in adapter.generate(prompt="hi", max_new_tokens=8, temperature=0.0)]

    deltas = [c for c in chunks if not c.done]
    terminal = [c for c in chunks if c.done]
    assert "".join(c.text_delta for c in deltas) == "Hello world"
    assert deltas[0].is_first is True
    assert len(terminal) == 1
    assert terminal[0].finish_reason == "length"
    assert terminal[0].prompt_tokens == 3
    assert terminal[0].completion_tokens == 2


async def test_generate_collects_to_result(adapter: MLXGenerationAdapter) -> None:
    with patch(
        "sie_server.adapters.mlx.generation.httpx.AsyncClient",
        return_value=_FakeClient(_SSE_LINES),
    ):
        result = await collect_generation(adapter.generate(prompt="hi", max_new_tokens=8))
    assert result.text == "Hello world"
    assert result.finish_reason == "length"
    assert result.completion_tokens == 2


async def test_generate_rejects_images(adapter: MLXGenerationAdapter) -> None:
    with pytest.raises(ValueError, match="vision"):
        gen = adapter.generate(prompt="hi", max_new_tokens=8, images=[{"data": b"x", "format": "png"}])
        await gen.__anext__()


async def test_generate_rejects_unsupported_min_new_tokens(adapter: MLXGenerationAdapter) -> None:
    with pytest.raises(ValueError, match="min_new_tokens is not supported"):
        gen = adapter.generate(prompt="hi", max_new_tokens=8, min_new_tokens=2)
        await gen.__anext__()


async def test_generate_unloaded_raises() -> None:
    a = MLXGenerationAdapter(model_name_or_path="m", mlx_repo="r")  # not loaded (no _server_url)
    with pytest.raises(RuntimeError):
        gen = a.generate(prompt="hi", max_new_tokens=8)
        await gen.__anext__()


def test_unload_terminates_subprocess(adapter: MLXGenerationAdapter) -> None:
    sentinel = object()
    adapter._process = sentinel  # type: ignore[assignment]
    with patch("sie_server.adapters.mlx.generation._server.terminate_process") as term:
        adapter.unload()
    term.assert_called_once_with(sentinel)
    assert adapter._process is None
    assert adapter._server_url is None


def test_memory_footprint_zero(adapter: MLXGenerationAdapter) -> None:
    assert adapter.memory_footprint() == 0


def test_build_sampling_body_merges_defaults() -> None:
    a = MLXGenerationAdapter(
        model_name_or_path="m",
        mlx_repo="repo",
        # presence_penalty mirrors the curated Qwen3.5-4B profile — it must NOT reach
        # mlx_lm.server (it's a CUDA/SGLang-only knob).
        default_sampling={"top_p": 0.8, "temperature": 0.7, "presence_penalty": 1.5},
        stop_tokens=["<|im_end|>"],
    )
    body = a._build_sampling_body(
        "prompt", max_new_tokens=16, temperature=0.0, top_p=1.0, top_k=None, stop=["X"], seed=None
    )
    assert body["model"] == "repo"
    assert body["max_tokens"] == 16
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    # explicit request values win over defaults; stop_tokens merged in
    assert body["temperature"] == 0.0
    assert "X" in body["stop"]
    assert "<|im_end|>" in body["stop"]
    # SGLang/OpenAI-only default sampling keys are filtered out for the MLX child.
    assert "presence_penalty" not in body


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (-1, (1 << 64) - 1),
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
    ],
)
def test_build_sampling_body_converts_signed_seed_for_mlx(seed: int, expected: int) -> None:
    adapter = MLXGenerationAdapter(model_name_or_path="m", mlx_repo="repo")
    body = adapter._build_sampling_body(
        "prompt",
        max_new_tokens=16,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        stop=None,
        seed=seed,
    )
    assert body["seed"] == expected


def test_build_sampling_body_converts_default_seed_for_mlx() -> None:
    adapter = MLXGenerationAdapter(
        model_name_or_path="m",
        mlx_repo="repo",
        default_sampling={"seed": -1},
    )
    body = adapter._build_sampling_body(
        "prompt",
        max_new_tokens=16,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        stop=None,
        seed=None,
    )
    assert body["seed"] == (1 << 64) - 1


@pytest.mark.parametrize("seed", [True, 1 << 63, -(1 << 63) - 1])
def test_build_sampling_body_rejects_invalid_seed_for_mlx(seed: int) -> None:
    adapter = MLXGenerationAdapter(model_name_or_path="m", mlx_repo="repo")
    with pytest.raises(ValueError, match="signed 64-bit"):
        adapter._build_sampling_body(
            "prompt",
            max_new_tokens=16,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            stop=None,
            seed=seed,
        )
