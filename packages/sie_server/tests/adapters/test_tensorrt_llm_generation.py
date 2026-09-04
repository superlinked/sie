from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from sie_server.adapters._generation_base import (
    GenerationDrainingError,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.tensorrt_llm import _server
from sie_server.adapters.tensorrt_llm.generation import TensorRTLLMGenerationAdapter, _completion_logprobs
from transformers import AutoTokenizer

_REVISION = "a" * 40


class _FixtureTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((prompt, add_special_tokens))
        return list(range(len(prompt.split()) + int(add_special_tokens)))


def _adapter(**kwargs: Any) -> TensorRTLLMGenerationAdapter:
    adapter = TensorRTLLMGenerationAdapter(
        "fixture/t5",
        revision=_REVISION,
        max_seq_length=256,
        max_input_len=128,
        max_batch_size=2,
        max_num_tokens=256,
        encoder_max_num_tokens=256,
        decoder_cuda_graph_batch_sizes=[1, 2],
        encoder_cuda_graph_batch_sizes=[1, 2],
        encoder_cuda_graph_num_tokens=[64, 128],
        encoder_cuda_graph_seq_lens=[64, 128],
        **kwargs,
    )
    adapter._server_url = "http://127.0.0.1:30200"
    adapter._process = object()  # type: ignore[assignment]
    adapter._tokenizer = _FixtureTokenizer()
    return adapter


async def _collect(iterator: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in iterator]


def _sse(*events: object) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_context_length_accounting_is_independent() -> None:
    assert TensorRTLLMGenerationAdapter.context_length_accounting == "independent"


def test_runtime_config_requires_dual_bf16_and_bounded_encoder_decoder_profile() -> None:
    config = _adapter().runtime_config

    assert config == {
        "attn_backend": "TRTLLM",
        "dtype": "bfloat16",
        "model_kwargs": {"torch_dtype": "bfloat16"},
        "enable_chunked_prefill": False,
        "max_batch_size": 2,
        "max_beam_width": 1,
        "max_input_len": 128,
        "max_num_tokens": 256,
        "max_seq_len": 256,
        "encoder_max_batch_size": 2,
        "encoder_max_num_tokens": 256,
        "cuda_graph_config": {"batch_sizes": [1, 2], "enable_padding": True},
        "encoder_cuda_graph_config": {
            "batch_sizes": [1, 2],
            "num_tokens": [64, 128],
            "seq_lens": [64, 128],
            "enable_padding": True,
        },
        "enable_encoder_decoder_mixed_cuda_graph": True,
        "disable_overlap_scheduler": False,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.85,
            "cross_kv_cache_fraction": 0.5,
            "use_kv_cache_manager_v2": False,
        },
        "scheduler_config": {"use_python_scheduler": True},
    }


def test_runtime_config_supports_fp32_with_encoder_graphs_only() -> None:
    config = _adapter(
        compute_precision="float32",
        enable_decoder_cuda_graphs=False,
        enable_encoder_decoder_mixed_cuda_graph=False,
    ).runtime_config

    assert config["dtype"] == "float32"
    assert config["model_kwargs"] == {"torch_dtype": "float32"}
    assert config["cuda_graph_config"] is None
    assert config["encoder_cuda_graph_config"] == {
        "batch_sizes": [1, 2],
        "num_tokens": [64, 128],
        "seq_lens": [64, 128],
        "enable_padding": True,
    }
    assert config["enable_encoder_decoder_mixed_cuda_graph"] is False


def test_runtime_config_supports_fp32_without_cuda_graphs() -> None:
    config = _adapter(
        compute_precision="float32",
        enable_decoder_cuda_graphs=False,
        enable_encoder_cuda_graphs=False,
        enable_encoder_decoder_mixed_cuda_graph=False,
    ).runtime_config

    assert config["dtype"] == "float32"
    assert config["model_kwargs"] == {"torch_dtype": "float32"}
    assert config["cuda_graph_config"] is None
    assert config["encoder_cuda_graph_config"] is None
    assert config["enable_encoder_decoder_mixed_cuda_graph"] is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"enable_decoder_cuda_graphs": False, "enable_encoder_decoder_mixed_cuda_graph": False},
            {"cuda_graph_config": None},
        ),
        (
            {"enable_encoder_cuda_graphs": False, "enable_encoder_decoder_mixed_cuda_graph": False},
            {"encoder_cuda_graph_config": None},
        ),
        (
            {"decoder_cuda_graph_enable_padding": False},
            {"cuda_graph_config": {"batch_sizes": [1, 2], "enable_padding": False}},
        ),
        (
            {"encoder_cuda_graph_enable_padding": False},
            {
                "encoder_cuda_graph_config": {
                    "batch_sizes": [1, 2],
                    "num_tokens": [64, 128],
                    "seq_lens": [64, 128],
                    "enable_padding": False,
                }
            },
        ),
        ({"enable_encoder_decoder_mixed_cuda_graph": False}, {"enable_encoder_decoder_mixed_cuda_graph": False}),
        ({"disable_overlap_scheduler": True}, {"disable_overlap_scheduler": True}),
    ],
)
def test_runtime_config_exposes_independent_generic_graph_controls(
    kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    values = {
        "revision": _REVISION,
        "max_seq_length": 256,
        "max_input_len": 128,
        "max_batch_size": 2,
        "max_num_tokens": 256,
        "encoder_max_num_tokens": 256,
        "decoder_cuda_graph_batch_sizes": [1, 2],
        "encoder_cuda_graph_batch_sizes": [1, 2],
        "encoder_cuda_graph_num_tokens": [64, 128],
        "encoder_cuda_graph_seq_lens": [64, 128],
    }
    config = TensorRTLLMGenerationAdapter("fixture/t5", **values, **kwargs).runtime_config

    assert {name: config[name] for name in expected} == expected


@pytest.mark.parametrize(
    "name",
    [
        "enable_decoder_cuda_graphs",
        "enable_encoder_cuda_graphs",
        "decoder_cuda_graph_enable_padding",
        "encoder_cuda_graph_enable_padding",
        "enable_encoder_decoder_mixed_cuda_graph",
        "disable_overlap_scheduler",
    ],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
@pytest.mark.parametrize("compute_precision", ["bfloat16", "float32"])
def test_graph_controls_require_actual_booleans(name: str, value: object, compute_precision: ComputePrecision) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        TensorRTLLMGenerationAdapter(  # type: ignore[arg-type]
            "fixture/t5",
            revision=_REVISION,
            compute_precision=compute_precision,
            **{name: value},
        )


@pytest.mark.parametrize("disabled_graph", ["enable_decoder_cuda_graphs", "enable_encoder_cuda_graphs"])
def test_mixed_graphs_require_both_graph_families(disabled_graph: str) -> None:
    with pytest.raises(ValueError, match="requires both CUDA graph families"):
        TensorRTLLMGenerationAdapter("fixture/t5", revision=_REVISION, **{disabled_graph: False})


@pytest.mark.parametrize(
    ("enable_decoder_cuda_graphs", "enable_encoder_decoder_mixed_cuda_graph", "message"),
    [
        (True, True, "enable_decoder_cuda_graphs=false"),
        (True, False, "enable_decoder_cuda_graphs=false"),
        (False, True, "enable_encoder_decoder_mixed_cuda_graph=false"),
    ],
)
def test_fp32_rejects_decoder_and_mixed_cuda_graphs(
    enable_decoder_cuda_graphs: bool,
    enable_encoder_decoder_mixed_cuda_graph: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TensorRTLLMGenerationAdapter(
            "fixture/t5",
            revision=_REVISION,
            compute_precision="float32",
            enable_decoder_cuda_graphs=enable_decoder_cuda_graphs,
            enable_encoder_decoder_mixed_cuda_graph=enable_encoder_decoder_mixed_cuda_graph,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revision": "main"}, "immutable"),
        ({"revision": _REVISION, "compute_precision": "float16"}, "bfloat16 or compute_precision=float32"),
        ({"revision": _REVISION, "compute_precision": []}, "bfloat16 or compute_precision=float32"),
        ({"revision": _REVISION, "compute_precision": {}}, "bfloat16 or compute_precision=float32"),
        ({"revision": _REVISION, "max_seq_length": 64, "max_input_len": 65}, "max_input_len"),
        ({"revision": _REVISION, "cross_kv_cache_fraction": 1.0}, "cross_kv_cache_fraction"),
    ],
)
def test_profile_validation_fails_closed(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TensorRTLLMGenerationAdapter("fixture/t5", **kwargs)


def test_factory_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        TensorRTLLMGenerationAdapter.create_for_device("cpu", model_name_or_path="fixture/t5", revision=_REVISION)


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:not-an-index"])
def test_factory_rejects_invalid_cuda_device(device: str) -> None:
    with pytest.raises(ValueError, match="invalid CUDA device"):
        TensorRTLLMGenerationAdapter.create_for_device(
            device,
            model_name_or_path="fixture/t5",
            revision=_REVISION,
        )


def test_load_uses_exact_cli_warmup_and_cleans_generated_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(
        enable_decoder_cuda_graphs=False,
        enable_encoder_cuda_graphs=False,
        enable_encoder_decoder_mixed_cuda_graph=False,
    )
    adapter._server_url = None
    adapter._process = None
    captured: dict[str, Any] = {}
    process = object()
    tokenizer = _FixtureTokenizer()

    monkeypatch.setattr(_server, "reserve_port", lambda: 30241)
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda: tokenizer)

    def launch(command: list[str], **kwargs: Any) -> object:
        captured["command"] = command
        captured.update(kwargs)
        return process

    def wait_until_ready(child: object, url: str, **kwargs: Any) -> None:
        assert child is process
        assert url == "http://127.0.0.1:30241"
        assert kwargs["served_model_name"] == "fixture/t5"
        config_path = Path(captured["command"][captured["command"].index("--config") + 1])
        captured["config_path"] = config_path
        written_config = json.loads(config_path.read_text())
        assert written_config == adapter.runtime_config
        assert written_config["cuda_graph_config"] is None
        assert written_config["encoder_cuda_graph_config"] is None

    monkeypatch.setattr(_server, "launch", launch)
    monkeypatch.setattr(_server, "wait_until_ready", wait_until_ready)
    monkeypatch.setattr(_server, "terminate", lambda child: captured.setdefault("terminated", child))
    monkeypatch.setattr(_server, "release_port", lambda port: captured.setdefault("released", port))

    adapter.load("cuda:1")

    assert captured["command"] == [
        "trtllm-serve",
        "fixture/t5",
        "--backend",
        "pytorch",
        "--host",
        "127.0.0.1",
        "--port",
        "30241",
        "--config",
        str(captured["config_path"]),
        "--hf_revision",
        _REVISION,
        "--served_model_name",
        "fixture/t5",
        "--no-telemetry",
    ]
    assert captured["device_index"] == 1
    assert captured["environment"]["SIE_TRTLLM_RC24_COMPAT"] == "1"
    assert captured["environment"]["PYTHONPATH"].split(":")[0].endswith("tensorrt_llm/_compat")
    config_path = captured["config_path"]
    log_path = Path(adapter._output_file.name)  # type: ignore[union-attr]
    adapter.unload()
    assert captured["terminated"] is process
    assert captured["released"] == 30241
    assert adapter._tokenizer is None
    assert not config_path.exists()
    assert not log_path.exists()


def test_startup_warmup_uses_only_remaining_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monotonic_values = iter((10.0, 11.0, 13.0))

    class Process:
        @staticmethod
        def poll() -> None:
            return None

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, list[object]]:
            return {"choices": []}

    monkeypatch.setattr(_server.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(_server.requests, "get", lambda *_args, **_kwargs: Response())

    def post(*_args: Any, **kwargs: Any) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(_server.requests, "post", post)

    _server.wait_until_ready(
        Process(),  # type: ignore[arg-type]
        "http://127.0.0.1:30200",
        served_model_name="fixture/t5",
        output_file=None,
        timeout_s=5.0,
    )

    assert captured["timeout"] == 2.0


def test_log_tail_reads_only_a_bounded_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    output_file = _server.open_output_log()
    try:
        output_file.write(("prefix-" + ("x" * 100_000) + "-suffix").encode())
        monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("read whole log"))

        assert _server.log_tail(output_file, chars=12) == "xxxxx-suffix"
    finally:
        output_file.close()
        Path(output_file.name).unlink()


def test_default_http_client_has_bounded_stream_read_timeout() -> None:
    adapter = _adapter()

    client = asyncio.run(adapter._get_or_create_http_client())

    assert client.timeout.read == 300.0
    asyncio.run(adapter.aclose_client())


def test_tokenizer_load_is_revision_pinned_and_disables_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    tokenizer = _FixtureTokenizer()
    captured: dict[str, Any] = {}
    compatibility_calls = 0

    def install_compatibility() -> None:
        nonlocal compatibility_calls
        compatibility_calls += 1

    def from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FixtureTokenizer:
        captured["model_name_or_path"] = model_name_or_path
        captured.update(kwargs)
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        "sie_server.adapters.tensorrt_llm.generation.install_transformers_5_t5_tokenizer_compatibility",
        install_compatibility,
    )

    assert adapter._load_tokenizer() is tokenizer
    assert compatibility_calls == 1
    assert captured == {
        "model_name_or_path": "fixture/t5",
        "revision": _REVISION,
        "trust_remote_code": False,
    }


def test_load_clears_lifecycle_state_when_port_reservation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    adapter._server_url = None
    adapter._process = None

    def fail_reservation() -> int:
        raise RuntimeError("no ports")

    monkeypatch.setattr(_server, "reserve_port", fail_reservation)

    with pytest.raises(RuntimeError, match="no ports"):
        adapter.load("cuda:0")

    assert adapter._device is None
    assert adapter._port is None
    assert adapter._server_url is None
    assert adapter._tokenizer is None


def test_load_releases_reserved_port_when_tokenizer_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    adapter._server_url = None
    adapter._process = None
    released: list[int | None] = []

    monkeypatch.setattr(_server, "reserve_port", lambda: 30242)
    monkeypatch.setattr(_server, "release_port", released.append)

    def fail_tokenizer_load() -> Any:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(adapter, "_load_tokenizer", fail_tokenizer_load)

    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        adapter.load("cuda:0")

    assert released == [30242]
    assert adapter._device is None
    assert adapter._port is None
    assert adapter._server_url is None
    assert adapter._tokenizer is None


def test_terminate_signals_process_group_after_launcher_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    waits: list[float] = []

    class ExitedLauncher:
        pid = 2345

        def poll(self) -> int:
            return 1

        def wait(self, *, timeout: float) -> int:
            waits.append(timeout)
            return 1

    monkeypatch.setattr(_server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(_server, "_wait_for_process_group_exit", lambda *_args, **_kwargs: True)

    _server.terminate(ExitedLauncher(), timeout_s=3.0)  # type: ignore[arg-type]

    assert signals == [(2345, signal.SIGTERM)]
    assert len(waits) == 1
    assert 0 <= waits[0] <= 3.0


def test_terminate_escalates_timed_out_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    waits: list[float] = []

    class StuckLauncher:
        pid = 3456

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            waits.append(timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(_server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    group_waits = iter((False, True))
    monkeypatch.setattr(
        _server,
        "_wait_for_process_group_exit",
        lambda *_args, **_kwargs: next(group_waits),
    )

    _server.terminate(StuckLauncher(), timeout_s=2.0)  # type: ignore[arg-type]

    assert signals == [(3456, signal.SIGTERM), (3456, signal.SIGKILL)]
    assert len(waits) == 1
    assert 0 <= waits[0] <= _server.PROCESS_GROUP_KILL_TIMEOUT_S


def test_terminate_raises_when_process_group_survives_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[tuple[int, signal.Signals]] = []

    class StuckLauncher:
        pid = 4567

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            return -signal.SIGKILL

    monkeypatch.setattr(_server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(_server, "_wait_for_process_group_exit", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        _server.terminate(cast("subprocess.Popen[bytes]", StuckLauncher()), timeout_s=0.0)

    assert signals == [(4567, signal.SIGTERM), (4567, signal.SIGKILL)]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux child-subreaper semantics")
def test_terminate_reaps_adopted_descendant_after_launcher_exits() -> None:
    server_script = """
import signal
import subprocess
import sys
import time

descendant = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(60)",
], stdout=subprocess.PIPE, text=True)
assert descendant.stdout is not None
assert descendant.stdout.readline().strip() == "READY"
print(descendant.pid, flush=True)
time.sleep(60)
"""
    subreaper_script = f"""
import ctypes
import os
import subprocess
import sys

from sie_server.adapters.tensorrt_llm import _server

PR_SET_CHILD_SUBREAPER = 36
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
process = subprocess.Popen(
    [sys.executable, "-c", {server_script!r}],
    stdout=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
assert process.stdout is not None
descendant_pid = int(process.stdout.readline())
_server.terminate(process, timeout_s=0.2)
if _server._process_group_exists(process.pid):
    raise RuntimeError("GROUP_SURVIVED")
try:
    os.kill(descendant_pid, 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError("DESCENDANT_SURVIVED")
print("GROUP_GONE")
"""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", subreaper_script],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "GROUP_GONE\n"


def test_generation_maps_completion_controls_text_finish_usage_and_logprobs() -> None:
    adapter = _adapter()
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": " bonjour",
                            "finish_reason": None,
                            "logprobs": {
                                "tokens": [" bonjour"],
                                "token_logprobs": [-0.2],
                                "top_logprobs": [{" bonjour": -0.2, " salut": -1.0}],
                                "text_offset": [0],
                            },
                        }
                    ]
                },
                {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        )

    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = asyncio.run(
        _collect(
            adapter.generate(
                "<2fr> hello",
                max_new_tokens=9,
                temperature=0.2,
                top_p=0.9,
                top_k=12,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                repetition_penalty=1.1,
                min_new_tokens=1,
                stop=["!"],
                seed=7,
                logit_bias={"12": -1.5},
                logprobs=True,
                top_logprobs=2,
            )
        )
    )

    assert captured["path"] == "/v1/completions"
    assert captured["body"] == {
        "model": "fixture/t5",
        "prompt": "<2fr> hello",
        "max_tokens": 9,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        "stop": ["!"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "top_k": 12,
        "repetition_penalty": 1.1,
        "min_tokens": 1,
        "seed": 7,
        "logit_bias": {"12": -1.5},
        "logprobs": 2,
    }
    assert len(chunks) == 2
    assert chunks[0].text_delta == " bonjour"
    assert chunks[0].is_first is True
    assert chunks[0].logprobs == (
        {
            "token": " bonjour",
            "logprob": -0.2,
            "bytes": list(b" bonjour"),
            "top_logprobs": [
                {"token": " bonjour", "logprob": -0.2, "bytes": list(b" bonjour")},
                {"token": " salut", "logprob": -1.0, "bytes": list(b" salut")},
            ],
        },
    )
    assert chunks[1].done is True
    assert chunks[1].finish_reason == "stop"
    assert (chunks[1].prompt_tokens, chunks[1].completion_tokens) == (3, 1)
    assert isinstance(adapter._tokenizer, _FixtureTokenizer)
    assert adapter._tokenizer.calls == [("<2fr> hello", True)]

    asyncio.run(adapter.aclose_client())


def test_completion_logprobs_normalizes_null_top_alternative_per_token() -> None:
    assert _completion_logprobs(
        {
            "tokens": [" bonjour", " monde"],
            "token_logprobs": [-0.2, -0.3],
            "top_logprobs": [None, {" monde": -0.3}],
        }
    ) == (
        {
            "token": " bonjour",
            "logprob": -0.2,
            "bytes": list(b" bonjour"),
            "top_logprobs": [],
        },
        {
            "token": " monde",
            "logprob": -0.3,
            "bytes": list(b" monde"),
            "top_logprobs": [
                {"token": " monde", "logprob": -0.3, "bytes": list(b" monde")},
            ],
        },
    )


@pytest.mark.parametrize(
    ("top_logprobs", "message"),
    [
        ([None], "misaligned top-logprob arrays"),
        ([None, []], "token-to-logprob objects"),
        ([None, 0], "token-to-logprob objects"),
        ([None, False], "token-to-logprob objects"),
        ([None, "bad"], "token-to-logprob objects"),
        ([None, {" monde": "invalid"}], "invalid entry"),
    ],
)
def test_completion_logprobs_rejects_malformed_non_null_and_misaligned_alternatives(
    top_logprobs: object,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _completion_logprobs(
            {
                "tokens": [" bonjour", " monde"],
                "token_logprobs": [-0.2, -0.3],
                "top_logprobs": top_logprobs,
            }
        )


def test_stream_completion_normalizes_null_top_alternatives() -> None:
    adapter = _adapter()
    adapter._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=_sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "text": " bonjour monde",
                                "finish_reason": None,
                                "logprobs": {
                                    "tokens": [" bonjour", " monde"],
                                    "token_logprobs": [-0.2, -0.3],
                                    "top_logprobs": [None, {" monde": -0.3, " monde!": -1.0}],
                                },
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                ),
            )
        )
    )

    chunks = asyncio.run(_collect(adapter.generate("<2fr> hello", max_new_tokens=4, logprobs=True)))

    assert chunks[0].logprobs == (
        {
            "token": " bonjour",
            "logprob": -0.2,
            "bytes": list(b" bonjour"),
            "top_logprobs": [],
        },
        {
            "token": " monde",
            "logprob": -0.3,
            "bytes": list(b" monde"),
            "top_logprobs": [
                {"token": " monde", "logprob": -0.3, "bytes": list(b" monde")},
                {"token": " monde!", "logprob": -1.0, "bytes": list(b" monde!")},
            ],
        },
    )
    assert chunks[1].done is True
    asyncio.run(adapter.aclose_client())


def test_unload_closes_http_client_before_child_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    class CloseTrackingClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    adapter = _adapter()
    client = CloseTrackingClient()
    adapter._http_client = cast("httpx.AsyncClient", client)
    close_state_at_teardown: list[bool] = []
    monkeypatch.setattr(adapter, "_teardown_child", lambda: close_state_at_teardown.append(client.closed))

    adapter.unload()

    assert client.closed
    assert close_state_at_teardown == [True]


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            ({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},),
            "ended without finish reason and usage",
        ),
        (
            ({"choices": [{"index": 0, "text": "x", "finish_reason": "stop"}]},),
            "ended without finish reason and usage",
        ),
        (({"error": {"message": "engine rejected"}, "choices": []},), "completion error: engine rejected"),
        (({"choices": [{"index": 1, "text": "x", "finish_reason": None}]},), "unsupported multiple choices"),
    ],
)
def test_stream_fails_closed_on_incomplete_or_invalid_events(events: tuple[object, ...], message: str) -> None:
    adapter = _adapter()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(*events))

    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4)))
    asyncio.run(adapter.aclose_client())


def test_stream_fails_when_eof_arrives_without_done() -> None:
    adapter = _adapter()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"index":0,"text":"x","finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
        )

    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match=r"before \[DONE\]"):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4)))
    asyncio.run(adapter.aclose_client())


@pytest.mark.parametrize(
    "token_ids",
    [None, [], [True], [-1], ["1"], [[1]]],
)
def test_generation_fails_before_dispatch_when_encoder_token_ids_are_invalid(token_ids: object) -> None:
    adapter = _adapter()
    adapter._tokenizer = type("MalformedTokenizer", (), {"encode": lambda *_args, **_kwargs: token_ids})()

    with pytest.raises(RuntimeError, match="invalid token ids"):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4)))

    assert adapter._http_client is None


def test_generation_fails_before_dispatch_when_encoder_tokenization_raises() -> None:
    adapter = _adapter()

    def fail_tokenization(*_args: Any, **_kwargs: Any) -> list[int]:
        raise ValueError("bad tokenizer state")

    adapter._tokenizer = type("FailingTokenizer", (), {"encode": fail_tokenization})()

    with pytest.raises(RuntimeError, match="tokenization failed"):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4)))

    assert adapter._http_client is None


def test_generation_allows_encoder_limit_after_special_token_accounting() -> None:
    adapter = _adapter()
    prompt = " ".join(["token"] * 127)

    adapter._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=_sse(
                    {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 128, "completion_tokens": 1}},
                ),
            )
        )
    )
    chunks = asyncio.run(_collect(adapter.generate(prompt, max_new_tokens=4)))

    assert chunks[-1].prompt_tokens == 128
    asyncio.run(adapter.aclose_client())


def test_generation_rejects_encoder_overflow_before_dispatch_after_special_token_accounting() -> None:
    adapter = _adapter()
    prompt = " ".join(["token"] * 128)

    with pytest.raises(
        GenerationInputTooLongError,
        match=r"prompt token count \(129\) exceeds max_input_len \(128\)",
    ) as exc_info:
        asyncio.run(_collect(adapter.generate(prompt, max_new_tokens=4)))

    assert exc_info.value.code == "INPUT_TOO_LONG"
    assert exc_info.value.param == "prompt"
    assert adapter._http_client is None


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "x",
                            "finish_reason": None,
                            "logprobs": {"tokens": ["x"], "token_logprobs": []},
                        }
                    ]
                },
            ),
            "misaligned token arrays",
        ),
        (
            (
                {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                {"choices": [{"index": 0, "text": "late", "finish_reason": None}]},
            ),
            "text after its terminal choice",
        ),
        (
            (
                {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": True, "completion_tokens": 1}},
            ),
            "usage is invalid",
        ),
        (
            (
                {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            ),
            "duplicate usage",
        ),
    ],
)
def test_stream_rejects_malformed_terminal_logprob_and_usage_shapes(events: tuple[object, ...], message: str) -> None:
    adapter = _adapter()
    adapter._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=_sse(*events)))
    )

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4, logprobs=True)))
    asyncio.run(adapter.aclose_client())


@pytest.mark.parametrize(
    ("status_code", "content", "error"),
    [
        (200, b"event: completion\n\n", RuntimeError),
        (503, b"unavailable", httpx.HTTPStatusError),
    ],
)
def test_stream_rejects_non_sse_and_http_errors(status_code: int, content: bytes, error: type[Exception]) -> None:
    adapter = _adapter()
    adapter._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(status_code, content=content))
    )

    with pytest.raises(error):
        asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=4)))
    asyncio.run(adapter.aclose_client())


@pytest.mark.parametrize(
    ("kwargs", "error_type", "param", "code"),
    [
        ({"grammar": object()}, GenerationUnsupportedFieldError, "grammar", "unsupported_field"),
        ({"images": [{"data": b"image"}]}, GenerationUnsupportedFieldError, "images", "unsupported_field"),
        ({"lora_path": "adapter"}, GenerationUnsupportedFieldError, "lora_path", "unsupported_field"),
        ({"n": 2}, GenerationUnsupportedFieldError, "n", "unsupported_field"),
        ({"best_of": 2}, GenerationUnsupportedFieldError, "best_of", "unsupported_field"),
        ({"top_logprobs": 2}, GenerationInvalidRequestError, "top_logprobs", "invalid_request"),
        ({"min_new_tokens": 5}, GenerationInvalidRequestError, "min_new_tokens", "invalid_request"),
    ],
)
@pytest.mark.parametrize("through_preflight", [True, False])
def test_request_refusals_are_typed_before_dispatch(
    kwargs: dict[str, Any],
    error_type: type[Exception],
    param: str,
    code: str,
    *,
    through_preflight: bool,
) -> None:
    adapter = _adapter()
    parameters = {"prompt": "prompt", "max_new_tokens": 4, **kwargs}

    with pytest.raises(error_type) as exc_info:
        if through_preflight:
            adapter.preflight_generate(parameters, stream=False)
        else:
            asyncio.run(_collect(adapter.generate(**parameters)))

    error = cast("Any", exc_info.value)
    assert error.code == code
    assert error.param == param
    if param == "lora_path":
        assert "lora_path" not in str(error)
        assert "lora_adapter" in str(error)
    assert adapter._http_client is None


def test_preflight_prompt_count_is_reused_by_matching_dispatch() -> None:
    async def scenario() -> list[tuple[str, bool]]:
        adapter = _adapter()
        adapter._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_sse(
                        {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                    ),
                )
            )
        )
        parameters = {"prompt": "translate this", "max_new_tokens": 4}
        preflight_result = adapter.preflight_generate(parameters, stream=False)
        chunks = adapter.generate_with_preflight(parameters, preflight_result)
        await asyncio.create_task(_collect(chunks))
        assert adapter._active_generations == 0
        await adapter.aclose_client()
        return adapter._tokenizer.calls

    assert asyncio.run(scenario()) == [("translate this", True)]


def test_preflight_result_is_consumed_once_even_for_an_identical_second_dispatch() -> None:
    async def scenario() -> list[tuple[str, bool]]:
        adapter = _adapter()
        adapter._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_sse(
                        {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                    ),
                )
            )
        )
        parameters = {"prompt": "translate this", "max_new_tokens": 4}
        preflight_result = adapter.preflight_generate(parameters, stream=False)

        await _collect(adapter.generate_with_preflight(parameters, preflight_result))
        await _collect(adapter.generate_with_preflight(parameters, preflight_result))

        assert adapter._active_generations == 0
        await adapter.aclose_client()
        return adapter._tokenizer.calls

    assert asyncio.run(scenario()) == [
        ("translate this", True),
        ("translate this", True),
    ]


def test_abandoned_unstarted_preflight_dispatch_cannot_reuse_prompt_count() -> None:
    async def scenario() -> list[tuple[str, bool]]:
        adapter = _adapter()
        adapter._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_sse(
                        {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                    ),
                )
            )
        )
        parameters = {"prompt": "same request text", "max_new_tokens": 4}
        preflight_result = adapter.preflight_generate(parameters, stream=False)

        abandoned = adapter.generate_with_preflight(parameters, preflight_result)
        await abandoned.aclose()
        await _collect(adapter.generate_with_preflight(parameters, preflight_result))

        assert adapter._active_generations == 0
        await adapter.aclose_client()
        return adapter._tokenizer.calls

    assert asyncio.run(scenario()) == [
        ("same request text", True),
        ("same request text", True),
    ]


def test_preflight_result_cannot_cross_adapter_ownership() -> None:
    async def scenario() -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
        first = _adapter()
        second = _adapter()
        second._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_sse(
                        {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
                    ),
                )
            )
        )
        parameters = {"prompt": "same prompt", "max_new_tokens": 4}
        preflight_result = first.preflight_generate(parameters, stream=False)

        await _collect(second.generate_with_preflight(parameters, preflight_result))

        await second.aclose_client()
        return first._tokenizer.calls, second._tokenizer.calls

    first_calls, second_calls = asyncio.run(scenario())
    assert first_calls == [("same prompt", True)]
    assert second_calls == [("same prompt", True)]


def test_preflight_cache_cannot_bypass_changed_dispatch_controls() -> None:
    async def scenario() -> None:
        adapter = _adapter()
        parameters = {"prompt": "translate this", "max_new_tokens": 4}
        preflight_result = adapter.preflight_generate(parameters, stream=False)
        changed_parameters = {**parameters, "grammar": object()}
        with pytest.raises(GenerationUnsupportedFieldError) as exc_info:
            await _collect(adapter.generate_with_preflight(changed_parameters, preflight_result))
        assert exc_info.value.param == "grammar"
        assert adapter._http_client is None

    asyncio.run(scenario())


def test_preflight_results_are_request_scoped_across_concurrent_child_tasks() -> None:
    async def scenario() -> list[tuple[str, bool]]:
        adapter = _adapter()
        adapter._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=_sse(
                        {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
                    ),
                )
            )
        )

        async def run(parameters: dict[str, Any]) -> None:
            preflight_result = adapter.preflight_generate(parameters, stream=True)
            chunks = adapter.generate_with_preflight(parameters, preflight_result)
            await asyncio.create_task(_collect(chunks))

        await asyncio.gather(
            run({"prompt": "same prompt", "max_new_tokens": 4}),
            run({"prompt": "same prompt", "max_new_tokens": 8}),
        )
        assert adapter._active_generations == 0
        await adapter.aclose_client()
        return adapter._tokenizer.calls

    assert asyncio.run(scenario()) == [("same prompt", True), ("same prompt", True)]


class _CancellableStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False
        self.block = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"index":0,"text":"x","finish_reason":null}]}\n\n'
        await self.block.wait()

    async def aclose(self) -> None:
        self.closed = True
        self.block.set()


def test_closing_generation_closes_incomplete_upstream_response() -> None:
    async def scenario() -> bool:
        adapter = _adapter()
        stream = _CancellableStream()
        adapter._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
        )
        iterator = adapter.generate("prompt", max_new_tokens=4)
        first = await anext(iterator)
        assert first.text_delta == "x"
        await iterator.aclose()
        await adapter.aclose_client()
        return stream.closed

    assert asyncio.run(scenario()) is True


def test_generation_drain_blocks_new_admission_and_waits_for_active_stream() -> None:
    async def scenario() -> tuple[bool, bool, int]:
        adapter = _adapter()
        stream = _CancellableStream()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)))
        adapter._http_client = client
        parameters = {"prompt": "prompt", "max_new_tokens": 4}
        preflight_result = adapter.preflight_generate(parameters, stream=True)
        iterator = adapter.generate_with_preflight(parameters, preflight_result)
        first = await anext(iterator)
        assert first.text_delta == "x"

        drain = asyncio.create_task(adapter.drain_generation())
        await asyncio.sleep(0)
        assert not drain.done()
        with pytest.raises(GenerationDrainingError):
            adapter.preflight_generate(parameters, stream=True)

        await iterator.aclose()
        await drain
        return stream.closed, client.is_closed, adapter._active_generations

    assert asyncio.run(scenario()) == (True, True, 0)


class _BlockingCloseFailureCompletion:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self) -> _BlockingCloseFailureCompletion:
        return self

    async def __anext__(self) -> Any:
        self.started.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("secondary close failure")


def test_cancellation_is_not_masked_by_completion_close_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[int, int]:
        adapter = _adapter()
        completion = _BlockingCloseFailureCompletion()
        monkeypatch.setattr(adapter, "_stream_completion", lambda *_args, **_kwargs: completion)
        consume = asyncio.create_task(anext(adapter.generate("prompt", max_new_tokens=4)))
        await completion.started.wait()
        consume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consume
        return completion.close_calls, adapter._active_generations

    assert asyncio.run(scenario()) == (1, 0)


class _TerminalCloseFailureCompletion:
    def __init__(self) -> None:
        self.yielded = False
        self.close_calls = 0

    def __aiter__(self) -> _TerminalCloseFailureCompletion:
        return self

    async def __anext__(self) -> Any:
        if self.yielded:
            raise StopAsyncIteration
        self.yielded = True
        return type("Terminal", (), {"done": True})()

    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("secondary close failure")


def test_terminal_outcome_is_not_replaced_by_completion_close_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[list[Any], int, int]:
        adapter = _adapter()
        completion = _TerminalCloseFailureCompletion()
        monkeypatch.setattr(adapter, "_stream_completion", lambda *_args, **_kwargs: completion)
        chunks = await _collect(adapter.generate("prompt", max_new_tokens=4))
        return chunks, completion.close_calls, adapter._active_generations

    chunks, close_calls, active = asyncio.run(scenario())
    assert len(chunks) == 1
    assert chunks[0].done is True
    assert (close_calls, active) == (1, 0)


class _TerminalBlockingCloseCompletion:
    def __init__(self) -> None:
        self.yielded = False
        self.close_started = asyncio.Event()

    def __aiter__(self) -> _TerminalBlockingCloseCompletion:
        return self

    async def __anext__(self) -> Any:
        if self.yielded:
            raise StopAsyncIteration
        self.yielded = True
        return type("Terminal", (), {"done": True})()

    async def aclose(self) -> None:
        self.close_started.set()
        await asyncio.Event().wait()


def test_cancellation_during_completion_cleanup_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> int:
        adapter = _adapter()
        completion = _TerminalBlockingCloseCompletion()
        monkeypatch.setattr(adapter, "_stream_completion", lambda *_args, **_kwargs: completion)
        consume = asyncio.create_task(_collect(adapter.generate("prompt", max_new_tokens=4)))
        await completion.close_started.wait()
        consume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consume
        return adapter._active_generations

    assert asyncio.run(scenario()) == 0


def test_independent_requests_share_client_and_reach_transport_concurrently() -> None:
    async def scenario() -> int:
        adapter = _adapter()
        active = 0
        peak = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(
                200,
                content=_sse(
                    {"choices": [{"index": 0, "text": "x", "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                ),
            )

        adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await asyncio.gather(
            _collect(adapter.generate("one", max_new_tokens=2)),
            _collect(adapter.generate("two", max_new_tokens=2)),
        )
        await adapter.aclose_client()
        return peak

    assert asyncio.run(scenario()) == 2
