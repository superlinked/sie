"""CPU contract smoke; these checks do not establish CUDA acceptance."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from peft import PeftModel
from safetensors.torch import load_file
from sie_sdk._msgpack import packb as pack_msgpack
from sie_sdk.client.errors import RequestError, ServerError
from sie_server.adapters.sglang import _server
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter, resolve_adapter_class, validate_loadtime_options
from transformers import Qwen3Config, Qwen3ForCausalLM

from tools.benchmarks.runtime_lora import probe
from tools.benchmarks.runtime_lora.adapters import ObservedGenerationAdapter
from tools.benchmarks.runtime_lora.fixtures import MODELS, digest, write_adapter
from tools.benchmarks.runtime_lora.observer import patch_scheduler
from tools.benchmarks.runtime_lora.probe import (
    BOUNDARY_PROMPT,
    Client,
    Evidence,
    batch_rows,
    check_buffered_native,
    check_effect,
    configs,
    embedding_delta,
    mixed_batches,
    probability,
    record_check,
    sequence_comparison,
    token_distribution_comparison,
)


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def test_fixture_weights_load_in_peft_change_logits_and_restore_base(tmp_path: Path) -> None:
    config = tiny_config()
    first = write_adapter(tmp_path / "a", config.to_dict(), "tiny-qwen", "test", 17)
    write_adapter(tmp_path / "same", config.to_dict(), "tiny-qwen", "test", 17)
    write_adapter(tmp_path / "b", config.to_dict(), "tiny-qwen", "test", 18)
    assert first["files"]["adapter_model.safetensors"] == digest(tmp_path / "same/adapter_model.safetensors")
    tensors = load_file(tmp_path / "a/adapter_model.safetensors")
    assert len(tensors) == 8
    assert all(torch.count_nonzero(value) == value.numel() for value in tensors.values())
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        base = model(inputs).logits.clone()
    model = PeftModel.from_pretrained(model, tmp_path / "a", adapter_name="a")
    model.load_adapter(tmp_path / "b", adapter_name="b")
    with torch.no_grad():
        model.set_adapter("a")
        a = model(inputs).logits.clone()
        model.set_adapter("b")
        b = model(inputs).logits.clone()
        with model.disable_adapter():
            restored = model(inputs).logits
    assert torch.max(torch.abs(base - a)) > 1e-5
    assert torch.max(torch.abs(base - b)) > 1e-5
    assert torch.max(torch.abs(a - b)) > 1e-5
    torch.testing.assert_close(restored, base, rtol=0, atol=1e-6)


def test_catalog_pins_and_declared_fixture_config(tmp_path: Path) -> None:
    generated = configs(tmp_path / "models", tmp_path / "fixtures")
    for kind, (name, revision) in MODELS.items():
        data = generated[kind]
        assert data["hf_id"] == name
        assert data["hf_revision"] == revision
        options = data["profiles"]["default"]["adapter_options"]["loadtime"]
        assert options["preload_loras"] == ["pin"]
        assert set(options["lora_paths"]) == {"pin", "a", "b", "c", "d"}
        assert options["max_loras_per_batch"] == 2
        assert options["max_loaded_loras"] == 3
        assert options["mem_fraction_static"] == 0.5


@pytest.mark.parametrize("kind", ["generation", "embedding"])
def test_generated_config_resolves_validates_and_constructs_without_engine_launch(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "models"
    generated = configs(directory, tmp_path / "fixtures")
    config = ModelConfig.model_validate(generated[kind])
    expected = SGLangGenerationAdapter if kind == "generation" else SGLangEmbeddingAdapter
    assert issubclass(resolve_adapter_class(config, directory), expected)
    validate_loadtime_options(config, directory)
    adapter = load_adapter(config, directory, device="cuda:0")
    try:
        assert isinstance(adapter, expected)
        assert adapter._process is None
        assert adapter._lora.max_per_batch == 2
        assert adapter._lora.max_loaded == 3
        assert adapter._lora.preloaded == {str(tmp_path / "fixtures" / kind / "pin")}
        if kind == "generation":
            assert generated[kind]["tasks"]["generate"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert Path(adapter._compat_pythonpath_entries()[0], "sitecustomize.py").is_file()
        else:
            assert "causal" not in config.resolve_profile("default").loadtime
    finally:
        adapter._lora.close()


@pytest.mark.parametrize("kind", ["generation", "embedding"])
def test_generated_profiles_only_add_one_kv_cap_to_actual_engine_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    commands = []

    class LaunchCapturedError(Exception):
        pass

    def capture_command(cmd, **kwargs):
        commands.append(list(cmd))
        raise LaunchCapturedError

    monkeypatch.setattr(_server, "find_free_port", lambda *args: 30000)
    monkeypatch.setattr(_server, "launch_sglang_server", capture_command)
    directory = tmp_path / "models"
    generated = configs(directory, tmp_path / "fixtures")[kind]
    for capped in (False, True):
        data = deepcopy(generated)
        profile = data["profiles"]["default"]
        if not capped:
            if kind == "generation":
                extra = profile["adapter_options"]["loadtime"]["extra_launch_args"]
                assert extra[-2:] == ["--max-total-tokens", "4096"]
                del extra[-2:]
            else:
                profile["adapter_path"] = "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter"
        config = ModelConfig.model_validate(data)
        validate_loadtime_options(config, directory)
        adapter = load_adapter(config, directory, device="cuda:0")
        # Source materialization is separately tested; use the declared path so
        # temporary snapshot names do not obscure the launch-argv comparison.
        monkeypatch.setattr(adapter._lora, "fetch", lambda name: name)
        expected_reservation = adapter._memory_budget.required_bytes(
            device_type="cuda", device_total_bytes=80 * 1024**3
        )
        try:
            with pytest.raises(LaunchCapturedError):
                adapter.load("cuda:0")
            assert adapter._process is None
            assert (
                adapter._memory_budget.required_bytes(device_type="cuda", device_total_bytes=80 * 1024**3)
                == expected_reservation
            )
        finally:
            adapter._lora.close()
    uncapped, capped = commands
    assert "--max-total-tokens" not in uncapped
    assert capped.count("--max-total-tokens") == 1
    index = capped.index("--max-total-tokens")
    assert capped[index + 1] == "4096"
    del capped[index : index + 2]
    assert capped == uncapped
    assert capped[capped.index("--mem-fraction-static") + 1] == "0.5"


def test_sdk_observer_captures_cold_503_and_native_retry(tmp_path: Path) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                503, json={"error": {"code": "LORA_LOADING", "message": "loading"}}, headers={"Retry-After": "0"}
            )
        return httpx.Response(200, json={"model": MODELS["generation"][0], "text": "ok"})

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        assert client.call("generation", MODELS["generation"][0], "a", "generate") == {"text": "ok", "usage": None}
        assert [row["status"] for row in client.responses] == [503, 200]
        assert all(request["lora_adapter"] == "a" for request in calls)
        assert client.responses[0]["code"] == "LORA_LOADING"
    finally:
        client.close()


@pytest.mark.parametrize("route", ["native_stream", "native_boundary"])
def test_native_stream_sdk_retains_terminal_logprobs_and_retries_only_http_cold_refusal(
    tmp_path: Path, route: str
) -> None:
    calls = []
    first = {"token": "one", "logprob": -0.1}
    last = {"token": "two", "logprob": -0.2}

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                503, json={"error": {"code": "LORA_LOADING", "message": "loading"}}, headers={"Retry-After": "0"}
            )
        chunks = [
            {"text_delta": "one", "logprobs": [first], "done": False},
            {
                "text_delta": "two",
                "logprobs": [last],
                "done": True,
                "finish_reason": "length",
                "usage": {"completion_tokens": 2},
            },
        ]
        return httpx.Response(
            200,
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        result = client.call("generation", MODELS["generation"][0], "a", route)
        assert result["text"] == "onetwo"
        assert result["logprobs"]["content"] == [first, last]
        assert [row["status"] for row in client.responses] == [503, 200]
        assert all(request["stream"] and request["logprobs"] and request["top_logprobs"] == 20 for request in calls)
        assert all(
            request["lora_adapter"] == "a" and request["temperature"] == 0 and request["seed"] == 17
            for request in calls
        )
        assert calls[-1]["max_new_tokens"] == (1 if route == "native_boundary" else 24)
        if route == "native_boundary":
            assert calls[-1]["prompt"] == BOUNDARY_PROMPT
    finally:
        client.close()


def test_native_stream_does_not_retry_an_error_after_http_success(tmp_path: Path) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        error = {"done": True, "finish_reason": "error", "error": {"code": "LORA_LOADING", "message": "late failure"}}
        return httpx.Response(
            200, text=f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n", headers={"Content-Type": "text/event-stream"}
        )

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ServerError):
            client.call("generation", MODELS["generation"][0], "a", "native_stream")
        assert len(calls) == 1
        assert "fresh_sdk_generation_retry" not in (tmp_path / "events.jsonl").read_text()
    finally:
        client.close()


@pytest.mark.parametrize("error_content_type", ["application/json", "application/msgpack"])
def test_sdk_observer_captures_msgpack_selection_and_automatic_encode_retry(
    tmp_path: Path, error_content_type: str
) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            error = {"error": {"code": "LORA_LOADING", "message": "loading"}}
            content = pack_msgpack(error) if error_content_type == "application/msgpack" else json.dumps(error).encode()
            return httpx.Response(
                503, content=content, headers={"Content-Type": error_content_type, "Retry-After": "0"}
            )
        return httpx.Response(
            200,
            content=pack_msgpack({"items": [{"dense": {"dims": 2, "values": np.array([0.5, 0.25])}}] * 2}),
            headers={"Content-Type": "application/msgpack"},
        )

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        assert client.call("embedding", MODELS["embedding"][0], "a") == {"vectors": [[0.5, 0.25]] * 2}
        assert [row["status"] for row in client.responses] == [503, 200]
        assert all(row["selection"] == "a" for row in client.responses)
        assert client.responses[0]["code"] == "LORA_LOADING"
        assert all(request.headers["content-type"] == "application/msgpack" for request in calls)
        assert "fresh_sdk_generation_retry" not in (tmp_path / "events.jsonl").read_text()
    finally:
        client.close()


def test_observer_records_actual_successful_batch_and_does_not_change_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "batches.jsonl"
    result = object()
    seen = []

    class Scheduler:
        def run_batch(self, batch, extra=None):
            seen.append((batch, extra))
            if extra == "fail":
                raise ValueError("forward failed")
            return result

    module = ModuleType("fake_scheduler")
    module.Scheduler = Scheduler
    patch_scheduler(module, str(trace))
    scheduler = Scheduler()
    uid_a, uid_b = "62f6770ac6ac4f718ff17fe577616e01", "ad52764f1b98498595d08d1243932632"
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            lora_manager=SimpleNamespace(
                lora_refs={uid_a: SimpleNamespace(lora_name="a"), uid_b: SimpleNamespace(lora_name="b")},
                memory_pool=SimpleNamespace(buffer_id_to_uid=[uid_a, uid_b], uid_to_buffer_id={uid_a: 0, uid_b: 1}),
            )
        )
    )

    def req(rid, uid):
        return SimpleNamespace(
            rid=rid,
            lora_id=uid,
            extra_key=uid,
            cached_tokens=3,
            prefix_indices=[0, 1, 2],
            extend_input_len=2,
            origin_input_ids=[1, 2, 3, 4, 5],
            output_ids=[],
        )

    batch = SimpleNamespace(forward_mode="DECODE", reqs=[req("1", uid_a), req("2", uid_b)])
    assert scheduler.run_batch(batch, extra=3) is result
    assert seen == [(batch, 3)]
    with pytest.raises(ValueError, match="forward failed"):
        scheduler.run_batch(batch, extra="fail")
    rows = batch_rows(trace)
    assert len(rows) == 1
    assert rows[0]["gpu_slot_count"] == 2
    assert rows[0]["requests"][0] == {
        "rid": "1",
        "uid": uid_a,
        "lora": "a",
        "cache_key": uid_a,
        "cached_tokens": 3,
        "prefix_length": 3,
        "extend_input_length": 2,
        "input_length": 5,
        "output_length": 0,
    }
    assert rows[0]["gpu_slots"] == [{"uid": uid_a, "lora": "a", "slot": 0}, {"uid": uid_b, "lora": "b", "slot": 1}]
    assert mixed_batches(rows, 0, 2**64) == rows
    rows[0]["gpu_slot_uids"][1] = uid_a
    assert mixed_batches(rows, 0, 2**64) == []
    rows[0]["gpu_slot_uids"][1] = uid_b
    rows[0]["gpu_slots"][1]["slot"] = 0
    assert mixed_batches(rows, 0, 2**64) == []
    rows[0]["gpu_slots"][1]["slot"] = 1
    rows[0]["requests"][1]["lora"] = "a"
    assert mixed_batches(rows, 0, 2**64) == []
    pool_module = ModuleType("sglang.srt.lora.mem_pool")
    pool_module.EMPTY_SLOT = object()
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    pool = scheduler.tp_worker.model_runner.lora_manager.memory_pool
    pool.buffer_id_to_uid = [None, pool_module.EMPTY_SLOT]
    pool.uid_to_buffer_id = {None: 0}
    batch.reqs = [req("base", None)]
    assert scheduler.run_batch(batch) is result
    base_row = batch_rows(trace)[-1]
    assert base_row["gpu_slot_uids"] == [None, {"empty_slot": True}]
    assert base_row["gpu_slots"] == [{"lora": None, "slot": 0, "uid": None}]
    assert mixed_batches([base_row], 0, 2**64) == []


def test_child_observer_defers_import_and_preserves_production_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic upstream package checks startup composition without CUDA."""
    for package in ("sglang", "sglang/srt", "sglang/srt/managers", "sglang/srt/lora"):
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    metadata = tmp_path / "sglang-0.5.10.post1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.10.post1\n")
    # The real slot-release implementation has its own production tests. This
    # sentinel exercises the observer's explicit prerequisite and evidence.
    (tmp_path / "sglang/srt/lora/lora_manager.py").write_text(
        "class LoRAManager:\n    _sie_lora_slot_release_compat = True\n"
    )
    (tmp_path / "sglang/srt/managers/scheduler.py").write_text(
        "from sglang.srt.lora.lora_manager import LoRAManager\n"
        "class Scheduler:\n    def run_batch(self, batch):\n        return 47\n"
    )
    trace = tmp_path / "child-batches.jsonl"
    monkeypatch.delenv("PYTHONPATH", raising=False)
    adapter = ObservedGenerationAdapter("unused", mem_fraction_static=0.5)
    entries = adapter._compat_pythonpath_entries()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((*entries, str(tmp_path))),
        "SIE_RUNTIME_LORA_BATCH_TRACE": str(trace),
        "SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT": "1",
        "SIE_SGLANG_LORA_COMPAT": "",
    }
    script = (
        "import sys\nassert 'sglang' not in sys.modules\n"
        "from sglang.srt.managers.scheduler import Scheduler\n"
        "import builtins\nfrom types import SimpleNamespace as S\n"
        "assert getattr(builtins.__import__, '_sie_mm_process_config_deferred_compat', False)\n"
        "scheduler = Scheduler()\n"
        "scheduler.tp_worker = S(model_runner=S(lora_manager=S(lora_refs={}, "
        "memory_pool=S(buffer_id_to_uid=[None], uid_to_buffer_id={None: 0}))))\n"
        "req = S(rid='r', lora_id=None, extra_key=None, cached_tokens=0, prefix_indices=[], "
        "extend_input_len=1, origin_input_ids=[1], output_ids=[])\n"
        "assert scheduler.run_batch(S(forward_mode='DECODE', reqs=[req])) == 47\n"
    )
    result = subprocess.run(  # noqa: S603 — fixed synthetic package and script, no external input
        [sys.executable, "-c", script], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    rows = batch_rows(trace)
    assert [row["event"] for row in rows] == ["observer_installed", "forward"]
    assert rows[0]["slot_release_compat"] is True
    assert rows[1]["requests"][0]["uid"] is None
    assert rows[1]["requests"][0]["prefix_length"] == 0


def test_effect_checks_fail_for_ignored_adapters_and_contaminated_base() -> None:
    base = {"vectors": [[0.1, 0.2]]}
    a = {"vectors": [[0.2, 0.3]]}
    b = {"vectors": [[0.3, 0.4]]}
    assert check_effect("embedding", base, a, b, base)["base_reset_max_abs"] == 0
    with pytest.raises(AssertionError, match="did not differ"):
        check_effect("embedding", base, base, b, base)
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("embedding", base, a, b, a)
    assert np.isfinite(check_effect("embedding", base, a, b, base)["a_vs_b_max_abs"])


def test_embedding_warm_baseline_is_stable_and_sets_effect_floor() -> None:
    def result(value: float) -> dict:
        return {"vectors": [[0.1, value]]}

    base, a, b = result(0.5), result(0.6), result(0.7)
    effects = check_effect("embedding", base, a, b, base, result(0.50001))
    assert effects["baseline_max_abs"] == pytest.approx(1e-5)
    assert effects["required_effect_minimum"] == pytest.approx(5e-5)
    with pytest.raises(AssertionError, match="baseline was not stable"):
        check_effect("embedding", base, a, b, base, result(0.5001))
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("embedding", base, a, b, result(0.5001), base)
    with pytest.raises(AssertionError, match="beyond base drift"):
        check_effect("embedding", base, result(0.5001), result(0.5003), base, result(0.50004))


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_embedding_comparison_rejects_nonfinite_baseline_or_output(invalid: float) -> None:
    base = {"vectors": [[0.1, 0.5]]}
    a, b = {"vectors": [[0.1, 0.6]]}, {"vectors": [[0.1, 0.7]]}
    invalid_result = {"vectors": [[0.1, invalid]]}
    with pytest.raises(AssertionError, match="non-finite"):
        embedding_delta(base, invalid_result)
    with pytest.raises(AssertionError, match="non-finite"):
        check_effect("embedding", base, a, b, base, invalid_result)
    with pytest.raises(AssertionError, match="non-finite"):
        check_effect("embedding", base, a, b, invalid_result, base)


@pytest.mark.parametrize("vectors", [[], [[0.1]], [[0.1, 0.5], [0.1, 0.5]]])
def test_embedding_comparison_rejects_empty_or_broadcastable_shape_mismatch(vectors: list) -> None:
    with pytest.raises(AssertionError, match="shapes differ or are empty"):
        embedding_delta({"vectors": [[0.1, 0.5]]}, {"vectors": vectors})


def test_embedding_probe_chooses_warm_reference_before_adapter_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selections = []
    ready = ["pin"]
    model = MODELS["embedding"][0]
    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: process.returncode

    class FakeClient:
        def __init__(self, *args):
            self.sdk = self
            self.responses = []

        def call(self, kind, name, selection, route):
            assert kind == "embedding"
            assert name == model
            if selection == "undeclared":
                raise RequestError("undeclared", "INVALID_INPUT", 400)
            selections.append(selection)
            if selection is None:
                return {"vectors": [[0.48 if len(selections) == 1 else 0.5]]}
            if selection not in ready:
                self.responses.append({"selection": selection, "status": 503, "code": "LORA_LOADING"})
                ready.append(selection)
                if len(ready) > 3:
                    del ready[1]
            return {"vectors": [[0.6 if selection == "a" else 0.7]]}

        def list_models(self):
            return [{"name": model, "lora_adapters": list(ready)}]

        def close(self):
            pass

    async def unload(name):
        assert name == model
        process.returncode = 0

    monkeypatch.setattr(probe, "Client", FakeClient)
    registry = SimpleNamespace(get=lambda name: SimpleNamespace(_process=process), unload_async=unload)
    result = asyncio.run(
        probe.run_model("embedding", registry, "http://unused", Evidence(tmp_path), tmp_path / "trace")
    )
    assert selections[:5] == [None, None, None, "a", "b"]
    assert result["passed"] is True
    assert result["effects"]["baseline_max_abs"] == result["effects"]["base_reset_max_abs"] == 0
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    baseline = next(row for row in rows if row["event"] == "baseline_comparison")
    assert baseline["first_vs_warm_max_abs"] == pytest.approx(0.02)
    assert baseline["warm_repeat_max_abs"] == 0


def test_effect_checks_require_delta_beyond_observed_numeric_drift() -> None:
    def embedding(value: float) -> dict:
        return {"vectors": [[value]]}

    def generation(value: float) -> dict:
        return {"text": "same", "logprobs": {"content": [{"token": "same", "logprob": math.log(value)}]}}

    for kind, build, a, b, reset in (
        ("embedding", embedding, 0.0001, 0.0003, 0.00004),
        ("generation", generation, 0.0002, 0.0004, 0.00005),
    ):
        base = build(0.5)
        assert check_effect(kind, base, build(0.5 + a), build(0.5 + b), base)
        with pytest.raises(AssertionError, match="beyond base drift"):
            check_effect(kind, base, build(0.5 + a), build(0.5 + b), build(0.5 + reset))


def test_generation_restoration_checks_later_tokens_and_pre_adapter_baseline() -> None:
    def result(first: float, second: float = 0.6, text: str = "same") -> dict:
        return {
            "text": text,
            "logprobs": {
                "content": [
                    {"token": "first", "logprob": math.log(first)},
                    {"token": "second", "logprob": math.log(second)},
                ]
            },
        }

    base, a, b = result(0.5), result(0.6), result(0.7)
    assert check_effect("generation", base, a, b, base, base)
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("generation", base, a, b, result(0.5, second=0.55), base)
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("generation", base, a, b, result(0.5, text="different later text"), base)
    with pytest.raises(AssertionError, match="baseline was not stable"):
        check_effect("generation", base, a, b, base, result(0.5, second=0.55))
    with pytest.raises(AssertionError, match="beyond base drift"):
        check_effect("generation", base, result(0.501), result(0.502), base, result(0.5003))
    shifted_alternate = result(0.5)
    base["logprobs"]["content"][1]["top_logprobs"] = [
        {"token": "second", "logprob": math.log(0.6)},
        {"token": "alternate", "logprob": math.log(0.2)},
    ]
    shifted_alternate["logprobs"]["content"][1]["top_logprobs"] = [
        {"token": "second", "logprob": math.log(0.6)},
        {"token": "alternate", "logprob": math.log(0.3)},
    ]
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("generation", base, a, b, shifted_alternate, base)
    with pytest.raises(AssertionError, match="baseline was not stable"):
        check_effect("generation", base, a, b, base, shifted_alternate)


def test_sequence_comparison_stops_after_context_diverges() -> None:
    def result(tokens: list[str]) -> dict:
        return {
            "text": "".join(tokens),
            "logprobs": {"content": [{"token": token, "logprob": -0.2} for token in tokens]},
        }

    comparison = sequence_comparison(result(["same", "a", "suffix"]), result(["same", "b", "suffix"]))
    assert comparison["same_token_sequence"] is False
    assert len(comparison["matching_context_positions"]) == 2
    assert comparison["matching_context_positions"][-1]["chosen_probability_delta"] is None


def test_top_k_boundary_changes_do_not_imply_zero_probability() -> None:
    def token(alternate: str, value: float) -> dict:
        return {
            "token": "chosen",
            "logprob": math.log(0.6),
            "top_logprobs": [
                {"token": "chosen", "logprob": math.log(0.6)},
                {"token": alternate, "logprob": math.log(value)},
            ],
        }

    comparison = token_distribution_comparison(token("a", 0.01), token("b", 0.01001))
    assert comparison["shared_probability_delta"] == 0
    assert comparison["probability_delta_lower_bound"] == pytest.approx(0.00001)
    assert len(comparison["support_changes"]) == 2
    assert token_distribution_comparison(token("a", 0.01), token("b", 0.2))[
        "probability_delta_lower_bound"
    ] == pytest.approx(0.19)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.1])
def test_later_invalid_logprobs_cannot_disappear_from_comparison(invalid: float) -> None:
    base = {"text": "same", "logprobs": {"content": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2}]}}
    invalid_result = {
        "text": "same",
        "logprobs": {"content": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": invalid}]},
    }
    with pytest.raises(AssertionError, match="invalid token log probability"):
        sequence_comparison(base, invalid_result)
    assert probability(float("-inf")) == 0


def test_native_failure_survives_later_independent_success_and_runtime_errors_abort(tmp_path: Path) -> None:
    evidence, failures = Evidence(tmp_path), []
    values = {
        name: {"text": text}
        for name, text in (("first", "cold"), ("warm_1", "warm"), ("warm_2", "warm"), ("reset", "wrong"))
    }
    native = record_check(evidence, failures, "native", lambda: check_buffered_native(values))
    assert native["passed"] is False
    for phase in ("mixed", "churn", "embedding"):
        assert record_check(evidence, failures, phase, lambda: {"independent_evidence": True})["passed"] is True
    assert failures == [{"phase": "native", "message": "native buffered base text did not restore"}]
    values["reset"]["text"] = "warm"
    assert check_buffered_native(values)["first_vs_warm_text_equal"] is False
    values["warm_1"]["text"] = "unstable"
    with pytest.raises(AssertionError, match="warm baseline"):
        check_buffered_native(values)

    def runtime_failure() -> dict:
        raise RuntimeError("engine stopped")

    with pytest.raises(RuntimeError, match="engine stopped"):
        record_check(evidence, failures, "runtime", runtime_failure)
