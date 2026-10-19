"""A request selects only the LoRA adapters its selected profile declares.

A tiny PEFT-capable model whose forward runs a real torch module serves the
requests, so a LoRA's effect, or its absence, shows in the encode output.
Every LoRA is a real PEFT adapter directory on disk. The direct HTTP tests run
a real SIE app on loopback; the queued and sidecar tests drive a real
registry, worker and queue executor.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
import torch
import yaml
from fastapi import FastAPI
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from sie_server.adapter_call_loop import handle_run_batch
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    RunBatchItem,
    RunBatchRequest,
    ScoreBatchItem,
)
from sie_server.queue_executor import QueueExecutor

MODEL = "acme/tiny-lora"
ADAPTER_SOURCE = """\
import hashlib

import numpy as np
import torch
from sie_server.adapters.fake.adapter import FakeAdapter
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.out = nn.Linear(4, 4)

    def forward(self, x):
        return self.out(torch.relu(self.query(x)))


def tiny_encoder() -> TinyEncoder:
    model = TinyEncoder()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return model


def features(text: str):
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return torch.randn(4, generator=torch.Generator().manual_seed(seed))


class TinyLoraAdapter(PEFTLoRAMixin, FakeAdapter):
    def load(self, device: str) -> None:
        super().load(device)
        self._model = tiny_encoder()
        self.lora_fetches = []

    def fetch_lora(self, lora_path, revision=None):
        self.lora_fetches.append(lora_path)
        return super().fetch_lora(lora_path, revision=revision)

    def encode(self, items, output_types, *, instruction=None, is_query=False, prepared_items=None, options=None):
        x = torch.stack([features(item.text or "") for item in items])
        with torch.no_grad():
            dense = self._model(x).numpy().astype(np.float32)
        return EncodeOutput(dense=dense, is_query=is_query, dense_dim=4)
"""


def _model_yaml(profiles: dict[str, dict[str, Any]]) -> str:
    config: dict[str, Any] = {
        "sie_id": MODEL,
        "package_backed": True,
        "inputs": {"text": True},
        "tasks": {"encode": {"dense": {"dim": 4}}, "extract": {}, "score": {}},
        "profiles": {
            "default": {
                "adapter_path": "tiny_lora_adapter.py:TinyLoraAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"dense_dim": 4}},
            }
        },
    }
    for name, spec in profiles.items():
        adapter_options: dict[str, Any] = {}
        runtime = {key: spec[key] for key in ("lora_id", "lora") if key in spec}
        if runtime:
            adapter_options["runtime"] = runtime
        if "lora_paths" in spec:
            adapter_options["loadtime"] = {"dense_dim": 4, "lora_paths": spec["lora_paths"]}
        config["profiles"][name] = {"extends": "default", "adapter_options": adapter_options}
    return yaml.safe_dump(config, sort_keys=False)


def _save_lora(helpers: ModuleType, path: Path, **config_overrides: Any) -> str:
    torch.manual_seed(1)
    trained = get_peft_model(
        helpers.tiny_encoder(), LoraConfig(r=2, target_modules=["query"], bias="none", init_lora_weights=False)
    )
    trained.save_pretrained(str(path))
    if config_overrides:
        config_path = path / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config.update(config_overrides)
        config_path.write_text(json.dumps(config))
    return str(path)


def _save_rank_mismatched_lora(helpers: ModuleType, path: Path) -> str:
    torch.manual_seed(1)
    trained = get_peft_model(
        helpers.tiny_encoder(),
        LoraConfig(r=2, target_modules=["query", "out"], bias="none", init_lora_weights=False),
    )
    trained.save_pretrained(str(path))
    weights = path / "adapter_model.safetensors"
    tensors = load_file(str(weights))
    tensors["base_model.model.out.lora_A.weight"] = torch.ones(3, 4)
    save_file(tensors, str(weights))
    return str(path)


def _base_vector(helpers: ModuleType, text: str) -> np.ndarray:
    with torch.no_grad():
        return helpers.tiny_encoder()(helpers.features(text).unsqueeze(0))[0].numpy()


def _encode(url: str, text: str, options: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"items": [{"text": text}]}
    if options is not None:
        body["params"] = {"options": options}
    return httpx.post(f"{url}/v1/encode/{MODEL}", json=body, headers={"Accept": "application/json"}, timeout=10)


def _extract(url: str, options: dict[str, Any] | None = None) -> httpx.Response:
    params: dict[str, Any] = {"labels": ["thing"]}
    if options is not None:
        params["options"] = options
    body = {"items": [{"text": "hello"}], "params": params}
    return httpx.post(f"{url}/v1/extract/{MODEL}", json=body, headers={"Accept": "application/json"}, timeout=10)


def _score(url: str, options: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"query": {"text": "hello"}, "items": [{"text": "world"}]}
    if options is not None:
        body["options"] = options
    return httpx.post(f"{url}/v1/score/{MODEL}", json=body, headers={"Accept": "application/json"}, timeout=10)


def _encode_when_ready(url: str, text: str, options: dict[str, Any] | None = None) -> httpx.Response:
    deadline = time.monotonic() + 30
    while True:
        response = _encode(url, text, options)
        if response.status_code != 503 or time.monotonic() > deadline:
            return response
        time.sleep(0.05)


def _dense(response: httpx.Response) -> np.ndarray:
    assert response.status_code == 200, response.text
    return np.asarray(response.json()["items"][0]["dense"]["values"], dtype=np.float32)


def _error_code(response: httpx.Response) -> str:
    return response.json()["detail"]["code"]


@dataclass
class Served:
    url: str
    app: FastAPI

    def loaded_ids(self) -> set[str]:
        return set(self.app.state.registry._loaded[MODEL].loras)

    def fetches(self) -> list[str]:
        return list(self.app.state.registry.get(MODEL).lora_fetches)


@pytest.fixture
def helpers(tmp_path: Path) -> ModuleType:
    source = tmp_path / "helpers" / "tiny_lora_adapter.py"
    source.parent.mkdir()
    source.write_text(ADAPTER_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tiny_lora_adapter_helpers", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, dict[str, Any]]], Path]:
    async def no_config_watcher(self: ModelRegistry) -> None:
        _ = self

    monkeypatch.setattr(ModelRegistry, "start_hot_reload", no_config_watcher)

    def write(profiles: dict[str, dict[str, Any]]) -> Path:
        models = tmp_path / "models"
        models.mkdir()
        (models / "tiny_lora_adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
        (models / "tiny-lora.yaml").write_text(_model_yaml(profiles), encoding="utf-8")
        return models

    return write


@pytest.fixture
def serve(
    models_dir: Callable[[dict[str, dict[str, Any]]], Path],
    serve_on_loopback: Callable[[FastAPI], AbstractContextManager[str]],
    _offline_apps: None,
) -> Callable[[dict[str, dict[str, Any]]], AbstractContextManager[Served]]:
    @contextmanager
    def start(profiles: dict[str, dict[str, Any]]) -> Iterator[Served]:
        app = AppFactory.create_app(AppStateConfig(models_dir=str(models_dir(profiles)), device="cpu"))
        with serve_on_loopback(app) as url:
            yield Served(url=url, app=app)

    return start


@pytest.fixture
def hub_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)


class TestDirectEncodeRoute:
    def test_a_declared_lora_serves_by_profile_and_by_request_option(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        base = _base_vector(helpers, "hello")

        with serve({"declared": {"lora_id": declared}}) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            by_profile = _dense(_encode_when_ready(server.url, "hello", {"profile": "declared"}))
            by_option = _dense(_encode_when_ready(server.url, "hello", {"profile": "declared", "lora_id": declared}))
            by_selection = _dense(_encode_when_ready(server.url, "hello", {"profile": "declared", "lora": declared}))

        assert not np.allclose(by_profile, base, atol=1e-3)
        assert np.allclose(by_option, by_profile, atol=1e-5)
        assert np.allclose(by_selection, by_profile, atol=1e-5)

    def test_an_undeclared_hub_repo_is_refused_without_a_fetch(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, hub_offline: None
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")

        with serve({"declared": {"lora_id": declared}}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"profile": "declared", "lora_id": "attacker-org/some-lora"})
            assert response.status_code == 400, response.text
            assert _error_code(response) == "INVALID_INPUT"
            assert server.loaded_ids() == {declared}
            assert server.fetches() == [declared]

    def test_an_undeclared_local_directory_is_refused_without_a_fetch(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        undeclared = _save_lora(helpers, tmp_path / "elsewhere" / "undeclared-lora")

        with serve({"declared": {"lora_id": declared}}) as server:
            base = _dense(_encode_when_ready(server.url, "hello"))
            for _ in range(3):
                response = _encode(server.url, "hello", {"lora_id": undeclared})
                assert response.status_code == 400, response.text
                assert _error_code(response) == "INVALID_INPUT"
            assert server.loaded_ids() == {declared}
            assert server.fetches() == [declared]
            assert np.allclose(_dense(_encode(server.url, "hello")), base, atol=1e-6)

    def test_a_lora_another_profile_declares_is_refused(self, tmp_path: Path, helpers: ModuleType, serve: Any) -> None:
        banking = _save_lora(helpers, tmp_path / "banking-lora")
        legal = _save_lora(helpers, tmp_path / "legal-lora")

        with serve({"banking": {"lora_id": banking}, "legal": {"lora_id": legal}}) as server:
            _encode_when_ready(server.url, "warm")
            for options in (
                {"lora_id": banking},
                {"lora": banking},
                {"profile": "legal", "lora_id": banking},
                {"profile": "legal", "lora": banking},
            ):
                response = _encode(server.url, "hello", options)
                assert response.status_code == 400, (options, response.text)
                assert _error_code(response) == "INVALID_INPUT"

    def test_an_undeclared_selection_is_refused_instead_of_hanging(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")

        with serve({"declared": {"lora_id": declared}}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"profile": "declared", "lora": "attacker/unknown"})
            assert response.status_code == 400, response.text
            assert _error_code(response) == "INVALID_INPUT"

    @pytest.mark.parametrize("initializer", ["pissa", "olora"])
    def test_a_declared_base_mutating_lora_leaves_the_shared_model_unchanged(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, initializer: str
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        mutating = _save_lora(helpers, tmp_path / "mutating-lora", init_lora_weights=initializer)
        base = _base_vector(helpers, "hello")

        with serve({"declared": {"lora_id": declared}, "mutating": {"lora_id": mutating}}) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            assert server.loaded_ids() == {declared}

    def test_a_declared_lora_whose_weights_reach_the_base_model_is_refused(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        overwriting = _save_lora(helpers, tmp_path / "overwriting-lora")
        weights = Path(overwriting) / "adapter_model.safetensors"
        tensors = load_file(str(weights))
        tensors["base_model.model.out.weight"] = torch.zeros(4, 4)
        tensors["base_model.model.out.bias"] = torch.full((4,), 7.0)
        save_file(tensors, str(weights))
        base = _base_vector(helpers, "hello")

        with serve({"declared": {"lora_id": declared}, "overwriting": {"lora_id": overwriting}}) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            assert server.loaded_ids() == {declared}

    def test_a_declared_lora_with_bin_weights_is_refused(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        pickled = _save_lora(helpers, tmp_path / "pickled-lora")
        weights = Path(pickled) / "adapter_model.safetensors"
        torch.save(load_file(str(weights)), Path(pickled) / "adapter_model.bin")
        weights.unlink()

        with serve({"declared": {"lora_id": declared}, "pickled": {"lora_id": pickled}}) as server:
            _encode_when_ready(server.url, "warm")
            assert server.loaded_ids() == {declared}

        assert "ships no adapter_model.safetensors" in caplog.text

    def test_a_failing_declared_lora_backs_off_with_a_retryable_answer(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        missing = str(tmp_path / "missing-lora")

        with serve({"declared": {"lora_id": declared}, "missing": {"lora_id": missing}}) as server:
            _encode_when_ready(server.url, "warm")
            first = _encode(server.url, "hello", {"profile": "missing"})
            assert first.status_code == 503, first.text
            assert _error_code(first) == "LORA_LOADING"
            assert "profile 'missing'" in first.text
            assert missing not in first.text
            deadline = time.monotonic() + 10
            while not server.app.state.registry.lora_load_failed(MODEL, missing):
                assert time.monotonic() < deadline, "the failing load was not recorded"
                time.sleep(0.05)
            fetches = server.fetches().count(missing)
            for _ in range(5):
                response = _encode(server.url, "hello", {"profile": "missing"})
                assert response.status_code == 503, response.text
                assert _error_code(response) == "LORA_LOADING"
                assert int(response.headers["Retry-After"]) > 1
                assert missing not in response.text
            assert server.fetches().count(missing) == fetches
            own = _encode(server.url, "hello", {"profile": "missing", "lora_id": missing})
            assert own.status_code == 503, own.text
            assert missing in own.text

    def test_a_caller_who_names_only_the_profile_is_not_told_its_adapter_source(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        selected = str(tmp_path / "selected-lora")
        loaded = str(tmp_path / "loaded-lora")

        with serve({"selects": {"lora_paths": [selected], "lora": selected}, "loads": {"lora_id": loaded}}) as server:
            _encode_when_ready(server.url, "warm")
            for profile in ("selects", "loads"):
                for _ in range(3):
                    response = _encode(server.url, "hello", {"profile": profile})
                    assert response.status_code == 503, response.text
                    assert _error_code(response) == "LORA_LOADING"
                    assert str(tmp_path) not in response.text
            deadline = time.monotonic() + 30
            while True:
                response = httpx.post(
                    f"{server.url}/v1/embeddings", json={"model": f"{MODEL}:loads", "input": "hello"}, timeout=10
                )
                assert response.status_code == 503, response.text
                assert str(tmp_path) not in response.text
                if "LORA_LOADING" in response.text:
                    break
                assert time.monotonic() < deadline, response.text
                time.sleep(0.05)

    @pytest.mark.parametrize("after_a_loaded_lora", [False, True])
    def test_a_declared_lora_whose_attach_fails_leaves_the_base_model_unchanged(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, after_a_loaded_lora: bool
    ) -> None:
        broken = _save_rank_mismatched_lora(helpers, tmp_path / "broken-lora")
        profiles: dict[str, dict[str, Any]] = {"broken": {"lora_id": broken}}
        if after_a_loaded_lora:
            profiles = {"declared": {"lora_id": _save_lora(helpers, tmp_path / "declared-lora")}, **profiles}
        base = _base_vector(helpers, "hello")

        with serve(profiles) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            failures = server.app.state.registry._loaded[MODEL].lora_failures
            for attempt in (1, 2):
                response = _encode(server.url, "hello", {"profile": "broken"})
                assert response.status_code == 503, response.text
                deadline = time.monotonic() + 10
                while broken not in failures or failures[broken].attempts < attempt:
                    assert time.monotonic() < deadline, "the failing load was not recorded"
                    time.sleep(0.05)
                failures[broken].retry_at = time.monotonic() - 1
                assert np.allclose(_dense(_encode(server.url, "hello")), base, atol=1e-5)
            assert broken not in server.loaded_ids()
            adapter = server.app.state.registry.get(MODEL)
            if after_a_loaded_lora:
                assert adapter._peft_adapter_name(broken) not in adapter._peft_model.peft_config
            else:
                assert adapter._peft_model is None

    def test_selecting_a_declared_lora_that_is_not_loaded_is_retryable_instead_of_hanging(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        missing = str(tmp_path / "missing-lora")

        with serve({"declared": {"lora_id": declared}, "missing": {"lora_paths": [missing]}}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"profile": "missing", "lora": missing})
            assert response.status_code == 503, response.text
            assert _error_code(response) == "LORA_LOADING"
            assert "Retry-After" in response.headers
            assert declared not in response.text
            assert _encode(server.url, "hello", {"profile": "declared", "lora": declared}).status_code == 200

    def test_selecting_a_declared_lora_when_none_is_loaded_does_not_serve_the_base_model(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        missing = str(tmp_path / "missing-lora")

        with serve({"missing": {"lora_paths": [missing]}}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"profile": "missing", "lora": missing})
            assert response.status_code == 503, response.text
            assert _error_code(response) == "LORA_LOADING"


class TestDirectExtractAndScoreRoutes:
    def test_an_undeclared_or_cross_profile_lora_is_refused_without_a_fetch(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, hub_offline: None
    ) -> None:
        banking = _save_lora(helpers, tmp_path / "banking-lora")
        legal = _save_lora(helpers, tmp_path / "legal-lora")

        with serve({"banking": {"lora_id": banking}, "legal": {"lora_id": legal}}) as server:
            _encode_when_ready(server.url, "warm")
            assert _extract(server.url).status_code == 200
            assert _score(server.url).status_code == 200
            for options in (
                {"lora_id": "attacker-org/some-lora"},
                {"lora": "attacker-org/some-lora"},
                {"lora_id": banking},
                {"profile": "legal", "lora_id": banking},
                {"profile": "legal", "lora": banking},
            ):
                for response in (_extract(server.url, options), _score(server.url, options)):
                    assert response.status_code == 400, (options, response.text)
                    assert _error_code(response) == "INVALID_INPUT"
            assert server.loaded_ids() == {banking, legal}
            assert sorted(server.fetches()) == sorted([banking, legal])


class TestQueuedAndSidecarPaths:
    """The worker's pre-formed batch path, reached through the queue executor and the sidecar's RunBatch."""

    @staticmethod
    async def _executor(models: Path) -> QueueExecutor:
        executor = QueueExecutor(ModelRegistry(models_dir=str(models), device="cpu"))
        deadline = time.monotonic() + 30
        while (state := await executor.ensure_model_ready(MODEL)) != "ready":
            assert state in {"loading_started", "loading_in_progress"}, state
            assert time.monotonic() < deadline, "the model did not load"
            await asyncio.sleep(0.05)
        return executor

    @staticmethod
    def _encode_item(options: dict[str, Any]) -> EncodeBatchItem:
        return EncodeBatchItem(
            work_item_id="req-1.0",
            request_id="req-1",
            item_index=0,
            total_items=1,
            timestamp=time.time(),
            item={"text": "hello"},
            output_types=["dense"],
            options=options,
        )

    @staticmethod
    def _extract_item(options: dict[str, Any]) -> ExtractBatchItem:
        return ExtractBatchItem(
            work_item_id="req-2.0",
            request_id="req-2",
            item_index=0,
            total_items=1,
            timestamp=time.time(),
            item={"text": "hello"},
            labels=["thing"],
            options=options,
        )

    @staticmethod
    def _score_item(options: dict[str, Any]) -> ScoreBatchItem:
        return ScoreBatchItem(
            work_item_id="req-3.0",
            request_id="req-3",
            item_index=0,
            total_items=1,
            timestamp=time.time(),
            query_item={"text": "hello"},
            score_items=[{"text": "world"}],
            options=options,
        )

    async def test_a_queued_score_refuses_an_undeclared_or_cross_profile_lora(
        self, tmp_path: Path, helpers: ModuleType, models_dir: Callable[[dict[str, dict[str, Any]]], Path]
    ) -> None:
        banking = _save_lora(helpers, tmp_path / "banking-lora")
        legal = _save_lora(helpers, tmp_path / "legal-lora")
        executor = await self._executor(models_dir({"banking": {"lora_id": banking}, "legal": {"lora_id": legal}}))

        (served,) = (
            await executor.process_score_batch(ProcessScoreBatchRequest(model_id=MODEL, items=[self._score_item({})]))
        ).outcomes
        assert served.error is None, served.error
        for options in (
            {"lora_id": "attacker-org/some-lora"},
            {"lora": "attacker-org/some-lora"},
            {"profile": "legal", "lora_id": banking},
            {"profile": "legal", "lora": banking},
        ):
            outcome = await executor.process_score_batch(
                ProcessScoreBatchRequest(model_id=MODEL, items=[self._score_item(options)])
            )
            (item,) = outcome.outcomes
            assert item.disposition == "publish_error_and_ack", options
            assert item.error_code == "INVALID_INPUT", (options, item.error)
        assert set(executor._registry._loaded[MODEL].loras) == {banking, legal}

    async def test_a_caller_who_names_only_the_profile_is_not_told_its_adapter_source(
        self, tmp_path: Path, helpers: ModuleType, models_dir: Callable[[dict[str, dict[str, Any]]], Path]
    ) -> None:
        selected = str(tmp_path / "selected-lora")
        executor = await self._executor(models_dir({"tenant": {"lora_paths": [selected], "lora": selected}}))
        options = {"profile": "tenant"}

        queued = await executor.process_encode_batch(
            ProcessEncodeBatchRequest(model_id=MODEL, items=[self._encode_item(options)])
        )
        extracted = await executor.process_extract_batch(
            ProcessExtractBatchRequest(model_id=MODEL, items=[self._extract_item(options)])
        )
        sidecar = await handle_run_batch(
            executor,
            RunBatchRequest(
                model_id=MODEL,
                batch_id=1,
                lora_key="",
                total_cost=1,
                items=[RunBatchItem(op="encode", encode=self._encode_item(options))],
            ),
        )

        for outcome in (*queued.outcomes, *extracted.outcomes, *sidecar.outcomes):
            assert outcome.disposition == "publish_error_and_ack"
            assert outcome.error_code == "LORA_LOADING", outcome.error
            assert outcome.error is not None
            assert str(tmp_path) not in outcome.error

    async def test_a_served_name_the_adapter_cannot_select_fails_without_naming_loaded_adapters(
        self, tmp_path: Path, helpers: ModuleType, models_dir: Callable[[dict[str, dict[str, Any]]], Path]
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "private-lora")
        executor = await self._executor(models_dir({"served": {"lora_paths": {"legal": declared}}}))
        options = {"profile": "served", "lora": "legal"}

        queued = await executor.process_encode_batch(
            ProcessEncodeBatchRequest(model_id=MODEL, items=[self._encode_item(options)])
        )
        extracted = await executor.process_extract_batch(
            ProcessExtractBatchRequest(model_id=MODEL, items=[self._extract_item(options)])
        )
        sidecar = await handle_run_batch(
            executor,
            RunBatchRequest(
                model_id=MODEL,
                batch_id=1,
                lora_key="legal",
                total_cost=1,
                items=[RunBatchItem(op="encode", encode=self._encode_item(options))],
            ),
        )

        for outcome in (*queued.outcomes, *extracted.outcomes, *sidecar.outcomes):
            assert outcome.disposition == "publish_error_and_ack"
            assert outcome.error_code == "INVALID_INPUT", outcome.error
            assert outcome.error is not None
            assert "private-lora" not in outcome.error

    async def test_a_declared_lora_that_is_not_loaded_is_retryable(
        self, tmp_path: Path, helpers: ModuleType, models_dir: Callable[[dict[str, dict[str, Any]]], Path]
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        missing = str(tmp_path / "missing-lora")
        executor = await self._executor(
            models_dir({"declared": {"lora_paths": [declared]}, "missing": {"lora_paths": [missing]}})
        )

        outcome = await handle_run_batch(
            executor,
            RunBatchRequest(
                model_id=MODEL,
                batch_id=1,
                lora_key=missing,
                total_cost=1,
                items=[RunBatchItem(op="encode", encode=self._encode_item({"profile": "missing", "lora": missing}))],
            ),
        )

        (item,) = outcome.outcomes
        assert item.disposition == "publish_error_and_ack"
        assert item.error_code == "LORA_LOADING", item.error
        assert item.retry_after_s is not None
        assert item.error is not None
        assert declared not in item.error
