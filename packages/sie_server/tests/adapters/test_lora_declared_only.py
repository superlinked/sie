"""A request selects only the LoRA adapters its model's configuration declares.

A real SIE app on loopback serves a tiny PEFT-capable model whose forward runs
a real torch module, so a LoRA's effect, or its absence, shows in the encode
output. Every LoRA is a real PEFT adapter directory on disk.
"""

from __future__ import annotations

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
from fastapi import FastAPI
from peft import LoraConfig, get_peft_model
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.registry import ModelRegistry

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
        self.lora_load_attempts = []

    def load_lora(self, lora_path, revision=None):
        self.lora_load_attempts.append(lora_path)
        return super().load_lora(lora_path, revision=revision)

    def encode(self, items, output_types, *, instruction=None, is_query=False, prepared_items=None, options=None):
        x = torch.stack([features(item.text or "") for item in items])
        with torch.no_grad():
            dense = self._model(x).numpy().astype(np.float32)
        return EncodeOutput(dense=dense, is_query=is_query, dense_dim=4)
"""


def _model_yaml(profiles: dict[str, str]) -> str:
    lines = [
        f"sie_id: {MODEL}",
        "package_backed: true",
        "inputs:",
        "  text: true",
        "tasks:",
        "  encode:",
        "    dense:",
        "      dim: 4",
        "profiles:",
        "  default:",
        "    adapter_path: tiny_lora_adapter.py:TinyLoraAdapter",
        "    max_batch_tokens: 8192",
        "    adapter_options:",
        "      loadtime:",
        "        dense_dim: 4",
    ]
    for name, lora in profiles.items():
        lines += [
            f"  {name}:",
            "    extends: default",
            "    adapter_options:",
            "      runtime:",
            f"        lora_id: {lora}",
        ]
    return "\n".join(lines) + "\n"


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


def _base_vector(helpers: ModuleType, text: str) -> np.ndarray:
    with torch.no_grad():
        return helpers.tiny_encoder()(helpers.features(text).unsqueeze(0))[0].numpy()


def _encode(url: str, text: str, options: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"items": [{"text": text}]}
    if options is not None:
        body["params"] = {"options": options}
    return httpx.post(f"{url}/v1/encode/{MODEL}", json=body, headers={"Accept": "application/json"}, timeout=10)


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

    def load_attempts(self) -> list[str]:
        return list(self.app.state.registry.get(MODEL).lora_load_attempts)


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
def serve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serve_on_loopback: Callable[[FastAPI], AbstractContextManager[str]],
    _offline_apps: None,
) -> Callable[[dict[str, str]], AbstractContextManager[Served]]:
    async def no_config_watcher(self: ModelRegistry) -> None:
        _ = self

    monkeypatch.setattr(ModelRegistry, "start_hot_reload", no_config_watcher)

    @contextmanager
    def start(profiles: dict[str, str]) -> Iterator[Served]:
        models = tmp_path / "models"
        models.mkdir()
        (models / "tiny_lora_adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
        (models / "tiny-lora.yaml").write_text(_model_yaml(profiles), encoding="utf-8")
        app = AppFactory.create_app(AppStateConfig(models_dir=str(models), device="cpu"))
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

        with serve({"declared": declared}) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            by_profile = _dense(_encode_when_ready(server.url, "hello", {"profile": "declared"}))
            by_option = _dense(_encode_when_ready(server.url, "hello", {"lora_id": declared}))
            by_selection = _dense(_encode_when_ready(server.url, "hello", {"lora": declared}))

        assert not np.allclose(by_profile, base, atol=1e-3)
        assert np.allclose(by_option, by_profile, atol=1e-5)
        assert np.allclose(by_selection, by_profile, atol=1e-5)

    def test_an_undeclared_hub_repo_is_refused_without_a_load(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, hub_offline: None
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")

        with serve({"declared": declared}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"lora_id": "attacker-org/some-lora"})
            assert response.status_code == 400, response.text
            assert _error_code(response) == "INVALID_INPUT"
            assert server.loaded_ids() == {declared}
            assert server.load_attempts() == [declared]

    def test_an_undeclared_local_directory_is_refused_without_a_load(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        undeclared = _save_lora(helpers, tmp_path / "elsewhere" / "undeclared-lora")

        with serve({"declared": declared}) as server:
            base = _dense(_encode_when_ready(server.url, "hello"))
            for _ in range(3):
                response = _encode(server.url, "hello", {"lora_id": undeclared})
                assert response.status_code == 400, response.text
                assert _error_code(response) == "INVALID_INPUT"
            assert server.loaded_ids() == {declared}
            assert server.load_attempts() == [declared]
            assert np.allclose(_dense(_encode(server.url, "hello")), base, atol=1e-6)

    def test_an_undeclared_selection_is_refused_instead_of_hanging(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")

        with serve({"declared": declared}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"lora": "attacker/unknown"})
            assert response.status_code == 400, response.text
            assert _error_code(response) == "INVALID_INPUT"

    @pytest.mark.parametrize("initializer", ["pissa", "olora"])
    def test_a_declared_base_mutating_lora_leaves_the_shared_model_unchanged(
        self, tmp_path: Path, helpers: ModuleType, serve: Any, initializer: str
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        mutating = _save_lora(helpers, tmp_path / "mutating-lora", init_lora_weights=initializer)
        base = _base_vector(helpers, "hello")

        with serve({"declared": declared, "mutating": mutating}) as server:
            assert np.allclose(_dense(_encode_when_ready(server.url, "hello")), base, atol=1e-5)
            assert server.loaded_ids() == {declared}

    def test_selecting_a_declared_lora_that_is_not_loaded_fails_instead_of_hanging(
        self, tmp_path: Path, helpers: ModuleType, serve: Any
    ) -> None:
        declared = _save_lora(helpers, tmp_path / "declared-lora")
        missing = str(tmp_path / "missing-lora")

        with serve({"declared": declared, "missing": missing}) as server:
            _encode_when_ready(server.url, "warm")
            response = _encode(server.url, "hello", {"lora": missing})
            assert response.status_code == 400, response.text
            assert _error_code(response) == "INVALID_INPUT"
            assert _encode(server.url, "hello", {"lora": declared}).status_code == 200
