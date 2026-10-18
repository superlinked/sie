"""Real detached/attached PEFT storage accounting and inconsistent-state controls."""

from __future__ import annotations

import pytest
import torch
from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter

from tools.benchmarks.cross_encoder_lora.fixtures import prepare
from tools.benchmarks.cross_encoder_lora.tests.test_probe import tiny_snapshot
from tools.benchmarks.cross_encoder_lora.worker import adapter_storage


@pytest.fixture
def resident(tmp_path):
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    prepare(fixtures, snapshot, smoke=True)
    adapter = CrossEncoderLoraAdapter(snapshot, lora_paths={name: str(fixtures / name) for name in ("a", "b")})
    adapter.load("cpu")
    adapter.load_lora("a")
    adapter.load_lora("b")
    yield adapter
    adapter.unload()


def test_storage_covers_detached_residents_without_double_counting_attached_layers(resident):
    assert not any(".lora_" in name for name, _ in resident._model.model.named_parameters())
    detached = adapter_storage(resident)
    assert detached == {
        "resident_adapters": 2,
        "storage_count": 16,
        "tensor_storage_bytes": 4096,
        "rounded_storage_bytes": 8192,
    }
    resident.set_active_lora("a")
    assert adapter_storage(resident) == detached
    resident.set_active_lora("b")
    assert adapter_storage(resident) == detached
    resident.set_active_lora(None)
    assert adapter_storage(resident) == detached
    resident.unload_lora("a")
    assert adapter_storage(resident) == {name: value // 2 for name, value in detached.items()}
    resident.unload_lora("b")
    assert adapter_storage(resident) == dict.fromkeys(detached, 0)


@pytest.mark.parametrize("damage", ["source", "bytes", "orphan"])
def test_storage_refuses_incomplete_resident_accounting(resident, damage):
    manager = resident._lora
    source = next(iter(manager._loaded))
    if damage == "source":
        del manager._parameters[source]
    elif damage == "bytes":
        manager._loaded[source] += 1
    else:
        layer = next(iter(manager._resident_layers.values()))
        layer.lora_A["unaccounted"] = torch.nn.Linear(16, 4, bias=False)
    with pytest.raises(AssertionError, match="resident"):
        adapter_storage(resident)
