"""Generic host-RAM parking of in-process tensors (``_host_offload`` and ``BaseAdapter``).

CI has no GPU, so these tests park from ``cpu`` to host memory: every
non-empty CPU tensor counts as "on the device", and each park and wake must
produce fresh storage while keeping values, parameter identity and structure.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch
from sie_server.adapters import _host_offload
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec

CPU = torch.device("cpu")


class _Tied(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.head.weight = self.embed.weight  # tied
        self.register_buffer("scale", torch.ones(4))
        self.cached = torch.arange(4.0)  # plain tensor attribute


class _Pipeline:
    """Stands in for a transformers pipeline: the module is its ``model``."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)


class _Holder:
    def __init__(self) -> None:
        self.model = _Tied()
        self.loose = torch.full((3,), 7.0)
        self.table = {"a": torch.ones(2), "nested": [torch.zeros(2)]}
        self.pipe = _Pipeline()
        self.name = "not a tensor"


def _ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_park_and_unpark_move_every_reachable_tensor_and_keep_identity() -> None:
    holder = _Holder()
    weight = holder.model.embed.weight
    before = {
        "weight": weight.detach().clone(),
        "scale": holder.model.scale.clone(),
        "cached": holder.model.cached.clone(),
        "loose": holder.loose.clone(),
        "a": holder.table["a"].clone(),
        "pipe": holder.pipe.model.weight.detach().clone(),
    }
    old_ptrs = {
        "weight": _ptr(weight),
        "scale": _ptr(holder.model.scale),
        "cached": _ptr(holder.model.cached),
        "loose": _ptr(holder.loose),
        "a": _ptr(holder.table["a"]),
        "nested": _ptr(holder.table["nested"][0]),
        "pipe": _ptr(holder.pipe.model.weight),
    }

    expected = _host_offload.parkable_bytes(holder, CPU)
    assert expected is not None
    parked = _host_offload.park(holder, CPU)
    # The tied weight moves once.
    assert parked.nbytes == expected
    assert holder.model.head.weight is holder.model.embed.weight is weight
    parked_ptrs = {
        "weight": _ptr(weight),
        "scale": _ptr(holder.model.scale),
        "cached": _ptr(holder.model.cached),
        "loose": _ptr(holder.loose),
        "a": _ptr(holder.table["a"]),
        "nested": _ptr(holder.table["nested"][0]),
        "pipe": _ptr(holder.pipe.model.weight),
    }
    assert all(parked_ptrs[key] != old_ptrs[key] for key in old_ptrs)

    _host_offload.unpark(parked)
    assert holder.model.head.weight is weight
    assert all(
        _ptr(tensor) != parked_ptrs[key]
        for key, tensor in {
            "weight": weight,
            "scale": holder.model.scale,
            "loose": holder.loose,
            "a": holder.table["a"],
            "pipe": holder.pipe.model.weight,
        }.items()
    )
    assert torch.equal(weight, before["weight"])
    assert torch.equal(holder.model.scale, before["scale"])
    assert torch.equal(holder.model.cached, before["cached"])
    assert torch.equal(holder.loose, before["loose"])
    assert torch.equal(holder.table["a"], before["a"])
    assert torch.equal(holder.pipe.model.weight, before["pipe"])
    assert holder.name == "not a tensor"


def test_unpark_again_is_a_no_op() -> None:
    holder = _Holder()
    parked = _host_offload.park(holder, CPU)
    _host_offload.unpark(parked)
    restored = _ptr(holder.model.embed.weight)
    _host_offload.unpark(parked)
    assert _ptr(holder.model.embed.weight) == restored


def test_a_tensor_in_a_tuple_makes_the_adapter_unparkable() -> None:
    holder = _Holder()
    holder.pair = (torch.ones(2), "label")
    assert _host_offload.parkable_bytes(holder, CPU) is None


def test_a_quantized_parameter_subclass_makes_the_adapter_unparkable() -> None:
    class _Quantized(torch.nn.Parameter):
        pass

    holder = _Holder()
    holder.model.extra = torch.nn.Linear(2, 2)
    holder.model.extra.weight = _Quantized(torch.ones(2, 2))
    assert _host_offload.parkable_bytes(holder, CPU) is None


def test_accelerate_hooks_make_the_adapter_unparkable() -> None:
    holder = _Holder()
    holder.model.embed._hf_hook = object()
    assert _host_offload.parkable_bytes(holder, CPU) is None


def test_nothing_on_the_device_is_unparkable() -> None:
    holder = _Holder()
    assert _host_offload.parkable_bytes(holder, torch.device("cuda", 0)) is None


class _GraphRunner:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


class _Adapter(BaseAdapter):
    spec: ClassVar[AdapterSpec] = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=("_model",))

    def __init__(self, device: str | None) -> None:
        self._model = torch.nn.Linear(4, 4)
        self._device = device
        self._graphs: Any = _GraphRunner()
        self._cuda_graphs = "bucketed"

    def load(self, device: str) -> None:
        self._device = device

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_base_adapter_does_not_park_off_cuda() -> None:
    assert _Adapter("cpu").host_offload_bytes() is None
    assert _Adapter("mps").host_offload_bytes() is None
    assert _Adapter(None).host_offload_bytes() is None


def test_base_adapter_parks_clears_graphs_and_wakes(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Adapter("cpu")
    monkeypatch.setattr(adapter, "_host_offload_device", lambda: CPU)
    weight = adapter._model.weight
    values = weight.detach().clone()
    size = adapter.host_offload_bytes()
    assert size == sum(p.nbytes for p in adapter._model.parameters())

    offload = adapter.offload_to_host()
    assert offload.host_bytes == size
    assert not offload.retains_device_memory
    assert adapter._graphs.cleared == 1
    # Parked: no second park, and the size check declines.
    assert adapter.host_offload_bytes() is None
    with pytest.raises(RuntimeError, match="nothing on a CUDA device"):
        adapter.offload_to_host()

    assert adapter.restore_from_host() == 0
    assert adapter._model.weight is weight
    assert torch.equal(weight, values)
    with pytest.raises(RuntimeError, match="not parked"):
        adapter.restore_from_host()


def test_a_graph_holder_without_clear_blocks_parking(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Adapter("cpu")
    monkeypatch.setattr(adapter, "_host_offload_device", lambda: CPU)
    adapter._prefix_graphs = object()
    assert adapter.host_offload_bytes() is None


def test_opting_out_disables_parking(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Adapter("cpu")
    monkeypatch.setattr(adapter, "_host_offload_device", lambda: CPU)
    monkeypatch.setattr(_Adapter, "host_offload", False)
    assert adapter.host_offload_bytes() is None


def test_unload_while_parked_drops_the_host_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Adapter("cpu")
    monkeypatch.setattr(adapter, "_host_offload_device", lambda: CPU)
    adapter.offload_to_host()
    adapter.unload()
    assert adapter._parked_tensors is None
    assert adapter._model is None


def test_weights_the_walk_cannot_reach_block_parking(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Adapter("cpu")
    monkeypatch.setattr(adapter, "_host_offload_device", lambda: CPU)
    reachable = adapter.host_offload_bytes()
    assert reachable is not None
    # The adapter reports more weights than the walk found.
    monkeypatch.setattr(adapter, "memory_footprint", lambda: reachable * 2)
    assert adapter.host_offload_bytes() is None


def test_aliases_of_one_tensor_are_copied_once_and_stay_aliases() -> None:
    holder = _Holder()
    holder.model.original_scale = holder.model.scale  # a plain-attribute alias of a buffer
    holder.weight_alias = holder.model.embed.weight.data  # a loose alias of a parameter's data

    expected = _host_offload.parkable_bytes(holder, CPU)
    parked = _host_offload.park(holder, CPU)
    assert parked.nbytes == expected
    assert _ptr(holder.model.original_scale) == _ptr(holder.model.scale)
    assert _ptr(holder.weight_alias) == _ptr(holder.model.embed.weight)

    _host_offload.unpark(parked)
    assert _ptr(holder.model.original_scale) == _ptr(holder.model.scale)
    assert _ptr(holder.weight_alias) == _ptr(holder.model.embed.weight)
