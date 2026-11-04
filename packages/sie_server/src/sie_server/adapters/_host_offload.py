"""Move an in-process adapter's tensors to host RAM and back.

This is the generic half of host-RAM parking for adapters that hold their
weights in this process (``BaseAdapter`` subclasses). It finds the tensors an
adapter references on its device and copies them to host memory, freeing the
device copies, then copies them back on wake.

What it finds, starting from the adapter's instance attributes:

- ``torch.nn.Module`` objects: every parameter, buffer and plain tensor
  attribute of every submodule. Parameters keep their identity (only
  ``.data`` moves), so references to the parameter objects held elsewhere,
  such as a PEFT wrapper around the same modules, follow along;
- loose tensors, and tensors or modules inside dicts and lists, a few levels
  deep;
- wrapper objects that hold a module as their ``model`` attribute
  (transformers pipelines, FlagEmbedding models).

An adapter is not parkable when a device tensor sits where it cannot be
rebound (inside a tuple), when a parameter is a quantized subclass the copy
would not preserve, when accelerate dispatch hooks manage placement, or when
nothing on the device was found at all. Device state outside this reach, such
as a tensor captured in a closure, stays where it is: it remains valid, but
its memory is not freed while the model is parked. Adapters with such state
override the parking hooks or set ``host_offload = False``.

Host copies are ordinary (pageable) CPU tensors. PyTorch's pinned-memory
allocator rounds every block up to a power of two and keeps freed blocks
cached, which would make the host tier's budget fiction; pageable copies are
accounted exactly and released to the OS when dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

# How deep to follow dicts and lists (and wrappers) below an adapter attribute.
_MAX_CONTAINER_DEPTH = 4


@dataclass(frozen=True)
class _Slot:
    """Where a loose tensor lives, so it can be rebound."""

    owner: Any
    key: Any
    attribute: bool

    def get(self) -> object:
        if self.attribute:
            return getattr(self.owner, self.key, None)
        try:
            return self.owner[self.key]
        except (IndexError, KeyError):
            return None

    def set(self, value: torch.Tensor) -> None:
        if self.attribute:
            setattr(self.owner, self.key, value)
        else:
            self.owner[self.key] = value


@dataclass
class _Reach:
    modules: list[torch.nn.Module] = field(default_factory=list)
    loose: list[tuple[_Slot, torch.Tensor]] = field(default_factory=list)
    unmovable_bytes: int = 0


@dataclass
class ParkedTensors:
    """An adapter's tensors parked in host RAM, and where they came from."""

    modules: list[torch.nn.Module]
    loose: list[tuple[_Slot, torch.Tensor]]
    # Host copies keyed by their storage address, with the device each came
    # from. Holding the copies keeps the addresses unique until restore.
    origins: dict[int, torch.device]
    copies: list[torch.Tensor]

    @property
    def nbytes(self) -> int:
        return sum(copy.nbytes for copy in self.copies)


def _on(device: torch.device, tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` holds memory on ``device`` (empty tensors hold none)."""
    if tensor.device.type != device.type or tensor.untyped_storage().nbytes() == 0:
        return False
    return device.index is None or tensor.device.index == device.index


def _collect(adapter: object, device: torch.device) -> _Reach:
    reach = _Reach()
    seen_modules: set[int] = set()
    seen: set[int] = set()

    def add_module(module: torch.nn.Module) -> None:
        if id(module) not in seen_modules:
            seen_modules.add(id(module))
            reach.modules.append(module)

    def visit(value: object, slot: _Slot | None, depth: int) -> None:
        if isinstance(value, torch.nn.Module):
            add_module(value)
            return
        if isinstance(value, torch.Tensor):
            if _on(device, value):
                if slot is None:
                    reach.unmovable_bytes += value.nbytes
                else:
                    reach.loose.append((slot, value))
            return
        if depth >= _MAX_CONTAINER_DEPTH or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in list(value.items()):
                visit(item, _Slot(value, key, attribute=False), depth + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, _Slot(value, index, attribute=False), depth + 1)
        elif isinstance(value, tuple):
            for item in value:
                visit(item, None, depth + 1)
        else:
            # A wrapper's own ``model`` attribute, read from its instance
            # dictionary so no property runs.
            wrapped = getattr(value, "__dict__", {}).get("model")
            if isinstance(wrapped, torch.nn.Module):
                add_module(wrapped)

    for name, value in list(vars(adapter).items()):
        visit(value, _Slot(adapter, name, attribute=True), 0)
    return reach


def _view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identity of a tensor view: two tensor objects with one key alias the same bytes."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _module_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for submodule in module.modules():
        tensors.extend(p for p in submodule._parameters.values() if p is not None)
        tensors.extend(b for b in submodule._buffers.values() if b is not None)
        tensors.extend(v for v in vars(submodule).values() if isinstance(v, torch.Tensor))
    return tensors


def parkable_bytes(adapter: object, device: torch.device) -> int | None:
    """Device bytes ``park`` would move to host RAM, or None when parking cannot be trusted."""
    reach = _collect(adapter, device)
    if reach.unmovable_bytes:
        return None
    # Counted per view, as ``park`` copies them: aliases of one view once.
    views: dict[tuple[Any, ...], int] = {}
    for module in reach.modules:
        for submodule in module.modules():
            if getattr(submodule, "_hf_hook", None) is not None:
                return None
        for tensor in _module_tensors(module):
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                return None
            if _on(device, tensor):
                views[_view_key(tensor)] = tensor.nbytes
    for _, tensor in reach.loose:
        views[_view_key(tensor)] = tensor.nbytes
    total = sum(views.values())
    return total or None


def park(adapter: object, device: torch.device) -> ParkedTensors:
    """Copy the adapter's device tensors to host RAM and rebind them there."""
    reach = _collect(adapter, device)
    origins: dict[int, torch.device] = {}
    copies: list[torch.Tensor] = []
    # Device views already copied. A second tensor object over the same view
    # (``self.original_inv_freq = self.inv_freq``, a loose ``weight.data``)
    # gets the same host copy, so the bytes are copied and charged once and
    # the aliasing survives the wake.
    moved: dict[tuple[Any, ...], torch.Tensor] = {}

    def to_host(tensor: torch.Tensor) -> torch.Tensor:
        key = _view_key(tensor)
        copy = moved.get(key)
        if copy is None:
            copy = tensor.detach().to("cpu", copy=True)
            origins[copy.untyped_storage().data_ptr()] = tensor.device
            copies.append(copy)
            moved[key] = copy
        return copy

    def still_on_device(tensor: torch.Tensor) -> bool:
        # A parameter shared by two modules, or a loose reference to one, has
        # already been rebound to its host copy by the time it is seen again.
        return _on(device, tensor) and tensor.untyped_storage().data_ptr() not in origins

    with torch.no_grad():
        for module in reach.modules:
            _move_module(module, still_on_device, to_host)
        for slot, tensor in reach.loose:
            if still_on_device(tensor):
                slot.set(to_host(tensor))
    return ParkedTensors(modules=reach.modules, loose=reach.loose, origins=origins, copies=copies)


def unpark(parked: ParkedTensors) -> None:
    """Copy parked tensors back to the devices they came from.

    Safe to call again after a failure part way through (an out-of-memory
    error): tensors already back on their device are skipped.
    """

    def parked_here(tensor: torch.Tensor) -> bool:
        storage = tensor.untyped_storage()
        return tensor.device.type == "cpu" and storage.nbytes() > 0 and storage.data_ptr() in parked.origins

    # One device tensor per host copy, so aliases stay aliases.
    restored: dict[int, torch.Tensor] = {}

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        ptr = tensor.untyped_storage().data_ptr()
        if ptr not in restored:
            restored[ptr] = tensor.to(parked.origins[ptr], copy=True)
        return restored[ptr]

    with torch.no_grad():
        for module in parked.modules:
            _move_module(module, parked_here, to_device)
        for slot, _ in parked.loose:
            current = slot.get()
            if isinstance(current, torch.Tensor) and parked_here(current):
                slot.set(to_device(current))
    for device in {origin for origin in parked.origins.values() if origin.type == "cuda"}:
        torch.cuda.synchronize(device)


def _move_module(module: torch.nn.Module, select: Any, move: Any) -> None:
    for submodule in module.modules():
        for key, param in submodule._parameters.items():
            if param is not None and select(param):
                param.data = move(param.data)
        for key, buffer in submodule._buffers.items():
            if buffer is not None and select(buffer):
                submodule._buffers[key] = move(buffer)
        for key, value in list(vars(submodule).items()):
            if isinstance(value, torch.Tensor) and select(value):
                setattr(submodule, key, move(value))


def release_cached_device_memory(device: torch.device) -> None:
    """Return freed blocks to the driver so device usage reflects the park."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
