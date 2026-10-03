"""Ordered accelerator groups claimed by one model.

A model whose profile declares a tensor-parallel width above one does not sit
on a device, it owns several of them for the life of the load. That is an
exception to the registry's ordinary placement, where devices are shared and a
model can be evicted to make room. The rules live here, engine-neutral, so the
adapter that launches the engine and the registry that reserves the hardware
derive the same group from the same anchor and cannot disagree about which
cards are taken.
"""

from __future__ import annotations

from collections.abc import Sequence

MAX_TENSOR_PARALLEL_SIZE = 8
"""Largest declarable width.

A surface limit rather than a hardware one. A declared width is an exclusive
claim on that many whole cards, so an unbounded field would let a typo claim
every card a worker has.
"""


def validate_tensor_parallel_size(value: object) -> int:
    """Return ``value`` as a usable tensor-parallel width, or raise.

    Rejects ``bool`` explicitly. ``True`` is an ``int`` in Python and would
    otherwise pass as width one, turning a mistyped flag into a silent
    single-device load on a deployment sized for the whole group.

    Raises:
        ValueError: If the value is not an integer in range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"tensor_parallel_size must be an integer, got {value!r}"
        raise ValueError(msg)
    if not 1 <= value <= MAX_TENSOR_PARALLEL_SIZE:
        msg = f"tensor_parallel_size must be between 1 and {MAX_TENSOR_PARALLEL_SIZE}, got {value}"
        raise ValueError(msg)
    return value


def resolve_device_group(anchor_index: int, width: int) -> list[int]:
    """Return the ordered CUDA indices a width-``width`` load claims from ``anchor_index``.

    The group is the contiguous block starting at the anchor. Contiguity is not
    a performance claim, it is what makes the mapping single-valued: two
    callers given the same anchor and width name the same cards. Whether those
    cards exist is the registry's check, made against its own device list.

    Args:
        anchor_index: The device the placement decision selected.
        width: Declared width. One yields ``[anchor_index]``.

    Raises:
        ValueError: If ``width`` is below one or the anchor is negative.
    """
    if width < 1:
        msg = f"tensor-parallel width must be at least 1, got {width}"
        raise ValueError(msg)
    if anchor_index < 0:
        msg = f"anchor device index must not be negative, got {anchor_index}"
        raise ValueError(msg)
    return list(range(anchor_index, anchor_index + width))


def format_device_mask(device_indices: Sequence[int]) -> str:
    """Return the ``CUDA_VISIBLE_DEVICES`` value for an ordered group.

    Order is preserved because rank N binds the Nth entry of the mask.

    Raises:
        ValueError: If the group is empty, or a device repeats. A repeat would
            give two ranks the same card, which deadlocks rather than fails.
    """
    if not device_indices:
        msg = "a device group must contain at least one device index"
        raise ValueError(msg)
    if len(set(device_indices)) != len(device_indices):
        msg = f"device indices must be distinct, got {list(device_indices)!r}"
        raise ValueError(msg)
    return ",".join(str(index) for index in device_indices)
