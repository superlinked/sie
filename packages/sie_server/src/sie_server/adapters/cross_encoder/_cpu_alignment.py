"""Load-time compatibility for unaligned BERT weights on Apple CPU BLAS.

Older safetensors files can expose float32 views at a non-four-byte-aligned
address. Contiguity does not correct that address, and Accelerate can return
incorrect results for it. Keep this correction before model publication; it is
not a forward hook or a general tensor-layout conversion.
"""

import torch
from transformers import BertForSequenceClassification


def normalize_bert_cpu_storage(model: torch.nn.Module | None) -> None:
    """Copy unsafe standard BERT float32 storage once, retaining tensor objects.

    The caller restricts this optional compatibility path to Darwin arm64 CPU.
    Distinct tensors sharing storage are intentionally unsupported: cloning them
    independently could break aliases. Repeated registrations of one tensor are safe.
    """
    if type(model) is not BertForSequenceClassification or getattr(model, "is_quantized", False):
        return
    if getattr(model, "peft_config", None) or getattr(model, "_hf_peft_config_loaded", False):
        return

    tensors: dict[int, tuple[str, torch.Tensor]] = {}
    for name, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        tensors.setdefault(id(tensor), (name, tensor))
    targets = [
        (name, tensor)
        for name, tensor in tensors.values()
        if tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tensor.layout == torch.strided
        and tensor.numel()
        and tensor.data_ptr() % tensor.element_size()
    ]
    if not targets:
        return

    # Storage ranges also catch separately constructed, cross-dtype buffer views
    # whose Python storage objects differ but whose underlying bytes overlap.
    ranges = []
    for name, tensor in tensors.values():
        if tensor.device.type == "cpu" and tensor.layout == torch.strided:
            storage = tensor.untyped_storage()
            ranges.append((name, tensor, storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    for name, tensor in targets:
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
            raise ValueError(f"Cannot align custom BERT tensor {name!r}")
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise ValueError(f"Cannot align BERT tensor {name!r}: expected contiguous storage at offset zero")
        storage = tensor.untyped_storage()
        start, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        for other_name, other, other_start, other_end in ranges:
            if other is not tensor and start < other_end and other_start < end:
                raise ValueError(f"Cannot align BERT tensor {name!r}: storage overlaps {other_name!r}")

    # Every allocation and layout check completes before the first mutation.
    # The old detached views keep rollback storage alive without copying it.
    copies = []
    with torch.no_grad():
        for name, tensor in targets:
            original = tensor.detach()
            replacement = original.clone()
            if (
                replacement.data_ptr() % replacement.element_size()
                or replacement.dtype != tensor.dtype
                or replacement.device != tensor.device
                or replacement.shape != tensor.shape
                or replacement.stride() != tensor.stride()
                or replacement.storage_offset() != 0
            ):
                raise ValueError(f"Could not create aligned BERT storage for {name!r}")
            copies.append((tensor, original, replacement))
        try:
            for tensor, _, replacement in copies:
                tensor.data = replacement
        except BaseException:
            for tensor, original, _ in copies:
                tensor.data = original
            raise
