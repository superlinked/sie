"""The effective generation payload shared by IPC admission and execution."""

from collections.abc import Mapping
from typing import Any


def extract_generate_params(work_item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prefer the current payload, retaining the legacy options wire shape."""
    params = work_item.get("generate")
    if isinstance(params, dict):
        return params
    options = work_item.get("options")
    if isinstance(options, dict) and ("prompt" in options or "messages" in options):
        return options
    return None
