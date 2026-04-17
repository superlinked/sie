from __future__ import annotations

from typing import Literal

# Canonical type aliases — import these instead of redefining per-adapter
ComputePrecision = Literal["float16", "bfloat16", "float32"]
PoolingStrategy = Literal["cls", "mean", "last"]
AttnImplementation = Literal["sdpa", "eager"]

# Standard error messages
ERR_NOT_LOADED = "Model not loaded. Call load() first."
ERR_REQUIRES_TEXT = "{adapter_name} requires text input"
