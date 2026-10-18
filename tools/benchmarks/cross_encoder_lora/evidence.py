"""Incremental observations and atomic terminal records for the bounded probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.protocol import require


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def vector_delta(left: list[float], right: list[float]) -> float:
    require(len(left) == len(right) and bool(left), "score vector shape changed")
    require(all(math.isfinite(value) for value in left + right), "nonfinite score")
    return max(abs(x - y) for x, y in zip(left, right, strict=True))


class Evidence:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {"event": event, "time_ns": time.time_ns(), **fields}

        # Preserve nonfinite observations as explicit strings in failed evidence.
        def finite(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(item) for item in value]
            return value

        row = finite(row)
        with self.lock, (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        return row
