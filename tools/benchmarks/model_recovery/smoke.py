"""CPU-only synthetic adapters for checking the probe, never performance evidence."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.load_errors import _BASE_COOLDOWN_S, LoadErrorClass


class SmokeAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None
        self._process: subprocess.Popen[bytes] | None = None

    def load(self, device: str) -> None:
        self._device = device
        time.sleep(0.15)
        self._process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(180)"])

    def unload(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        self._device = None

    def engine_exit_code(self) -> int | None:
        return self._process.poll() if self._process is not None else None

    def memory_footprint(self) -> int:
        return 1024 if self._process is not None else 0

    async def generate(self, prompt: str, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        await asyncio.sleep(0.01)
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("synthetic child exited")
        yield GenerationChunk(text_delta="safe", is_first=True)
        yield GenerationChunk(
            text_delta="", done=True, finish_reason="stop", prompt_tokens=len(prompt.split()), completion_tokens=1
        )


def install_smoke(stack: ExitStack) -> None:
    stack.enter_context(
        patch("sie_server.core.model_loader.load_adapter", side_effect=lambda *_a, **_kw: SmokeAdapter())
    )
    stack.enter_context(patch("sie_server.core.model_loader.ModelLoader.ensure_weights_cached", return_value=None))
    stack.enter_context(patch.dict(_BASE_COOLDOWN_S, {LoadErrorClass.ENGINE: 0.5}))
