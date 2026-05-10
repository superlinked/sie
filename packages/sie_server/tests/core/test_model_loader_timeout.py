"""Tests for the per-model load timeout in ModelLoader.

Covers:
- ``_read_load_timeout_from_env`` env parsing edge cases.
- ``_await_with_load_timeout`` fast path, timeout path, and executor swap.
- Constructor precedence (kwarg > env > default).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from sie_server.core.model_loader import (
    DEFAULT_MODEL_LOAD_TIMEOUT_S,
    ModelLoader,
    _read_load_timeout_from_env,
)


# ---------- _read_load_timeout_from_env ----------


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, "default"),
        ("", "default"),
        ("30", 30.0),
        ("0.5", 0.5),
        ("0", None),
        ("-1", None),
        ("abc", "default"),
    ],
)
def test_env_parser(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("SIE_MODEL_LOAD_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("SIE_MODEL_LOAD_TIMEOUT_S", env_value)

    result = _read_load_timeout_from_env()
    if expected == "default":
        assert result == DEFAULT_MODEL_LOAD_TIMEOUT_S
    else:
        assert result == expected


# ---------- _await_with_load_timeout ----------


def _make_loader(**overrides) -> ModelLoader:
    return ModelLoader(
        preprocessor_registry=MagicMock(),
        postprocessor_registry=MagicMock(),
        all_configs={},
        **overrides,
    )


@pytest.mark.asyncio
async def test_timeout_disabled_passes_through():
    loader = _make_loader(model_load_timeout_s=None)
    loader._model_load_timeout_s = None  # explicit override regardless of env

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(loader._load_executor, lambda: 42)
    assert await loader._await_with_load_timeout("m", fut) == 42


@pytest.mark.asyncio
async def test_fast_load_returns_value():
    loader = _make_loader(model_load_timeout_s=2.0)
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(loader._load_executor, lambda: "loaded")
    assert await loader._await_with_load_timeout("m", fut) == "loaded"


@pytest.mark.asyncio
async def test_timeout_fires_recreates_executor():
    loader = _make_loader(model_load_timeout_s=0.2)
    original_executor = loader._load_executor

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(loader._load_executor, lambda: time.sleep(2.0))

    start = time.monotonic()
    with pytest.raises(TimeoutError) as exc_info:
        await loader._await_with_load_timeout("my-model", fut)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"timeout fired late: {elapsed:.2f}s"
    msg = str(exc_info.value)
    assert "my-model" in msg
    assert "timeout=0.2s" in msg
    assert "executor was recreated" in msg
    assert isinstance(loader._load_executor, ThreadPoolExecutor)
    assert loader._load_executor is not original_executor


@pytest.mark.asyncio
async def test_executor_swap_lets_new_work_run():
    loader = _make_loader(model_load_timeout_s=0.2)
    loop = asyncio.get_running_loop()

    hung = loop.run_in_executor(loader._load_executor, lambda: time.sleep(2.0))
    with pytest.raises(TimeoutError):
        await loader._await_with_load_timeout("hung", hung)

    new_fut = loop.run_in_executor(loader._load_executor, lambda: "ok")
    assert await asyncio.wait_for(new_fut, timeout=2.0) == "ok"


# ---------- ModelLoader timeout wiring ----------


def test_constructor_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("SIE_MODEL_LOAD_TIMEOUT_S", "999")
    loader = _make_loader(model_load_timeout_s=5.0)
    assert loader._model_load_timeout_s == 5.0


def test_env_used_when_kwarg_none(monkeypatch):
    monkeypatch.setenv("SIE_MODEL_LOAD_TIMEOUT_S", "42")
    loader = _make_loader(model_load_timeout_s=None)
    assert loader._model_load_timeout_s == 42.0


def test_default_when_no_env_no_kwarg(monkeypatch):
    monkeypatch.delenv("SIE_MODEL_LOAD_TIMEOUT_S", raising=False)
    loader = _make_loader(model_load_timeout_s=None)
    assert loader._model_load_timeout_s == DEFAULT_MODEL_LOAD_TIMEOUT_S