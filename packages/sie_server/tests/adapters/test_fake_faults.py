"""Failure injection on the fake family (#1849).

Every fault must flow through real code paths: the injected OOM satisfies the
real ``is_oom_error`` classification, latches block real load/dispatch calls,
and config errors fail loudly at adapter construction.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from sie_server.adapters.fake.adapter import (
    SIE_FAKE_FAULTS_ENV,
    FakeAdapter,
    _resolve_faults,
)
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.core.oom import ResourceExhaustedError, is_oom_error
from sie_server.types.inputs import Item

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

ITEM = [Item(text="x")]


def _loaded_embed(**fault_kwargs: object) -> FakeAdapter:
    adapter = FakeAdapter(faults=dict(fault_kwargs))
    adapter.load("cpu")
    return adapter


# -- OOM on Nth dispatch --------------------------------------------------------


def test_oom_fires_on_exact_dispatch_and_recovers() -> None:
    adapter = _loaded_embed(oom_on_dispatch=3)
    adapter.encode(ITEM, output_types=["dense"])
    adapter.encode(ITEM, output_types=["dense"])
    with pytest.raises(RuntimeError) as excinfo:
        adapter.encode(ITEM, output_types=["dense"])
    assert is_oom_error(excinfo.value), "injected OOM must satisfy the real classifier"
    # Window is one dispatch wide by default: the retry succeeds.
    adapter.encode(ITEM, output_types=["dense"])


def test_oom_repeat_widens_failure_window() -> None:
    adapter = _loaded_embed(oom_on_dispatch=1, oom_repeat=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            adapter.encode(ITEM, output_types=["dense"])
    adapter.encode(ITEM, output_types=["dense"])


async def test_oom_applies_to_generation() -> None:
    adapter = FakeAdapter(faults={"oom_on_dispatch": 1})
    adapter.load("cpu")
    with pytest.raises(RuntimeError) as excinfo:
        async for _ in adapter.generate("p", max_new_tokens=4):
            pass
    assert is_oom_error(excinfo.value)


def test_oom_typed_variant_exercises_isinstance_tier() -> None:
    """`oom_typed` raises the typed ResourceExhaustedError — the isinstance
    tier of is_oom_error — instead of the substring-matched RuntimeError.
    """
    adapter = _loaded_embed(oom_on_dispatch=1, oom_typed=True)
    with pytest.raises(ResourceExhaustedError) as excinfo:
        adapter.encode(ITEM, output_types=["dense"])
    assert is_oom_error(excinfo.value)
    # The typed variant must NOT depend on the substring tier.
    assert "out of memory" not in str(excinfo.value).lower()


# -- Latency faults ---------------------------------------------------------------


def test_request_latency_delays_every_dispatch() -> None:
    adapter = FakeAdapter(request_latency_s=0.05)
    adapter.load("cpu")
    start = time.monotonic()
    adapter.encode(ITEM, output_types=["dense"])
    adapter.encode(ITEM, output_types=["dense"])
    assert time.monotonic() - start >= 0.09


async def test_inter_token_latency_paces_generation() -> None:
    adapter = FakeAdapter(default_completion_tokens=4, inter_token_latency_s=0.03)
    adapter.load("cpu")
    start = time.monotonic()
    async for _ in adapter.generate("p", max_new_tokens=8):
        pass
    assert time.monotonic() - start >= 4 * 0.03 * 0.9


# -- Load faults ----------------------------------------------------------------


def test_fail_load_raises() -> None:
    adapter = FakeAdapter(faults={"fail_load": True})
    with pytest.raises(RuntimeError, match="injected load failure"):
        adapter.load("cpu")


def test_slow_load_delays() -> None:
    adapter = FakeAdapter(faults={"slow_load_s": 0.05})
    start = time.monotonic()
    adapter.load("cpu")
    assert time.monotonic() - start >= 0.04


def test_teardown_hang_delays_unload() -> None:
    adapter = _loaded_embed(teardown_hang_s=0.05)
    start = time.monotonic()
    adapter.unload()
    assert time.monotonic() - start >= 0.04


# -- Latch faults -----------------------------------------------------------------


def test_load_latch_blocks_until_released(tmp_path: Path) -> None:
    latch = tmp_path / "release-load"
    adapter = FakeAdapter(faults={"load_latch_file": str(latch), "latch_timeout_s": 5.0})
    done = threading.Event()

    def _load() -> None:
        adapter.load("cpu")
        done.set()

    thread = threading.Thread(target=_load)
    thread.start()
    try:
        assert not done.wait(0.1), "load must hold while the latch file is absent"
        latch.touch()
        assert done.wait(5.0), "load must complete once the latch is released"
    finally:
        thread.join(timeout=5.0)


def test_dispatch_latch_blocks_until_released(tmp_path: Path) -> None:
    latch = tmp_path / "release-dispatch"
    adapter = _loaded_embed(dispatch_latch_file=str(latch), latch_timeout_s=5.0)
    done = threading.Event()

    def _encode() -> None:
        adapter.encode(ITEM, output_types=["dense"])
        done.set()

    thread = threading.Thread(target=_encode)
    thread.start()
    try:
        assert not done.wait(0.1), "dispatch must hold while the latch file is absent"
        latch.touch()
        assert done.wait(5.0), "dispatch must complete once the latch is released"
    finally:
        thread.join(timeout=5.0)


def test_latch_timeout_raises(tmp_path: Path) -> None:
    latch = tmp_path / "never-released"
    adapter = FakeAdapter(faults={"load_latch_file": str(latch), "latch_timeout_s": 0.05})
    with pytest.raises(TimeoutError, match="not released"):
        adapter.load("cpu")


# -- Config surfaces --------------------------------------------------------------


def test_nonfinite_fault_numerics_fail_loudly(tmp_path: Path) -> None:
    """json.loads accepts NaN — a NaN latch_timeout_s would make an
    unreleased latch hang forever instead of failing at the deadline.
    """
    with pytest.raises(ValueError, match="latch_timeout_s"):
        FakeAdapter(faults={"load_latch_file": str(tmp_path / "latch"), "latch_timeout_s": float("nan")})
    with pytest.raises(ValueError, match="slow_load_s"):
        FakeAdapter(faults={"slow_load_s": float("inf")})
    with pytest.raises(ValueError, match="oom_repeat"):
        FakeAdapter(faults={"oom_on_dispatch": 1, "oom_repeat": 0})


def test_unknown_fault_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown fake fault"):
        FakeAdapter(faults={"explode_on_tuesdays": True})


def test_env_override_merges_over_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIE_FAKE_FAULTS_ENV, '{"sie-fake": {"oom_on_dispatch": 1}}')
    faults = _resolve_faults({"slow_load_s": 0.5}, "sie-fake")
    assert faults.oom_on_dispatch == 1
    assert faults.slow_load_s == 0.5  # YAML values not named in the override survive


def test_env_override_ignores_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIE_FAKE_FAULTS_ENV, '{"sie-fake:small-a": {"fail_load": true}}')
    faults = _resolve_faults(None, "sie-fake")
    assert not faults.fail_load


def test_malformed_env_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIE_FAKE_FAULTS_ENV, "not json")
    with pytest.raises(ValueError, match=SIE_FAKE_FAULTS_ENV):
        _resolve_faults(None, "sie-fake")


# -- Pre-baked catalog scenarios ---------------------------------------------------


def test_prebaked_oom_scenario_via_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SIE_FAKE_FAULTS_ENV, raising=False)
    configs = load_model_configs(MODELS_DIR)
    adapter = load_adapter(configs["sie-fake:oom-3rd"], MODELS_DIR, device="cpu")
    adapter.load("cpu")
    adapter.encode(ITEM, output_types=["dense"])
    adapter.encode(ITEM, output_types=["dense"])
    with pytest.raises(RuntimeError) as excinfo:
        adapter.encode(ITEM, output_types=["dense"])
    assert is_oom_error(excinfo.value)


def test_env_override_reaches_adapter_via_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIE_FAKE_FAULTS_ENV, '{"sie-fake": {"fail_load": true}}')
    configs = load_model_configs(MODELS_DIR)
    adapter = load_adapter(configs["sie-fake"], MODELS_DIR, device="cpu")
    with pytest.raises(RuntimeError, match="injected load failure"):
        adapter.load("cpu")
