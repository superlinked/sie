"""Fake Engine regression test (#1850): the OOM-recovery ladder, end to end.

Table-driven walk of CACHE_CLEAR → EVICT_LRU → SPLIT_BATCH where the OOM
originates in a REAL adapter dispatch (sie-fake with ``oom_on_dispatch`` /
``oom_repeat``, #1849), is classified by the real ``is_oom_error`` matcher,
and EVICT_LRU performs a real eviction through a real ``ModelRegistry``.
Retires the mock-dispatch string-OOM as the only coverage at this seam
(tests/core/worker/test_oom_recovery.py:67-68).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import SIE_FAKE_MEMORY_BUDGET_ENV
from sie_server.core.oom import OomRecoveryConfig, OomRecoveryStats
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker.oom_recovery import BatchExecutor, ConfigGroup
from sie_server.types.inputs import Item

pytestmark = pytest.mark.fake_stack

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class _Metadata:
    """Minimal request-metadata: the executor touches ``future`` and
    ``_partial_results`` only (same shape the worker harness uses).
    """

    __slots__ = ("_partial_results", "future")

    def __init__(self) -> None:
        self.future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._partial_results: dict[int, Any] | None = None


class _Handler:
    @staticmethod
    def slice_output(output: Any, batch_idx: int) -> Any:
        return (output, batch_idx)


def _group(texts: list[str]) -> tuple[ConfigGroup, list[_Metadata]]:
    metas = [_Metadata() for _ in texts]
    # ignore justification: ConfigGroup's metadata slot is typed to the real
    # RequestMetadata; the executor only touches .future/._partial_results,
    # so the minimal _Metadata stand-in is intentionally substituted here
    # (same substitution the worker's own oom_recovery tests make).
    return (list(texts), metas, list(range(len(texts))), list(texts)), metas  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("oom_repeat", "batch", "expect_cache_clears", "expect_evictions", "expect_splits", "sibling_survives"),
    [
        # Rung 1: one OOM — CACHE_CLEAR's retry succeeds; nothing is evicted.
        (1, ["a"], 1, 0, 0, True),
        # Rung 2: two OOMs — cache-clear retry fails, EVICT_LRU really evicts
        # the sibling model, then the retry succeeds.
        (2, ["a"], 1, 1, 0, False),
        # Rung 3: three OOMs — both earlier rungs fail, SPLIT_BATCH halves the
        # two-item batch and the singles succeed.
        (3, ["a", "b"], 1, 1, 1, False),
    ],
)
async def test_oom_ladder_walks_real_seams(
    monkeypatch: pytest.MonkeyPatch,
    oom_repeat: int,
    batch: list[str],
    expect_cache_clears: int,
    expect_evictions: int,
    expect_splits: int,
    sibling_survives: bool,
) -> None:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        f'{{"sie-fake:small-a": {{"oom_on_dispatch": 1, "oom_repeat": {oom_repeat}}}}}',
    )

    configs = load_model_configs(MODELS_DIR)
    registry = ModelRegistry()
    # Adding the BASE config registers every sie-fake:<profile> variant.
    registry.add_config(configs["sie-fake"])
    # The sibling loads first so it is the LRU eviction candidate.
    await registry.load_async("sie-fake:small-b", device="cpu")
    await registry.load_async("sie-fake:small-a", device="cpu")
    adapter = registry.get("sie-fake:small-a")

    stats = OomRecoveryStats()
    executor = BatchExecutor(
        model_name="sie-fake:small-a",
        registry=registry,
        config=OomRecoveryConfig(),
        stats=stats,
    )

    async def dispatch(_handler: Any, group: ConfigGroup) -> Any:
        texts = group[0]
        return adapter.encode([Item(text=t) for t in texts], output_types=["dense"])

    group, metas = _group(batch)
    await executor.run(_Handler(), group, dispatch)

    assert stats.cache_clears == expect_cache_clears
    assert stats.evictions_triggered == expect_evictions
    assert stats.batch_splits == expect_splits
    assert stats.terminal_failures == 0
    assert stats.recoveries_succeeded >= 1
    for meta in metas:
        assert meta._partial_results, "every request must receive its slice after recovery"

    sibling_loaded = "sie-fake:small-b" in registry.memory_manager.loaded_models
    assert sibling_loaded == sibling_survives, (
        "EVICT_LRU must evict the real sibling exactly when the ladder reaches it"
    )
