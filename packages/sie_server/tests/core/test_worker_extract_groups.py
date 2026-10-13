"""A batch with several extract configurations is shown to the adapter whole before its per-configuration calls."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.prepared import ExtractPreparedItem
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import Item


def _adapter(*, primes: bool = True, error: Exception | None = None, width: int | None = None) -> MagicMock:
    """An adapter that records, in order, the calls the worker makes (``adapter.calls``)."""
    adapter = MagicMock()
    adapter.calls = []
    adapter.groups = []
    adapter.supports_lora.return_value = False
    if width is None:
        del adapter.max_concurrent_dispatch
    else:
        adapter.max_concurrent_dispatch.return_value = width

    def extract(items: list[Item], **kwargs: Any) -> ExtractOutput:
        adapter.calls.append(("extract", (tuple(kwargs["labels"]), [item.text for item in items])))
        return ExtractOutput(entities=[[] for _ in items])

    def prime(groups: list[dict[str, Any]]) -> None:
        adapter.calls.append(
            ("prime", [(tuple(group["labels"]), [item.text for item in group["items"]]) for group in groups])
        )
        adapter.groups = groups
        if error is not None:
            raise error

    adapter.extract.side_effect = extract
    if primes:
        adapter.prime_extract_groups.side_effect = prime
    else:
        del adapter.prime_extract_groups
    return adapter


async def _run(adapter: Any, requests: list[tuple[str, list[str], dict[str, Any] | None]]) -> list[Any]:
    """Submit one-item extract requests that land in one batch; their results."""
    worker = ModelWorker(adapter, WorkerConfig(max_batch_tokens=1000, max_batch_requests=16, max_batch_wait_ms=50))
    await worker.start()
    try:
        futures = [
            await worker.submit_extract(
                [ExtractPreparedItem(cost=len(text), original_index=0)],
                [Item(text=text)],
                labels=labels,
                options=options,
            )
            for text, labels, options in requests
        ]
        return await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=5.0)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_a_batch_of_several_configurations_is_primed_once_before_its_calls() -> None:
    adapter = _adapter()

    results = await _run(
        adapter,
        [("a", ["x", "y"], None), ("b", ["y", "x"], None), ("c", ["x", "y"], None), ("d", ["z"], {"threshold": 0.5})],
    )

    assert all(result.output.batch_size == 1 for result in results)
    groups = [(("x", "y"), ["a", "c"]), (("y", "x"), ["b"]), (("z",), ["d"])]
    assert adapter.calls == [("prime", groups), *(("extract", group) for group in groups)]
    # Each group carries the arguments its extract call gets.
    assert [group["options"] for group in adapter.groups] == [None, None, {"threshold": 0.5}]
    assert all(group["output_schema"] is None and group["instruction"] is None for group in adapter.groups)


@pytest.mark.asyncio
async def test_a_batch_of_one_configuration_is_not_primed() -> None:
    adapter = _adapter()

    await _run(adapter, [("a", ["x", "y"], None), ("b", ["x", "y"], None)])

    assert adapter.calls == [("extract", (("x", "y"), ["a", "b"]))]


@pytest.mark.asyncio
async def test_an_adapter_without_the_method_is_left_alone() -> None:
    adapter = _adapter(primes=False)

    results = await _run(adapter, [("a", ["x"], None), ("b", ["y"], None)])

    assert all(result.output.batch_size == 1 for result in results)
    assert [name for name, _ in adapter.calls] == ["extract", "extract"]


@pytest.mark.asyncio
async def test_a_priming_error_leaves_every_configuration_to_its_own_call(caplog: pytest.LogCaptureFixture) -> None:
    adapter = _adapter(error=RuntimeError("no memory for one pass"))

    with caplog.at_level("WARNING"):
        results = await _run(adapter, [("a", ["x"], None), ("b", ["y"], None)])

    assert all(result.output.batch_size == 1 for result in results)
    assert [name for name, _ in adapter.calls] == ["prime", "extract", "extract"]
    assert "Priming 2 extract configurations failed" in caplog.text


@pytest.mark.asyncio
async def test_a_worker_that_runs_batches_side_by_side_does_not_prime() -> None:
    # What an adapter keeps from priming belongs to one batch; two batches at once would share it.
    adapter = _adapter(width=2)

    await _run(adapter, [("a", ["x"], None), ("b", ["y"], None)])

    assert "prime" not in [name for name, _ in adapter.calls]


def test_the_handler_primes_nothing_for_a_single_group() -> None:
    adapter = _adapter()

    ExtractHandler().prime(adapter, [([Item(text="a")], (("x",), None, None, None), [])])

    assert adapter.calls == []
