"""Real Sentence Transformers forwarding and exact timed-path restoration."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

import pytest
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter
from sie_server.adapters.lora import LoraStateUncertainError
from sie_server.types.inputs import Item

from tools.benchmarks.cross_encoder_lora.adapter import (
    ObservedCrossEncoder,
    require_forward_coverage,
    require_forward_selection,
)
from tools.benchmarks.cross_encoder_lora.fixtures import prepare
from tools.benchmarks.cross_encoder_lora.runtime_adapter import ObservedRuntimeCrossEncoder
from tools.benchmarks.cross_encoder_lora.tests.test_probe import tiny_snapshot
from tools.benchmarks.cross_encoder_lora.worker import unobserved


def pairs(count: int = 4) -> tuple[list[Item], list[Item]]:
    return (
        [Item(id=f"q-{index}", text="how do i reset my password ?") for index in range(count)],
        [Item(id=f"d-{index}", text="use the password reset link .") for index in range(count)],
    )


def test_real_predict_bypasses_root_hook_but_bound_forward_covers_every_row(tmp_path: Path) -> None:
    adapter = ObservedCrossEncoder(tiny_snapshot(tmp_path / "model"))
    adapter.load("cpu")
    try:
        assert adapter._model is not None
        model = adapter._model.model
        hook_calls = []
        handle = model.register_forward_hook(lambda *_args: hook_calls.append(True))
        try:
            # This reproduces the old observer's failure using the actual
            # Sentence Transformers predict path, not a fake direct call.
            with unobserved(adapter):
                result = adapter.score_pairs(*pairs(70))
                assert len(result.scores) == 70
                assert not hook_calls
                assert not adapter.forward_observations
            with pytest.raises(AssertionError, match="observations are missing"):
                require_forward_coverage({"item_ids": list(range(70)), "forwards": []})
            adapter.score_pairs(*pairs(70))
            batch = adapter.batch_observations[-1]
            require_forward_selection(batch, None)
            assert len(batch["forwards"]) > 1  # predict internally chunks the outer batch.
            assert sum(row["input_shape"][0] for row in batch["forwards"]) == 70
            assert not hook_calls
        finally:
            handle.remove()
    finally:
        adapter.unload()


@pytest.mark.parametrize("instance_override", [False, True])
@pytest.mark.parametrize("fail_inside_timing", [False, True])
def test_timing_restores_exact_original_lookup_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, instance_override: bool, fail_inside_timing: bool
) -> None:
    original_load = CrossEncoderAdapter.load
    original_attribute = {}

    def load(adapter, device):
        original_load(adapter, device)
        model = adapter._model.model
        if instance_override:
            target = model.forward

            @wraps(target)
            def original_override(*args, **kwargs):
                return target(*args, **kwargs)

            model.forward = original_override
        original_attribute["had_override"] = "forward" in model.__dict__
        original_attribute["value"] = model.__dict__.get("forward")
        original_attribute["target"] = model.forward

    monkeypatch.setattr(CrossEncoderAdapter, "load", load)
    adapter = ObservedCrossEncoder(tiny_snapshot(tmp_path / "model"))
    adapter.load("cpu")
    assert adapter._model is not None
    model = adapter._model.model

    def assert_original():
        assert ("forward" in model.__dict__) == original_attribute["had_override"]
        if instance_override:
            assert model.__dict__["forward"] is original_attribute["value"]
        current, target = model.forward, original_attribute["target"]
        assert getattr(current, "__self__", None) is getattr(target, "__self__", None)
        assert getattr(current, "__func__", current) is getattr(target, "__func__", target)

    try:
        adapter.score_pairs(*pairs())
        before = (
            len(adapter.batch_observations),
            len(adapter.forward_observations),
            len(adapter.selection_observations),
        )
        expected_error = (
            pytest.raises(RuntimeError, match="injected timed-phase failure") if fail_inside_timing else nullcontext()
        )
        with expected_error:
            with unobserved(adapter):
                assert_original()
                assert adapter.score_pairs.__func__ is CrossEncoderAdapter.score_pairs
                assert adapter.set_active_lora.__func__ is CrossEncoderAdapter.set_active_lora
                adapter.set_active_lora(None)
                adapter.score_pairs(*pairs())
                assert_original()
                if fail_inside_timing:
                    raise RuntimeError("injected timed-phase failure")
        assert before == (
            len(adapter.batch_observations),
            len(adapter.forward_observations),
            len(adapter.selection_observations),
        )
        assert model.__dict__["forward"] is adapter._observer_wrapper
        adapter.score_pairs(*pairs())
        assert len(adapter.forward_observations) > before[1]
    finally:
        adapter.unload()
    assert_original()
    assert adapter._original_forward is None
    assert adapter._forward_override is None
    assert adapter._observer_wrapper is None


def observed_batch() -> dict:
    return {
        "item_ids": ["0", "1"],
        "forwards": [{"input_shape": [2, 8], "logits": [[0.1], [0.2]], "enabled_adapters": []}],
    }


def test_runtime_timing_uses_actual_opt_in_selection_and_preserves_poison_refusal(tmp_path: Path) -> None:
    snapshot = tiny_snapshot(tmp_path / "model")
    fixtures = tmp_path / "fixtures"
    prepare(fixtures, snapshot, smoke=True)
    adapter = ObservedRuntimeCrossEncoder(snapshot, lora_paths={"a": str(fixtures / "a")})
    adapter.load("cpu")
    try:
        adapter.load_lora("a")
        before = len(adapter.selection_observations)
        with unobserved(adapter):
            assert adapter.production_adapter_class is CrossEncoderLoraAdapter
            assert adapter.score_pairs.__func__ is CrossEncoderAdapter.score_pairs
            assert adapter.set_active_lora.__func__ is CrossEncoderLoraAdapter.set_active_lora
            adapter.set_active_lora("a")
            assert adapter._lora._active == str(fixtures / "a")
            adapter.set_active_lora(None)
            assert adapter._lora._active is None
            adapter.score_pairs(*pairs())
        assert len(adapter.selection_observations) == before
        adapter._lora._uncertain = True
        with pytest.raises(LoraStateUncertainError), unobserved(adapter):
            adapter.set_active_lora(None)
        assert adapter._observer_wrapper is not None
    finally:
        adapter.unload()


@pytest.mark.parametrize("failure", ["empty", "rows", "logits", "nonfinite", "wrong_adapter", "enabled_base"])
def test_missing_partial_or_wrong_tuner_observation_cannot_qualify(failure: str) -> None:
    batch = copy.deepcopy(observed_batch())
    expected = None
    if failure == "empty":
        batch["forwards"] = []
    elif failure == "rows":
        batch["item_ids"].append("2")
    elif failure == "logits":
        batch["forwards"][0]["logits"] = [[0.1]]
    elif failure == "nonfinite":
        batch["forwards"][0]["logits"][0][0] = float("nan")
    elif failure == "wrong_adapter":
        expected = "wanted"
        batch["forwards"][0]["enabled_adapters"] = [{"module": "query", "adapters": ["other"]}]
    else:
        batch["forwards"][0]["enabled_adapters"] = [{"module": "query", "adapters": ["unexpected"]}]
    with pytest.raises(AssertionError):
        require_forward_selection(batch, expected)
