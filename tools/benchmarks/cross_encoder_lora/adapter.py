"""Identical benchmark observation on both main and candidate adapters."""

from __future__ import annotations

import math
import time
from functools import wraps
from typing import Any

from sie_server.adapters.cross_encoder import CrossEncoderAdapter

from tools.benchmarks.cross_encoder_lora.protocol import require

INPUT_DIMENSIONS = 2


def production_class(arm: str) -> type[CrossEncoderAdapter]:
    require(arm in {"B", "N", "R"}, "unknown benchmark arm")
    if arm == "R":
        # Optional production adapter does not exist in the archived B source.
        from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter  # noqa: PLC0415

        return CrossEncoderLoraAdapter
    return CrossEncoderAdapter


def require_forward_coverage(batch: dict[str, Any]) -> None:
    """An outer score batch alone does not prove the model forward was observed."""
    forwards = batch["forwards"]
    require(bool(forwards), "model forward observations are missing")
    rows = 0
    for forward in forwards:
        shape = forward["input_shape"]
        require(
            isinstance(shape, list)
            and len(shape) == INPUT_DIMENSIONS
            and all(isinstance(n, int) and n > 0 for n in shape),
            "invalid observed forward input shape",
        )
        logits = forward.get("logits")
        require(
            isinstance(logits, list)
            and len(logits) == shape[0]
            and all(isinstance(row, list) and len(row) == 1 and math.isfinite(row[0]) for row in logits),
            "observed forward logits are missing, nonfinite or have the wrong shape",
        )
        rows += shape[0]
    require(rows == len(batch["item_ids"]), "observed forward rows do not cover the score batch")


def require_forward_selection(batch: dict[str, Any], expected_adapter: str | None) -> None:
    require_forward_coverage(batch)
    for forward in batch["forwards"]:
        enabled = forward["enabled_adapters"]
        require(bool(enabled) == bool(expected_adapter), "actual adapter enable state differs at model forward")
        require(all(item["adapters"] == [expected_adapter] for item in enabled), "wrong PEFT adapter at forward")


class ObservedCrossEncoder(CrossEncoderAdapter):
    production_adapter_class = CrossEncoderAdapter

    def load(self, device: str) -> None:
        self.batch_observations = []
        self.selection_observations = []
        self.observed_selection = None
        self.observation_phase = "load"
        self.active_forward = False
        self.mutation_overlap = False
        self.forward_observations = []
        self.fetch_observations = []
        super().load(device)
        assert self._model is not None
        model = self._model.model
        self._forward_had_override = "forward" in model.__dict__
        self._forward_override = model.__dict__.get("forward")
        self._original_forward = model.forward
        self._observer_wrapper = None
        self.install_forward_observer()

    def assert_original_forward(self) -> None:
        assert self._model is not None
        model = self._model.model
        if self._forward_had_override:
            require(model.__dict__.get("forward") is self._forward_override, "original forward override changed")
        else:
            require("forward" not in model.__dict__, "forward descriptor lookup was not restored")
        current = model.forward
        original = self._original_forward
        require(
            getattr(current, "__self__", None) is getattr(original, "__self__", None)
            and getattr(current, "__func__", current) is getattr(original, "__func__", original),
            "original forward target changed",
        )

    def install_forward_observer(self) -> None:
        assert self._model is not None
        require(self._observer_wrapper is None, "forward observer is already installed")
        self.assert_original_forward()
        model = self._model.model
        original = self._original_forward
        assert callable(original)

        # Sentence Transformers calls model.forward directly, bypassing root
        # Module.__call__ hooks. Preserve its signature and wrap that exact target.
        @wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            enabled = [
                {"module": name, "adapters": list(module.active_adapters)}
                for name, module in model.named_modules()
                if hasattr(module, "lora_A") and not getattr(module, "disable_adapters", False)
            ]
            inputs = kwargs.get("input_ids", args[0] if args else None)
            row = {
                "selection": self.observed_selection,
                "enabled_adapters": enabled,
                "input_shape": list(inputs.shape) if inputs is not None else None,
            }
            result = original(*args, **kwargs)
            row["logits"] = result.logits.detach().float().cpu().tolist()
            self.forward_observations.append(row)
            return result

        self._observer_wrapper = observed
        model.forward = observed

    def remove_forward_observer(self) -> None:
        assert self._model is not None
        model = self._model.model
        if self._observer_wrapper is not None:
            require(model.__dict__.get("forward") is self._observer_wrapper, "forward observer target changed")
            if self._forward_had_override:
                override = self._forward_override
                assert callable(override)
                model.forward = override
            else:
                del model.forward
            self._observer_wrapper = None
        self.assert_original_forward()

    def set_active_lora(self, lora_name: str | None) -> None:
        if self.active_forward:
            self.mutation_overlap = True
        super().set_active_lora(lora_name)
        self.observed_selection = lora_name
        self.selection_observations.append({"time_ns": time.time_ns(), "selection": lora_name})

    def score_pairs(self, queries: list, docs: list, **kwargs: Any) -> Any:
        started = time.time_ns()
        first_forward = len(self.forward_observations)
        self.active_forward = True
        try:
            result = super().score_pairs(queries, docs, **kwargs)
            batch = {
                "start_ns": started,
                "end_ns": time.time_ns(),
                "selection": self.observed_selection,
                "phase": self.observation_phase,
                "queries": [item.text for item in queries],
                "documents": [item.text for item in docs],
                "item_ids": [item.id for item in docs],
                "scores": result.scores.tolist(),
                "input_token_counts": result.input_token_counts,
                "forwards": self.forward_observations[first_forward:],
            }
            self.batch_observations.append(batch)
            require_forward_coverage(batch)
            return result
        finally:
            self.active_forward = False

    def load_lora(self, *args: Any, **kwargs: Any) -> int:
        if self.active_forward:
            self.mutation_overlap = True
        return super().load_lora(*args, **kwargs)

    def fetch_lora(self, *args: Any, **kwargs: Any) -> str:
        self.fetch_observations.append(
            {"time_ns": time.time_ns(), "name": args[0] if args else kwargs.get("lora_name")}
        )
        return super().fetch_lora(*args, **kwargs)

    def unload_lora(self, *args: Any, **kwargs: Any) -> None:
        if self.active_forward:
            self.mutation_overlap = True
        super().unload_lora(*args, **kwargs)

    def unload(self) -> None:
        if getattr(self, "_original_forward", None) is not None:
            self.remove_forward_observer()
            # Bound methods and closures otherwise keep the unloaded model alive.
            self._original_forward = None
            self._forward_override = None
        super().unload()
