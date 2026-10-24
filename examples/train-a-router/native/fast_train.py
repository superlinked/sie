"""Train the explicit fast recipe; retain reference FP32 SDPA validation and exports."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import bbeh_native as reference
import fast_protocol as protocol
import fast_runtime as runtime


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.events = []
        self.started = time.perf_counter()
        self.phase = "initialization"

    def record(self, event):
        event = dict(event, elapsed_seconds_since_wrapper_start=time.perf_counter() - self.started)
        self.events.append(event)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        if event.get("event") in {"epoch", "phase_start", "phase_complete"}:
            print(json.dumps({k: event[k] for k in ("event", "phase", "epoch", "loss", "training_seconds",
                             "actual_optimizer_steps", "attempted_optimizer_windows") if k in event}), flush=True)

    def snapshot(self):
        epochs = [e for e in self.events if e["event"] == "epoch"]
        validations = [e for e in self.events if e["event"] == "reference_validation"]
        return {
            "format": "laya-fast-runtime-events-v1",
            "recipe_id": protocol.RECIPE_ID,
            "events": self.events,
            "training_seconds": sum(e["training_seconds"] for e in epochs),
            "synchronization_seconds": sum(e["synchronization_seconds"] for e in epochs),
            "model_copy_seconds": sum(e["model_copy_seconds"] for e in self.events if e["event"] == "phase_start"),
            "training_phase_wall_seconds_including_cleanup": {
                e["phase"]: e["elapsed_seconds"] for e in self.events if e["event"] == "training_phase_return"
            },
            "reference_fp32_validation_seconds": sum(e["elapsed_seconds"] for e in validations),
            "reference_fp32_validation_calls": len(validations),
            "actual_optimizer_steps": sum(e["actual_optimizer_steps"] for e in epochs),
            "attempted_optimizer_windows": sum(e["attempted_optimizer_windows"] for e in epochs),
            "peak_allocated_bytes": max((e["peak_allocated_bytes"] for e in epochs + validations), default=0),
            "peak_reserved_bytes": max((e["peak_reserved_bytes"] for e in epochs + validations), default=0),
            "elapsed_seconds": time.perf_counter() - self.started,
            "calibration_or_final_outcomes_accessed": False,
        }


@contextlib.contextmanager
def fast_execution(recorder):
    import laya.train as lt
    import torch

    original_train, original_records, original_select = lt.train_model, lt.calibration_records, reference.select_epoch
    calls = 0

    def train_model(*args, **kwargs):
        nonlocal calls
        recorder.phase = "selection" if calls == 0 else "refit"
        phase = recorder.phase
        calls += 1
        started = time.perf_counter()
        result = runtime.train_model(lt, *args, **kwargs, phase=phase, record=recorder.record)
        recorder.record({"event": "training_phase_return", "phase": phase,
                         "elapsed_seconds": time.perf_counter() - started})
        recorder.phase = "selection_complete" if phase == "selection" else "lora_export"
        return result

    def calibration_records(model, *args, **kwargs):
        runtime.require_sdpa_fp32(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = original_records(model, *args, **kwargs)
        torch.cuda.synchronize()
        recorder.record({"event": "reference_validation", "phase": recorder.phase,
                         "elapsed_seconds": time.perf_counter() - started, "decision_items": len(result),
                         "attention": "sdpa", "dtype": "float32", "autocast": False,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        return result

    def select_epoch(history):
        selected = original_select(history)
        recorder.record({"event": "development_selection", "selected_epoch": selected,
                         "candidate_epochs": [r["epoch"] for r in history]})
        recorder.phase = "base_export"
        return selected

    lt.train_model, lt.calibration_records, reference.select_epoch = train_model, calibration_records, select_epoch
    try:
        yield
    finally:
        lt.train_model, lt.calibration_records, reference.select_epoch = original_train, original_records, original_select


def prepare(run):
    run = Path(run)
    protocol.checked_reference()
    plan, _ = reference.core.checked_plan(run)
    os.environ.update(TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1")
    inventory = protocol.runtime_inventory()
    recipe = protocol.manifest(reference.core.sha(run / "plan.json"), inventory)
    reference.core.write(run / "fast-recipe.json", recipe)
    recorder = Recorder(run / "fast-runtime")
    return run, plan, recipe, recorder


def train(run, training_grades, development_grades, initial_checkpoint, *, commit=lambda: None):
    run, plan, recipe, recorder = prepare(run)
    out = run / "native"
    if out.exists():
        raise ValueError("Use a new run directory; native outputs already exist")

    def persist():
        recipe_path = out / "fast-recipe.json"
        if not recipe_path.exists():
            shutil.copyfile(run / "fast-recipe.json", recipe_path)
        binding = {"runtime_recipe_id": protocol.RECIPE_ID, "fast_recipe_sha256": protocol.sha(recipe_path),
                   "reference_plan_sha256": recipe["reference_plan_sha256"],
                   "fast_implementation_sources_sha256": recipe["implementation_sources_sha256"]}
        start_path = out / "training-start.json"
        if start_path.exists():
            start = reference.core.read(start_path)
            if "runtime_recipe_id" not in start:
                shutil.copyfile(start_path, out / "training-start.reference.json")
                start["reference_native_config"] = start.pop("config")
                start["config"] = dict(plan["native"], **protocol.EFFECTIVE)
                start.update(binding)
                replace_json(start_path, start)
        for name in ("selection-config.json", "refit-config.json"):
            path = out / name
            if path.exists():
                config = reference.core.read(path)
                if config["gradient_checkpointing"]:
                    backup = path.with_name(path.stem + ".reference.json")
                    if not backup.exists():
                        shutil.copyfile(path, backup)
                    config["gradient_checkpointing"] = False
                    replace_json(path, config)
        runtime_path = out / "fast-runtime.json"
        replace_json(runtime_path, recorder.snapshot())
        trained_path = out / "trained.json"
        if trained_path.exists():
            trained = reference.core.read(trained_path)
            trained.update(binding, format="bbeh-paired-native-fast-v1", fast_runtime_sha256=protocol.sha(runtime_path),
                           reference_training_start_sha256=protocol.sha(out / "training-start.reference.json"),
                           training_start_sha256=protocol.sha(start_path),
                           runtime_recipe_scope="New implementation; historical plan is the source/population reference")
            replace_json(trained_path, trained)
        elif (out / "base-export.json").exists() and recorder.phase == "base_export":
            recorder.record({"event": "base_export_complete"})
            recorder.phase = "lora_export"
        commit()

    try:
        with fast_execution(recorder):
            reference.train(run, training_grades, development_grades, initial_checkpoint, commit=persist)
        persist()
    except BaseException as error:
        recorder.record({"event": "failure", "error_type": type(error).__name__, "error": str(error)})
        if out.exists():
            persist()
        raise
    return reference.core.read(out / "trained.json")


def memory_probe(run, initial_checkpoint, lengths_path, output):
    run = Path(run)
    protocol.checked_reference()
    inventory = protocol.runtime_inventory()
    recorder = Recorder(Path(output).parent / "fast-memory-runtime")
    with fast_execution(recorder):
        receipt = reference.memory_probe(run, initial_checkpoint, lengths_path, output)
    receipt.update(format="laya-fast-memory-probe-v1", runtime_recipe_id=protocol.RECIPE_ID,
                   gradient_checkpointing=False, training_attention="flash_attention_2",
                   validation_attention="sdpa", validation_dtype="float32", runtime=inventory,
                   runtime_measurements=recorder.snapshot())
    replace_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-grades", type=Path, required=True)
    parser.add_argument("--development-grades", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.run, args.training_grades, args.development_grades, args.initial_checkpoint)
    print(json.dumps({key: result[key] for key in ("format", "runtime_recipe_id", "selected_epoch", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
