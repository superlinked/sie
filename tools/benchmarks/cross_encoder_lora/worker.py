"""One fresh-process SDK/worker trial; invoked by runner.py, never provisions compute."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
import uvicorn
import yaml
from sie_sdk.client.errors import RequestError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.types.inputs import Item

from tools.benchmarks.cross_encoder_lora import client as client_workload
from tools.benchmarks.cross_encoder_lora.adapter import production_class, require_forward_selection
from tools.benchmarks.cross_encoder_lora.client import (
    REQUEST_ITEMS,
    Client,
    describe_response,
)
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest, fingerprint, vector_delta
from tools.benchmarks.cross_encoder_lora.group_control import Member, validate_source
from tools.benchmarks.cross_encoder_lora.isolation import IsolatedLoadgen
from tools.benchmarks.cross_encoder_lora.protocol import (
    CONCURRENCIES,
    CPU_ORACLE,
    DOCUMENTS,
    EFFECT_DRIFT_MULTIPLIER,
    EFFECT_FLOOR,
    EXPECTED_VERSIONS,
    MAX_BATCH_ITEMS,
    MAX_TRIAL_SECONDS,
    MODEL,
    ORACLE_ATOL,
    PERFORMANCE_FUTURE_TIMEOUT_S,
    QUERY,
    RESTORE_ATOL,
    SEQUENTIAL,
    STEADY_SECONDS,
    THREADS,
    WARMUP,
    WEIGHTS_SHA256,
    require,
)

RESPONSE_CAPTURE_MODE = client_workload.RESPONSE_CAPTURE_MODE
HERE = Path(__file__).resolve().parent
ORDER = sorted(range(len(DOCUMENTS)), key=lambda index: len(DOCUMENTS[index]))
RESIDENT_CAPACITY = 2


def error_record(error: BaseException) -> dict[str, Any]:
    record: dict[str, Any] = {"type": type(error).__name__, "message": str(error), "context": []}
    seen = {id(error)}
    cause = error.__cause__ or error.__context__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        record["context"].append({"type": type(cause).__name__, "message": str(cause)})
        cause = cause.__cause__ or cause.__context__
    return record


def ordered_workload(copies: int) -> tuple[list[Item], list[Item]]:
    indices = [index for index in ORDER for _ in range(copies)]
    return (
        [Item(text=QUERY["text"]) for _ in indices],
        [Item(id=f"oracle:{j}", text=DOCUMENTS[index]) for j, index in enumerate(indices)],
    )


def batch_key(row: dict[str, Any]) -> str:
    require(len(row["documents"]) % 4 == 0, "unexpected partial four-document request batch")
    copies = len(row["documents"]) // 4
    require(1 <= copies <= MAX_BATCH_ITEMS // 4, "unexpected batch size")
    queries, documents = ordered_workload(copies)
    workload = {"queries": [item.text for item in queries], "documents": [item.text for item in documents]}
    require(
        row["queries"] == workload["queries"] and row["documents"] == workload["documents"],
        "batch order/workload differs from its preregistered oracle",
    )
    return fingerprint(workload)


def memory() -> dict[str, Any]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def telemetry() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "qualification": False}
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)  # noqa: S603 -- fixed nvidia-smi argv
    return {
        "query": command[1],
        "returncode": result.returncode,
        "csv": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def tensor_state(adapter: Any) -> dict[str, Any]:
    outer = adapter._model
    model = outer.model
    tensors = {}
    for name, tensor in model.named_parameters():
        if ".lora_" not in name:
            value = tensor.detach().cpu().contiguous()
            tensors[str(id(tensor))] = {
                "name": name.replace(".base_layer.", "."),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
    return {
        "wrapper_id": id(outer),
        "model_id": id(model),
        "tokenizer_id": id(outer.tokenizer),
        "head_id": id(model.classifier),
        "activation": str(outer.activation_fn),
        "tensors": tensors,
        "tree": {
            name: [type(module).__module__ + "." + type(module).__qualname__, id(module)]
            for name, module in model.named_modules()
        },
        "hf_peft_config_loaded": bool(getattr(model, "_hf_peft_config_loaded", False)),
        "peft_config_present": bool(getattr(model, "peft_config", None)),
    }


def base_unchanged(before: dict[str, Any], after: dict[str, Any], *, restored: bool = False) -> None:
    for key in ("wrapper_id", "model_id", "tokenizer_id", "head_id", "activation", "tensors"):
        require(before[key] == after[key], f"base state changed: {key}")
    if restored:
        for key in ("tree", "hf_peft_config_loaded", "peft_config_present"):
            require(before[key] == after[key], f"final unload did not restore {key}")


def adapter_storage(adapter: Any) -> dict[str, int]:
    """Inspect actual resident tensors outside timing, including detached layers."""
    parameters = {id(tensor): tensor for name, tensor in adapter._model.model.named_parameters() if ".lora_" in name}
    manager = getattr(adapter, "_lora", None)
    resident_count = 0
    if manager is not None:
        require(set(manager._parameters) == set(manager._loaded), "resident parameter sources differ from admission")
        declared = {}
        for source, values in manager._parameters.items():
            require(bool(values), "resident adapter has no parameter evidence")
            require(
                sum(value.numel() * value.element_size() for value in values) == manager._loaded[source],
                "resident parameter bytes differ from admission",
            )
            declared.update({id(value): value for value in values})
        for layer in getattr(manager, "_resident_layers", {}).values():
            parameters.update(
                {id(value): value for name, value in layer.named_parameters() if name.startswith("lora_")}
            )
        require(set(parameters) == set(declared), "actual resident tensors differ from declared parameter ownership")
        resident_count = len(manager._loaded)
    else:
        require(not parameters, "a model without a manager contains LoRA tensors")
    storages = {}
    for tensor in parameters.values():
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return {
        "resident_adapters": resident_count,
        "storage_count": len(storages),
        "tensor_storage_bytes": sum(storages.values()),
        "rounded_storage_bytes": sum(math.ceil(size / 512) * 512 for size in storages.values()),
    }


@contextmanager
def unobserved(adapter: Any) -> Iterator[None]:
    """Time production methods with the original forward attribute lookup restored."""
    score_pairs, select = adapter.score_pairs, adapter.set_active_lora
    production = adapter.production_adapter_class
    before = (len(adapter.batch_observations), len(adapter.forward_observations), len(adapter.selection_observations))
    adapter.remove_forward_observer()
    adapter.score_pairs = production.score_pairs.__get__(adapter)
    adapter.set_active_lora = production.set_active_lora.__get__(adapter)
    try:
        require(adapter.score_pairs.__func__ is production.score_pairs, "timed scoring is instrumented")
        require(adapter.set_active_lora.__func__ is production.set_active_lora, "timed selection is instrumented")
        adapter.assert_original_forward()
        yield
        adapter.assert_original_forward()
        after = (
            len(adapter.batch_observations),
            len(adapter.forward_observations),
            len(adapter.selection_observations),
        )
        require(before == after, "observer activity occurred during production timing")
    finally:
        adapter.score_pairs, adapter.set_active_lora = score_pairs, select
        adapter.install_forward_observer()


def matching_vector(scores: list[float], oracles: dict[str, Any]) -> dict[str, Any] | None:
    """Return one deterministic diagnostic witness, never actual batch attribution.

    The full oracle corpus and raw score vector are retained separately, so all
    matches (including ambiguity) can be recomputed without repeating up to 136
    equivalent descriptions on every response.
    """
    for key in sorted(oracles):
        oracle_row = oracles[key]
        copies = oracle_row["copies"]
        for position in range(copies):
            vector = [oracle_row["scores"][ORDER.index(index) * copies + position] for index in range(REQUEST_ITEMS)]
            if vector_delta(scores, vector) <= RESTORE_ATOL:
                return {"workload_sha256": key, "copies": copies, "request_position": position}
    return None


def performance(url: str, concurrency: int, evidence: Evidence, *, smoke: bool) -> dict[str, Any]:
    """Retain the historical in-process diagnostic with its explicit settings."""
    return client_workload.performance(
        url,
        concurrency,
        evidence,
        smoke=smoke,
        warmup=WARMUP,
        sequential=SEQUENTIAL,
        steady_seconds=STEADY_SECONDS,
        future_timeout=PERFORMANCE_FUTURE_TIMEOUT_S,
    )


def validate_usage(rows: list[dict[str, Any]], batches: list[dict[str, Any]]) -> None:
    tokens = {
        item_id: count
        for batch in batches
        for item_id, count in zip(batch["item_ids"], batch["input_token_counts"], strict=True)
    }
    for row in rows:
        expected = sum(tokens[f"{row['label']}:{i}"] for i in range(4))
        require(
            row["result"]["usage"]["input_tokens"] == expected, "SDK token usage differs from real tokenizer counts"
        )


def replay(url: str, concurrency: int) -> list[dict[str, Any]]:
    clients = [Client(url) for _ in range(concurrency)]
    rows = []
    try:
        for repeat in range(2):
            barrier = threading.Barrier(concurrency)

            def lane(index: int) -> dict[str, Any]:
                barrier.wait(timeout=20)
                return clients[index].score(None, f"replay-C{concurrency}-{repeat}-{index}")

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                rows.extend(executor.map(lane, range(concurrency)))
        return rows
    finally:
        for client in clients:
            client.close()


async def oracle(adapter: Any, worker: Any, selection: str | None, *, smoke: bool) -> dict[str, Any]:
    result = {}
    for copies in range(1, MAX_BATCH_ITEMS // 4 + 1):
        queries, documents = ordered_workload(copies)

        def forward() -> Any:
            adapter.set_active_lora(selection)
            return adapter.score_pairs(queries, documents)

        values = []
        for _ in range(2):
            output = await worker.run_adapter_mutation(forward)
            values.append(output.scores.tolist())
        require(vector_delta(*values) <= RESTORE_ATOL, "preselection shape oracle is unstable")
        key = fingerprint({"queries": [item.text for item in queries], "documents": [item.text for item in documents]})
        result[key] = {
            "copies": copies,
            "scores": values[0],
            "repeat": values[1],
            "input_token_counts": output.input_token_counts,
        }
    return result


def configs(directory: Path, source: Path, fixtures: Path, arm: str, snapshot: Path, *, smoke: bool) -> None:
    directory.mkdir()
    config = yaml.safe_load(
        (source / "packages/sie_server/models/cross-encoder__ms-marco-MiniLM-L-6-v2.yaml").read_text()
    )
    config.pop("hf_id")
    config["weights_path"] = str(snapshot)
    if smoke:
        config["max_sequence_length"] = 64
    profile = config["profiles"]["default"]
    profile["adapter_path"] = f"{HERE / 'adapter.py'}:ObservedCrossEncoder"
    if arm == "R":
        profile["adapter_path"] = f"{HERE / 'runtime_adapter.py'}:ObservedRuntimeCrossEncoder"
        profile["adapter_options"]["loadtime"] = {
            "lora_paths": {name: str(fixtures / name) for name in ("a", "b", "c")}
        }
    (directory / "model.yaml").write_text(yaml.safe_dump(config))


def preflight(args: argparse.Namespace, evidence: Evidence, *, actual_arm: str) -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    try:
        torchao = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao = None
    production = production_class(actual_arm)
    module_path = sys.modules[production.__module__].__file__
    assert module_path is not None
    actual_source = Path(module_path).resolve()
    require(actual_source.is_relative_to(args.source.resolve()), "adapter was imported from a different source tree")
    data = {
        "versions": versions,
        "torchao": torchao,
        "adapter_source": str(actual_source),
        "adapter_sha256": digest(actual_source),
        "adapter_class": f"{production.__module__}:{production.__name__}",
        "python": sys.version,
        "threads": torch.get_num_threads(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "torch_cuda": torch.version.cuda,
        "qualification": not args.smoke
        and (getattr(args, "group", None) is None or args.group.binding["mode"] == "qualification"),
        "installed_distributions": sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        ),
    }
    data["installed_distributions_sha256"] = fingerprint(data["installed_distributions"])
    evidence.emit("environment", **data)
    if not args.smoke:
        require(versions == EXPECTED_VERSIONS, "installed stack differs from the exact public lock")
        require(torchao is None, "torchao must be absent; benchmark does not mutate dependencies")
        require(torch.cuda.is_available() and "L4" in data["device"], "qualification requires the declared L4")
    return data


async def experiment(args: argparse.Namespace, evidence: Evidence, summary: dict[str, Any]) -> None:
    group: Member | None = getattr(args, "group", None)
    fixtures = args.fixtures.resolve()
    manifest = json.loads((fixtures / "manifest.json").read_text())
    snapshot = args.snapshot.resolve() if args.snapshot else Path(manifest["snapshot"])
    require(bool(manifest["smoke"]) == args.smoke, "fixture smoke/qualification mode differs")
    for filename, expected in manifest["model_files"].items():
        require(digest(snapshot / filename) == expected, "model file changed after fixture preparation")
    if not args.smoke:
        require(digest(snapshot / "model.safetensors") == WEIGHTS_SHA256, "public checkpoint hash differs")
    for name, record in manifest["adapters"].items():
        for filename, expected in record["files"].items():
            require(digest(fixtures / name / filename) == expected, "LoRA fixture changed")
    summary["fixtures"] = manifest
    configs(evidence.directory / "models", args.source, fixtures, args.arm, snapshot, smoke=args.smoke)
    app = AppFactory.create_app(
        AppStateConfig(
            models_dir=evidence.directory / "models",
            device="cpu" if args.smoke else "cuda:0",
            remote_serving=False,
            bind_host="127.0.0.1",
        )
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=20, access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    client = Client(url)
    adapter = None
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("server stopped before readiness")
            await asyncio.sleep(0.05)
        registry = app.state.registry
        load_start = time.monotonic()
        adapter = await registry.load_async(MODEL, "cpu" if args.smoke else "cuda:0")
        summary["load_s"] = time.monotonic() - load_start
        loaded = registry._loaded[MODEL]
        worker = loaded.worker
        require(worker is not None, "actual model worker is missing")
        original = await asyncio.to_thread(tensor_state, adapter)
        evidence.emit("original_state", state=original)
        expected_dtype = "torch.float32" if args.smoke else "torch.float16"
        require(
            {value["dtype"] for value in original["tensors"].values()} == {expected_dtype},
            "actual base tensor precision differs from the protocol",
        )
        summary["actual_precision"] = expected_dtype
        summary["loaded_before_forward_memory"] = memory()
        summary["manager_present"] = getattr(adapter, "_lora", None) is not None
        require(summary["manager_present"] == (args.arm == "R"), "unexpected manager activation")
        require(adapter.production_adapter_class is production_class(args.arm), "wrong production adapter class")

        async def call(selection: str | None, label: str, *, cold: bool = False) -> dict[str, Any]:
            row = await asyncio.to_thread(client.score, selection, label, cold=cold)
            evidence.emit("request", **row)
            validate_usage([row], adapter.batch_observations)
            return row

        adapter.observation_phase = "preselection"
        bases = [await call(None, f"base-{i}") for i in range(3)]
        baseline = bases[0]["scores"]
        drift = max(vector_delta(baseline, row["scores"]) for row in bases)
        require(drift <= RESTORE_ATOL, "preselection SDK baseline is unstable")
        if not args.smoke:
            require(vector_delta(baseline, CPU_ORACLE) <= ORACLE_ATOL, "finite public-checkpoint oracle differs")
        summary["base_scores"] = baseline
        summary["base_drift"] = drift
        summary["shape_oracles"] = await oracle(adapter, worker, None, smoke=args.smoke)
        evidence.emit("shape_oracles", values=summary["shape_oracles"])
        # The fixed base memory reference is warm and precedes any adapter
        # selection; CUDA library initialization is not counted as LoRA bytes.
        summary["initial_memory"] = memory()
        selected_oracles = {None: summary["shape_oracles"]}
        if args.arm == "R":
            before_fetch = len(adapter.fetch_observations)
            try:
                await call("undeclared", "undeclared")
            except RequestError as error:
                require(error.status_code == HTTPStatus.BAD_REQUEST, "undeclared selection did not return 400")
                evidence.emit("undeclared_denied", status=error.status_code, code=error.code)
            else:
                raise AssertionError("undeclared LoRA accepted")
            require(len(adapter.fetch_observations) == before_fetch, "undeclared source reached fetch")
            adapted = {}
            for alias in ("a", "b"):
                if alias == "a":
                    cold_clients = [Client(url) for _ in range(4)]
                    cold_barrier = threading.Barrier(4)

                    def cold_call(index: int) -> dict[str, Any]:
                        cold_barrier.wait(timeout=20)
                        return cold_clients[index].score("a", f"singleflight-{index}", cold=True)

                    try:
                        cold_rows = await asyncio.gather(*(asyncio.to_thread(cold_call, index) for index in range(4)))
                    finally:
                        for other in cold_clients:
                            other.close()
                    evidence.emit("singleflight_requests", requests=cold_rows)
                    require(
                        sum(any(item.get("code") == "LORA_LOADING" for item in row["responses"]) for row in cold_rows)
                        >= RESIDENT_CAPACITY,
                        "cold duplicate requests did not overlap source loading",
                    )
                    require(
                        len(adapter.fetch_observations) - before_fetch == 1, "duplicate cold requests fetched twice"
                    )
                adapted[alias] = await call(alias, f"cold-{alias}", cold=True)
                if alias == "b":
                    require(
                        any(
                            row.get("code") == "LORA_LOADING" and row["status"] == HTTPStatus.SERVICE_UNAVAILABLE
                            for row in adapted[alias]["responses"]
                        ),
                        "cold request did not expose LORA_LOADING",
                    )
                restored = await call(None, f"reset-{alias}")
                require(vector_delta(baseline, restored["scores"]) <= RESTORE_ATOL, "base restoration failed")
            effects = {
                "a_base": vector_delta(adapted["a"]["scores"], baseline),
                "b_base": vector_delta(adapted["b"]["scores"], baseline),
                "a_b": vector_delta(adapted["a"]["scores"], adapted["b"]["scores"]),
            }
            evidence.emit("effects", **effects)
            require(
                min(effects.values()) >= max(EFFECT_FLOOR, EFFECT_DRIFT_MULTIPLIER * drift),
                "adapter effects not qualified",
            )
            require(
                len(adapter.fetch_observations) - before_fetch == RESIDENT_CAPACITY,
                "cold adapters fetched more than once",
            )
            for alias in ("a", "b"):
                selected_oracles[alias] = await oracle(adapter, worker, alias, smoke=args.smoke)
            adapter.observation_phase = "concurrent-isolation"
            first = len(adapter.batch_observations)
            clients = [Client(url) for _ in range(12)]
            barrier = threading.Barrier(len(clients))
            selections = [None, "a", "b"] * 4

            def concurrent(index: int) -> dict[str, Any]:
                barrier.wait(timeout=20)
                return clients[index].score(selections[index], f"isolation-{index}")

            try:
                rows = await asyncio.gather(*(asyncio.to_thread(concurrent, index) for index in range(len(clients))))
            finally:
                for other in clients:
                    other.close()
            evidence.emit("concurrent_requests", requests=rows)
            batches = adapter.batch_observations[first:]
            evidence.emit("concurrent_batches", batches=batches)
            validate_usage(rows, batches)
            combined = set()
            for batch in batches:
                selections_in_batch = {selections[int(item.split(":")[0].split("-")[-1])] for item in batch["item_ids"]}
                require(len(selections_in_batch) == 1, "one forward mixed adapter selections")
                alias = next(iter(selections_in_batch))
                source = str(fixtures / alias) if alias else None
                require(batch["selection"] in (alias, source), "actual worker selection differs from request")
                key = batch_key(batch)
                require(
                    vector_delta(batch["scores"], selected_oracles[alias][key]["scores"]) <= RESTORE_ATOL,
                    "concurrent selected scores differ from shape-matched oracle",
                )
                expected_peft = PEFTLoRAMixin._peft_adapter_name(source) if source else None
                require_forward_selection(batch, expected_peft)
                if len({item.split(":")[0] for item in batch["item_ids"]}) > 1:
                    combined.add(alias)
            require(combined == {None, "a", "b"}, "did not observe cross-request batches for base, A and B")
            require(
                max(row["start_ns"] for row in rows) < min(row["end_ns"] for row in rows), "SDK calls did not overlap"
            )
            await call(None, "before-performance-base")
            require(set(registry.loaded_lora_names(MODEL)) == {"a", "b"}, "resident set before R measurement differs")
            summary["adapter_storage"] = adapter_storage(adapter)
            require(
                summary["adapter_storage"]["resident_adapters"] == RESIDENT_CAPACITY,
                "storage evidence does not cover every resident adapter",
            )
            base_unchanged(original, await asyncio.to_thread(tensor_state, adapter))

        summary["untimed_replays"] = {}
        adapter.observation_phase = "untimed-shape-qualification"
        for concurrency in (1, *CONCURRENCIES):
            first = len(adapter.batch_observations)
            rows = await asyncio.to_thread(replay, url, concurrency)
            batches = adapter.batch_observations[first:]
            evidence.emit("untimed_replay", concurrency=concurrency, requests=rows, batches=batches)
            summary["untimed_replays"][str(concurrency)] = {"requests": rows, "batches": batches}
            validate_usage(rows, batches)
            for batch in batches:
                key = batch_key(batch)
                require(
                    vector_delta(batch["scores"], summary["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                    "untimed SDK replay differs from preselection shape oracle",
                )
                require(batch["selection"] is None, "untimed base replay selected a LoRA")
                require_forward_selection(batch, None)
            if concurrency > 1:
                require(
                    any(len({item.split(":")[0] for item in batch["item_ids"]}) > 1 for batch in batches),
                    "untimed concurrent replay did not combine independent SDK requests",
                )
        summary["resident_memory"] = memory()
        summary["performance"] = {}
        summary["pre_timing_history"] = {
            "batches": len(adapter.batch_observations),
            "selections": len(adapter.selection_observations),
            "fetches": len(adapter.fetch_observations),
            "main_client_wire_attempts": len(client.responses),
            "retained_untimed_replays": sum(len(value["requests"]) for value in summary["untimed_replays"].values()),
            "policy": "Existing server histories retained; memory() just completed its existing gc.collect/CUDA synchronization outside timing. No new collection or history release is introduced.",
        }
        with unobserved(adapter):
            async with IsolatedLoadgen(
                url,
                args.source,
                evidence,
                smoke=args.smoke,
                idle_deadline=group.binding["work_deadline"] if group is not None else None,
            ) as loadgen:
                if group is not None:
                    group.acknowledge(
                        actual_arm=args.arm,
                        manager_present=summary["manager_present"],
                        config_sha256=digest(evidence.directory / "models/model.yaml"),
                        sdk_identity=loadgen.receipt["identity"],
                        quiet=group_quiet(worker),
                    )
                for concurrency in (1, *CONCURRENCIES):
                    if group is not None:
                        await group.grant(f"C{concurrency}")
                    phase = f"performance-C{concurrency}"
                    evidence.emit("gpu_before_phase", phase=phase, telemetry=telemetry())
                    record = await loadgen.phase(concurrency)
                    evidence.emit("gpu_after_phase", phase=phase, telemetry=telemetry())
                    if group is not None:
                        group.acknowledge(record=record, quiet=group_quiet(worker))
                if group is not None:
                    # No SDK shutdown, raw payload loading, churn, unload or
                    # finalization until all nine group phase ACKs exist.
                    await group.grant("finish")
            # Reaping and all raw-row materialization/validation happen after
            # every measured phase, so their allocations never enter the
            # serving heap during a subsequent timed phase.
            summary["loadgen"] = loadgen.receipt
            summary["performance"] = loadgen.results()
        for concurrency in (1, *CONCURRENCIES):
            measured = summary["performance"][str(concurrency)]
            evidence.emit("performance", phase=f"performance-C{concurrency}", **measured)
            atomic_json(evidence.directory / "trial.json", summary)
            for row in measured["samples"]:
                require(row["result"]["usage"] == bases[0]["result"]["usage"], "timed SDK usage differs")
                if concurrency == 1:
                    require(vector_delta(row["scores"], baseline) <= RESTORE_ATOL, "timed C1 base output differs")
                row["diagnostic_matching_oracle"] = matching_vector(row["scores"], summary["shape_oracles"])
            measured["actual_timed_batch_shape"] = "unobserved"
            evidence.emit(
                "timed_output_diagnostics",
                concurrency=concurrency,
                witnesses=[
                    {"label": row["label"], "match": row["diagnostic_matching_oracle"]} for row in measured["samples"]
                ],
            )
        adapter.observation_phase = "lifecycle"
        if args.arm == "R":
            previous = set(registry.loaded_lora_names(MODEL))
            evicted = set()
            for index, alias in enumerate(("c", "a", "b", "c")):
                await call(alias, f"churn-{index}", cold=True)
                ready = set(registry.loaded_lora_names(MODEL))
                require(alias in ready and len(ready) == RESIDENT_CAPACITY, "LRU capacity/readiness differs")
                evicted.update(previous - ready)
                evidence.emit("resident_set", selected=alias, ready=sorted(ready))
                previous = ready
            require(evicted == {"a", "b", "c"}, "LRU churn failed to evict each dynamic adapter")

            async def unload_adapters() -> None:
                for name in list(loaded.loras):
                    await registry._unload_lora(loaded, name, model=MODEL)
                    loaded.loras.pop(name)

            await unload_adapters()
            final_state = await asyncio.to_thread(tensor_state, adapter)
            evidence.emit("restored_state", state=final_state)
            base_unchanged(original, final_state, restored=True)
            restored = await call(None, "final-unloaded-base")
            require(vector_delta(baseline, restored["scores"]) <= RESTORE_ATOL, "final unload changed base output")
            summary["unloaded_memory"] = memory()
            require(
                summary["unloaded_memory"]["allocated"] == summary["initial_memory"]["allocated"],
                "adapter unload did not restore allocated GPU bytes",
            )
            await call("a", "same-instance-reload", cold=True)
            require(id(adapter._model) == original["wrapper_id"], "reload replaced outer wrapper")
            await unload_adapters()
            base_unchanged(original, await asyncio.to_thread(tensor_state, adapter), restored=True)
        require(not adapter.mutation_overlap, "adapter mutation overlapped a score forward")
        evidence.emit(
            "all_batches",
            batches=adapter.batch_observations,
            selections=adapter.selection_observations,
            fetches=adapter.fetch_observations,
        )
        inner_ref = weakref.ref(adapter._model.model)
        outer_ref = weakref.ref(adapter._model)
        await registry.unload_async(MODEL)
        gc.collect()
        require(
            adapter._model is None and outer_ref() is None and inner_ref() is None,
            "full model unload retained wrapper or inner model",
        )
        summary["model_unloaded_memory"] = memory()
        summary["passed"] = True
    finally:
        try:
            evidence.emit(
                "client_wire_responses", responses=[describe_response(response) for response in client.responses]
            )
            if adapter is not None:
                evidence.emit(
                    "final_batch_snapshot",
                    batches=adapter.batch_observations,
                    fetches=adapter.fetch_observations,
                    mutation_overlap=adapter.mutation_overlap,
                )
        finally:
            try:
                client.close()
            finally:
                server.should_exit = True
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=30)
                    summary["shutdown_complete"] = True
                finally:
                    sock.close()


def group_quiet(worker: Any) -> dict[str, Any]:
    """A lease ends only after drained requests and an outside-timer CUDA fence."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    state = {
        "pending": worker.pending_count,
        "in_flight": len(worker._in_flight),
        "cuda_synchronized": torch.cuda.is_available(),
    }
    require(state["pending"] == 0 and state["in_flight"] == 0, "worker was not quiet at lease boundary")
    return state


async def run(args: argparse.Namespace) -> dict[str, Any]:
    group: Member | None = getattr(args, "group", None)
    evidence = Evidence(args.output.resolve())
    summary: dict[str, Any] = {
        "arm": group.binding["logical_role"] if group is not None else args.arm,
        "pair": args.pair,
        "passed": False,
        "shutdown_complete": False,
        "smoke": args.smoke,
        "pid": os.getpid(),
    }
    if group is not None:
        summary["group"] = group.binding
        summary["actual_arm"] = args.arm
    os.environ.update(
        {
            "SIE_TELEMETRY_DISABLED": "1",
            "SIE_DISK_CACHE_ENABLED": "false",
            "SIE_MAX_LORAS_PER_MODEL": "2",
            "SIE_MAX_BATCH_REQUESTS": str(MAX_BATCH_ITEMS),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    try:
        if group is not None:
            await group.grant("prepare")
            validate_source(group.binding, args.source, args.arm, args.pair)
        summary["environment"] = preflight(args, evidence, actual_arm=args.arm)
        timeout = max(0, group.binding["work_deadline"] - time.monotonic()) if group is not None else args.deadline_s
        async with asyncio.timeout(timeout):
            await experiment(args, evidence, summary)
    except BaseException as error:
        summary["passed"] = False
        summary["error"] = error_record(error)
        evidence.emit("failure", **summary["error"])
        raise
    finally:
        atomic_json(evidence.directory / "trial.json", summary)
        evidence.emit("finished", passed=summary["passed"], shutdown_complete=summary["shutdown_complete"])
        if group is not None and summary["passed"] and summary["shutdown_complete"]:
            group.acknowledge(terminal_sha256=digest(evidence.directory / "trial.json"))
    return summary


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("B", "N", "R"), required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--deadline-s", type=float, default=MAX_TRIAL_SECONDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    require(
        0 < args.deadline_s <= MAX_TRIAL_SECONDS,
        f"worker deadline must be in (0, {MAX_TRIAL_SECONDS}]",
    )
    return args


def main() -> None:
    asyncio.run(run(arguments()))


if __name__ == "__main__":
    main()
