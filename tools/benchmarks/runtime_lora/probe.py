"""Exercise runtime LoRA through public SIE SDK routes on one CUDA device."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import socket
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import torch
import uvicorn
import yaml
from msgpack.exceptions import UnpackException
from sie_sdk import SIEClient
from sie_sdk._msgpack import unpackb as unpack_msgpack
from sie_sdk.client.errors import RequestError, ServerError
from sie_sdk.types import ChatMessage
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import ModelConfig

from tools.benchmarks.runtime_lora.adapters import KV_CAPACITY, STATIC_MEMORY_FRACTION
from tools.benchmarks.runtime_lora.fixtures import ALIASES, MODELS, RANK, digest, prepare
from tools.benchmarks.runtime_lora.observer import TRACE_ENV

ROOT = Path(__file__).resolve().parents[3]
MESSAGES: list[ChatMessage] = [{"role": "user", "content": "Explain why the sky is blue in two short sentences."}]
PROMPT = "The sky appears blue because"
BOUNDARY_PROMPT = PROMPT + " of the reflection of light. This is an example of"
TEXTS = ["A red fox walks through the forest.", "Neural networks encode text as vectors."]
MAX_RESIDENT = 3
EMBEDDING_ARRAY_NDIM = 2
MIN_EMBEDDING_EFFECT = 1e-5
MAX_EMBEDDING_RESET_ERROR = 5e-5
MIN_PROBABILITY_EFFECT = 1e-4
MAX_PROBABILITY_RESET_ERROR = 1e-3
MAX_GENERATION_RETRIES = 10
MAX_RETRY_DELAY_S = 5
EFFECT_DRIFT_MULTIPLIER = 5


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wire_object(message: httpx.Request | httpx.Response) -> dict[str, Any]:
    try:
        if message.headers.get("content-type", "").split(";", 1)[0] == "application/msgpack":
            value = unpack_msgpack(message.content, numeric_arrays=False)
        else:
            value = json.loads(message.content)
    except (ValueError, UnpackException, httpx.RequestNotRead):
        return {}
    return value if isinstance(value, dict) else {}


class Evidence:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {"event": event, "time_ns": time.time_ns(), **fields}
        encoded = json.dumps(row, sort_keys=True)
        with self._lock, (self.directory / "events.jsonl").open("a") as file:
            file.write(encoded + "\n")
        print("RUNTIME_LORA " + encoded, flush=True)
        return row


class Client:
    def __init__(self, url: str, evidence: Evidence, *, transport: httpx.BaseTransport | None = None) -> None:
        self.evidence = evidence
        self.responses: list[dict[str, Any]] = []
        http = httpx.Client(base_url=url, timeout=180, transport=transport, event_hooks={"response": [self.observe]})
        self.sdk = SIEClient(url, timeout_s=180, http_client=http)

    def observe(self, response: httpx.Response) -> None:
        code = None
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            response.read()
            body = wire_object(response)
            error = body.get("error", body.get("detail", body))
            code = error.get("code") if isinstance(error, dict) else None
        request = wire_object(response.request)
        options = (request.get("params") or {}).get("options") or request.get("options") or {}
        row = {
            "path": response.request.url.path,
            "status": response.status_code,
            "code": code,
            "selection": request.get("lora_adapter", options.get("lora")),
        }
        self.responses.append(row)
        self.evidence.emit("http_response", **row)

    def call(self, kind: str, model: str, selection: str | None, route: str = "chat") -> dict[str, Any]:
        for attempt in range(MAX_GENERATION_RETRIES + 1):
            response_start = len(self.responses)
            try:
                return self._call(kind, model, selection, route)
            except ServerError as error:
                if (
                    kind != "generation"
                    or error.status_code != HTTPStatus.SERVICE_UNAVAILABLE
                    or error.code != "LORA_LOADING"
                    or len(self.responses) == response_start
                    or self.responses[-1]["status"] != HTTPStatus.SERVICE_UNAVAILABLE
                    or self.responses[-1]["code"] != "LORA_LOADING"
                    or attempt == MAX_GENERATION_RETRIES
                ):
                    raise
                delay = error.retry_after if error.retry_after is not None else 0.2
                require(0 <= delay <= MAX_RETRY_DELAY_S, "LoRA retry delay is outside this bounded probe")
                self.evidence.emit(
                    "fresh_sdk_generation_retry", selection=selection, route=route, attempt=attempt + 1, delay_s=delay
                )
                time.sleep(delay)
        raise AssertionError("unreachable retry state")

    def _call(self, kind: str, model: str, selection: str | None, route: str) -> dict[str, Any]:
        if kind == "embedding":
            result = self.sdk.encode(
                model,
                [{"text": text} for text in TEXTS],
                output_types=["dense"],
                options={"lora": selection},
                max_oom_retries=0,
            )
            return {"vectors": np.stack([item["dense"] for item in result]).tolist()}
        if route == "generate":
            result = self.sdk.generate(
                model,
                PROMPT,
                max_new_tokens=24,
                temperature=0.0,
                seed=17,
                lora_adapter=selection,
                max_oom_retries=0,
            )
            return {"text": result["text"], "usage": result.get("usage")}
        if route in {"native_stream", "native_boundary"}:
            boundary = route == "native_boundary"
            chunks = list(
                self.sdk.stream_generate(
                    model,
                    BOUNDARY_PROMPT if boundary else PROMPT,
                    max_new_tokens=1 if boundary else 24,
                    temperature=0.0,
                    seed=17,
                    logprobs=True,
                    top_logprobs=20,
                    lora_adapter=selection,
                    max_oom_retries=0,
                )
            )
            require(chunks and chunks[-1].get("done"), "native stream had no successful terminal event")
            require(chunks[-1].get("finish_reason") in {"stop", "length"}, "native stream did not finish successfully")
            content = [item for chunk in chunks for item in chunk.get("logprobs", [])]
            require(content, "native stream did not return token probabilities")
            require(
                len(content) == (chunks[-1].get("usage") or {}).get("completion_tokens"),
                "native stream omitted token probabilities",
            )
            return {
                "text": "".join(chunk.get("text_delta", "") for chunk in chunks),
                "logprobs": {"content": content},
                "usage": chunks[-1].get("usage"),
                "chunk_count": len(chunks),
            }
        result = self.sdk.chat_completions(
            model,
            MESSAGES,
            max_tokens=24,
            temperature=0.0,
            seed=17,
            logprobs=True,
            top_logprobs=5,
            lora_adapter=selection,
            max_oom_retries=0,
        )
        choice = result["choices"][0]
        return {"text": choice["message"].get("content") or "", "logprobs": choice.get("logprobs")}

    def stream(self, model: str, selection: str, barrier: threading.Barrier) -> dict[str, Any]:
        barrier.wait(timeout=10)
        start = time.time_ns()
        chunks = list(
            self.sdk.stream_chat_completions(
                model,
                [{"role": "user", "content": "Count from 1 to 200, separated by commas. Do not explain."}],
                max_tokens=128,
                temperature=0.0,
                seed=17,
                lora_adapter=selection,
                extra_body={"min_tokens": 96},
                max_oom_retries=0,
            )
        )
        end = time.time_ns()
        text = "".join(
            choice.get("delta", {}).get("content") or "" for chunk in chunks for choice in chunk.get("choices", [])
        )
        require(chunks and text, "stream returned no content")
        return {"selection": selection, "start_ns": start, "end_ns": end, "chunk_count": len(chunks), "text": text}

    def close(self) -> None:
        self.sdk.close()


def token_content(result: dict[str, Any]) -> list[dict[str, Any]]:
    content = (result.get("logprobs") or {}).get("content") or []
    require(content, "generation must return token logprobs")
    return content


def probability(value: Any) -> float:
    logprob = float(value)
    require(not math.isnan(logprob) and logprob <= 0, "invalid token log probability")
    return math.exp(logprob)


def token_distribution_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    def observed(value: dict[str, Any]) -> tuple[dict[str, float], float]:
        top = {item["token"]: probability(item["logprob"]) for item in value.get("top_logprobs") or []}
        cutoff = min(top.values()) if top else 1.0
        return {**top, value["token"]: probability(value["logprob"])}, cutoff

    a, cutoff_a = observed(left)
    b, cutoff_b = observed(right)
    shared_delta = max((abs(a[token] - b[token]) for token in a.keys() & b.keys()), default=0.0)
    changes: list[dict[str, Any]] = [
        {"token": token, "observed_in": side, "probability": known[token], "other_upper_bound": cutoff}
        for known, missing, side, cutoff in ((a, b, "left", cutoff_b), (b, a, "right", cutoff_a))
        for token in sorted(known.keys() - missing.keys())
    ]
    # An absent token is below the other top-k cutoff, not probability zero.
    lower_bound = max([shared_delta, *(max(0.0, item["probability"] - item["other_upper_bound"]) for item in changes)])
    return {
        "shared_probability_delta": shared_delta,
        "probability_delta_lower_bound": lower_bound,
        "support_changes": changes,
    }


def distribution_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return token_distribution_comparison(token_content(left)[0], token_content(right)[0])[
        "probability_delta_lower_bound"
    ]


def sequence_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare probabilities only while the preceding token contexts match."""
    a, b = token_content(left), token_content(right)
    deltas = []
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        same_token = x["token"] == y["token"]
        comparison = token_distribution_comparison(x, y)
        deltas.append(
            {
                "index": index,
                "same_token": same_token,
                "chosen_probability_delta": abs(probability(x["logprob"]) - probability(y["logprob"]))
                if same_token
                else None,
                **comparison,
            }
        )
        if not same_token:
            break
    return {
        "text_equal": left["text"] == right["text"],
        "same_token_sequence": len(a) == len(b) == len(deltas) and all(row["same_token"] for row in deltas),
        "max_aligned_chosen_probability_delta": max(
            (row["chosen_probability_delta"] or 0.0 for row in deltas), default=0.0
        ),
        "max_aligned_shared_top_probability_delta": max(
            (row["shared_probability_delta"] for row in deltas), default=0.0
        ),
        "max_aligned_probability_delta_lower_bound": max(
            (row["probability_delta_lower_bound"] for row in deltas), default=0.0
        ),
        "matching_context_positions": deltas,
    }


def embedding_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Compare raw SDK vectors without normalizing away output differences."""
    a, b = (np.asarray(value["vectors"], dtype=np.float64) for value in (left, right))
    require(
        a.ndim == b.ndim == EMBEDDING_ARRAY_NDIM and a.shape == b.shape and a.size > 0,
        "embedding shapes differ or are empty",
    )
    require(np.isfinite(a).all() and np.isfinite(b).all(), "non-finite embedding")
    return float(np.max(np.abs(a - b)))


def check_effect(
    kind: str,
    base: dict[str, Any],
    a: dict[str, Any],
    b: dict[str, Any],
    restored: dict[str, Any],
    baseline_repeat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = baseline_repeat if baseline_repeat is not None else base
    if kind == "embedding":
        ab, aa, bb = (embedding_delta(x, y) for x, y in ((a, b), (base, a), (base, b)))
        baseline_drift = embedding_delta(base, baseline)
        reset = embedding_delta(base, restored)
        require(baseline_drift < MAX_EMBEDDING_RESET_ERROR, "warm embedding baseline was not stable")
        require(reset < MAX_EMBEDDING_RESET_ERROR, "base embeddings were not restored")
        minimum = max(MIN_EMBEDDING_EFFECT, EFFECT_DRIFT_MULTIPLIER * max(reset, baseline_drift))
        require(min(ab, aa, bb) > minimum, "LoRA embeddings did not differ beyond base drift from base and each other")
        return {
            "a_vs_b_max_abs": ab,
            "a_vs_base_max_abs": aa,
            "b_vs_base_max_abs": bb,
            "base_reset_max_abs": reset,
            "baseline_max_abs": baseline_drift,
            "required_effect_minimum": minimum,
        }
    aa, bb, ab = (distribution_delta(x, y) for x, y in ((base, a), (base, b), (a, b)))
    reset = distribution_delta(base, restored)
    baseline_drift = distribution_delta(base, baseline)
    repeat_comparison = sequence_comparison(base, baseline)
    reset_comparison = sequence_comparison(base, restored)
    minimum = max(MIN_PROBABILITY_EFFECT, EFFECT_DRIFT_MULTIPLIER * max(reset, baseline_drift))
    require(
        repeat_comparison["text_equal"]
        and repeat_comparison["same_token_sequence"]
        and repeat_comparison["max_aligned_chosen_probability_delta"] < MAX_PROBABILITY_RESET_ERROR
        and repeat_comparison["max_aligned_probability_delta_lower_bound"] < MAX_PROBABILITY_RESET_ERROR,
        "warm generation baseline was not stable",
    )
    require(
        reset_comparison["text_equal"]
        and reset_comparison["same_token_sequence"]
        and reset_comparison["max_aligned_chosen_probability_delta"] < MAX_PROBABILITY_RESET_ERROR
        and reset_comparison["max_aligned_probability_delta_lower_bound"] < MAX_PROBABILITY_RESET_ERROR
        and reset < MAX_PROBABILITY_RESET_ERROR,
        "base generation was not restored",
    )
    require(min(aa, bb, ab) > minimum, "LoRA generation had no observable effect beyond base drift")
    return {
        "a_vs_base_first_token_probability_delta_lower_bound": aa,
        "b_vs_base_first_token_probability_delta_lower_bound": bb,
        "a_vs_b_first_token_probability_delta_lower_bound": ab,
        "base_reset_first_token_probability_delta_lower_bound": reset,
        "baseline_first_token_probability_delta_lower_bound": baseline_drift,
        "baseline_comparison": repeat_comparison,
        "reset_comparison": reset_comparison,
        "required_effect_minimum": minimum,
        "base_text_restored": True,
    }


def record_check(
    evidence: Evidence, failures: list[dict[str, str]], phase: str, check: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Continue independent proof phases after numeric assertions, never runtime errors."""
    try:
        result = check()
    except AssertionError as error:
        failure = {"phase": phase, "message": str(error)}
        failures.append(failure)
        evidence.emit("assertion_failed", **failure)
        return {"passed": False, **failure}
    evidence.emit("assertion_pass", phase=phase, **result)
    return {"passed": True, **result}


def check_buffered_native(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(results["warm_1"]["text"] == results["warm_2"]["text"], "native buffered warm baseline text was not stable")
    require(results["warm_2"]["text"] == results["reset"]["text"], "native buffered base text did not restore")
    return {
        "first_vs_warm_text_equal": results["first"]["text"] == results["warm_2"]["text"],
        "warm_text_restored": True,
    }


def configs(directory: Path, fixtures: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind, (model, revision) in MODELS.items():
        data = yaml.safe_load((ROOT / "packages/sie_server/models" / f"{model.replace('/', '__')}.yaml").read_text())
        require(data["hf_revision"] == revision, "catalog model revision differs from benchmark pin")
        profile = data["profiles"]["default"]
        profile["adapter_path"] = (
            f"{Path(__file__).with_name('adapters.py')}:ObservedGenerationAdapter"
            if kind == "generation"
            else f"{Path(__file__).with_name('adapters.py')}:CappedEmbeddingAdapter"
        )
        profile["max_batch_tokens"] = 4096
        profile["kv_budget_tokens"] = KV_CAPACITY
        loadtime = profile.setdefault("adapter_options", {}).setdefault("loadtime", {})
        if kind == "embedding":
            # The catalog's original PyTorch adapter owns the causal option.
            loadtime.pop("causal", None)
        else:
            loadtime["extra_launch_args"] = [
                *loadtime.get("extra_launch_args", []),
                "--max-total-tokens",
                str(KV_CAPACITY),
            ]
        loadtime.update(
            {
                "mem_fraction_static": STATIC_MEMORY_FRACTION,
                "max_seq_length": 4096,
                "lora_paths": {alias: str((fixtures / kind / alias).resolve()) for alias in ALIASES},
                "preload_loras": ["pin"],
                "max_lora_rank": RANK,
                "lora_target_modules": ["q_proj", "v_proj"],
                "max_loras_per_batch": 2,
                "max_loaded_loras": 3,
            }
        )
        data["profiles"] = {"default": profile}
        data["max_sequence_length"] = 4096
        ModelConfig.model_validate(data)
        (directory / f"{kind}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        result[kind] = data
    return result


def source_manifest() -> dict[str, Any]:
    roots = [ROOT / "packages/sie_server/src", ROOT / "packages/sie_sdk/src", Path(__file__).parent]
    files = [path for root in roots for path in root.rglob("*.py") if "__pycache__" not in path.parts]
    files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock"))
    files.extend(
        ROOT / "packages/sie_server/models" / f"{model.replace('/', '__')}.yaml" for model, _ in MODELS.values()
    )
    files.sort()
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "files": hashes}


def batch_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def mixed_batches(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row.get("event") != "forward" or not start <= row["start_ns"] <= end:
            continue
        selected = {req["uid"]: req["lora"] for req in row["requests"] if req["lora"] in {"a", "b"}}
        if not {"a", "b"}.issubset(selected.values()):
            continue
        slots = {entry["uid"]: entry for entry in row["gpu_slots"]}
        if not all(
            uid in slots
            and slots[uid]["lora"] == alias
            and 0 <= slots[uid]["slot"] < row["gpu_slot_count"]
            and row["gpu_slot_uids"][slots[uid]["slot"]] == uid
            for uid, alias in selected.items()
        ):
            continue
        if len({slots[uid]["slot"] for uid in selected}) == len(selected):
            result.append(row)
    return result


async def run_model(kind: str, registry: Any, url: str, evidence: Evidence, trace: Path) -> dict[str, Any]:
    model = MODELS[kind][0]
    client = Client(url, evidence)
    others = []
    pid = None
    failures: list[dict[str, str]] = []

    async def call(selection: str | None, phase: str, route: str = "chat") -> dict[str, Any]:
        start = time.time_ns()
        value = await asyncio.to_thread(client.call, kind, model, selection, route)
        end = time.time_ns()
        evidence.emit(
            "result", kind=kind, phase=phase, route=route, selection=selection, start_ns=start, end_ns=end, result=value
        )
        if kind == "generation":
            matching = [
                row
                for row in batch_rows(trace)
                if row.get("event") == "forward"
                and start <= row["start_ns"] <= end
                and any(req["lora"] == selection for req in row["requests"])
            ]
            require(matching, f"{route} did not forward with its selected LoRA")
        if pid is not None:
            require(registry.get(model)._process.pid == pid, "engine PID changed during LoRA lifecycle")
        return value

    async def listing(phase: str) -> set[str]:
        models = await asyncio.to_thread(client.sdk.list_models)
        current = cast("dict[str, Any]", next(item for item in models if item["name"] == model))
        require("lora_adapters" in current, "local listing did not expose ready LoRAs")
        names = set(current["lora_adapters"])
        require("pin" in names and len(names) <= MAX_RESIDENT, "pinned adapter or residency bound violated")
        evidence.emit("residency", kind=kind, phase=phase, names=sorted(names), pid=registry.get(model)._process.pid)
        return names

    try:
        base = await call(None, "base_before")
        process = registry.get(model)._process
        pid = process.pid
        require(await listing("initial") == {"pin"}, "dynamic adapters must start unloaded")
        try:
            await call("undeclared", "undeclared")
        except RequestError as error:
            require(error.status_code == HTTPStatus.BAD_REQUEST, "undeclared adapter was not rejected as invalid input")
            evidence.emit("undeclared_denied", kind=kind, status=error.status_code, code=error.code)
        else:
            raise AssertionError("undeclared adapter was accepted")
        native: dict[str, dict[str, dict[str, Any]]] = {}
        baseline_repeat = None
        if kind == "embedding":
            baseline_repeat = await call(None, "embedding_base_warm_1")
            warm = await call(None, "embedding_base_warm_2")
            evidence.emit(
                "baseline_comparison",
                route="embedding",
                first_vs_warm_max_abs=embedding_delta(base, warm),
                warm_repeat_max_abs=embedding_delta(baseline_repeat, warm),
            )
            base = warm
        else:
            baseline_repeat = await call(None, "chat_base_warm_1")
            warm = await call(None, "chat_base_warm_2")
            evidence.emit(
                "baseline_comparison",
                route="chat",
                first_vs_warm=sequence_comparison(base, warm),
                warm_repeat=sequence_comparison(baseline_repeat, warm),
            )
            base = warm
            for route in ("generate", "native_stream", "native_boundary"):
                native[route] = {
                    phase: await call(None, f"{route}_base_{phase}", route) for phase in ("first", "warm_1", "warm_2")
                }
                if route != "generate":
                    evidence.emit(
                        "baseline_comparison",
                        route=route,
                        first_vs_warm=sequence_comparison(native[route]["first"], native[route]["warm_2"]),
                        warm_repeat=sequence_comparison(native[route]["warm_1"], native[route]["warm_2"]),
                    )
            for route, values in native.items():
                values["a"] = await call("a", f"{route}_a", route)
        a = await call("a", "a")
        b = await call("b", "cold_b")
        require(await listing("a_b_ready") == {"pin", "a", "b"}, "ready listing omitted a selected adapter")
        reset = await call(None, "base_after_a_b")
        if kind == "generation":
            evidence.emit("restoration_comparison", route="chat", comparison=sequence_comparison(base, reset))
        effects = record_check(
            evidence, failures, f"{kind}_effects", lambda: check_effect(kind, base, a, b, reset, baseline_repeat)
        )

        native_effects = {}
        if kind == "generation":
            for route, values in native.items():
                values["b"] = await call("b", f"{route}_b", route)
                values["reset"] = await call(None, f"{route}_reset", route)
                if route == "generate":
                    native_effects[route] = record_check(
                        evidence, failures, route, lambda: check_buffered_native(values)
                    )
                else:
                    evidence.emit(
                        "restoration_comparison",
                        route=route,
                        comparison=sequence_comparison(values["warm_2"], values["reset"]),
                    )
                    native_effects[route] = record_check(
                        evidence,
                        failures,
                        route,
                        lambda: check_effect(
                            "generation", values["warm_2"], values["a"], values["b"], values["reset"], values["warm_1"]
                        ),
                    )
            barrier = threading.Barrier(2)
            others = [Client(url, evidence), Client(url, evidence)]
            streams = await asyncio.gather(
                *(
                    asyncio.to_thread(item.stream, model, alias, barrier)
                    for item, alias in zip(others, ("a", "b"), strict=True)
                )
            )
            start, end = max(item["start_ns"] for item in streams), min(item["end_ns"] for item in streams)
            require(start < end, "A/B client streams did not overlap")
            batches = mixed_batches(batch_rows(trace), start, end)
            require(batches, "no successful SGLang forward contained both A and B")
            evidence.emit(
                "mixed_engine_batch_pass", streams=streams, forward_count=len(batches), first_batch=batches[0]
            )

        previous = await listing("before_churn")
        evicted = set()
        for index, selection in enumerate(("c", "d", "a", "b", "c", "d", "pin")):
            await call(selection, f"churn_{index}")
            ready = await listing(f"churn_{index}")
            require(selection in ready, "successful request missing from ready listing")
            evicted.update(previous - ready)
            previous = ready
        require(evicted and "pin" not in evicted, "churn did not evict dynamic adapters while preserving pin")
        restored = await call(None, "base_after_churn")
        if kind == "generation":
            evidence.emit(
                "restoration_comparison", route="chat_after_churn", comparison=sequence_comparison(base, restored)
            )
        post_churn_effects = record_check(
            evidence,
            failures,
            f"{kind}_post_churn_effects",
            lambda: check_effect(kind, base, a, b, restored, baseline_repeat),
        )
        cold = {
            row["selection"]
            for row in client.responses
            if row["status"] == HTTPStatus.SERVICE_UNAVAILABLE and row["code"] == "LORA_LOADING"
        }
        require(
            {"a", "b", "c", "d"}.issubset(cold),
            "cold adapter requests did not expose 503 LORA_LOADING before SDK retry",
        )
        result = {
            "model": model,
            "pid": pid,
            "effects": effects,
            "post_churn_effects": post_churn_effects,
            "cold_retried": sorted(cold),
            "evicted": sorted(evicted),
            "pin_preserved": True,
            "assertion_failures": failures,
            "passed": not failures,
        }
        if kind == "generation":
            forwards = [row for row in batch_rows(trace) if row.get("event") == "forward"]
            capacity = max(row["gpu_slot_count"] for row in forwards)
            seen = {req["lora"] for row in forwards for req in row["requests"]} - {None}
            require(
                set(ALIASES).issubset(seen) and len(seen) > capacity,
                "successful churn did not exceed actual GPU slot count",
            )
            result.update(
                {
                    "gpu_slot_count": capacity,
                    "distinct_forward_adapters": sorted(seen),
                    "native_effects": native_effects,
                }
            )
        evidence.emit("model_result", kind=kind, **result)
        await registry.unload_async(model)
        require(process.poll() is not None, "owned engine still running after sequential unload")
        evidence.emit("engine_stopped", kind=kind, pid=pid, returncode=process.poll())
        return result
    finally:
        client.close()
        for item in others:
            item.close()


async def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    evidence = Evidence(output)
    os.environ.update(
        {
            "SIE_TELEMETRY_DISABLED": "1",
            "SIE_DISK_CACHE_ENABLED": "false",
            "SIE_MAX_LORAS_PER_MODEL": "3",
            "SIE_MODEL_LOAD_TIMEOUT_S": "360",
            "SIE_SGLANG_STARTUP_TIMEOUT_S": "330",
            TRACE_ENV: str(output / "batches.jsonl"),
        }
    )
    require(
        "preload_loras" in inspect.signature(SGLangGenerationAdapter).parameters,
        "runtime LoRA implementation is absent from this source tree",
    )
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sglang", "peft")}
    require(versions["sglang"] == "0.5.10.post1", "unexpected SGLang version")
    require(torch.cuda.is_available(), "CUDA is required for acceptance; run test_probe.py for CPU smoke")
    evidence.emit(
        "environment",
        versions=versions,
        device=torch.cuda.get_device_name(0),
        total_memory=torch.cuda.get_device_properties(0).total_memory,
    )
    manifest = source_manifest()
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    evidence.emit("source", sha256=manifest["sha256"])
    fixtures = args.fixtures.resolve() if args.fixtures else output / "fixtures"
    if not (fixtures / "manifest.json").is_file():
        prepare(fixtures, cache_dir=args.cache_dir)
    fixture_manifest = json.loads((fixtures / "manifest.json").read_text())
    for kind in MODELS:
        for alias, metadata in fixture_manifest[kind]["adapters"].items():
            require(
                all(digest(fixtures / kind / alias / name) == sha for name, sha in metadata["files"].items()),
                "fixture hash mismatch",
            )
    (output / "fixtures-manifest.json").write_text(json.dumps(fixture_manifest, indent=2) + "\n")
    configs(output / "models", fixtures)
    app = AppFactory.create_app(
        AppStateConfig(models_dir=output / "models", device="cuda:0", remote_serving=False, bind_host="127.0.0.1")
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="info", timeout_graceful_shutdown=30))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    summary: dict[str, Any] = {"passed": False, "versions": versions, "source_sha256": manifest["sha256"], "models": {}}
    try:
        async with asyncio.timeout(args.deadline_s):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("SIE stopped before readiness")
                await asyncio.sleep(0.1)
            for kind in MODELS:
                summary["models"][kind] = await run_model(
                    kind, app.state.registry, url, evidence, output / "batches.jsonl"
                )
            require(
                all(result["passed"] for result in summary["models"].values()),
                "required acceptance assertions failed; see model results and events",
            )
            summary["passed"] = True
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        summary["assertions_passed"] = summary["passed"]
        summary["shutdown_complete"] = False
        (output / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
        server.should_exit = True
        try:
            async with asyncio.timeout(40):
                await task
            summary["shutdown_complete"] = True
        except BaseException as error:
            summary["passed"] = False
            summary["shutdown_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            (output / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
            evidence.emit("finished", passed=summary["passed"], shutdown_complete=summary["shutdown_complete"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument("--deadline-s", type=float, default=780)
    args = parser.parse_args()
    require(not args.output.exists() or not any(args.output.iterdir()), "use a fresh evidence directory")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
