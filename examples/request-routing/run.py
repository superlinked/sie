#!/usr/bin/env python3
"""Run a small SIE cascade smoke check using the supplied frozen routing assets.

    export SIE_BASE_URL=http://localhost:8000 SIE_API_KEY=your-key
    uv run --frozen python run.py
    uv run --frozen python run.py --limit 100 --output run-output/another.jsonl

This spends SIE API credits: each input sends one embedding request and, below
confidence .29, at most one generation request. The default is 20 inputs;
--limit explicitly permits up to 3,000. There are no retries or rival calls.
SIE_BASE_URL selects the deployment exposing both models. Actual charges depend
on the endpoint; missing usage or failed-call charges remain unknown.

The evidence directory must contain assets.json, inputs.jsonl and the three
front-{clinc150,banking77,massive}.npz files. Frozen coefficients and route order
are used directly, never refitted. Results contain answers, observed usage and
fixed failure codes, without input text, exception text or latency/quality claims.
An existing output is never overwritten. This is one serving observation,
not a reproduction of a recorded score or proof of repeated-request stability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import numpy as np
from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
DATASETS = ("clinc150", "banking77", "massive")
ENCODER = "Qwen/Qwen3-Embedding-4B"
GENERATOR = "Qwen/Qwen3.8-27B-FP8"
INSTRUCTION = "Given a user request, find the requests that ask for the same action"
THRESHOLD = 0.29
TIMEOUT = 45


class RetryDisabled(RuntimeError):
    """The SDK requested an additional inference attempt."""


class NoRetrySIEClient(SIEClient):
    def _record_retry(self) -> None:
        # Current SDK retry branches call this hook before sleeping/resending.
        raise RetryDisabled


def require(ok: bool) -> None:
    if not ok:
        raise ValueError


def load_assets(directory: Path) -> tuple[dict, dict]:
    assets = json.loads((directory / "assets.json").read_text())
    fronts = {}
    for dataset in DATASETS:
        with np.load(directory / f"front-{dataset}.npz", allow_pickle=False) as archive:
            model = {name: archive[name] for name in ("coef", "intercept", "classes", "labels")}
        labels = assets[dataset]["labels"]
        require(len(labels) >= 10 and len(set(labels)) == len(labels))
        require(all(isinstance(label, str) and label for label in labels))
        front_labels = [label for label in labels if label != "none of these"]
        require(list(map(str, model["labels"])) == front_labels)
        require(model["coef"].shape == (len(model["classes"]), 2560))
        require(model["intercept"].shape == (len(model["classes"]),))
        require(bool(np.isfinite(model["coef"]).all() and np.isfinite(model["intercept"]).all()))
        require(np.issubdtype(model["classes"].dtype, np.integer))
        require(len(set(model["classes"])) == len(model["classes"]))
        require(bool(((model["classes"] >= 0) & (model["classes"] < len(front_labels))).all()))
        for label in labels:
            if label == "none of these":
                continue
            examples = assets[dataset]["examples"][label]
            require(1 <= len(examples) <= 10 and all(isinstance(text, str) for text in examples))
        fronts[dataset] = model
    return assets, fronts


def score_front(vector, model: dict) -> dict:
    vector = np.asarray(vector, dtype=np.float32)
    require(vector.shape == (2560,) and bool(np.isfinite(vector).all()))
    require(abs(float(np.linalg.norm(vector)) - 1) <= 0.01)
    query = vector.reshape(1, -1).astype(np.float16).astype(np.float32)
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    require(bool(np.isfinite(query).all() and (norm > 0).all()))
    query /= norm
    logits = query @ model["coef"].T + model["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    scores = np.zeros((1, len(model["labels"])), np.float32)
    scores[:, model["classes"]] = exp / exp.sum(axis=1, keepdims=True)
    require(bool(np.isfinite(scores).all()) and bool(np.allclose(scores.sum(), 1, atol=1e-6, rtol=0)))
    ranks = np.argsort(-scores, axis=1)[0, :10]
    top10 = [str(model["labels"][index]) for index in ranks]
    return {"pred": top10[0], "conf": float(scores.max()), "top10": top10}


def routing_body(row: dict, front: dict, assets: dict) -> dict:
    data = assets[row["dataset"]]
    labels, examples = data["labels"], data["examples"]
    require(len(front["top10"]) == 10 and len(set(front["top10"])) == 10 and set(front["top10"]) <= set(labels))
    lines = [
        label + ": " + " | ".join(" ".join(text.split()) for text in examples[label])
        if label in front["top10"]
        else label
        for label in labels
    ]
    return {
        "messages": [
            {"role": "system", "content": "Route the request. Answer with exactly one of the routes, copied exactly."},
            {
                "role": "user",
                "content": "Routes, each with example requests:\n"
                + "\n".join(lines)
                + "\n\nRequest:\n"
                + row["text"][:2000],
            },
        ],
        "max_tokens": 64,
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "route",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"route": {"type": "string", "enum": labels}},
                    "required": ["route"],
                    "additionalProperties": False,
                },
            },
        },
    }


def observed_usage(result: Mapping) -> dict | None:
    request = result.get("request") or {}
    usage = result.get("usage") or request.get("usage") or {}
    observed = {
        name: usage[name]
        for name in ("input_tokens", "prompt_tokens", "completion_tokens", "total_tokens", "credits_charged")
        if type(usage.get(name)) is int and usage[name] >= 0
    }
    if type(request.get("credits_debited")) is int and request["credits_debited"] >= 0:
        observed["credits_debited"] = request["credits_debited"]
    return observed or None


def route(client: SIEClient, row: dict, assets: dict, fronts: dict) -> dict:
    result = {
        "dataset": row.get("dataset") if isinstance(row.get("dataset"), str) and row["dataset"] in DATASETS else None,
        "id": row.get("id") if isinstance(row.get("id"), str) else None,
        "answer": None,
        "error": None,
        "embedding_usage": None,
        "generation_usage": None,
    }
    stage = "input"
    try:
        require(row.get("dataset") in DATASETS and isinstance(row.get("id"), str) and isinstance(row.get("text"), str))
        stage = "encode"
        encoded = client.encode(
            ENCODER,
            [{"text": row["text"]}],
            output_types=["dense"],
            output_dtype="float32",
            instruction=INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            provision_timeout_s=TIMEOUT,
            max_oom_retries=0,
        )
        if not isinstance(encoded, list) or len(encoded) != 1:
            raise TypeError
        result["embedding_usage"] = observed_usage(encoded[0])
        stage = "front"
        front = score_front(encoded[0]["dense"], fronts[row["dataset"]])
        result["front"] = front
        result["generation_used"] = front["conf"] < THRESHOLD
        if not result["generation_used"]:
            result["answer"] = front["pred"]
            return result
        stage = "generate"
        response = client.chat_completions(
            model=GENERATOR,
            **routing_body(row, front, assets),
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=TIMEOUT,
        )
        result["generation_usage"] = observed_usage(response)
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError
        answer = json.loads(content)
        require(
            isinstance(answer, dict)
            and set(answer) == {"route"}
            and answer["route"] in assets[row["dataset"]]["labels"]
        )
        usage = response.get("usage") or {}
        require(type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] >= 0)
        require(type(usage.get("completion_tokens")) is int and 0 <= usage["completion_tokens"] <= 64)
        result["answer"] = answer["route"]
    except Exception as error:  # noqa: BLE001 — SDK failures must not expose request or credential details.
        result["error"] = "sdk_retry_blocked" if isinstance(error, RetryDisabled) else stage + "_failed"
    return result


def limit(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 3000:
        raise argparse.ArgumentTypeError("--limit must be between 1 and 3000")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SIE cascade smoke run; API calls spend credits. Default:20 inputs, no retries."
    )
    parser.add_argument("--limit", type=limit, default=20, help="input count, from 1 to 3000 (default:20)")
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence")
    parser.add_argument("--output", type=Path, default=HERE / "run-output/cascade.jsonl")
    args = parser.parse_args()
    key = os.environ.get("SIE_API_KEY", "").strip()
    if not key:
        parser.error("SIE_API_KEY is required; running this example can spend API credits")
    base_url = os.environ.get("SIE_BASE_URL", "").strip()
    if not base_url:
        parser.error("SIE_BASE_URL is required; choose a deployment exposing both routing models")
    try:
        if args.output.exists():
            parser.error("output already exists; choose a new --output path")
        assets, fronts = load_assets(args.evidence)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with (
            (args.evidence / "inputs.jsonl").open() as inputs,
            args.output.open("x") as output,
            NoRetrySIEClient(
                base_url,
                api_key=key,
                timeout_s=TIMEOUT,
                max_connections=1,
            ) as client,
        ):
            print(f"API credits may be spent: at most {args.limit} embedding and {args.limit} generation requests.")
            count = failures = 0
            for line in islice(inputs, args.limit):
                try:
                    row = json.loads(line)
                    require(isinstance(row, dict))
                    answer = route(client, row, assets, fronts)
                except Exception:  # noqa: BLE001 — Persist a fixed failure code without untrusted input details.
                    answer = {"dataset": None, "id": None, "answer": None, "error": "input_failed"}
                output.write(json.dumps(answer, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                count += 1
                failures += answer["error"] is not None
        print(f"Wrote {count} input rows; {failures} failures. No score or latency claim is computed.")
        return 1 if failures else 0
    except Exception:  # noqa: BLE001 — Setup failures may contain deployment credentials.
        print(
            "Setup or output failed; existing result rows are retained. Exception details are suppressed.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
