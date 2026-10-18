"""Compute count-based decode rates without treating SSE events as tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools.benchmarks.stream_decode.observer import SOURCE_FILES, SOURCE_SHA256, VERSION, digest, save

MIN_OBSERVATIONS = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count(value: Any) -> bool:
    return type(value) is int and value >= 0


def analyze_trial(
    client: dict[str, Any], backend: dict[str, Any], prompt_tokens: int, output_tokens: int
) -> dict[str, Any]:
    require(client.get("completed") is True and backend.get("completed") is True, "incomplete stream")
    require(client.get("retry_count") == 0 and client.get("http_requests") == 1, "retried or multiple HTTP requests")
    require(
        bool(client.get("clock_domain")) and client["clock_domain"] == backend.get("clock_domain"),
        "different or unknown clock domain",
    )
    start, end = client["started_ns"], client["finished_ns"]
    require(type(start) is int and type(end) is int and start < end, "invalid client interval")
    backend_start, backend_end = backend["started_ns"], backend["finished_ns"]
    require(
        type(backend_start) is int and type(backend_end) is int and backend_start < backend_end,
        "invalid backend interval",
    )
    require(start <= backend_start <= end, "backend call outside client interval")
    chunks = client["chunks"]
    require(bool(chunks), "missing SDK chunks")
    text, first_visible, last_visible = "", None, None
    usage = []
    finishes = []
    client_ids = set()
    previous = start
    for row in chunks:
        at = row["at_ns"]
        require(type(at) is int and previous <= at <= end, "invalid SDK chunk clock")
        previous = at
        chunk = row["chunk"]
        require(isinstance(chunk, dict), "invalid SDK chunk")
        require(isinstance(chunk.get("id"), str) and bool(chunk["id"]), "missing SDK request id")
        client_ids.add(chunk["id"])
        if chunk.get("usage") is not None:
            usage.append(chunk["usage"])
        choices = chunk.get("choices")
        require(isinstance(choices, list) and len(choices) <= 1, "expected one completion")
        if not choices:
            continue
        choice = choices[0]
        require(choice.get("index") == 0, "unexpected completion index")
        delta = choice.get("delta", {})
        require(
            not delta.get("reasoning_content") and not delta.get("tool_calls"),
            "benchmark requires plain answer-only text",
        )
        content = delta.get("content")
        require(content is None or isinstance(content, str), "invalid text delta")
        if content:
            text += content
            first_visible = at if first_visible is None else first_visible
            last_visible = at
        if choice.get("finish_reason") is not None:
            finishes.append(choice["finish_reason"])
    require(first_visible is not None and last_visible is not None, "no visible output")
    require(finishes == ["length"], "output did not reach its fixed token cap")
    require(len(usage) == 1, "terminal usage absent or duplicated")
    require(
        all(
            count(u.get("prompt_tokens"))
            and count(u.get("completion_tokens"))
            and u["prompt_tokens"] == prompt_tokens
            and u["completion_tokens"] == output_tokens
            for u in usage
        ),
        "prompt or completion length differs from fixed protocol",
    )
    events = backend["events"]
    require(bool(events), "backend token events absent")
    advances = []
    previous_count, previous_text, previous_time = 0, "", backend["started_ns"]
    terminal = []
    identifiers = set()
    incremental = backend.get("incremental_streaming_output")
    require(type(incremental) is bool, "unknown backend streaming mode")
    for row in events:
        at, event = row["at_ns"], row["event"]
        require(type(at) is int and previous_time <= at <= min(end, backend_end), "invalid backend event clock")
        previous_time = at
        require(isinstance(event, dict) and isinstance(event.get("meta_info"), dict), "invalid backend event")
        meta = event["meta_info"]
        tokens = meta.get("completion_tokens")
        require(
            count(tokens) and previous_count <= tokens <= output_tokens, "missing or regressing backend token count"
        )
        require(
            count(meta.get("prompt_tokens")) and meta["prompt_tokens"] == prompt_tokens, "backend prompt length changed"
        )
        cumulative = event.get("text")
        require(isinstance(cumulative, str), "missing backend text")
        if incremental:
            cumulative = previous_text + cumulative
        require(isinstance(cumulative, str) and cumulative.startswith(previous_text), "backend text is not cumulative")
        if tokens > previous_count:
            advances.append({"at_ns": at, "tokens": tokens, "new_tokens": tokens - previous_count})
        previous_count, previous_text = tokens, cumulative
        require(isinstance(meta.get("id"), str) and bool(meta["id"]), "missing backend request id")
        identifiers.add(meta["id"])
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        if finish is not None:
            terminal.append(finish)
    require(len(identifiers) == 1 and client_ids == identifiers, "mixed or mismatched stream request ids")
    require(terminal == ["length"] and previous_count == output_tokens, "backend stream incomplete or wrong length")
    require(previous_text == text, "SDK and backend output differ")
    require(len(advances) >= MIN_OBSERVATIONS, "too few token observations to measure decode")
    first, last = advances[0], advances[-1]
    interval = (last["at_ns"] - first["at_ns"]) / 1e9
    require(math.isfinite(interval) and interval > 0, "nonpositive decode interval")
    rate = (last["tokens"] - first["tokens"]) / interval
    return {
        "client_ttft_s": (first_visible - start) / 1e9,
        "client_total_s": (end - start) / 1e9,
        "completion_tokens_per_total_s": output_tokens / ((end - start) / 1e9),
        "backend_first_observation_tokens": first["tokens"],
        "backend_decode_interval_s": interval,
        "backend_after_first_observation_tokens_per_s": rate,
        "backend_after_first_token_tokens_per_s": rate if first["tokens"] == 1 else None,
        "exact_first_token_observed": first["tokens"] == 1,
        "backend_token_advance_groups": len(advances),
        "largest_token_advance_group": max(a["new_tokens"] for a in advances),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def bindings(client: dict[str, Any], backend: dict[str, Any], plan: dict[str, Any], prompt_tokens: int) -> str:
    """Check the actual wire request and prepared engine input, independently."""
    require(client.get("plan_sha256") == digest(plan), "client plan changed")
    require(client.get("clock_domain") == plan["clock_domain"], "planned clock domain changed")
    require(client.get("cohort") == backend.get("cohort") == plan["cohort"], "cohort changed")
    require(backend.get("server_manifest_sha256") == plan["server_manifest_sha256"], "server source changed")
    require(
        backend.get("source_sha256") == SOURCE_SHA256 and backend.get("version") == VERSION, "unreviewed engine source"
    )
    wire, child = client["request_body"], backend["chat_body"]
    messages = [{"role": "user", "content": plan["prompt"]}]
    require(wire.get("model") == plan["model"], "SDK model changed")
    require(child.get("model") == backend["served_model_name"], "child model changed")
    require(wire.get("messages") == child.get("messages") == messages, "prompt changed")
    for name, expected in plan["sampling"].items():
        require(wire.get(name) == child.get(name) == expected, f"wire sampling changed: {name}")
    for body in (wire, child):
        require(
            body.get("stream") is True and body.get("stream_options") == {"include_usage": True},
            "stream contract changed",
        )
        require(body.get("chat_template_kwargs", {}).get("enable_thinking") is False, "thinking was not disabled")
    tokenized = backend["tokenized"]
    ids = tokenized["input_ids"]
    require(
        isinstance(ids, list) and len(ids) == prompt_tokens and all(count(i) for i in ids),
        "prepared prompt tokens differ",
    )
    require(tokenized["has_multimodal_inputs"] is False, "unexpected multimodal input")
    require(
        {tokenized["rid"]} == {row["event"]["meta_info"]["id"] for row in backend["events"]},
        "prepared request id differs",
    )
    effective = tokenized["sampling"]
    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "sampling_seed": 17,
        "n": 1,
        "min_new_tokens": plan["completion_tokens"],
        "max_new_tokens": plan["completion_tokens"],
    }
    require(all(effective.get(k) == v for k, v in expected.items()), "effective sampling changed")
    require(
        all(
            effective.get(k) is None
            for k in ("json_schema", "regex", "ebnf", "structural_tag", "custom_params", "logit_bias")
        ),
        "unexpected constrained sampling",
    )
    # IDs vary per request; model, full prepared IDs and sampling must not.
    return digest(
        {
            "model_path": backend["model_path"],
            "served_model_name": backend["served_model_name"],
            "input_ids": ids,
            "sampling": effective,
            "adapted_sampling": backend["adapted_sampling"],
            "chat_body": child,
        }
    )


def analyze(client_dir: Path, server_dir: Path, prompt_tokens: int, output_tokens: int) -> dict[str, Any]:
    plan = json.loads((client_dir / "plan.json").read_text())
    manifest = json.loads((server_dir / "server.json").read_text())
    require(digest(manifest) == plan["server_manifest_sha256"], "server manifest changed")
    require(plan["cohort"] == manifest["cohort"], "server cohort differs")
    require(plan["prompt_sha256"] == hashlib.sha256(plan["prompt"].encode()).hexdigest(), "prompt hash differs")
    require(plan["completion_tokens"] == output_tokens, "output length differs from client plan")
    controller = json.loads((client_dir / "controller.json").read_text())
    require(controller.get("completed") is True, "controller incomplete")
    children = controller["children"]
    require(
        len(children) == plan["warmups"] + plan["repeats"]
        and all(
            row["index"] == i and row["returncode"] == 0 and row["timed_out"] is False and not row["signals"]
            for i, row in enumerate(children)
        ),
        "failed or missing child",
    )
    clients = [json.loads(p.read_text()) for p in sorted(client_dir.glob("trial-*.json"))]
    require(len(clients) == plan["warmups"] + plan["repeats"], "missing or extra client trials")
    servers = [json.loads(p.read_text()) for p in sorted(server_dir.glob("stream-*.json"))]
    installed = {row["pid"]: row for row in (json.loads(p.read_text()) for p in server_dir.glob("installed-*.json"))}
    matched = set()
    signatures = set()
    rows = []
    for index, client in enumerate(clients):
        require(client["index"] == index and client["warmup"] == (index < plan["warmups"]), "trial order changed")
        if index:
            require(clients[index - 1]["finished_ns"] <= client["started_ns"], "client trials overlap")
        ids = {row["chunk"].get("id") for row in client["chunks"]}
        require(len(ids) == 1 and None not in ids, "missing or mixed SDK ids")
        candidates = [s for s in servers if {row["event"].get("meta_info", {}).get("id") for row in s["events"]} == ids]
        require(len(candidates) == 1, "client interval does not bind exactly one backend stream")
        server = candidates[0]
        require(server["id"] not in matched, "backend stream reused")
        matched.add(server["id"])
        marker = installed.get(server["pid"], {})
        require(
            marker.get("server_manifest_sha256") == plan["server_manifest_sha256"]
            and marker.get("source_sha256") == SOURCE_SHA256
            and marker.get("source_files") == SOURCE_FILES
            and marker.get("version") == VERSION,
            "matching observer installation absent",
        )
        signatures.add(bindings(client, server, plan, prompt_tokens))
        rows.append(
            {
                "index": index,
                "warmup": client["warmup"],
                "backend_id": server["id"],
                **analyze_trial(client, server, prompt_tokens, output_tokens),
            }
        )
    measured = [r for r in rows if not r["warmup"]]
    require(bool(measured), "no measured trials")
    require(len(signatures) == 1, "prepared input/model/sampling changed between trials")

    def overlaps(record: dict[str, Any]) -> bool:
        start, end = record["started_ns"], record["finished_ns"]
        require(
            type(start) is int and (end is None or (type(end) is int and end >= start)),
            "invalid background interval",
        )
        require(record["clock_domain"] == plan["clock_domain"], "background clock domain differs")
        return start <= clients[-1]["finished_ns"] and (end is None or end >= clients[0]["started_ns"])

    inside_window = [s for s in servers if overlaps(s)]
    require(len(inside_window) == len(matched), "unexpected backend stream during measured schedule")
    background = [json.loads(p.read_text()) for p in sorted(server_dir.glob("background-*.json"))]
    require(not any(overlaps(row) for row in background), "background generation overlaps measured schedule")
    return {
        "valid": True,
        "trials": rows,
        "measured_trials": len(measured),
        "client_ttft_median_s": statistics.median(r["client_ttft_s"] for r in measured),
        "after_first_observation_median_tokens_per_s": statistics.median(
            r["backend_after_first_observation_tokens_per_s"] for r in measured
        ),
        "exact_first_token_observed_in_all_trials": all(r["exact_first_token_observed"] for r in rows),
        "unmatched_backend_streams": len(servers) - len(matched),
        "background_calls_outside_schedule": len(background),
        "prepared_request_sha256": next(iter(signatures)),
        "server_manifest_sha256": digest(manifest),
        "scope": "Instrumented local SIE SDK chat streaming; cumulative token counts and host timestamps observed at SGLang TokenizerManager before chat formatting. TTFT is first visible SDK text. No one-token-per-SSE assumption, tokenization estimate, GPU kernel-time claim or speedup inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-directory", type=Path, required=True)
    parser.add_argument("--server-directory", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--completion-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, analyze(args.client_directory, args.server_directory, args.prompt_tokens, args.completion_tokens))


if __name__ == "__main__":
    main()
