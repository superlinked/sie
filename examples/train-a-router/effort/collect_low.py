"""Collect Luna-low on the published final questions without repeating uncertain calls.

Run from the example root with ``python -m effort.collect_low --help``.
Only --execute makes requests; an unchanged output directory is resumable.
"""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time

import httpx

import collect
import evidence
import openai_population as original
from reader_io import exclusive_run, read, require, sha, sync_files

ROOT = Path(collect.__file__).resolve().parent
MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/responses"
RELEASE_REVISION = "86dd7494a4c2396bc9ffa890f994b9722610b357"
MANIFEST_SHA256 = "4fd8a49d760384ad745c1bd49b3ee61cb58fd8d98c71cef6552ff6dbfda66b55"
INPUT_MEMBER = "source/question-only/final-inputs.json"
SOURCE_MEMBER = "source/source-plan.json"
PROTOCOL = {
    "model": MODEL, "reasoning_effort": "low", "max_output_tokens": 16384,
    "service_tier": "default", "store": False, "endpoint": ENDPOINT,
    "workers": 16, "max_attempts": 3, "request_timeout_seconds": 600,
    "connect_timeout_seconds": 30, "backoff_seconds": [5, 15], "retry_after_cap_seconds": 60,
    "first_http_200_only": True, "quality_retries": False, "uncertain_execution_retries": False,
    "reference_answers_on_input": False, "original_medium_or_sol_requests": False,
    "timing": "Per-attempt wall time including client creation, HTTP and parsing; excluding checkpointing/backoff",
}
PRE_EXECUTION_ERRORS = {"ConnectTimeout", "ConnectError", "PoolTimeout"}


def source_hashes():
    return {"effort/collect_low.py": sha(__file__), "collect.py": sha(collect.__file__),
            "openai_population.py": sha(original.__file__), "reader_io.py": sha(ROOT / "reader_io.py"),
            "evidence.py": sha(evidence.__file__)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def body_for(item):
    body = collect.body_for(item, "luna")
    require(body["model"] == MODEL and body["reasoning"] == {"effort": "medium"}
            and body["max_output_tokens"] == 16384 and body["service_tier"] == "default",
            "The original answer-model protocol changed")
    return {**body, "reasoning": {"effort": "low"}}


def prepare(inputs_path, manifest_path, source_plan_path, *, release_path=None, pair_path=None,
            budget_usd="20.00", deadline_seconds=5400):
    """Verify only public question inputs and metadata, never references or prior answers."""
    release_path, pair_path = Path(release_path or ROOT / "release.json"), Path(pair_path or ROOT / "pair.json")
    release, pair, source = read(release_path), read(pair_path), read(source_plan_path)
    require(release["format"] == "train-a-router-release-v1"
            and release["manifest_sha256"] == MANIFEST_SHA256
            and release["manifest_url"] == f"https://huggingface.co/datasets/superlinked/train-a-router-bbeh/resolve/{RELEASE_REVISION}/manifest.json",
            "Use the original immutable public BBEH release")
    manifest = evidence.checked_manifest(Path(manifest_path).read_bytes(), MANIFEST_SHA256)
    entries = {row["path"]: row for row in manifest["files"]}
    for member, path in ((INPUT_MEMBER, inputs_path), (SOURCE_MEMBER, source_plan_path)):
        require(member in entries and sha(path) == entries[member]["sha256"]
                and Path(path).stat().st_size == entries[member]["size"], "Published input/source binding changed")
    require(sha(inputs_path) == source["files_sha256"]["question-only/final-inputs.json"], "Final inputs changed")
    require(pair["models"] == {"cheap": MODEL, "strong": "gpt-6.1-sol"}
            and pair["settings"] == collect.OPENAI_SETTINGS and pair["system_prompt"] == collect.SYSTEM
            and pair["prices_usd_per_million"] == collect.PRICES, "Original pair recipe changed")
    items = read(inputs_path)
    require(isinstance(items, list) and len(items) == 975, "Use all 975 published final questions")
    ids = [item["id"] for item in items]
    require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids), "Invalid/duplicate ID")
    bodies = {item["id"]: digest(body_for(item)) for item in items}
    cap = Decimal(str(budget_usd))
    require(cap.is_finite() and 0 < cap <= 20, "Budget must be positive and no more than USD20")
    require(type(deadline_seconds) is int and 600 < deadline_seconds <= 5400, "Deadline must be601..5400seconds")
    plan = {
        "format": "bbeh-luna-low-acquisition-v1", "role": "final", "n": len(items),
        "protocol": PROTOCOL, "ordered_ids": ids, "inputs_sha256": sha(inputs_path),
        "inputs_canonical_sha256": digest(items), "request_body_sha256": bodies,
        "original_medium_body_sha256": {item["id"]: digest(collect.body_for(item, "luna")) for item in items},
        "release_sha256": sha(release_path), "release": release, "manifest_sha256": sha(manifest_path),
        "source_plan_sha256": sha(source_plan_path), "pair_sha256": sha(pair_path),
        "prices_usd_per_million": collect.PRICES[MODEL], "budget_usd": str(cap),
        "deadline_seconds": deadline_seconds, "deadline_scope": "Wall clock from first execute, including resumes",
        "only_request_change": {"reasoning.effort": {"from": "medium", "to": "low"}},
        "source_files_sha256": source_hashes(),
        "no_quality_selection": True, "new_routing_fit": False,
    }
    return items, plan


def acquire(item, client_factory=None):
    """One physical request, readable response fields only; no automatic HTTP retry."""
    body = body_for(item)
    row = {"id": item["id"], "arm": "luna", "model": MODEL, "reasoning_effort": "low",
           "body_sha256": digest(body), "created_unix": time.time(), "terminal_response": False,
           "execution_unknown": True, "request_not_sent": False}
    started = time.monotonic()
    try:
        factory = client_factory or (lambda: httpx.Client(timeout=httpx.Timeout(600, connect=30),
            follow_redirects=False, transport=httpx.HTTPTransport(retries=0), trust_env=False))
        with factory() as client:
            response = client.post(ENDPOINT, json=body, headers={
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
        row.update(http_status=response.status_code,
                   retry_after_seconds=original.retry_after(response.headers.get("retry-after")))
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            row.update(status="provider_error", execution_unknown=False,
                       error_code=original.safe_code(error.get("code")), error_type=original.safe_code(error.get("type")))
            return row
        row["terminal_response"] = True
        require(isinstance(payload, dict), "Invalid provider JSON object")
        reasoning = payload.get("reasoning")
        row.update(provider_status=payload.get("status"), resolved_model=payload.get("model"),
                   service_tier=payload.get("service_tier"),
                   resolved_reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None)
        messages = []
        for output in payload.get("output", []):
            if output.get("type") == "message":
                for part in output.get("content", []):
                    if part.get("type") in {"output_text", "refusal"}:
                        key = "text" if part["type"] == "output_text" else "refusal"
                        require(isinstance(part.get(key), str), "Invalid readable output content")
                        messages.append({"type": part["type"], key: part[key]})
        row["messages"] = messages
        usage = payload.get("usage")
        require(isinstance(usage, dict) and original.numeric_usage(usage), "Usage must be a readable numeric object")
        row["usage"] = usage
        row["cost"] = collect.openai_cost(usage, body)
        row["execution_unknown"] = False
        details = payload.get("incomplete_details")
        row["incomplete_reason"] = original.safe_code(details.get("reason")) if isinstance(details, dict) else None
        text = "".join(part["text"] for part in messages if part["type"] == "output_text")
        row["answer"] = collect.parse_answer(text, item["response_contract"]) if row["provider_status"] == "completed" else None
        row.update(status="complete", valid_answer=row["answer"] is not None)
        if Decimal(row["cost"]["budget_usd"]) > collect.reservation(body, "luna"):
            row["status"] = "charge_exceeds_reservation"
    except Exception as error:
        row.update(status="receipt_incomplete" if row["terminal_response"] else "failed_or_unknown",
                   error_type=type(error).__name__)
        if type(error).__name__ in PRE_EXECUTION_ERRORS:
            row.update(request_not_sent=True, execution_unknown=False)
    finally:
        row["elapsed_seconds"] = time.monotonic() - started
    return row


def decision(row, attempt):
    if (row.get("http_status") in {401, 403} or row.get("error_code") in original.FATAL_CODES
            or row.get("error_type") in original.FATAL_CODES or row.get("binding_valid") is False
            or row.get("status") == "charge_exceeds_reservation"):
        return "fatal"
    if row.get("terminal_response"):
        if (row.get("resolved_model") != MODEL or row.get("service_tier") != "default"
                or row.get("resolved_reasoning_effort") not in (None, "low")):
            return "fatal"
        return "terminal"
    if row.get("execution_unknown", True):
        return "uncertain"
    transient = (row.get("http_status") in original.TRANSIENT_HTTP
                 or row.get("request_not_sent") is True and row.get("error_type") in PRE_EXECUTION_ERRORS)
    if not transient:
        return "incomplete"
    if attempt >= PROTOCOL["max_attempts"]:
        return "exhausted"
    if (row.get("retry_after_seconds") or 0) > PROTOCOL["retry_after_cap_seconds"]:
        return "deferred"
    return "retry"


def accounting(row, reserve, body):
    """Retain a bound for every unpriced physical attempt; never call missing usage free."""
    if row.get("cost") is not None:
        expected = collect.openai_cost(row["usage"], body)
        require(row["cost"] == expected, "Stored usage/cost mismatch")
        return Decimal(expected["usd_estimate"]), Decimal(expected["budget_usd"]), Decimal(0)
    if row.get("request_not_sent") is True:
        require(row.get("error_type") in PRE_EXECUTION_ERRORS and not row.get("terminal_response")
                and not row.get("http_status"), "Cannot classify a dispatched request as free")
        return Decimal(0), Decimal(0), Decimal(0)
    return Decimal(0), Decimal(0), reserve


def run(items, output, plan, commit=lambda: None, *, call=acquire, emit=True):
    """Rolling16 requests; callback persists Volume writes before dispatch and after receipts."""
    with exclusive_run(output):
        return _run(items, Path(output).resolve(), plan, commit, call, emit)


def _run(items, root, plan, commit, call, emit):
    require(plan["format"] == "bbeh-luna-low-acquisition-v1" and plan["protocol"] == PROTOCOL,
            "Changed low-effort protocol")
    require(plan["ordered_ids"] == [row["id"] for row in items] and plan["n"] == len(items)
            and len({row["id"] for row in items}) == len(items)
            and plan["inputs_canonical_sha256"] == digest(items), "Changed fixed population")
    require(plan["source_files_sha256"] == source_hashes(), "Collector source changed")
    cap = Decimal(plan["budget_usd"])
    require(cap.is_finite() and 0 < cap <= 20 and 600 < plan["deadline_seconds"] <= 5400, "Invalid budget/deadline")
    bodies = {row["id"]: body_for(row) for row in items}
    require(plan["request_body_sha256"] == {key: digest(body) for key, body in bodies.items()}, "Request bodies changed")
    dirty = set()

    def write(path, value, *, immutable=False):
        path = root / path
        require(not path.is_symlink() and path.resolve().is_relative_to(root), "Output path escapes collection")
        if immutable and path.exists():
            require(path.read_bytes() == original.encoded(value), "Cannot change prior immutable evidence")
            return
        original.dump(path, value)
        dirty.add(path)

    def checkpoint():
        if dirty:
            sync_files(dirty, root)
            commit()
            dirty.clear()

    if not (root / "plan.json").exists():
        require(not any(path.name != ".collector.lock" for path in root.iterdir()), "Use an empty directory or exact prior plan")
    write("plan.json", plan, immutable=True)
    write("inputs.json", items, immutable=True)
    plan_hash = sha(root / "plan.json")
    start_path = root / "execution.json"
    if start_path.exists():
        execution = read(start_path)
        require(execution["plan_sha256"] == plan_hash, "Execution plan changed")
    else:
        execution = {"plan_sha256": plan_hash, "started_unix": time.time(), "deadline_seconds": plan["deadline_seconds"]}
        write("execution.json", execution, immutable=True)
    require(execution["deadline_seconds"] == plan["deadline_seconds"], "Execution deadline changed")
    checkpoint()
    if (root / "selection.json").exists():
        previous = read(root / "selection.json")
        require(previous["plan_sha256"] == plan_hash and previous["ordered_ids"] == plan["ordered_ids"], "Prior selection changed")
        for state in previous["entries"]:
            for entry in state["attempts"]:
                for field in ("intent", "record"):
                    if entry[field + "_path"] is not None:
                        path = root / entry[field + "_path"]
                        require(path.resolve().is_relative_to(root) and not path.is_symlink()
                                and sha(path) == entry[field + "_sha256"], "Prior attempt bytes changed")
    states, item_map, queue = {}, {row["id"]: row for row in items}, deque()
    known_list, known_budget, unknown = Decimal(0), Decimal(0), Decimal(0)
    expected_files, stop = set(), None

    def record(state, intent_path, response_path, row, number, reserve):
        nonlocal known_list, known_budget, unknown, stop
        body = bodies[state["id"]]
        matched = all(row.get(k) == value for k, value in {
            "id": state["id"], "arm": "luna", "model": MODEL, "reasoning_effort": "low", "body_sha256": digest(body)}.items())
        require(row.get("binding_valid", matched) == matched, "Receipt binding flag changed")
        amount, budget, bound = accounting(row, reserve, body)
        known_list += amount
        known_budget += budget
        unknown += bound
        action = decision({**row, "binding_valid": matched}, number)
        if budget > reserve:
            action = "fatal"
        entry = {"attempt": number, "intent_path": intent_path.relative_to(root).as_posix(),
                 "intent_sha256": sha(intent_path), "record_path": response_path.relative_to(root).as_posix() if response_path.exists() else None,
                 "record_sha256": sha(response_path) if response_path.exists() else None,
                 "status": row["status"], "action": action, "known_list_usd": str(amount) if not bound else None,
                 "known_budget_usd": str(budget) if not bound else None, "unknown_reservation_usd": str(bound)}
        state["attempts"].append(entry)
        if row.get("terminal_response"):
            state["selected_attempt"] = number
        state["status"] = "complete" if action == "terminal" and row["status"] == "complete" else action
        if action == "fatal":
            stop = "provider_auth_quota_or_identity_error"
        elif action == "deferred":
            stop = "retry_after_exceeds_supported_cap"
        if action == "retry":
            delay = max(PROTOCOL["backoff_seconds"][number - 1], row.get("retry_after_seconds") or 0)
            state["not_before_unix"] = row["created_unix"] + row["elapsed_seconds"] + delay
        return action

    for item in items:
        identity = item["id"]
        state = {"id": identity, "status": "pending", "selected_attempt": None, "attempts": [], "not_before_unix": 0}
        states[identity] = state
        directory = root / "attempts" / hashlib.sha256(identity.encode()).hexdigest()
        intents = sorted(directory.glob("*-intent.json"))
        require(len(intents) <= PROTOCOL["max_attempts"], "Physical attempt cap exceeded")
        for number, path in enumerate(intents, 1):
            require(path.name == f"{number:03d}-intent.json" and not path.is_symlink(), "Invalid attempt sequence")
            require(not state["attempts"] or state["attempts"][-1]["action"] == "retry", "Unauthorized repeat after terminal/uncertain attempt")
            intent = read(path)
            reserve = collect.reservation(bodies[identity], "luna")
            require(all(intent.get(k) == v for k, v in {
                "id": identity, "arm": "luna", "attempt": number, "model": MODEL, "reasoning_effort": "low",
                "body_sha256": digest(bodies[identity]), "plan_sha256": plan_hash, "reserved_usd": str(reserve)}.items()), "Prior intent changed")
            response = path.with_name(f"{number:03d}-response.json")
            expected_files.add(path.relative_to(root).as_posix())
            if response.exists():
                require(not response.is_symlink(), "Response must be a regular file")
                expected_files.add(response.relative_to(root).as_posix())
                row = read(response)
                require(row.get("intent_sha256") == sha(path) and row.get("attempt") == number, "Response intent changed")
            else:
                row = {"id": identity, "arm": "luna", "model": MODEL, "reasoning_effort": "low",
                       "body_sha256": digest(bodies[identity]), "status": "interrupted_unknown", "execution_unknown": True,
                       "terminal_response": False, "created_unix": intent["created_unix"], "elapsed_seconds": None}
            action = record(state, path, response, row, number, reserve)
        if not intents or action == "retry":
            queue.append(identity)
    actual_files = {path.relative_to(root).as_posix() for path in (root / "attempts").rglob("*") if path.is_file()}
    require(actual_files == expected_files, "Unaccounted attempt/response files")
    require(known_budget + unknown <= cap, "Preserved cost exposure exceeds budget")
    pending, completed_since_progress = {}, 0
    pause_until = 0
    dispatch_deadline = execution["started_unix"] + plan["deadline_seconds"] - PROTOCOL["request_timeout_seconds"]
    with ThreadPoolExecutor(max_workers=PROTOCOL["workers"]) as pool:
        while queue or pending:
            if time.time() >= dispatch_deadline:
                stop = stop or "dispatch_deadline"
            candidates = []
            reserved = sum((entry[3] for entry in pending.values()), Decimal(0))
            for _ in range(len(queue)):
                if stop or time.time() < pause_until or len(pending) + len(candidates) >= PROTOCOL["workers"]:
                    break
                identity = queue.popleft()
                state = states[identity]
                reserve = collect.reservation(bodies[identity], "luna")
                if time.time() < state["not_before_unix"] or known_budget + unknown + reserved + reserve > cap:
                    queue.append(identity)
                    continue
                number = len(state["attempts"]) + 1
                directory = Path("attempts") / hashlib.sha256(identity.encode()).hexdigest()
                path, response = directory / f"{number:03d}-intent.json", directory / f"{number:03d}-response.json"
                require(number <= PROTOCOL["max_attempts"] and state["selected_attempt"] is None
                        and not (root / path).exists() and not (root / response).exists(), "Never overwrite/repeat a physical request")
                write(path, {"id": identity, "arm": "luna", "model": MODEL, "reasoning_effort": "low",
                    "attempt": number, "body_sha256": digest(bodies[identity]), "plan_sha256": plan_hash,
                    "reserved_usd": str(reserve), "created_unix": time.time()}, immutable=True)
                candidates.append((identity, path, response, reserve, number))
                reserved += reserve
            if candidates:
                checkpoint()  # Bounded group of durable intents, before any request in the group.
                for identity, path, response, reserve, number in candidates:
                    pending[pool.submit(call, item_map[identity])] = (identity, path, response, reserve, number)
            if not pending:
                if stop or not queue:
                    break
                if not any(known_budget + unknown + collect.reservation(bodies[key], "luna") <= cap for key in queue):
                    stop = "budget_bound"
                    break
                time.sleep(0.1)
                continue
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                identity, path, response, reserve, number = pending.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    row = {"id": identity, "arm": "luna", "model": MODEL, "reasoning_effort": "low",
                        "body_sha256": digest(bodies[identity]), "status": "failed_or_unknown", "execution_unknown": True,
                        "terminal_response": False, "error_type": type(error).__name__, "created_unix": time.time(), "elapsed_seconds": None}
                row.update(intent_sha256=sha(root / path), attempt=number)
                write(response, row, immutable=True)
                action = record(states[identity], root / path, root / response, row, number, reserve)
                if action == "retry":
                    queue.append(identity)
                    if row.get("http_status") in {429, 503}:
                        pause_until = max(pause_until, states[identity]["not_before_unix"])
                completed_since_progress += 1
            if done:
                checkpoint()
            if emit and completed_since_progress >= 32:
                completed_since_progress = 0
                progress = {"selected": sum(state["selected_attempt"] is not None for state in states.values()),
                    "expected": len(items), "known_list_usd": str(known_list), "unknown_reservation_usd": str(unknown),
                    "inflight": len(pending), "stop": stop}
                write("progress.json", progress)
                checkpoint()
                print(json.dumps(progress), flush=True)
    for state in states.values():
        if state["status"] in {"pending", "retry"}:
            state["status"] = "incomplete_" + (stop or "unavailable")
    selection = {"format": "bbeh-luna-low-selection-v1", "role": "final", "arm": "luna", "model": MODEL,
        "reasoning_effort": "low", "plan_sha256": plan_hash, "ordered_ids": plan["ordered_ids"],
        "entries": list(states.values()), "first_terminal_response_only": True, "no_quality_retries": True}
    write("selection.json", selection)
    complete = sum(state["status"] == "complete" for state in states.values())
    summary = {"format": "bbeh-luna-low-collection-v1", "status": "complete" if complete == len(items) and stop is None else "incomplete",
        "expected_responses": len(items), "complete_receipts": complete,
        "selected_responses": sum(state["selected_attempt"] is not None for state in states.values()),
        "physical_attempts": sum(len(state["attempts"]) for state in states.values()),
        "status_counts": dict(Counter(state["status"] for state in states.values())),
        "known_list_usd": str(known_list), "known_budget_usd": str(known_budget), "unknown_reservation_usd": str(unknown),
        "cost_upper_bound_usd": str(known_budget + unknown), "budget_usd": plan["budget_usd"],
        "stop": stop, "plan_sha256": plan_hash, "selection_sha256": sha(root / "selection.json"),
        "elapsed_seconds": time.time() - execution["started_unix"], "correctness_inspected": False}
    write("collection.json", summary)
    checkpoint()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", default="20.00")
    parser.add_argument("--deadline-seconds", type=int, default=5400)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    items, plan = prepare(args.inputs, args.manifest, args.source_plan, release_path=args.release,
        pair_path=args.pair, budget_usd=args.budget_usd, deadline_seconds=args.deadline_seconds)
    if args.execute:
        require(bool(os.environ.get("OPENAI_API_KEY")), "Set OPENAI_API_KEY before --execute")
        result = run(items, args.output, plan)
    else:
        with exclusive_run(args.output):
            path = args.output / "plan.json"
            if path.exists():
                require(path.read_bytes() == original.encoded(plan), "Cannot replace an existing plan")
            else:
                require(not any(p.name != ".collector.lock" for p in args.output.iterdir()), "Use an empty plan directory")
                original.dump(path, plan)
                sync_files([path], args.output)
        result = {"prepared_only": True, "n": len(items), "plan_sha256": sha(path), "paid_calls": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
