"""Resumable question-only acquisition; frozen bodies, explicit physical attempts."""

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx

from collect import ARMS, body_for, openai_cost, parse_answer, reservation

MAX_ATTEMPTS = 3
WORKERS_PER_ARM = 16
TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
TRANSIENT_ERRORS = {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError", "WriteError", "RemoteProtocolError"}
FATAL_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached", "invalid_api_key", "account_deactivated"}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(encoded(value))
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


def retry_after(value, now=None):
    if not isinstance(value, str) or len(value) > 128:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = when.timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if 0 <= seconds < float("inf") else None


def numeric_usage(value):
    if isinstance(value, dict):
        return all(isinstance(key, str) and numeric_usage(item) for key, item in value.items())
    return type(value) is int and value >= 0


def acquire(item, arm, client_factory=None):
    """Equivalent request body/endpoint; retain Retry-After and readable responses."""
    body = body_for(item, arm)
    row = {"id": item["id"], "arm": arm, "model": body["model"],
           "body_sha256": digest(json.dumps(body, sort_keys=True).encode()), "created_unix": time.time(),
           "terminal_response": False, "execution_unknown": True}
    started = time.monotonic()
    try:
        factory = client_factory or (lambda: httpx.Client(timeout=httpx.Timeout(600, connect=30),
            follow_redirects=False, transport=httpx.HTTPTransport(retries=0), trust_env=False))
        with factory() as client:
            response = client.post("https://api.openai.com/v1/responses", json=body,
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
        row["http_status"] = response.status_code
        row["retry_after_seconds"] = retry_after(response.headers.get("retry-after"))
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            row.update(status="provider_error", error_code=safe_code(error.get("code")), error_type=safe_code(error.get("type")))
            return row
        # Any HTTP200 response is retained. Parsing/format/truncation never triggers another answer.
        row["terminal_response"] = True
        require(isinstance(payload, dict), "Invalid provider JSON object")
        row.update(provider_status=payload.get("status"), resolved_model=payload.get("model"), service_tier=payload.get("service_tier"))
        messages = [{"type": part["type"], **({"text": part.get("text")} if part["type"] == "output_text" else {"refusal": part.get("refusal")})}
                    for output in payload.get("output", []) if output.get("type") == "message"
                    for part in output.get("content", []) if part.get("type") in {"output_text", "refusal"}]
        row["messages"] = messages
        usage = payload.get("usage")
        require(isinstance(usage, dict) and numeric_usage(usage), "Usage must be a readable numeric object")
        row["usage"] = usage
        row["cost"] = openai_cost(usage, body)
        row["execution_unknown"] = False
        details = payload.get("incomplete_details") or {}
        row["incomplete_reason"] = safe_code(details.get("reason")) if isinstance(details, dict) else None
        text = "".join(part.get("text") or "" for part in messages if part["type"] == "output_text")
        row["answer"] = parse_answer(text, item["response_contract"]) if row["provider_status"] == "completed" else None
        row.update(status="complete", valid_answer=row["answer"] is not None)
        if Decimal(row["cost"]["budget_usd"]) > reservation(body, arm):
            row["status"] = "charge_exceeds_reservation"
    except Exception as error:
        row.update(status="receipt_incomplete" if row["terminal_response"] else "failed_or_unknown", error_type=type(error).__name__)
    finally:
        row["elapsed_seconds"] = time.monotonic() - started
    return row


def decision(row, attempt):
    if row.get("http_status") in {401, 403} or row.get("error_code") in FATAL_CODES or row.get("error_type") in FATAL_CODES:
        return "fatal"
    if row.get("status") == "charge_exceeds_reservation" or row.get("binding_valid") is False:
        return "fatal"
    if row.get("terminal_response"):
        if row.get("resolved_model") != row["model"] or row.get("service_tier") != "default":
            return "fatal"
        return "terminal"
    transient = row.get("http_status") in TRANSIENT_HTTP or row.get("error_type") in TRANSIENT_ERRORS or row.get("status") == "interrupted_unknown"
    if not transient:
        return "incomplete"
    if attempt >= MAX_ATTEMPTS:
        return "exhausted"
    if (row.get("retry_after_seconds") or 0) > 60:
        return "defer_server_delay"
    return "retry"


def retry_delay(row, attempt, identity):
    base = (5, 15)[min(attempt - 1, 1)]
    jitter = int(digest(f"{identity}:{attempt}".encode())[:4], 16) / 65535
    return max(base, row.get("retry_after_seconds") or 0) + jitter


def fits_budget(known, unknown, pending, reserve, cap):
    return known + unknown + pending + reserve <= cap


def run(items_by_split, root, plan, commit, *, call=acquire, dispatch_seconds=20900, emit=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    plan_bytes = encoded(plan)
    if plan_path.exists():
        require(plan_path.read_bytes() == plan_bytes, "Cannot change a resumed collection plan")
    else:
        plan_path.write_bytes(plan_bytes)
    plan_sha = digest(plan_bytes)
    states, items, queues = {}, {}, {arm: deque() for arm in ("luna", "sol")}
    retries = {arm: deque() for arm in queues}
    known, unknown, known_list = Decimal(0), Decimal(0), Decimal(0)
    fatal, pause_until = None, 0.0
    for split, split_items in items_by_split.items():
        for item in split_items:
            for arm in queues:
                key = (split, arm, item["id"])
                items[key] = item
                body = body_for(item, arm)
                body_sha = digest(json.dumps(body, sort_keys=True).encode())
                directory = root / split / arm / "attempts" / digest(item["id"].encode())
                state = {"id": item["id"], "arm": arm, "split": split, "attempts": [], "selected_attempt": None,
                         "status": "pending", "not_before_unix": 0.0}
                states[key] = state
                paths = sorted(directory.glob("*-intent.json")) if directory.exists() else []
                require(len(paths) <= MAX_ATTEMPTS, "Attempt cap violated")
                for number, intent_path in enumerate(paths, 1):
                    require(intent_path.name == f"{number:03d}-intent.json", "Attempt sequence changed")
                    intent = json.loads(intent_path.read_bytes())
                    expected = {"id": item["id"], "arm": arm, "split": split, "attempt": number,
                                "body_sha256": body_sha, "plan_sha256": plan_sha}
                    require(all(intent.get(k) == v for k, v in expected.items()), "Prior intent binding changed")
                    reserve = Decimal(intent["reserved_usd"])
                    require(reserve == reservation(body, arm), "Prior reservation changed")
                    response_path = directory / f"{number:03d}-response.json"
                    row = json.loads(response_path.read_bytes()) if response_path.exists() else {
                        "id": item["id"], "arm": arm, "model": body["model"], "body_sha256": body_sha,
                        "status": "interrupted_unknown", "terminal_response": False,
                        "created_unix": intent["created_unix"], "error_type": "ProcessInterrupted"}
                    if response_path.exists():
                        require(row.get("intent_sha256") == file_hash(intent_path)
                                and all(row.get(k) == v for k, v in {"id": item["id"], "arm": arm, "model": body["model"], "body_sha256": body_sha}.items()),
                                "Prior response binding changed")
                    amount = row.get("cost", {}).get("budget_usd")
                    if amount is None:
                        unknown += reserve
                    else:
                        amount = Decimal(amount)
                        require(amount.is_finite() and amount >= 0, "Invalid prior charge")
                        known += amount
                        known_list += Decimal(row["cost"]["usd_estimate"])
                    action = decision(row, number)
                    attempt = {"attempt": number, "intent_path": intent_path.relative_to(root).as_posix(), "intent_sha256": file_hash(intent_path),
                               "record_path": response_path.relative_to(root).as_posix() if response_path.exists() else None,
                               "record_sha256": file_hash(response_path) if response_path.exists() else None, "status": row["status"],
                               "action": action, "known_list_usd": row.get("cost", {}).get("usd_estimate"),
                               "known_budget_usd": None if amount is None else str(amount), "unknown_reservation_usd": str(reserve) if amount is None else "0"}
                    state["attempts"].append(attempt)
                    if action == "terminal":
                        require(state["selected_attempt"] is None, "A terminal response was followed by an unauthorized retry")
                        state.update(selected_attempt=number, status="complete" if row.get("status") == "complete" else "incomplete_receipt")
                    elif state["selected_attempt"] is not None:
                        raise ValueError("Attempt after terminal provider response")
                    elif action == "fatal":
                        fatal = "preserved_fatal_provider_or_identity_error"
                    else:
                        state["status"] = action
                        delay = retry_delay(row, number, item["id"]) if action == "retry" else (row.get("retry_after_seconds") or 0)
                        finished = row.get("created_unix", intent["created_unix"]) + (row.get("elapsed_seconds") or 0)
                        if row["status"] == "interrupted_unknown":
                            finished = time.time()
                        state["not_before_unix"] = max(state["not_before_unix"], finished + delay)
                if state["selected_attempt"] is None and (not state["attempts"] or state["attempts"][-1]["action"] in {"retry", "defer_server_delay"}):
                    (retries if state["attempts"] else queues)[arm].append(key)
    checkpoint_count, checkpoint_seconds, dirty_responses = 0, 0.0, 0
    last_checkpoint = time.monotonic()

    def checkpoint():
        nonlocal checkpoint_count, checkpoint_seconds, dirty_responses, last_checkpoint
        checkpoint_start = time.monotonic()
        commit()
        checkpoint_seconds += time.monotonic() - checkpoint_start
        checkpoint_count += 1
        dirty_responses = 0
        last_checkpoint = time.monotonic()

    checkpoint()
    cap = Decimal(plan["budget_usd"])
    require(known + unknown <= cap, "Preserved acquisition already exceeds its cap")
    pending = {}
    physical_count = sum(len(state["attempts"]) for state in states.values())
    selected_count = sum(state["selected_attempt"] is not None for state in states.values())
    selected_by_split = Counter(state["split"] for state in states.values() if state["selected_attempt"] is not None)
    selections, written_splits = {}, set()

    def write_split(split):
        split_items = items_by_split[split]
        for arm in queues:
            rows = [states[(split, arm, item["id"])] for item in split_items]
            path = root / split / arm / "selection.json"
            dump(path, {"format": "routerbench-openai-attempt-selection-v1", "split": split, "arm": arm,
                        "model": ARMS[arm]["model"], "plan_sha256": plan_sha, "ordered_ids": [item["id"] for item in split_items],
                        "entries": rows, "first_terminal_response_only": True, "no_quality_retries": True})
            selections[f"{split}/{arm}"] = {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path),
                "expected": len(rows), "selected": sum(row["selected_attempt"] is not None for row in rows),
                "complete_receipts": sum(row["status"] == "complete" for row in rows)}
        split_selections = {arm: selections[f"{split}/{arm}"] for arm in queues}
        dump(root / split / "collection.json", {"format": "routerbench-openai-split-collection-v1", "split": split,
            "status": "complete" if all(row["complete_receipts"] == len(split_items) for row in split_selections.values()) else "incomplete",
            "expected_pairs": len(split_items), "plan_sha256": plan_sha, "selections": split_selections,
            "pair_spec_sha256": plan.get("pair_spec_sha256"), "acquisition_protocol_sha256": plan_sha,
            "source_plan_sha256": plan.get("source_plan_sha256"), "correctness_inspected": False})
        written_splits.add(split)

    start, last_progress, last_refill = time.monotonic(), 0, 0.0
    stop = fatal
    with ThreadPoolExecutor(max_workers=WORKERS_PER_ARM * 2) as pool:
        while any(queues.values()) or any(retries.values()) or pending:
            if time.monotonic() - start >= dispatch_seconds:
                stop = stop or "dispatch_deadline"
            candidates = []
            reserved_pending = sum((value[2] for value in pending.values()), Decimal(0))
            counts = Counter(key[1] for key, _, _ in pending.values())
            vacancies = WORKERS_PER_ARM * 2 - len(pending)
            refill_ready = not pending or vacancies >= 8 or time.monotonic() - last_refill >= 0.5
            if stop is None and time.time() >= pause_until and refill_ready:
                for arm, queue in queues.items():
                    candidates_for_arm = retries[arm]
                    examined = len(candidates_for_arm)
                    retry_keys = []
                    while candidates_for_arm and examined:
                        key = candidates_for_arm.popleft()
                        examined -= 1
                        if time.time() < states[key]["not_before_unix"]:
                            candidates_for_arm.append(key)
                        else:
                            retry_keys.append(key)
                    queue.extendleft(reversed(retry_keys))
                    while queue and counts[arm] < WORKERS_PER_ARM:
                        key = queue.popleft()
                        state, item = states[key], items[key]
                        if time.time() < state["not_before_unix"]:
                            retries[arm].append(key)
                            continue
                        body = body_for(item, arm)
                        reserve = reservation(body, arm)
                        if not fits_budget(known, unknown, reserved_pending, reserve, cap):
                            queue.appendleft(key)
                            break
                        number = len(state["attempts"]) + 1
                        require(number <= MAX_ATTEMPTS and state["selected_attempt"] is None, "Ineligible new attempt")
                        directory = root / key[0] / arm / "attempts" / digest(item["id"].encode())
                        intent_path = directory / f"{number:03d}-intent.json"
                        response_path = directory / f"{number:03d}-response.json"
                        require(not intent_path.exists() and not response_path.exists(), "Never overwrite a physical attempt")
                        dump(intent_path, {"id": item["id"], "arm": arm, "split": key[0], "attempt": number,
                             "body_sha256": digest(json.dumps(body, sort_keys=True).encode()), "plan_sha256": plan_sha,
                             "created_unix": time.time(), "reserved_usd": str(reserve)})
                        candidates.append((key, (intent_path, response_path, number), reserve))
                        reserved_pending += reserve
                        counts[arm] += 1
            if candidates:
                checkpoint()  # Every intent is durable before its physical request starts.
                last_refill = time.monotonic()
                for key, paths, reserve in candidates:
                    pending[pool.submit(call, items[key], key[1])] = (key, paths, reserve)
            if not pending:
                if stop or not (any(queues.values()) or any(retries.values())):
                    break
                remaining = [key for queue in list(queues.values()) + list(retries.values()) for key in queue]
                if not any(fits_budget(known, unknown, Decimal(0), reservation(body_for(items[key], key[1]), key[1]), cap) for key in remaining):
                    stop = "budget_bound"
                    break
                time.sleep(min(1, max(0.01, pause_until-time.time())))
                continue
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if not done:
                if dirty_responses and time.monotonic() - last_checkpoint >= 2:
                    checkpoint()
                continue
            for future in done:
                key, (intent_path, response_path, number), reserve = pending.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    body = body_for(items[key], key[1])
                    row = {"id": key[2], "arm": key[1], "model": body["model"], "body_sha256": digest(json.dumps(body, sort_keys=True).encode()),
                           "created_unix": time.time(), "status": "failed_or_unknown", "terminal_response": False, "error_type": type(error).__name__}
                expected_body = body_for(items[key], key[1])
                row["binding_valid"] = all(row.get(k) == v for k, v in {
                    "id": key[2], "arm": key[1], "model": expected_body["model"],
                    "body_sha256": digest(json.dumps(expected_body, sort_keys=True).encode())}.items())
                row["intent_sha256"] = file_hash(intent_path)
                row["attempt"] = number
                dump(response_path, row)
                dirty_responses += 1
                amount = row.get("cost", {}).get("budget_usd")
                if amount is None:
                    unknown += reserve
                else:
                    amount = Decimal(amount)
                    require(amount.is_finite() and amount >= 0, "Invalid charge")
                    known += amount
                    known_list += Decimal(row["cost"]["usd_estimate"])
                    if amount > reserve:
                        stop = "charge_exceeds_reservation"
                action = decision(row, number)
                state = states[key]
                state["attempts"].append({"attempt": number, "intent_path": intent_path.relative_to(root).as_posix(),
                    "intent_sha256": file_hash(intent_path), "record_path": response_path.relative_to(root).as_posix(),
                    "record_sha256": file_hash(response_path), "status": row["status"], "action": action,
                    "known_list_usd": row.get("cost", {}).get("usd_estimate"), "known_budget_usd": None if amount is None else str(amount),
                    "unknown_reservation_usd": str(reserve) if amount is None else "0"})
                state["status"] = action
                physical_count += 1
                if action == "terminal":
                    state.update(selected_attempt=number, status="complete" if row.get("status") == "complete" else "incomplete_receipt")
                    selected_count += 1
                    selected_by_split[key[0]] += 1
                elif action == "fatal":
                    stop = "provider_auth_quota_or_identity_error"
                elif action == "defer_server_delay":
                    state["not_before_unix"] = time.time() + row["retry_after_seconds"]
                    stop = "server_delay_exceeds_supported_cap"
                elif action == "retry":
                    state["not_before_unix"] = time.time() + retry_delay(row, number, key[2])
                    retries[key[1]].append(key)
                    if row.get("http_status") in {429, 503}:
                        pause_until = max(pause_until, state["not_before_unix"])
            if dirty_responses >= 16 or time.monotonic() - last_checkpoint >= 2 or not pending:
                checkpoint()
            for split, split_items in items_by_split.items():
                if split not in written_splits and selected_by_split[split] == len(split_items) * 2:
                    write_split(split)
                    checkpoint()
                    if emit:
                        print(json.dumps({"split_ready": split, "pairs": len(split_items),
                                          "collection_sha256": file_hash(root / split / "collection.json")}), flush=True)
            if emit and physical_count-last_progress >= 32:
                last_progress = physical_count
                progress = {"physical_attempts": physical_count, "selected_responses": selected_count,
                            "expected_responses": len(states), "known_list_usd": str(known_list), "known_budget_usd": str(known),
                            "unknown_reservation_usd": str(unknown), "inflight": len(pending), "elapsed_seconds": time.monotonic()-start,
                            "checkpoints": checkpoint_count, "checkpoint_seconds": checkpoint_seconds, "stop": stop}
                dump(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    for split, split_items in items_by_split.items():
        if split not in written_splits:
            for row in (states[(split, arm, item["id"])] for arm in queues for item in split_items):
                if row["selected_attempt"] is None and row["status"] in {"pending", "retry"}:
                    row["status"] = "incomplete_" + (stop or "unavailable")
            write_split(split)
    summary = {"format": "routerbench-openai-population-collection-v1", "status": "complete" if all(state["status"] == "complete" for state in states.values()) else "incomplete",
               "expected_responses": len(states), "selected_responses": sum(state["selected_attempt"] is not None for state in states.values()),
               "physical_attempts": sum(len(state["attempts"]) for state in states.values()), "known_list_usd": str(known_list),
               "known_budget_usd": str(known), "unknown_reservation_usd": str(unknown), "stop": stop,
               "selections": selections, "plan_sha256": plan_sha, "elapsed_seconds": time.monotonic()-start,
               "final_dispatched": "final" in items_by_split, "correctness_inspected": False,
               "checkpoint_metrics_before_terminal_commit": {"count": checkpoint_count, "seconds": checkpoint_seconds},
               "checkpoint_policy": {"intent_before_dispatch": True, "refill_coalescing_seconds": 0.5,
                    "refill_immediate_vacancies": 8, "completion_count": 16, "completion_seconds": 2}}
    dump(root / "collection.json", summary)
    checkpoint()
    return summary
