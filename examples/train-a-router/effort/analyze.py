"""Post-publication Luna-low comparison and unchanged medium-trained route replay.

From the example root: uv sync --frozen --group collection --group outcomes
Then: uv run --frozen python -m effort.analyze --data DATA --collection RUN --output REPORT
This command has no network, model-fitting, threshold-selection or inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

import collect
import evidence
import grade_v2
import normalize
from effort import collect_low
from evaluation import frontier, utility_router

ROOT = Path(__file__).resolve().parents[1]
N = 975
RELEASE_REVISION = "86dd7494a4c2396bc9ffa890f994b9722610b357"
MANIFEST_SHA256 = "4fd8a49d760384ad745c1bd49b3ee61cb58fd8d98c71cef6552ff6dbfda66b55"
OFFICIAL_SHA256 = "4b4f06e5babb015de2ba639bae995a5526182ffb5b5890af54dfcf038580eb34"
METHODS = (
    "base", "lora", "knn", "embedding", "lora__textcost", "embedding__textcost",
    "text_length__textcost", "text_topic__textcost",
)
ARMS = ("luna_low", "luna_medium", "sol_medium")
PROTOCOL = {
    "study": "Post-publication comparison on all 975 existing final questions; not a new blind test.",
    "change": "New gpt-6-luna low reasoning; saved Luna-medium and Sol-medium answers are reused.",
    "grading": "Unchanged source-only normalize.py followed by grade_v2.decimal_match and pinned BBEH scorer.",
    "selection": "No fitting or threshold selection. Exact saved Fraction rules and input scores are replayed.",
    "policies": "Original README eight 80%-calibration-answer-cost-cap rules plus embedding__textcost at 60%.",
    "transfer": "Routers and expected-cost estimates were trained for medium effort, not low effort.",
    "denominator": "All eligible IDs, including invalid, truncated and unavailable answers; none dropped.",
    "bootstrap": "2000 paired whole-family draws within each task, seed 89; fixed observed task weights.",
    "uncertainty": "Pointwise intervals conditional on these tasks; no simultaneous or unseen-task guarantee.",
    "latency": "Selected physical attempt wall time, not a concurrently measured routed service or TTFT.",
    "confounding": "Low and medium were collected on different days; latency differences also reflect service/load changes.",
    "cost": "Token-accounted standard list-price estimate, not invoice; reasoning is included in output tokens.",
    "failures": "Unavailable answers count operationally incorrect; unknown usage/cost stays unknown, never zero-imputed.",
    "router_cost": "Saved singleton encoder fees unchanged; native and small CPU hosting remain unpriced here.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return evidence.sha(path)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def bound(root, name, expected):
    path = evidence.artifact_path(root, name)
    require(path.is_file() and sha(path) == expected, f"Changed or missing artifact: {name}")
    return path


def checked_data(directory):
    """Verify just the needed immutable public files; no weight or feature loading."""
    root = Path(directory)
    manifest = evidence.checked_manifest((root / "manifest.json").read_bytes(), MANIFEST_SHA256)
    entries = {item["path"]: item for item in manifest["files"]}
    names = (
        "source/source-plan.json", "source/meta.json", "source/gold.json",
        "source/question-only/final-inputs.json", "source/provenance/bbeh/evaluate.py", "pair.json",
        "grades/grades-final.json", "reference/frozen-frontiers.json", "reference/final-report.json",
        "metadata/presentation-policies.json",
    )
    hashes = {}
    for name in names:
        entry = entries[name]
        path = bound(root, name, entry["sha256"])
        require(path.stat().st_size == entry["size"], f"Published size changed: {name}")
        hashes[name] = entry["sha256"]
    source, pair = read(root / names[0]), read(root / "pair.json")
    inputs = read(root / "source/question-only/final-inputs.json")
    meta, gold = read(root / "source/meta.json"), read(root / "source/gold.json")
    by_meta, by_gold = ({row["id"]: row for row in records} for records in (meta, gold))
    require(len(by_meta) == len(meta) and len(by_gold) == len(gold), "Duplicate source/reference ID")
    for name in ("meta.json", "gold.json", "question-only/final-inputs.json"):
        require(source["files_sha256"][name] == hashes["source/" + name], "Source file binding changed")
    ids = [row["id"] for row in inputs]
    require(len(ids) == N and len(set(ids)) == N, "Use the complete ordered 975-question population")
    rows = [by_meta[identity] for identity in ids]
    require({r["id"] for r in meta if r["retained"] and r["role"] == "final"} == set(ids),
            "Final source population changed")
    for item, row in zip(inputs, rows, strict=True):
        require(row["retained"] is True and row["role"] == "final"
                and row["request_sha256"] == hashlib.sha256(item["request"].encode()).hexdigest(),
                "Inactive, cross-role or changed question")
        collect.body_for(item, "luna")  # Validate the original exact three-field input contract.
        require(isinstance(by_gold[item["id"]]["target"], str), "Non-string reference")
    grades = read(root / "grades/grades-final.json")
    frozen, report = (read(root / f"reference/{name}.json") for name in ("frozen-frontiers", "final-report"))
    choices = read(root / "metadata/presentation-policies.json")
    source_sha, pair_sha = hashes["source/source-plan.json"], hashes["pair.json"]
    require(pair["models"] == {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"}
            and pair["settings"] == collect.OPENAI_SETTINGS and pair["system_prompt"] == collect.SYSTEM
            and pair["prices_usd_per_million"] == collect.PRICES, "Original answer recipe changed")
    require(grades["format"] == "bbeh-paired-grades-v1" and grades["role"] == "final"
            and grades["n"] == N and [r["id"] for r in grades["rows"]] == ids
            and grades["models"] == pair["models"] and grades["source_plan_sha256"] == source_sha
            and grades["pair_sha256"] == pair_sha and grades["grader_sha256"] == sha(grade_v2.__file__)
            and grades["normalizer_sha256"] == sha(normalize.__file__)
            and grades["official_grader_sha256"] == OFFICIAL_SHA256
            and grades["normalization_version"] == normalize.VERSION, "Original grading lineage changed")
    require(frozen["format"] == "bbeh-frozen-frontiers-v1" and frozen["ids"] == ids
            and frozen["frozen_before_final_outcomes"] is True and frozen["final_outcomes_accessed"] is False
            and frozen["source_plan_sha256"] == source_sha and frozen["pair_sha256"] == pair_sha
            and frozen["plan_sha256"] == entries["reference/embedding/plan.json"]["sha256"]
            and frozen["implementation_sha256"]["utility"] == sha(utility_router.__file__),
            "Original routing lineage changed")
    require(report["frozen_sha256"] == hashes["reference/frozen-frontiers.json"]
            and report["grades_sha256"] == hashes["grades/grades-final.json"]
            and report["n"] == N, "Original report binding changed")
    require(choices["format"] == "bbeh-presentation-policies-v1" and choices["final_answers_collected"] is False
            and choices["frozen_sha256"] == report["frozen_sha256"]
            and choices["calibration_sha256"] == frozen["calibration_sha256"], "Presentation rules changed")
    for row, old, saved in zip(rows, grades["rows"], frozen["source_rows"], strict=True):
        require(all(old[key] == saved[key] == row[key] for key in ("id", "task", "family", "role")),
                "Source/family identity changed")
    official = grade_v2.load_official(root / "source/provenance/bbeh/evaluate.py", OFFICIAL_SHA256)
    return {"inputs": inputs, "rows": rows, "gold": by_gold, "grades": grades, "frozen": frozen,
            "report": report, "choices": choices, "hashes": hashes, "official": official, "pair": pair}


def checked_plan(root, original):
    plan = read(root / "plan.json")
    inputs, hashes = original["inputs"], original["hashes"]
    require(plan["format"] == "bbeh-luna-low-acquisition-v1" and plan["protocol"] == collect_low.PROTOCOL
            and plan["role"] == "final" and plan["n"] == len(inputs)
            and plan["ordered_ids"] == [r["id"] for r in inputs]
            and plan["inputs_sha256"] == hashes["source/question-only/final-inputs.json"]
            and plan["inputs_canonical_sha256"] == collect_low.digest(inputs)
            and plan["manifest_sha256"] == MANIFEST_SHA256
            and plan["source_plan_sha256"] == hashes["source/source-plan.json"]
            and plan["pair_sha256"] == hashes["pair.json"], "Low acquisition population/lineage changed")
    release = read(ROOT / "release.json")
    require(plan["release"] == release and plan["release_sha256"] == sha(ROOT / "release.json")
            and release["manifest_sha256"] == MANIFEST_SHA256
            and f"/{RELEASE_REVISION}/" in release["manifest_url"], "Public release changed")
    require(plan["prices_usd_per_million"] == collect.PRICES["gpt-6-luna"]
            and plan["only_request_change"] == {"reasoning.effort": {"from": "medium", "to": "low"}}
            and plan["no_quality_selection"] is True and plan["new_routing_fit"] is False,
            "Low-effort comparison specification changed")
    sources = ("effort/collect_low.py", "collect.py", "openai_population.py", "reader_io.py", "evidence.py")
    require(set(plan["source_files_sha256"]) == set(sources), "Unknown collector source closure")
    for name in sources:
        require(plan["source_files_sha256"][name] == sha(ROOT / name), f"Collector source changed: {name}")
    require(read(root / "inputs.json") == inputs, "Collected question bytes changed")
    require(plan["request_body_sha256"] == {r["id"]: collect_low.digest(collect_low.body_for(r)) for r in inputs}
            and plan["original_medium_body_sha256"] == {
                r["id"]: collect_low.digest(collect.body_for(r, "luna")) for r in inputs}, "Request body changed")
    return plan


def checked_attempts(root, plan, inputs):
    """Reconcile the complete physical ledger, including attempts without usage."""
    root = Path(root)
    summary, selection = read(root / "collection.json"), read(root / "selection.json")
    plan_sha, ids = sha(root / "plan.json"), [r["id"] for r in inputs]
    require(summary["format"] == "bbeh-luna-low-collection-v1"
            and summary["plan_sha256"] == plan_sha and summary["selection_sha256"] == sha(root / "selection.json"),
            "Collection/selection binding changed")
    require(selection["format"] == "bbeh-luna-low-selection-v1" and selection["plan_sha256"] == plan_sha
            and selection["ordered_ids"] == ids and [r["id"] for r in selection["entries"]] == ids
            and selection["role"] == "final" and selection["arm"] == "luna"
            and selection["model"] == "gpt-6-luna" and selection["reasoning_effort"] == "low"
            and selection["first_terminal_response_only"] is True and selection["no_quality_retries"] is True,
            "Selection contract changed")
    chosen, ledger, declared = {}, [], set()
    totals = [Decimal(0), Decimal(0), Decimal(0)]
    for item, state in zip(inputs, selection["entries"], strict=True):
        body, first, previous = collect_low.body_for(item), None, None
        reserve = collect.reservation(body, "luna")
        require(len(state["attempts"]) <= plan["protocol"]["max_attempts"], "Physical attempt cap exceeded")
        for number, attempt in enumerate(state["attempts"], 1):
            require(first is None and (previous is None or previous == "retry"),
                    "Terminal or uncertain response was retried")
            prefix = f"attempts/{hashlib.sha256(item['id'].encode()).hexdigest()}/{number:03d}"
            require(attempt["attempt"] == number and attempt["intent_path"] == prefix + "-intent.json",
                    "Attempt ID/order changed")
            intent = read(bound(root, attempt["intent_path"], attempt["intent_sha256"]))
            declared.add(attempt["intent_path"])
            identity = {"id": item["id"], "arm": "luna", "model": "gpt-6-luna", "reasoning_effort": "low",
                        "body_sha256": collect_low.digest(body)}
            require(all(intent.get(k) == v for k, v in {
                **identity, "attempt": number, "plan_sha256": plan_sha, "reserved_usd": str(reserve),
            }.items()), "Intent/source binding changed")
            if attempt["record_path"] is None:
                require(attempt["record_sha256"] is None and not (root / (prefix + "-response.json")).exists(),
                        "Undeclared response")
                response = {**identity, "status": "interrupted_unknown", "execution_unknown": True,
                            "terminal_response": False, "created_unix": intent["created_unix"],
                            "elapsed_seconds": None}
            else:
                require(attempt["record_path"] == prefix + "-response.json", "Response ID/path changed")
                response = read(bound(root, attempt["record_path"], attempt["record_sha256"]))
                declared.add(attempt["record_path"])
                require(all(response.get(k) == v for k, v in {
                    **identity, "attempt": number, "intent_sha256": attempt["intent_sha256"],
                }.items()) and response.get("binding_valid", True) is True, "Response identity changed")
            elapsed = response.get("elapsed_seconds")
            require(elapsed is None or type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                    "Invalid physical elapsed time")
            amount, budget, unknown = collect_low.accounting(response, reserve, body)
            previous = collect_low.decision(response, number)
            if budget > reserve:
                previous = "fatal"
            require(attempt["action"] == previous and attempt["status"] == response["status"],
                    "Physical attempt decision changed")
            expected = (str(amount) if not unknown else None, str(budget) if not unknown else None, str(unknown))
            require(tuple(attempt[k] for k in ("known_list_usd", "known_budget_usd", "unknown_reservation_usd"))
                    == expected, "Physical accounting changed")
            totals = [a + b for a, b in zip(totals, (amount, budget, unknown), strict=True)]
            ledger.append({"id": item["id"], **attempt, "elapsed_seconds": elapsed,
                           "http_status": response.get("http_status"), "error_type": response.get("error_type"),
                           "error_code": response.get("error_code"), "request_not_sent": response.get("request_not_sent", False)})
            if response.get("terminal_response"):
                require(response.get("http_status") == 200 and previous == "terminal",
                        "Provider/model/effort/tier mismatch: cannot compare this response")
                first = (number, response, attempt)
        require(state["selected_attempt"] == (first[0] if first else None), "Not the first terminal response")
        if first:
            expected_state = "complete" if first[1]["status"] == "complete" else "terminal"
            require(state["status"] == expected_state, "Selected receipt status changed")
        chosen[item["id"]] = {"response": first[1] if first else None,
                              "attempt": first[2] if first else None, "state": state,
                              "unknown_bound_usd": str(sum((Decimal(a["unknown_reservation_usd"])
                                                            for a in state["attempts"]), Decimal(0)))}
    actual = {p.relative_to(root).as_posix() for p in (root / "attempts").rglob("*") if p.is_file()}
    require(actual == declared, "Missing or undeclared physical attempt files")
    states = selection["entries"]
    require(summary["expected_responses"] == len(ids) and summary["physical_attempts"] == len(ledger)
            and summary["selected_responses"] == sum(x["selected_attempt"] is not None for x in states)
            and summary["complete_receipts"] == sum(x["status"] == "complete" for x in states)
            and summary["status_counts"] == dict(Counter(x["status"] for x in states)), "Coverage totals changed")
    for key, value in zip(("known_list_usd", "known_budget_usd", "unknown_reservation_usd"), totals, strict=True):
        require(Decimal(summary[key]) == value, "Acquisition accounting totals changed")
    require(Decimal(summary["cost_upper_bound_usd"]) == totals[1] + totals[2]
            and totals[1] + totals[2] <= Decimal(plan["budget_usd"]), "Acquisition budget changed")
    require(summary["status"] == ("complete" if summary["complete_receipts"] == len(ids)
                                  and summary["stop"] is None else "incomplete"), "Completion status changed")
    return chosen, ledger, summary


def grade_low(item, meta, target, selected, official):
    response = selected["response"] or {"status": selected["state"]["status"]}
    text = "".join(p.get("text") or "" for p in response.get("messages", []) if p["type"] == "output_text")
    answer = collect.parse_answer(text, item["response_contract"]) if response.get("provider_status") == "completed" else None
    if response.get("status") == "complete":
        require(response.get("answer") == answer and response.get("valid_answer") == (answer is not None),
                "Saved final-answer parse changed")
    valid = response.get("status") == "complete" and answer is not None
    canonical, rule = normalize.canonicalize(answer, item["request"], meta["task"])
    cost = response.get("cost")
    amount = cost["usd_estimate"] if cost else None
    if amount is not None:
        upper = amount
    elif selected["state"]["attempts"]:
        upper = selected["unknown_bound_usd"]
    else:
        upper = None  # A never-dispatched answer has no observed deployment-cost bound.
    return {
        "correct": int(valid and canonical is not None and grade_v2.decimal_match(official, canonical, target)),
        "official_raw_correct": int(valid and official.evaluate_correctness(answer, target)),
        "literal_correct": int(valid and answer.strip() == target.strip()),
        "normalized_answer": canonical, "normalization_rule": rule, "valid": valid,
        "status": response["status"], "provider_status": response.get("provider_status"),
        "truncated": response.get("provider_status") == "incomplete", "usage": response.get("usage"),
        "cost_usd": amount, "cost_upper_bound_usd": upper, "elapsed_seconds": response.get("elapsed_seconds"),
        "record_sha256": selected["attempt"]["record_sha256"] if selected["attempt"] else None,
        "available_answer": selected["response"] is not None and response.get("status") == "complete",
    }


def original_arm(row, prefix, item):
    values = {key.removeprefix(prefix + "_"): value for key, value in row.items() if key.startswith(prefix + "_")}
    for name in ("correct", "official_raw_correct", "literal_correct"):
        require(type(values[name]) is int and values[name] in (0, 1), "Nonbinary original correctness")
    body = collect.body_for(item, "luna" if prefix == "cheap" else "sol")
    cost = collect.openai_cost(values["usage"], body)
    require(Decimal(str(values["cost_usd"])) == Decimal(cost["usd_estimate"]), "Original token cost mismatch")
    return {**values, "cost_usd": cost["usd_estimate"], "cost_upper_bound_usd": cost["usd_estimate"],
            "available_answer": values["status"] == "complete"}


def numbers(values):
    known = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) and v >= 0 for v in known), "Invalid nonnegative measurement")
    return {"observed_n": len(known), "missing_n": len(values) - len(known),
            "mean": float(np.mean(known)) if known else None,
            "p50": float(np.quantile(known, .5)) if known else None,
            "p95": float(np.quantile(known, .95)) if known else None,
            "known_sum": float(sum(known))}


def usage_values(arms, name):
    paths = {
        "input_tokens": ("input_tokens",), "output_tokens": ("output_tokens",),
        "reasoning_tokens": ("output_tokens_details", "reasoning_tokens"),
        "cached_input_tokens": ("input_tokens_details", "cached_tokens"),
        "cache_write_tokens": ("input_tokens_details", "cache_write_tokens"),
    }
    result = []
    for arm in arms:
        usage = arm.get("usage")
        value = usage
        for key in paths[name]:
            value = value.get(key) if isinstance(value, dict) else None
        require(value is None or type(value) is int and value >= 0, "Invalid token measurement")
        result.append(value)
    return result


def summarize(arms, bootstrap=None):
    n = len(arms)
    correct = np.array([r["correct"] for r in arms], dtype=float)
    costs = [r["cost_usd"] for r in arms]
    known = sum((Decimal(str(x)) for x in costs if x is not None), Decimal(0))
    bounded = all(r.get("cost_upper_bound_usd") is not None for r in arms)
    upper = sum((Decimal(str(r["cost_upper_bound_usd"])) for r in arms), Decimal(0)) if bounded else None
    full = all(x is not None for x in costs)
    result = {
        "n": n, "correct": int(correct.sum()), "accuracy_percent": 100 * float(correct.mean()),
        "official_raw_accuracy_percent": 100 * sum(r["official_raw_correct"] for r in arms) / n,
        "literal_accuracy_percent": 100 * sum(r["literal_correct"] for r in arms) / n,
        "unavailable_n": sum(not r["available_answer"] for r in arms),
        "invalid_n": sum(not r["valid"] for r in arms), "invalid_percent": 100 * sum(not r["valid"] for r in arms) / n,
        "truncated_n": sum(r["truncated"] for r in arms),
        "truncated_percent": 100 * sum(r["truncated"] for r in arms) / n,
        "status_counts": dict(Counter(r["status"] for r in arms)),
        "provider_status_counts": dict(Counter(str(r["provider_status"]) for r in arms)),
        "cost_status": "exact_selected_usage" if full else "incomplete_selected_usage",
        "answer_usd_per_10000": float(known * 10000 / n) if full else None,
        "known_selected_cost_usd": str(known), "cost_missing_n": sum(x is None for x in costs),
        "selected_cost_bounds_usd": {"lower": str(known), "upper": str(upper) if upper is not None else None},
        "selected_elapsed_seconds": numbers([r.get("elapsed_seconds") for r in arms]),
        "tokens": {name: numbers(usage_values(arms, name)) for name in (
            "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens", "cache_write_tokens")},
    }
    if bootstrap is not None:
        result["ci95"] = {"accuracy_percent": frontier.interval(100 * bootstrap.means(correct[:, None])[:, 0])}
        for key, values, scale in (
            ("answer_usd_per_10000", costs, 10000),
            ("selected_elapsed_mean_seconds", [r.get("elapsed_seconds") for r in arms], 1),
            ("output_tokens_mean", usage_values(arms, "output_tokens"), 1),
            ("reasoning_tokens_mean", usage_values(arms, "reasoning_tokens"), 1),
        ):
            result["ci95"][key] = mean_interval(bootstrap, values, scale)
    return result


def mean_interval(bootstrap, values, scale=1):
    if any(v is None for v in values):
        return None
    array = np.asarray(values, dtype=float)
    return frontier.interval(scale * bootstrap.means(array[:, None])[:, 0])


def compare(first, second, bootstrap):
    """Paired differences use the same complete population and bootstrap draw."""
    y0, y1 = (np.array([r["correct"] for r in rows], dtype=float) for rows in (first, second))
    result = {
        "n": len(first), "both_correct": int(np.sum((y0 == 1) & (y1 == 1))),
        "both_wrong": int(np.sum((y0 == 0) & (y1 == 0))),
        "first_only_correct": int(np.sum((y0 == 1) & (y1 == 0))),
        "second_only_correct": int(np.sum((y0 == 0) & (y1 == 1))),
        "accuracy_difference_pp": 100 * float(np.mean(y0 - y1)),
        "accuracy_difference_pp_ci95": mean_interval(bootstrap, y0 - y1, 100),
        "direction": "first minus second; negative cost/time difference means first is lower/faster",
    }
    for name, field, scale in (("answer_usd_per_10000", "cost_usd", 10000),
                               ("selected_elapsed_mean_seconds", "elapsed_seconds", 1)):
        a, b = ([r.get(field) for r in rows] for rows in (first, second))
        if any(v is None for v in a + b):
            result[name + "_difference"] = result[name + "_difference_ci95"] = None
            result[name + "_reduction_percent"] = result[name + "_reduction_percent_ci95"] = None
        else:
            matrix = np.column_stack((np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
            means = bootstrap.means(matrix)
            result[name + "_difference"] = scale * float(np.mean(matrix[:, 0] - matrix[:, 1]))
            result[name + "_difference_ci95"] = frontier.interval(scale * (means[:, 0] - means[:, 1]))
            result[name + "_reduction_percent"] = 100 * (1 - float(matrix[:, 0].mean() / matrix[:, 1].mean())) \
                if matrix[:, 1].mean() else None
            with np.errstate(divide="ignore", invalid="ignore"):
                result[name + "_reduction_percent_ci95"] = frontier.interval(100 * (1 - means[:, 0] / means[:, 1]))
    return result


def route_metrics(arms, routes, fees):
    value = summarize(arms)
    value["strong_share"] = float(np.mean(routes))
    value["router_usd_per_10000"] = None if fees is None else 10000 * float(np.mean(fees))
    value["total_usd_per_10000"] = None if fees is None or value["answer_usd_per_10000"] is None else (
        value["answer_usd_per_10000"] + value["router_usd_per_10000"])
    return value


def replay_policies(original, rows, bootstrap):
    frozen, choices, published = original["frozen"], original["choices"], original["report"]
    definitions = [(name, "0.8") for name in METHODS] + [("embedding__textcost", "0.6")]
    result = []
    for method, cap in definitions:
        selected, saved = choices["choices"][method][cap], frozen["methods"][method]
        candidate = next(r for r in saved["candidates"] if r["candidate_id"] == selected["candidate_id"])
        require(candidate["policy"] == selected["policy"] and candidate["calibration"] == selected["calibration"]
                and candidate["candidate_id"] in saved["calibration_frontier_candidate_ids"],
                "Presentation candidate differs from saved calibration rule")
        scores, rule = saved["scores"], candidate["policy"]
        require(scores["metadata_assisted"] is False, "Metadata-only policy is not a README deployable rule")
        routes = utility_router.replay(rule, scores["probabilities"], scores["expected_usd"])
        require(len(routes) == len(rows), "Route population changed")
        original_arms = [row["sol_medium" if use_sol else "luna_medium"] for row, use_sol in zip(rows, routes, strict=True)]
        changed_arms = [row["sol_medium" if use_sol else "luna_low"] for row, use_sol in zip(rows, routes, strict=True)]
        fees = frozen["encoder_fees_usd"] if scores["router_kind"] == "embedding" else None
        if rule["mode"].startswith("always_"):
            fees = [0.] * len(rows)
        old, new = (route_metrics(a, routes, fees) for a in (original_arms, changed_arms))
        for value, arm_rows in ((old, original_arms), (new, changed_arms)):
            correct = np.array([r["correct"] for r in arm_rows], dtype=float)
            sol_correct = np.array([r["sol_medium"]["correct"] for r in rows], dtype=float)
            value["quality_gap_vs_sol_pp"] = 100 * float(np.mean(sol_correct - correct))
            value["accuracy_percent_ci95"] = mean_interval(bootstrap, correct, 100)
            value["quality_gap_vs_sol_pp_ci95"] = mean_interval(bootstrap, sol_correct - correct, 100)
        prior = next(r for r in published["methods"][method]["points"] if r["candidate_id"] == candidate["candidate_id"])
        require(prior["rule"] == rule, "Published policy rule differs")
        errors = {}
        for key, current in (("answer_accuracy_percent", old["accuracy_percent"]),
                             ("answer_usd_per_10000", old["answer_usd_per_10000"]),
                             ("strong_share", old["strong_share"]),
                             ("router_usd_per_10000", old["router_usd_per_10000"]),
                             ("total_usd_per_10000", old["total_usd_per_10000"])):
            expected = prior[key]
            require((current is None) == (expected is None), "Original priced/unpriced scope changed")
            errors[key] = None if current is None else abs(current - expected)
            require(current is None or math.isclose(current, expected, rel_tol=0, abs_tol=1e-10),
                    f"Original replay parity failed: {method}/{key}")
        comparison = compare(changed_arms, original_arms, bootstrap)
        result.append({"method": method, "calibration_answer_cap_fraction": float(cap),
                       "candidate_id": candidate["candidate_id"], "rule": rule,
                       "routes_sha256": collect_low.digest(routes.astype(int).tolist()),
                       "strong_n": int(routes.sum()), "cheap_n": int((~routes).sum()),
                       "original_medium": old, "substituted_low": new,
                       "low_minus_medium": comparison, "original_replay_absolute_errors": errors,
                       "router_kind": scores["router_kind"], "same_routes_and_router_overhead": True,
                       "latency_excludes_router_and_retries": True})
    return result


def markdown(report):
    def fmt(value, digits=3):
        return "unknown" if value is None else f"{value:.{digits}f}"

    lines = ["# Luna reasoning-effort comparison", "", PROTOCOL["study"], "", PROTOCOL["confounding"], "",
             (f"Collection status: **{report['new_acquisition']['status']}**; "
             f"{report['new_acquisition']['selected_responses']}/{report['n']} selected responses; "
             f"{report['arms']['luna_low']['unavailable_n']} unavailable answers counted operationally incorrect."), "",
             "Accuracy uses all questions. Costs are selected-answer list-price estimates; times are selected-attempt wall time.",
             "", "| Answer model | Accuracy % | USD / 10k | Mean s | p50 s | p95 s | Invalid / truncated |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["arms"].items():
        timing = value["selected_elapsed_seconds"]
        lines.append(f"| {name} | {fmt(value['accuracy_percent'])} | {fmt(value['answer_usd_per_10000'])} | "
                     f"{fmt(timing['mean'])} | {fmt(timing['p50'])} | {fmt(timing['p95'])} | "
                     f"{value['invalid_n']} / {value['truncated_n']} |")
    delta = report["comparisons"]["luna_low_vs_luna_medium"]
    ci = delta["accuracy_difference_pp_ci95"]
    lines += ["", (f"Luna low minus medium accuracy: {fmt(delta['accuracy_difference_pp'])} pp "
              f"(paired 95% interval {fmt(ci['lower'])} to {fmt(ci['upper'])})."),
              "", "## Fixed-route transfer", "", PROTOCOL["transfer"], "",
              ("The original calibration rules and all route masks stay unchanged. Native/CPU hosting is unpriced; "
              "embedding fees stay unchanged. No operating point is selected on these new answers."), "",
              "| Method / calibration cap | Medium → low accuracy % | Medium → low answer USD / 10k | Medium → low mean s | Sol share % |",
              "|---|---:|---:|---:|---:|"]
    for point in report["policies"]:
        a, b = point["original_medium"], point["substituted_low"]
        lines.append(f"| {point['method']} / {point['calibration_answer_cap_fraction']:.0%} | "
                     f"{fmt(a['accuracy_percent'])} → {fmt(b['accuracy_percent'])} | "
                     f"{fmt(a['answer_usd_per_10000'])} → {fmt(b['answer_usd_per_10000'])} | "
                     f"{fmt(a['selected_elapsed_seconds']['mean'])} → {fmt(b['selected_elapsed_seconds']['mean'])} | "
                     f"{100 * a['strong_share']:.2f} |")
    ledger = report["new_acquisition"]
    lines += ["", (f"Acquisition: {ledger['physical_attempts']} physical attempts; known usage-priced subtotal "
              f"${ledger['known_list_usd']}; additional unknown-attempt reservation bound "
              f"${ledger['unknown_reservation_usd']}. This is separate from selected-answer deployment estimates."),
              "", PROTOCOL["uncertainty"], "", ("Medians and p95s are descriptive. Missing values and their coverage "
              "are explicit in report.json; observed-only timing summaries do not remove rows from accuracy."),
              "", "No new model fitting, threshold calibration, or fresh blind-test claim is made.", ""]
    return "\n".join(lines)


def analyze(data, collection, output, *, draws=2000):
    require(type(draws) is int and draws > 0, "Invalid bootstrap draw count")
    output = Path(output)
    require(not output.exists(), "Preserve previous reports; use a new output directory")
    original = checked_data(data)
    root = Path(collection)
    plan = checked_plan(root, original)
    selected, ledger, acquisition = checked_attempts(root, plan, original["inputs"])
    rows = []
    for item, meta, old in zip(original["inputs"], original["rows"], original["grades"]["rows"], strict=True):
        row = {key: meta[key] for key in ("id", "task", "family", "role")}
        row["luna_low"] = grade_low(item, meta, original["gold"][item["id"]]["target"],
                                    selected[item["id"]], original["official"])
        row["luna_medium"], row["sol_medium"] = original_arm(old, "cheap", item), original_arm(old, "strong", item)
        rows.append(row)
    bootstrap = frontier.Bootstrap(rows, draws)
    arms = {name: [row[name] for row in rows] for name in ARMS}
    by_task = {}
    for task in sorted({r["task"] for r in rows}):
        part = [r for r in rows if r["task"] == task]
        task_bootstrap = frontier.Bootstrap(part, draws)
        by_task[task] = {"n": len(part), "families": len({r["family"] for r in part}),
                         "arms": {name: summarize([r[name] for r in part]) for name in ARMS},
                         "low_minus_medium": compare([r["luna_low"] for r in part],
                                                      [r["luna_medium"] for r in part], task_bootstrap)}
    policies = replay_policies(original, rows, bootstrap)
    report = {
        "format": "bbeh-luna-effort-comparison-v1", "protocol": PROTOCOL, "bootstrap_draws": draws,
        "n": len(rows), "families": len({r["family"] for r in rows}), "tasks": len(by_task),
        "arms": {name: summarize(values, bootstrap) for name, values in arms.items()},
        "comparisons": {f"{first}_vs_{second}": compare(arms[first], arms[second], bootstrap)
                        for first, second in (("luna_low", "luna_medium"), ("luna_low", "sol_medium"),
                                              ("luna_medium", "sol_medium"))},
        "by_task": by_task, "policies": policies, "new_acquisition": acquisition,
        "physical_attempt_errors": dict(Counter(f"{r['http_status']}:{r['error_type']}:{r['error_code']}"
                                                for r in ledger if r["action"] != "terminal")),
        "physical_attempt_elapsed_seconds": numbers([r["elapsed_seconds"] for r in ledger]),
        "original_acquisition_ledger": original["grades"]["acquisition_ledger"],
        "original_replay_verified": True, "new_fit": False, "new_policy_selection": False,
        "provenance": {"release_revision": RELEASE_REVISION, "manifest_sha256": MANIFEST_SHA256,
                       "original_files_sha256": original["hashes"], "collection_sha256": sha(root / "collection.json"),
                       "plan_sha256": sha(root / "plan.json"), "selection_sha256": sha(root / "selection.json"),
                       "analyzer_sha256": sha(__file__), "collector_source_sha256": plan["source_files_sha256"],
                       "normalizer_sha256": sha(normalize.__file__), "grade_v2_sha256": sha(grade_v2.__file__),
                       "official_sha256": OFFICIAL_SHA256, "utility_sha256": sha(utility_router.__file__),
                       "bootstrap_sha256": sha(frontier.__file__), "numpy_version": np.__version__,
                       "original_acquisition_protocol_sha256": original["grades"]["acquisition_protocol_sha256"],
                       "low_attempt_created_unix_range": [min((r["response"]["created_unix"] for r in selected.values()
                                                               if r["response"]), default=None),
                                                          max((r["response"]["created_unix"] for r in selected.values()
                                                               if r["response"]), default=None)]},
    }
    output.mkdir(parents=True)
    write(output / "graded-rows.json", {"format": "bbeh-luna-effort-rows-v1", "rows": rows,
                                       "physical_attempts": ledger, "provenance": report["provenance"]})
    report["graded_rows_sha256"] = sha(output / "graded-rows.json")
    write(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.data, args.collection, args.output)
    print(json.dumps({"n": result["n"], "original_replay_verified": result["original_replay_verified"],
                      "collection_status": result["new_acquisition"]["status"],
                      "report_sha256": sha(args.output / "report.json")}))


if __name__ == "__main__":
    main()
