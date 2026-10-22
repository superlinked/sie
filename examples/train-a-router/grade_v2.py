"""Verify all physical attempts and grade every prospectively retained BBEH row."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

import collect
import normalize
import openai_population as acquisition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(root, relative, expected):
    path = root / relative
    require(not Path(relative).is_absolute() and root.resolve() in path.resolve().parents, "Unsafe receipt path")
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, "Receipt hash mismatch")
    return path


def load_official(path, expected):
    require(sha(path) == expected, "Official BBEH grader changed")
    spec = importlib.util.spec_from_file_location("pinned_bbeh_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    # The audited upstream file has eight demonstration prints, but no other side effects.
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def decimal_match(module, answer, target):
    prediction = module.preprocess_sample(answer)
    reference = module.preprocess_reference(target)
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        a, b = Decimal(prediction), Decimal(reference)
        if a.is_finite() and b.is_finite() and a == b:
            return True
    except InvalidOperation:
        pass
    return (prediction.replace("'", "") == reference.replace("'", "")
            or f"[{reference}]" == prediction or f"[{prediction}]" == reference
            or (prediction.endswith("?") and prediction[:-1] == reference))


def checked_arm(root, plan, collection, inputs, arm, role):
    binding = collection["selections"][f"{role}/{arm}"]
    selection_path = bound(root, binding["path"], binding["sha256"])
    selection = read(selection_path)
    ids = [item["id"] for item in inputs]
    plan_sha = sha(root / "plan.json")
    require(selection["ordered_ids"] == ids and [entry["id"] for entry in selection["entries"]] == ids,
            "Selection population/order changed")
    require(selection["plan_sha256"] == plan_sha and selection["model"] == collect.ARMS[arm]["model"]
            and selection["arm"] == arm and selection["split"] == role
            and selection["first_terminal_response_only"] and selection["no_quality_retries"], "Selection contract changed")
    rows, ledger, declared = {}, [], set()
    for item, entry in zip(inputs, selection["entries"], strict=True):
        body = collect.body_for(item, arm)
        body_sha = acquisition.digest(json.dumps(body, sort_keys=True).encode())
        first = None
        require(len(entry["attempts"]) <= acquisition.MAX_ATTEMPTS, "Attempt cap exceeded")
        for number, attempt in enumerate(entry["attempts"], 1):
            require(first is None, "A terminal answer was retried")
            require(attempt["attempt"] == number, "Attempt order changed")
            prefix = f"{role}/{arm}/attempts/{acquisition.digest(item['id'].encode())}/{number:03d}"
            require(attempt["intent_path"] == prefix + "-intent.json", "Intent ID/path mismatch")
            intent_path = bound(root, attempt["intent_path"], attempt["intent_sha256"])
            declared.add(intent_path.relative_to(root).as_posix())
            intent = read(intent_path)
            expected = {"id": item["id"], "arm": arm, "split": role, "attempt": number,
                        "body_sha256": body_sha, "plan_sha256": plan_sha}
            require(all(intent.get(key) == value for key, value in expected.items()), "Intent binding changed")
            reserve = collect.reservation(body, arm)
            require(Decimal(intent["reserved_usd"]) == reserve, "Reservation changed")
            if attempt["record_path"] is None:
                require(not (root / (prefix + "-response.json")).exists(), "Undeclared physical response")
                response = {"status": "interrupted_unknown", "terminal_response": False,
                            "error_type": "ProcessInterrupted", "created_unix": intent["created_unix"]}
            else:
                require(attempt["record_path"] == prefix + "-response.json", "Response ID/path mismatch")
                path = bound(root, attempt["record_path"], attempt["record_sha256"])
                declared.add(path.relative_to(root).as_posix())
                response = read(path)
                require(all(response.get(key) == value for key, value in {
                    "id": item["id"], "arm": arm, "model": body["model"], "body_sha256": body_sha,
                    "intent_sha256": attempt["intent_sha256"], "attempt": number}.items()), "Response binding changed")
                if response.get("terminal_response"):
                    require(response.get("resolved_model") == body["model"],
                            "Returned model differs from the requested model")
                    require(response.get("service_tier") == body["service_tier"] == "default",
                            "Returned service tier differs from standard pricing")
            action = acquisition.decision(response, number)
            require(attempt["action"] == action and attempt["status"] == response["status"], "Attempt decision changed")
            cost = response.get("cost")
            if cost is not None:
                require(cost == collect.openai_cost(response["usage"], body), "Usage cost changed")
                require(attempt["known_list_usd"] == cost["usd_estimate"]
                        and Decimal(attempt["known_budget_usd"]) == Decimal(cost["budget_usd"])
                        and Decimal(attempt["unknown_reservation_usd"]) == 0, "Known attempt ledger changed")
            else:
                require(attempt["known_list_usd"] is None and attempt["known_budget_usd"] is None
                        and Decimal(attempt["unknown_reservation_usd"]) == reserve, "Unknown attempt ledger changed")
            ledger.append({"id": item["id"], "arm": arm, "role": role, **attempt})
            if action == "terminal":
                require(response.get("http_status") == 200 and response.get("binding_valid") is True,
                        "Invalid terminal identity receipt")
                first = (number, response, attempt["record_sha256"])
        require(entry["selected_attempt"] == (first[0] if first else None), "Selected response is not first terminal")
        if first:
            response = first[1]
            text = "".join(part.get("text") or "" for part in response.get("messages", []) if part["type"] == "output_text")
            answer = collect.parse_answer(text, item["response_contract"]) if response.get("provider_status") == "completed" else None
            if response.get("status") == "complete":
                require(response.get("answer") == answer and response.get("valid_answer") == (answer is not None),
                        "Saved final-answer parse changed")
            rows[item["id"]] = {"response": response, "record_sha256": first[2], "answer": answer}
        else:
            rows[item["id"]] = {"response": {"status": entry["status"]}, "record_sha256": None, "answer": None}
    actual = {path.relative_to(root).as_posix() for path in (root / role / arm / "attempts").rglob("*.json")}
    require(actual == declared, "Undeclared or missing physical attempt files")
    return rows, ledger


def grade(prepared, root, official, pair_path, output):
    prepared, root = Path(prepared), Path(root)
    source, pair = read(prepared / "source-plan.json"), read(pair_path)
    for name in ("meta.json", "gold.json"):
        bound(prepared, name, source["files_sha256"][name])
    metadata, gold = (read(prepared / name) for name in ("meta.json", "gold.json"))
    meta_by_id, gold_by_id = ({row["id"]: row for row in rows} for rows in (metadata, gold))
    require(len(meta_by_id) == len(metadata) == len(gold_by_id) == len(gold), "Duplicate or inconsistent source IDs")
    plan, collection = read(root / "plan.json"), read(root / "collection.json")
    require(plan["source_plan_sha256"] == sha(prepared / "source-plan.json")
            and collection["plan_sha256"] == sha(root / "plan.json"), "Source/acquisition plan mismatch")
    require(plan["pair_sha256"] == sha(pair_path), "Pair specification changed")
    require(pair["models"] == {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"}, "Answer models changed")
    for name in ("collect.py", "openai_population.py"):
        require(plan["files_sha256"][name] == sha(Path(__file__).parent / name), "Executed collection source changed")
    require(set(plan["roles"]) <= {"train", "development", "calibration", "final"}, "Unknown phase role")
    module = load_official(Path(official), "4b4f06e5babb015de2ba639bae995a5526182ffb5b5890af54dfcf038580eb34")
    ledger, outputs = [], {}
    output = Path(output)
    require(not output.exists(), "Preserve prior grade outputs")
    output.mkdir(parents=True)
    for role in plan["roles"]:
        input_path = prepared / "question-only" / f"{role}-inputs.json"
        require(sha(input_path) == plan["files_sha256"][f"{role}-inputs.json"], "Collected inputs changed")
        bound(prepared, f"question-only/{role}-inputs.json", source["files_sha256"][f"question-only/{role}-inputs.json"])
        inputs = read(input_path)
        require(len(inputs) == plan["counts"][role] and len({row["id"] for row in inputs}) == len(inputs), "Role population changed")
        arms, role_ledger = {}, []
        for arm in ("luna", "sol"):
            arms[arm], physical = checked_arm(root, plan, collection, inputs, arm, role)
            role_ledger.extend(physical)
        ledger.extend(role_ledger)
        result_rows = []
        for item in inputs:
            meta, reference = meta_by_id[item["id"]], gold_by_id[item["id"]]
            require(meta["role"] == role and meta["retained"] is True, "Inactive or cross-role row in acquisition")
            require(meta["request_sha256"] == hashlib.sha256(item["request"].encode()).hexdigest(), "Source request changed")
            row = {"id": item["id"], "task": meta["task"], "family": meta["family"], "role": role, "dataset": "bbeh",
                   "source_format_flags": normalize.source_flags(item["request"], meta["task"])}
            for arm, label in (("luna", "cheap"), ("sol", "strong")):
                selected = arms[arm][item["id"]]
                response, answer = selected["response"], selected["answer"]
                valid = answer is not None and response.get("status") == "complete"
                cost = response.get("cost")
                normalized, normalization_rule = normalize.canonicalize(answer, item["request"], meta["task"])
                elapsed = response.get("elapsed_seconds")
                require(elapsed is None or (type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0),
                        "Invalid selected-response elapsed time")
                values = {"correct": int(valid and normalized is not None and decimal_match(module, normalized, reference["target"])),
                    "official_raw_correct": int(valid and module.evaluate_correctness(answer, reference["target"])),
                    "literal_correct": int(valid and answer.strip() == reference["target"].strip()),
                    "decimal_correct": int(valid and normalized is not None and decimal_match(module, normalized, reference["target"])),
                    "normalized_answer": normalized, "normalization_rule": normalization_rule,
                    "valid": valid, "status": response["status"], "provider_status": response.get("provider_status"),
                    "truncated": response.get("provider_status") == "incomplete", "usage": response.get("usage"),
                    "cost_usd": float(Decimal(cost["usd_estimate"])) if cost else None,
                    "elapsed_seconds": elapsed,
                    "record_sha256": selected["record_sha256"]}
                row.update({label + "_" + key: value for key, value in values.items()})
            result_rows.append(row)
        result = {"format": "bbeh-paired-grades-v1", "role": role, "n": len(result_rows),
            "source_plan_sha256": sha(prepared / "source-plan.json"), "pair_sha256": sha(pair_path),
            "collection_sha256": sha(root / "collection.json"), "acquisition_protocol_sha256": sha(root / "plan.json"),
            "grader_sha256": sha(Path(__file__)), "official_grader_sha256": sha(official),
            "normalizer_sha256": sha(normalize.__file__), "normalization_version": normalize.VERSION,
            "grading_semantics": "Source-only option/tuple normalization, then upstream string normalization with exact Decimal numeric equivalence. Original raw official score retained separately. No answer regeneration or target-aware repair.",
            "cost_scope": pair["cost_scope"], "models": pair["models"],
            "latency_scope": "Selected physical attempt wall time inside acquire(): client construction, HTTP request and response processing; excludes queueing, retry delays and checkpoint commits. Missing measurements remain null.",
            "prices_usd_per_million": {"cheap": collect.PRICES["gpt-6-luna"], "strong": collect.PRICES["gpt-6.1-sol"]},
            "rows": result_rows, "acquisition_ledger": {"physical_attempts": len(role_ledger), "attempts": role_ledger}}
        acquisition.dump(output / f"grades-{role}.json", result)
        outputs[role] = {"path": f"grades-{role}.json", "sha256": sha(output / f"grades-{role}.json"), "n": len(result_rows)}
    sums = {key: sum((Decimal(row[key] or "0") for row in ledger), Decimal(0))
            for key in ("known_list_usd", "known_budget_usd", "unknown_reservation_usd")}
    require(len(ledger) == collection["physical_attempts"]
            and all(value == Decimal(collection[key]) for key, value in sums.items()), "Physical acquisition totals changed")
    require(sums["known_budget_usd"] + sums["unknown_reservation_usd"] <= Decimal(plan["budget_usd"]), "Budget cap exceeded")
    index = {"format": "bbeh-phase-grading-v1", "status": collection["status"], "roles": outputs,
             "physical_attempts": len(ledger), **{key: str(value) for key, value in sums.items()},
             "cap_usd": plan["budget_usd"], "source_plan_sha256": sha(prepared / "source-plan.json"),
             "acquisition_protocol_sha256": sha(root / "plan.json"), "pair_sha256": sha(pair_path)}
    acquisition.dump(output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = grade(args.prepared, args.collection, args.official, args.pair, args.output)
    print(json.dumps({"roles": result["roles"], "status": result["status"]}))
