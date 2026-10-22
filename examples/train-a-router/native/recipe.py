"""Small BBEH adapter: fixed source roles, paired correctness and USD costs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

ARMS = ("cheap", "strong")
ROLES = ("train", "development", "calibration", "final")
CONTEXTS = (512, 1024, 2048, 4096, 8192)
BASE = "convaiinnovations/laya"
REVISION = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"
INITIAL_WEIGHTS_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
NATIVE = {
    "base": BASE,
    "revision": REVISION,
    "head_max_len": 256,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "encoder_lr": 0.0002,
    "head_lr": 0.0001,
    "weight_decay": 0.01,
    "micro_batch": 4,
    "grad_accum": 4,
    "evaluation_batch": 4,
    "gradient_checkpointing": True,
    "selection_epochs": 8,
    "seed": 89,
    "selection": "Minimum development summed binary NLL over epochs0..8; exact ties choose earlier epoch",
    "refit": "Fresh pinned base/seed; train+development for selected epoch count; no calibration/final labels",
    "versions": {"laya": "0.3.29", "torch": "2.9.1", "transformers": "4.57.6", "peft": "0.18.1"},
}
PROTOCOL = {
    "objective": "Two binary correctness questions, unweighted summed Bernoulli NLL; cheap-only wins retained",
    "policy": "p_strong-p_cheap-lambda*(expected_USD_strong-expected_USD_cheap)>=0; lambda>=0",
    "cost_model": "Four train+development arm-by-correctness USD means, shared by base and LoRA; an absent class uses that arm's unconditional mean; no cost floor",
    "selection": "Development selects epoch; refit train+development; full utility frontier on calibration",
    "final": "Input-only predictions may be saved; final labels never enter fit, selection, timing or calibration",
    "native": NATIVE,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sources():
    return {name: sha(Path(__file__).parent / name) for name in ("recipe.py", "bbeh_native.py")}


def questions(pair):
    models = pair.get("models") or {arm: pair["arms"][arm]["model"] for arm in ARMS}
    require(set(models) == set(ARMS) and all(isinstance(models[a], str) and models[a] for a in ARMS), "Bad model pair")
    return {
        f"{arm}_correct": {
            "type": "choice",
            "instructions": f"Predict whether {models[arm]} will answer this request correctly. Judge its likely answer quality; do not answer the request.",
            "criteria": {
                "incorrect": "The solver will give an incorrect answer.",
                "correct": "The solver will give the correct answer.",
            },
        }
        for arm in ARMS
    }


def source_rows(source):
    source = Path(source)
    plan = read(source / "source-plan.json")
    names = ["meta.json", *(f"question-only/{role}-inputs.json" for role in ROLES)]
    for name in names:
        require(sha(source / name) == plan["files_sha256"][name], f"Source binding differs: {name}")
    metadata = read(source / "meta.json")
    require(isinstance(metadata, list), "meta.json must be the source owner's flat list")
    by_id = {row["id"]: row for row in metadata}
    require(len(by_id) == len(metadata), "Duplicate source metadata ID")
    rows, families, ids = [], {}, set()
    for role in ROLES:
        inputs = read(source / f"question-only/{role}-inputs.json")
        require(isinstance(inputs, list) and inputs, f"Empty source role: {role}")
        expected = [row["id"] for row in metadata if row["role"] == role and row["retained"]]
        require([row["id"] for row in inputs] == expected, f"Source role/order differs: {role}")
        for item in inputs:
            require(set(item) == {"id", "request", "response_contract"}, "Unexpected question-only fields")
            require(
                item["id"] not in ids and isinstance(item["request"], str) and item["request"].strip(),
                "Bad source input",
            )
            row = by_id[item["id"]]
            require(text_sha(item["request"]) == row["request_sha256"], "Full request hash differs")
            family = row["family"]
            require(isinstance(family, str) and family, "Missing family")
            require(family not in families or families[family] == role, "A family crosses source roles")
            families[family] = role
            ids.add(item["id"])
            rows.append({**item, "role": role, "family": family, "task": row["task"]})
    return plan, rows


def audit(source, pair_path, output, checkpoint=None):
    """Remote CPU only: exact full sequence lengths, with no weights or outcomes."""
    require(not Path(output).exists(), "Preserve earlier length audit")
    import laya.train as lt
    from transformers import AutoTokenizer

    require(importlib.metadata.version("laya") == "0.3.29", "Use pinned Laya tokenizer serialization")
    require(importlib.metadata.version("transformers") == "4.57.6", "Use pinned Transformers")
    _, rows = source_rows(source)
    pair = read(pair_path)
    tokenizer = (
        AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer", local_files_only=True, trust_remote_code=False)
        if checkpoint
        else AutoTokenizer.from_pretrained(BASE, revision=REVISION, subfolder="tokenizer", trust_remote_code=False)
    )
    heads = {}
    for key, question in questions(pair).items():
        internal = lt.to_internal(key, question)
        head, _, _ = lt.build_head(tokenizer, internal, NATIVE["head_max_len"])
        uncut, _, _ = lt.build_head(tokenizer, internal, 8192)
        require(head == uncut, "Decision head exceeds the fixed256-token head cap")
        heads[key] = {"question": internal, "tokens": len(head)}
    lengths = []
    for row in rows:
        state = lt.encode_state(tokenizer, row["request"], 1_000_000)
        lengths.append(
            {
                "id": row["id"],
                "role": row["role"],
                "task": row["task"],
                "request_sha256": text_sha(row["request"]),
                "state_tokens": len(state),
                "sequence_tokens": len(state) + max(head["tokens"] for head in heads.values()) + 1,
            }
        )
    maximum = max(row["sequence_tokens"] for row in lengths)
    chosen = next((length for length in CONTEXTS if maximum <= length), None)
    result = {
        "format": "bbeh-native-length-audit-v1",
        "source_only": True,
        "model_weights_read": False,
        "source_plan_sha256": sha(Path(source) / "source-plan.json"),
        "pair_sha256": sha(pair_path),
        "audit_source_sha256": sha(__file__),
        "model": BASE,
        "revision": REVISION,
        "questions": questions(pair),
        "head_tokens": {key: value["tokens"] for key, value in heads.items()},
        "count": len(rows),
        "max_sequence_tokens": maximum,
        "selected_max_len": chosen,
        "overflow_counts": {
            str(length): sum(row["sequence_tokens"] > length for row in lengths) for length in CONTEXTS
        },
        "unsupported_ids": [row["id"] for row in lengths if row["sequence_tokens"] > 8192],
        "rows": lengths,
        "context_rule": "Smallest supported power of two covering both actual decision heads and every full request; never truncate or remove an overflow row",
    }
    write(output, result)
    return result


def prepare(source, pair_path, lengths_path, output, acquisition_protocol_sha256):
    output = Path(output)
    require(not output.exists(), "Preserve existing prepared study")
    _, rows = source_rows(source)
    audit, pair = read(lengths_path), read(pair_path)
    source_hash, pair_hash = sha(Path(source) / "source-plan.json"), sha(pair_path)
    require(
        audit["source_only"] and audit["source_plan_sha256"] == source_hash and audit["pair_sha256"] == pair_hash,
        "Length audit uses different source/pair",
    )
    require(audit["audit_source_sha256"] == sha(__file__), "Length auditor changed")
    require(
        audit["selected_max_len"] in CONTEXTS and not audit["unsupported_ids"],
        "Some full inputs exceed supported native context; report them, do not drop them",
    )
    require(
        [row["id"] for row in audit["rows"]] == [row["id"] for row in rows], "Length audit population/order differs"
    )
    require(
        all(a["request_sha256"] == text_sha(r["request"]) for a, r in zip(audit["rows"], rows, strict=True)),
        "Length audit request changed",
    )
    require(re.fullmatch(r"[0-9a-f]{64}", acquisition_protocol_sha256), "Missing acquisition protocol hash")
    output.mkdir(parents=True)
    write(output / "source-rows.json", rows)
    (output / "pair.json").write_bytes(Path(pair_path).read_bytes())
    (output / "lengths.json").write_bytes(Path(lengths_path).read_bytes())
    plan = {
        "format": "bbeh-native-plan-v1",
        "protocol": PROTOCOL,
        "native": {**NATIVE, "max_len": audit["selected_max_len"]},
        "source_plan_sha256": source_hash,
        "source_rows_sha256": sha(output / "source-rows.json"),
        "pair_sha256": pair_hash,
        "pair": pair,
        "questions": questions(pair),
        "acquisition_protocol_sha256": acquisition_protocol_sha256,
        "length_audit_sha256": sha(output / "lengths.json"),
        "source_code_sha256": sources(),
        "populations": {role: [row["id"] for row in rows if row["role"] == role] for role in ROLES},
        "initial_weights_sha256": INITIAL_WEIGHTS_SHA256,
    }
    write(output / "plan.json", plan)
    return plan


def checked_plan(run):
    run = Path(run)
    plan = read(run / "plan.json")
    require(
        plan["format"] == "bbeh-native-plan-v1"
        and plan["protocol"] == PROTOCOL
        and plan["source_code_sha256"] == sources(),
        "Native recipe changed",
    )
    require(
        plan["native"] == {**NATIVE, "max_len": plan["native"]["max_len"]} and plan["native"]["max_len"] in CONTEXTS,
        "Native settings changed",
    )
    for name, key in (
        ("source-rows.json", "source_rows_sha256"),
        ("pair.json", "pair_sha256"),
        ("lengths.json", "length_audit_sha256"),
    ):
        require(sha(run / name) == plan[key], f"Prepared bytes changed: {name}")
    rows = read(run / "source-rows.json")
    require(
        {role: [r["id"] for r in rows if r["role"] == role] for role in ROLES} == plan["populations"],
        "Native populations changed",
    )
    require(
        plan["pair"] == read(run / "pair.json") and plan["questions"] == questions(plan["pair"]),
        "Pair/criteria changed",
    )
    require(read(run / "lengths.json")["selected_max_len"] == plan["native"]["max_len"], "Context changed after audit")
    return plan, rows


def grade_arrays(plan, metadata, grades, role):
    import numpy as np

    require(role in ("train", "development"), "Native fitting cannot read calibration/final grades")
    require(grades["format"] == "bbeh-paired-grades-v1" and grades["role"] == role, "Wrong grade format/role")
    require(
        all(grades[key] == plan[key] for key in ("source_plan_sha256", "pair_sha256", "acquisition_protocol_sha256")),
        "Grade source/pair/acquisition differs",
    )
    models = plan["pair"].get("models") or {a: plan["pair"]["arms"][a]["model"] for a in ARMS}
    require(
        grades["models"] == models and grades["cost_scope"] == plan["pair"]["cost_scope"], "Wrong models/cost scope"
    )
    require(re.fullmatch(r"[0-9a-f]{64}", grades["grader_sha256"]), "Missing grader provenance")
    rows = [row for row in metadata if row["role"] == role]
    require(
        [row["id"] for row in grades["rows"]] == [row["id"] for row in rows],
        "Require complete grades in fixed source order",
    )
    labels, costs = [], []
    for grade in grades["rows"]:
        require(
            all(type(grade[f"{a}_correct"]) in (bool, int) and grade[f"{a}_correct"] in (0, 1) for a in ARMS),
            "Expected two binary labels",
        )
        values = [grade[f"{a}_cost_usd"] for a in ARMS]
        require(
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "Missing costs cannot become zero",
        )
        require(
            all(re.fullmatch(r"[0-9a-f]{64}", grade[f"{a}_record_sha256"]) for a in ARMS), "Missing response provenance"
        )
        labels.append([int(grade[f"{a}_correct"]) for a in ARMS])
        costs.append(values)
    return rows, np.asarray(labels, dtype=float), np.asarray(costs, dtype=float)


def conditional_usd(targets, costs):
    import numpy as np

    targets, costs = np.asarray(targets), np.asarray(costs, dtype=float)
    require(
        targets.ndim == 2
        and targets.shape[1] == 2
        and costs.shape == targets.shape
        and np.isin(targets, (0, 1)).all()
        and np.isfinite(costs).all()
        and (costs >= 0).all(),
        "Bad paired labels/costs",
    )
    means, counts = np.empty((2, 2)), np.empty((2, 2), dtype=int)
    for arm in range(2):
        for label in range(2):
            mask = targets[:, arm] == label
            means[arm, label] = costs[mask, arm].mean() if mask.any() else costs[:, arm].mean()
            counts[arm, label] = mask.sum()
    return {
        "unit": "USD",
        "arm_order": list(ARMS),
        "class_order": [0, 1],
        "means": means.tolist(),
        "counts": counts.tolist(),
    }


def expected_usd(probabilities, statistics):
    import numpy as np

    p, means = np.asarray(probabilities, dtype=float), np.asarray(statistics["means"], dtype=float)
    require(
        statistics["unit"] == "USD" and statistics["arm_order"] == list(ARMS) and statistics["class_order"] == [0, 1],
        "Wrong cost units/order",
    )
    require(
        p.ndim == 2
        and p.shape[1] == 2
        and means.shape == (2, 2)
        and np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all()
        and np.isfinite(means).all()
        and (means >= 0).all(),
        "Invalid cost predictions",
    )
    return (1 - p) * means[:, 0] + p * means[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--pair", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "audit":
            command.add_argument("--checkpoint", type=Path)
        else:
            command.add_argument("--lengths", type=Path, required=True)
            command.add_argument("--acquisition-protocol-sha256", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.source, args.pair, args.output, args.checkpoint)
        print(json.dumps({key: result[key] for key in ("count", "selected_max_len", "unsupported_ids")}))
    else:
        result = prepare(args.source, args.pair, args.lengths, args.output, args.acquisition_protocol_sha256)
        print(json.dumps({"plan": str(args.output / "plan.json"), "max_len": result["native"]["max_len"]}))


if __name__ == "__main__":
    main()
