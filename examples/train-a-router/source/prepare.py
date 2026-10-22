"""Prepare a prospective BBEH family split from pinned public source only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

from grouping import VERSION, digest, families, normalize

REVISION = "80d12ca916b7158f22293fcf3144f4d3d854d4be"
ROLES = ("train", "development", "calibration", "final")
WEIGHTS = (50, 15, 10, 25)
SEED = "bbeh-recipes-20261021-source-only-v1"
CONTRACT = {"type": "exact_string", "max_length": 2048}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def duplicate_key(text):
    # Case is semantic in some tasks. Case folding is used conservatively for
    # family links, never to delete a source problem as a duplicate.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def allocate(task, family_ids):
    ordered = sorted(set(family_ids), key=lambda family: digest([SEED, task, family]))
    n = len(ordered)
    if n < len(ROLES):
        raise ValueError(f"Task {task} lacks four independent retained families")
    counts = [n * weight // 100 for weight in WEIGHTS]
    residual = sorted(range(4), key=lambda i: (-(n * WEIGHTS[i] % 100), i))
    for index in residual[: n - sum(counts)]:
        counts[index] += 1
    if not all(counts):
        raise ValueError(f"Task {task} cannot populate every role under this allocation")
    result, start = {}, 0
    for role, count in zip(ROLES, counts, strict=True):
        result.update({family: role for family in ordered[start : start + count]})
        start += count
    return result


def load(source):
    manifests = [
        json.loads((source / name).read_bytes()) for name in ("source-download.json", "full-source-download.json")
    ]
    inventory = {}
    for manifest in manifests:
        if manifest["revision"] != REVISION:
            raise ValueError("Unexpected BBEH source revision")
        for name, item in manifest["files"].items():
            if sha(source / name) != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {name}")
            inventory[name] = item
    rows = []
    for name in sorted(inventory):
        if not name.endswith("/task.json"):
            continue
        task = Path(name).parent.name.removeprefix("bbeh_")
        for index, example in enumerate(json.loads((source / name).read_bytes())["examples"]):
            if not isinstance(example["input"], str) or not isinstance(example["target"], str):
                raise ValueError("Non-string source input/reference")
            rows.append(
                {
                    "id": f"bbeh:{task}:{index:03d}",
                    "task": task,
                    "index": index,
                    "source_id": f"{name}#examples/{index}",
                    "source_path": name,
                    "request": example["input"],
                    "target": example["target"],
                    "source_metadata": {k: v for k, v in example.items() if k not in {"input", "target"}},
                }
            )
    mini = json.loads((source / "bbeh/mini/data.json").read_bytes())["examples"]
    if len(rows) != 4520 or len({row["task"] for row in rows}) != 23 or len(mini) != 460:
        raise ValueError("Unexpected pinned source population")
    return rows, mini, inventory


def partition(rows, mini):
    # Grouping sees no reference labels, model responses, costs, or scores.
    identity, edges, coverage = families([{k: row[k] for k in ("id", "task", "request")} for row in rows])
    if not all(coverage.values()):
        raise ValueError("An input did not match its task's source-context grammar")
    mini_by_prompt = defaultdict(list)
    references = defaultdict(set)
    for index, row in enumerate(mini):
        mini_by_prompt[normalize(row["input"])].append(index)
        references[duplicate_key(row["input"])].add(row["target"])
    duplicates = defaultdict(list)
    for row in rows:
        key = duplicate_key(row["request"])
        duplicates[key].append(row)
        references[key].add(row["target"])
    conflicting = {key for key, targets in references.items() if len(targets) > 1}
    conflict_families = {identity[row["id"]] for row in rows if duplicate_key(row["request"]) in conflicting}
    mini_families = {identity[row["id"]] for row in rows if normalize(row["request"]) in mini_by_prompt}
    inactive = conflict_families | mini_families
    canonical = {key: min(group, key=lambda row: row["id"])["id"] for key, group in duplicates.items()}
    active = [
        row
        for row in rows
        if identity[row["id"]] not in inactive and canonical[duplicate_key(row["request"])] == row["id"]
    ]
    family_tasks = defaultdict(set)
    for row in active:
        family_tasks[identity[row["id"]]].add(row["task"])
    if any(len(tasks) != 1 for tasks in family_tasks.values()):
        raise ValueError("Cross-task family needs an explicit allocation rule")
    assignment = {}
    for task in sorted({row["task"] for row in rows}):
        assignment.update(allocate(task, [identity[row["id"]] for row in active if row["task"] == task]))
    meta = []
    for row in rows:
        family = identity[row["id"]]
        reasons = []
        if family in mini_families:
            reasons.append("prior_official_mini_problem_family")
        if family in conflict_families:
            reasons.append("reference_conflict_problem_family")
        original = canonical[duplicate_key(row["request"])]
        duplicate_of = original if original != row["id"] else None
        if duplicate_of:
            reasons.append("duplicate_source_input")
        retained = not reasons
        role = (
            assignment[family]
            if retained
            else ("excluded" if family in conflict_families or duplicate_of else "reserved")
        )
        meta.append(
            {k: row[k] for k in ("id", "task", "index", "source_id", "source_path", "source_metadata")}
            | {
                "dataset": "bbeh",
                "source": "bbeh",
                "family": family,
                "role": role,
                "retained": retained,
                "reservation_reasons": reasons,
                "duplicate_of": duplicate_of,
                "mini_indices": mini_by_prompt.get(normalize(row["request"]), []),
                "request_sha256": hashlib.sha256(row["request"].encode()).hexdigest(),
                "normalized_request_sha256": hashlib.sha256(duplicate_key(row["request"]).encode()).hexdigest(),
            }
        )
    conflicts = [
        {
            "normalized_input_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "full_source_ids": [row["id"] for row in duplicates.get(key, [])],
            "target_sha256": sorted(hashlib.sha256(target.encode()).hexdigest() for target in references[key]),
            "affected_family_ids": sorted({identity[row["id"]] for row in duplicates.get(key, [])}),
        }
        for key in sorted(conflicting)
    ]
    return meta, edges, conflicts


def prepare(source, output):
    if output.exists():
        raise FileExistsError("Preserve the existing prepared snapshot; use a new output directory")
    rows, mini, inventory = load(source)
    meta, edges, conflicts = partition(rows, mini)
    output.mkdir(parents=True)
    write(output / "meta.json", meta)
    write(output / "gold.json", [{"id": row["id"], "target": row["target"]} for row in rows])
    inputs = [
        {"id": row["id"], "request": row["request"], "response_contract": CONTRACT}
        for row, item in zip(rows, meta, strict=True)
        if item["retained"]
    ]
    role_by_id = {item["id"]: item["role"] for item in meta}
    for role in ROLES:
        write(
            output / "question-only" / f"{role}-inputs.json", [row for row in inputs if role_by_id[row["id"]] == role]
        )
    write(output / "question-only/all-inputs.json", inputs)
    write(output / "grouping-edges.json", edges)
    write(output / "reference-conflicts.json", conflicts)
    tasks = {}
    for task in sorted({item["task"] for item in meta}):
        items = [item for item in meta if item["task"] == task]
        counts = Counter(item["family"] for item in items)
        tasks[task] = {
            "source_rows": len(items),
            "source_families": len(counts),
            "largest_source_family": max(counts.values()),
            "rows_by_role": dict(Counter(item["role"] for item in items)),
            "families_by_role": {
                role: len({item["family"] for item in items if item["role"] == role}) for role in ROLES
            },
            "retained_rows": sum(item["retained"] for item in items),
            "reserved_mini_family_rows": sum(
                "prior_official_mini_problem_family" in item["reservation_reasons"] for item in items
            ),
        }
    audit = {
        "format": "bbeh-source-split-audit-v1",
        "source_only": True,
        "model_outcomes_accessed": False,
        "source_rows": len(rows),
        "retained_rows": len(inputs),
        "tasks_retained": len(tasks),
        "rows_by_role": dict(Counter(item["role"] for item in meta)),
        "families_by_role": {role: len({item["family"] for item in meta if item["role"] == role}) for role in ROLES},
        "reservation_reason_counts": dict(Counter(reason for item in meta for reason in item["reservation_reasons"])),
        "tasks": tasks,
        "context_rule_coverage": "all 4520 source rows",
        "grouping_edges_by_rule": dict(Counter(edge["rule"] for edge in edges)),
        "overgrouping_controls": [
            "A reused primitive, common task instruction, or generic template is not itself a family link.",
            "Time arithmetic groups the complete composed program, including numerical transformations; single subquestions do not link rows.",
            "SARC requires a shared post-and-reply pair, not a shared post alone; SportQA uses its main scenario, not a generic subquestion.",
            "Linguini shares original language-problem introductions/tables; causal variants share specific story text.",
        ],
        "limitations": [
            "Upstream provides no complete original-problem/template IDs; text-derived families do not prove absence of paraphrased or structurally isomorphic contexts.",
            "Within-task family splits measure performance on new problems from these tasks, not generalization to unseen tasks.",
            "Base-model pretraining overlap is unknown; this grouping constrains router training leakage only.",
            "Unequal family sizes make row proportions approximate. Final paired comparisons should bootstrap whole families and remain descriptive.",
        ],
    }
    write(output / "split-audit.json", audit)
    for name in ("source-download.json", "full-source-download.json", "LICENSE", "README.md", "bbeh/evaluate.py"):
        target = output / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    plan = {
        "format": "bbeh-recipes-source-plan-v1",
        "repository": "https://github.com/google-deepmind/bbeh",
        "revision": REVISION,
        "source_only": True,
        "model_outcomes_accessed": False,
        "split_seed": SEED,
        "role_proportions": dict(zip(ROLES, [weight / 100 for weight in WEIGHTS], strict=True)),
        "allocation": "Within each task: SHA256(seed,task,family)-ordered families; Hamilton integer apportionment in declared role order; keep whole families.",
        "row_order": "Sorted pinned task path, then original examples[] index; roles preserve this order.",
        "grouping_version": VERSION,
        "family_normalization": "Unicode NFKC + whitespace collapse + casefold for conservative context links",
        "duplicate_normalization": "Unicode NFKC + whitespace collapse, case preserved; retain smallest source ID unless whole family reserved/excluded",
        "prior_mini_policy": "Reserve the complete input-derived family of every official Mini row, including prior excluded/conflicting Mini rows.",
        "reference_conflict_policy": "Exclude the entire original-problem family when an identical normalized input has conflicting original references; no model outcomes used.",
        "response_contract": CONTRACT,
        "models": {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"},
        "grading": {
            "official_source_sha256": inventory["bbeh/evaluate.py"]["sha256"],
            "primary": "Unchanged official evaluate_correctness on the decoded answer string and original target",
            "sensitivities": [
                "literal answer.strip() == target.strip()",
                "official normalization with finite Decimal equality replacing binary float equality",
            ],
            "invalid": "Unsuccessful outcome with original usage retained; no response-based exclusions or quality retries",
        },
        "counts": {role: audit["rows_by_role"].get(role, 0) for role in ROLES},
        "family_counts": audit["families_by_role"],
        "tasks": list(tasks),
        "source_inventory": inventory,
        "preparation_sources_sha256": {
            name: sha(Path(__file__).parent / name) for name in ("prepare.py", "grouping.py", "test_source.py")
        },
        "files_sha256": {
            path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    write(output / "source-plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = prepare(args.source, args.output)
    print(
        json.dumps(
            {
                "source_plan_sha256": sha(args.output / "source-plan.json"),
                "counts": plan["counts"],
                "family_counts": plan["family_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
