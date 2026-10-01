#!/usr/bin/env python3
"""Score the recorded structured-output run offline. No API key, no network, no inference spend.

    python3 fetch.py
    uv run score.py                      # the recorded run
    uv run score.py --calls runs         # your own calls from run.py --record, same row format

Three sets, one table each:

    e1      500 records of the Structured Output Benchmark (SOB) test split, sent once to every model
    nhtsa   150 NHTSA vehicle complaints filed from 1 September 2026, after every model's training data
    repeat  the first 100 e1 records sent again, one at a time, for repeatability and latency

For e1 the figure is SOB's own: `evaluate_record` from SOB's evaluate.py, imported unmodified, gives each
record a `leaf_value_em`, the share of the gold answer's leaf values the model returned exactly (zero
unless the answer parses and validates against the record's schema). Value accuracy is its mean over the
500 records. Lenient accuracy relaxes the match (case, punctuation, articles, or one value containing
the other) and value token F1 is SOB's own softer measure; both are secondary.

For nhtsa each record scores the share of its gold fields the answer got right: the agency's coded
crash, fire and injury count, plus make, model and year when the narrative names them.

SIE minus each other model is a paired difference over records, with a 95% interval from a 10,000-draw
bootstrap. Each comparison draws from its own random.Random(1), so the intervals come out the same on
every machine and an arm's interval does not depend on which other arms are scored.

Needs `jsonschema` (for SOB's evaluate.py) and `pyarrow` (to read SOB's parquet): `uv run` installs both.
Python 3.12, as pyproject.toml pins: its sum() rounds differently from 3.11's, which moves the last digit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import fetch

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
SOB_CODE = EVIDENCE / "sob" / "code"
SOB_PARQUET = EVIDENCE / "sob" / "test.parquet"

# The SHA-256 of manifest.json at the dataset revision fetch.py pins. It lives here, outside the
# evidence, because a digest inside a file cannot authenticate that file. The manifest in turn carries
# the SHA-256 of every other file.
MANIFEST_SHA256 = "693d966b2edab5836acb570772d3cf1f13be17d7e0da31c24926ddee46d43cf1"

SIE_MODEL = "Qwen/Qwen3.8-27B-FP8"
SIE_FILE = "Qwen_Qwen3.8-27B-FP8"
REPEAT_N = 100
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1

NAMES = {
    SIE_MODEL: "SIE Qwen3.8 27B",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6-sol": "GPT-6 Sol",
}

# US dollars per 1M tokens: input, output, cached input. Read on 2026-09-30 from
#   OpenAI     https://openai.com/api/pricing/
#   Anthropic  https://docs.anthropic.com/en/docs/about-claude/pricing
#   SIE        https://superlinked.com/cloud
# Both vendors halve input and output for their batch APIs; SIE has no batch discount.
PRICES = {
    SIE_MODEL: (0.25, 2.00, 0.025),
    "gpt-6-luna": (0.10, 0.50, 0.01),
    "gpt-6-sol": (2.00, 10.00, 0.20),
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    # Opus 5.5 cache hits are 0.05x base input, not the usual 0.1x.
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
BATCH_DISCOUNT = {model: 0.5 for model in PRICES} | {SIE_MODEL: 1.0}
DOCUMENTS = 1_000_000


# --- evidence ---------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_evidence() -> dict[str, Any]:
    manifest_path = EVIDENCE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("No evidence/ yet. Run `python3 fetch.py` first.")
    if MANIFEST_SHA256.startswith("REPLACE"):
        print(
            "WARNING: score.py pins no manifest yet, so the evidence is checked only against itself\n", file=sys.stderr
        )
    elif sha256(manifest_path) != MANIFEST_SHA256:
        raise SystemExit("evidence/manifest.json is not the manifest this example pins; run fetch.py again")
    manifest = load_json(manifest_path)
    for name, digest in manifest["files_sha256"].items():
        path = EVIDENCE / name
        if not path.exists() or sha256(path) != digest:
            raise SystemExit(f"evidence/{name} is missing or does not match the manifest")
    if not SOB_PARQUET.exists() or sha256(SOB_PARQUET) != fetch.SOB_PARQUET_SHA256:
        raise SystemExit("evidence/sob/test.parquet is missing or is not the pinned SOB test split")
    for name, oid in fetch.SOB_CODE.items():
        path = SOB_CODE / name
        if not path.exists() or fetch.git_blob_oid(path.read_bytes()) != oid:
            raise SystemExit(f"evidence/sob/code/{name} is missing or is not SOB's file at {fetch.SOB_COMMIT}")
    return manifest


def load_sob() -> tuple[Any, Any, Any]:
    """SOB's prompts, schema utilities and scorer, imported unmodified from evidence/sob/code.

    Imported here rather than at module scope because the code is downloaded by fetch.py, not installed:
    it only becomes importable once its folder is on the path.
    """
    if str(SOB_CODE) not in sys.path:
        sys.path.insert(0, str(SOB_CODE))
    prompts = importlib.import_module("sob.common.prompts")
    schema_utils = importlib.import_module("sob.common.schema_utils")
    evaluate = importlib.import_module("evaluate")
    return prompts, schema_utils, evaluate


def load_rows() -> list[dict[str, Any]]:
    """Every row of the SOB test split, in file order. 49 record ids appear twice, so this is not a dict."""
    return pq.read_table(SOB_PARQUET).to_pylist()


def draw(records: list[dict[str, Any]], n: int, seed: str) -> list[str]:
    """The pre-registered sample: n records, stratified by difficulty x schema complexity, in proportion."""
    strata: dict[tuple[str, str], list[str]] = {}
    for record in records:
        strata.setdefault((record["question_difficulty"], record["schema_complexity"]), []).append(record["record_id"])
    total = sum(len(ids) for ids in strata.values())
    rng = random.Random(seed)
    keys = sorted(strata)
    quotas = {key: n * len(strata[key]) // total for key in keys}
    remainders = sorted(keys, key=lambda key: -(n * len(strata[key]) / total - quotas[key]))
    for key in remainders[: n - sum(quotas.values())]:
        quotas[key] += 1
    picked: list[str] = []
    for key in keys:
        ids = sorted(strata[key])
        rng.shuffle(ids)
        picked += ids[: quotas[key]]
    rng.shuffle(picked)
    return picked


def sob_items(set_name: str) -> dict[str, dict[str, Any]]:
    """{id: {system, user, schema, record}} for e1 or repeat, exactly as every model was sent them."""
    prompts, schema_utils, _ = load_sob()
    sample = load_json(EVIDENCE / "inputs" / "sob_sample.json")
    rows = load_rows()
    # The draw runs over every row, duplicates included; a record id then resolves to its last row.
    records = {row["record_id"]: row for row in rows}
    if draw(rows, sample["n"], sample["seed"]) != sample["e1"]:
        raise SystemExit("the sample in inputs/sob_sample.json is not the one its seed draws from the SOB split")
    ids = sample["e1"] if set_name == "e1" else sample["e1"][:REPEAT_N]
    items = {}
    for rid in ids:
        record = records[rid]
        schema = schema_utils.parse_if_string(record["json_schema"])
        items[rid] = {
            "system": prompts.SYSTEM_PROMPT,
            "user": prompts.build_user_message(record, schema=schema),
            "schema": schema_utils.normalize_schema_strict(schema),
            "record": record,
        }
    return items


def nhtsa_items() -> dict[str, dict[str, Any]]:
    return {
        item["id"]: {"system": item["system"], "user": item["user"], "schema": item["schema"], "record": item}
        for item in load_json(EVIDENCE / "inputs" / "nhtsa_records.json")
    }


def items_for(set_name: str) -> dict[str, dict[str, Any]]:
    return nhtsa_items() if set_name == "nhtsa" else sob_items(set_name)


def scored_rows(path: Path) -> dict[str, dict[str, Any]]:
    """One row per id: the first that succeeded, or else the last failure.

    The calls files keep every attempt, including failures later re-sent. A rejected schema is an
    answer (error null), so it is kept like any other.
    """
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in rows or rows[row["id"]].get("error") is not None:
            rows[row["id"]] = row
    return rows


def model_of(stem: str) -> str:
    base = stem.split("__")[0]
    return base.replace("_", "/", 1) if base.startswith("Qwen_") else base


def label(stem: str) -> str:
    base, _, variant = stem.partition("__")
    name = NAMES.get(model_of(base), model_of(base))
    return f"{name} ({variant})" if variant else name


# --- scoring ----------------------------------------------------------------------------------------


def _norm(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def loose_equal(pred: Any, gold: Any) -> bool:
    """Equal after SOB's normalisation (case, punctuation, articles), or one contains the other."""
    if pred == gold:
        return True
    if pred is None:
        return False
    a, b = _norm(pred), _norm(gold)
    return bool(b) and (a == b or b in a or a in b)


def nhtsa_equal(key: str, got: Any, gold: Any) -> bool:
    if key in {"make", "model"}:

        def norm(value: Any) -> str:
            return "".join(ch for ch in str(value or "").upper() if ch.isalnum())

        return norm(got) == norm(gold)
    return got == gold


class Tally:
    """Per-arm counts shared by the SOB and NHTSA scorers."""

    def __init__(self) -> None:
        self.errors = self.rejected = 0
        self.tokens_in = self.tokens_out = self.tokens_cached = 0

    def take(self, row: dict[str, Any] | None) -> bool:
        """Count one scored row; True when it carries an answer to score."""
        if row is not None and row.get("error") is None and row.get("rejected"):
            self.rejected += 1
            return False
        if row is None or row.get("error") is not None:
            self.errors += 1
            return False
        self.tokens_in += row["tokens_in"]
        self.tokens_out += row["tokens_out"]
        self.tokens_cached += row["tokens_cached"]
        return True

    def summary(self, n: int) -> dict[str, Any]:
        answered = max(1, n - self.errors - self.rejected)
        return {
            "answered": n - self.errors - self.rejected,
            "errors": self.errors,
            "schemas_rejected": self.rejected,
            "mean_tokens_in": self.tokens_in / answered,
            "mean_tokens_cached": self.tokens_cached / answered,
            "mean_tokens_out": self.tokens_out / answered,
        }


def called_ids(folder: Path) -> set[str]:
    return {rid for path in folder.glob("*.jsonl") for rid in scored_rows(path)}


def score_sob(set_name: str, calls: Path, partial: bool = False) -> dict[str, Any]:
    _, schema_utils, evaluate = load_sob()
    items = items_for(set_name)
    if partial:
        items = {rid: item for rid, item in items.items() if rid in called_ids(calls / set_name)}
    arms: dict[str, dict[str, Any]] = {}
    per_record: dict[str, dict[str, float]] = {}
    for path in sorted((calls / set_name).glob("*.jsonl")):
        rows = scored_rows(path)
        tally = Tally()
        scores: dict[str, float] = {}
        token_f1 = lenient = 0.0
        perfect = 0
        for rid, item in items.items():
            record = item["record"]
            row = rows.get(rid)
            candidate = schema_utils.extract_json(row["text"]) if tally.take(row) else None
            sob_row = {
                "metadata": {
                    "record_id": rid,
                    "model_id": path.stem,
                    "difficulty": record["question_difficulty"],
                    "schema_complexity": record["schema_complexity"],
                },
                "input": {"json_schema": schema_utils.parse_if_string(record["json_schema"])},
                "output": {
                    "candidate_response": candidate,
                    "ground_truth": schema_utils.parse_if_string(record["ground_truth"]),
                },
            }
            metrics = evaluate.evaluate_record(sob_row, "text").row
            scores[rid] = metrics["leaf_value_em"]
            token_f1 += metrics["value_token_f1"]
            perfect += metrics["strict_json_em"]
            gold = evaluate.flatten_leaf_paths(sob_row["output"]["ground_truth"])
            pred = evaluate.flatten_leaf_paths(candidate) if isinstance(candidate, (dict, list)) else {}
            lenient += sum(1 for k, v in gold.items() if k in pred and loose_equal(pred[k], v)) / max(1, len(gold))
        n = len(items)
        arms[path.stem] = {
            "value_accuracy": sum(scores.values()) / n,
            "lenient_value_accuracy": lenient / n,
            "value_token_f1": token_f1 / n,
            "perfect_records": perfect / n,
            **tally.summary(n),
        }
        per_record[path.stem] = scores
    return finish({"set": set_name, "n": len(items), "arms": arms}, per_record)


def score_nhtsa(calls: Path, partial: bool = False) -> dict[str, Any]:
    items = items_for("nhtsa")
    if partial:
        items = {rid: item for rid, item in items.items() if rid in called_ids(calls / "nhtsa")}
    arms: dict[str, dict[str, Any]] = {}
    per_record: dict[str, dict[str, float]] = {}
    for path in sorted((calls / "nhtsa").glob("*.jsonl")):
        rows = scored_rows(path)
        tally = Tally()
        scores: dict[str, float] = {}
        field_right: dict[str, int] = {}
        field_n: dict[str, int] = {}
        for rid, item in items.items():
            gold = item["record"]["gold"]
            row = rows.get(rid)
            answer: Any = {}
            if tally.take(row):
                try:
                    answer = json.loads(row["text"])
                except json.JSONDecodeError:
                    answer = {}
            right = 0
            for key, value in gold.items():
                ok = nhtsa_equal(key, answer.get(key) if isinstance(answer, dict) else None, value)
                right += ok
                field_right[key] = field_right.get(key, 0) + ok
                field_n[key] = field_n.get(key, 0) + 1
            scores[rid] = right / len(gold)
        n = len(items)
        arms[path.stem] = {
            "value_accuracy": sum(scores.values()) / n,
            "field_accuracy": {key: right / field_n[key] for key, right in field_right.items()},
            "field_n": field_n,
            **tally.summary(n),
        }
        per_record[path.stem] = scores
    return finish({"set": "nhtsa", "n": len(items), "arms": arms}, per_record)


def finish(report: dict[str, Any], per_record: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Add SIE minus each arm with its bootstrap interval, and the price per 1M documents."""
    if SIE_FILE in per_record:
        base = per_record[SIE_FILE]
        ids = sorted(base)
        for stem, scores in per_record.items():
            if stem == SIE_FILE:
                continue
            # Each arm draws from its own seed-1 stream, so its interval does not depend on the other arms.
            rng = random.Random(BOOTSTRAP_SEED)
            diffs = [base[i] - scores.get(i, 0.0) for i in ids]
            mean = sum(diffs) / len(diffs)
            boots = sorted(
                sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP_DRAWS)
            )
            report["arms"][stem]["sie_minus_arm"] = {
                "mean": mean,
                "ci95_low": boots[int(BOOTSTRAP_DRAWS * 0.025) - 1],
                "ci95_high": boots[int(BOOTSTRAP_DRAWS * 0.975) - 1],
            }
    for stem, summary in report["arms"].items():
        model = model_of(stem)
        if model not in PRICES:
            continue
        p_in, p_out, p_cached = PRICES[model]
        fresh = summary["mean_tokens_in"] - summary["mean_tokens_cached"]
        per_doc = (fresh * p_in + summary["mean_tokens_cached"] * p_cached + summary["mean_tokens_out"] * p_out) / 1e6
        summary["usd_per_1m_documents_list"] = per_doc * DOCUMENTS
        summary["usd_per_1m_documents_batch"] = per_doc * DOCUMENTS * BATCH_DISCOUNT[model]
    return report


def canonical(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except json.JSONDecodeError:
        return text


def score_repeat(calls: Path) -> dict[str, Any]:
    """The first 100 e1 records sent again, one at a time: same answer twice, and how long it took."""
    arms: dict[str, dict[str, Any]] = {}
    for path in sorted((calls / "repeat").glob("*.jsonl")):
        again = {
            rid: row for rid, row in scored_rows(path).items() if row.get("error") is None and not row.get("rejected")
        }
        first_path = calls / "e1" / path.name
        first = scored_rows(first_path) if first_path.exists() else {}
        paired = {
            rid: row
            for rid, row in again.items()
            if rid in first and first[rid].get("error") is None and not first[rid].get("rejected")
        }
        same = [rid for rid, row in paired.items() if canonical(first[rid]["text"]) == canonical(row["text"])]
        totals = sorted(row["total_s"] for row in again.values() if row.get("total_s") is not None)
        ttft = sorted(row["ttft_s"] for row in again.values() if row.get("ttft_s") is not None)

        def pick(values: list[float], q: float) -> float | None:
            return values[min(len(values) - 1, int(q * len(values)))] if values else None

        arms[path.stem] = {
            "n": len(again),
            "paired_n": len(paired),
            "identical": len(same) / len(paired) if paired else None,
            "p50_total_s": pick(totals, 0.5),
            "p90_total_s": pick(totals, 0.9),
            "p50_ttft_s": pick(ttft, 0.5),
            "mean_tokens_out": sum(row["tokens_out"] for row in again.values()) / max(1, len(again)),
        }
    return {"set": "repeat", "arms": arms}


# --- report -----------------------------------------------------------------------------------------


def pct(value: float) -> str:
    return f"{100 * value:5.1f}%"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.0f}"


def print_accuracy(report: dict[str, Any], title: str, secondary: bool) -> None:
    print(f"{title}, {report['n']} records")
    header = f"{'model':<34} {'value acc':>9}"
    if secondary:
        header += f" {'lenient':>8} {'token F1':>8}"
    header += f" {'SIE minus model, 95% CI':>26} {'failed':>6} {'rejected':>8} {'tokens in/out':>14}"
    print(header + f" {'$/1M docs':>10} {'batch':>9}")
    rows = sorted(report["arms"].items(), key=lambda kv: (kv[0] != SIE_FILE, -kv[1]["value_accuracy"]))
    for stem, arm in rows:
        line = f"{label(stem):<34} {pct(arm['value_accuracy']):>9}"
        if secondary:
            line += f" {pct(arm['lenient_value_accuracy']):>8} {pct(arm['value_token_f1']):>8}"
        diff = arm.get("sie_minus_arm")
        ci = (
            f"{100 * diff['mean']:+5.1f} ({100 * diff['ci95_low']:+5.1f} to {100 * diff['ci95_high']:+5.1f})"
            if diff
            else "-"
        )
        answered = arm["answered"] > 0
        tokens = f"{arm['mean_tokens_in']:,.0f} / {arm['mean_tokens_out']:,.0f}" if answered else "-"
        line += f" {ci:>26} {arm['errors']:>6} {arm['schemas_rejected']:>8} {tokens:>14}"
        usd_list = arm.get("usd_per_1m_documents_list") if answered else None
        usd_batch = arm.get("usd_per_1m_documents_batch") if answered else None
        line += f" {money(usd_list):>10} {money(usd_batch):>9}"
        print(line)
    incomplete = [label(stem) for stem, arm in report["arms"].items() if arm["errors"]]
    if incomplete:
        print(f"  failed: records with no answer after retries, scored 0 ({', '.join(incomplete)})")
    rejected = [label(stem) for stem, arm in report["arms"].items() if arm["schemas_rejected"]]
    if rejected:
        print(f"  rejected: records whose schema the vendor API refused, scored 0 ({', '.join(rejected)})")
    print()


def print_nhtsa_fields(report: dict[str, Any]) -> None:
    fields = list(next(iter(report["arms"].values()))["field_n"])
    print("NHTSA, share right per field (records with that field in the gold):")
    counts = next(iter(report["arms"].values()))["field_n"]
    print(f"{'model':<34} " + " ".join(f"{f'{field} ({counts[field]})':>19}" for field in fields))
    for stem, arm in sorted(report["arms"].items(), key=lambda kv: (kv[0] != SIE_FILE, kv[0])):
        print(f"{label(stem):<34} " + " ".join(f"{pct(arm['field_accuracy'][field]):>19}" for field in fields))
    print()


def print_repeat(report: dict[str, Any]) -> None:
    print("Repeat: the first 100 e1 records sent again, one request at a time")
    print(f"{'model':<34} {'answered':>8} {'same answer':>11} {'p50 s':>7} {'p90 s':>7} {'p50 first token s':>18}")
    for stem, arm in sorted(report["arms"].items(), key=lambda kv: (kv[0] != SIE_FILE, kv[0])):

        def seconds(value: float | None) -> str:
            return "-" if value is None else f"{value:.2f}"

        identical = "n/a" if arm["identical"] is None else pct(arm["identical"])
        print(
            f"{label(stem):<34} {arm['n']:>8} {identical:>11} {seconds(arm['p50_total_s']):>7} "
            f"{seconds(arm['p90_total_s']):>7} {seconds(arm['p50_ttft_s']):>18}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=Path, help="score calls from here (<set>/<model>.jsonl) instead")
    parser.add_argument("--json", type=Path, help="also write every figure to this JSON file")
    args = parser.parse_args()

    manifest = verify_evidence()
    calls = args.calls or EVIDENCE / "calls"
    if args.calls is None:
        print(f"Recorded run, {', '.join(manifest['run_dates'])}\n")
    # Your own calls may cover part of a set (run.py --limit); they are scored on the records they cover.
    partial = args.calls is not None
    reports = {}
    if (calls / "e1").is_dir():
        reports["e1"] = score_sob("e1", calls, partial)
        print_accuracy(reports["e1"], "E1: Structured Output Benchmark, SOB's leaf_value_em", secondary=True)
    if (calls / "nhtsa").is_dir():
        reports["nhtsa"] = score_nhtsa(calls, partial)
        print_accuracy(reports["nhtsa"], "NHTSA complaints filed from 2026-09-01, fields right", secondary=False)
        print_nhtsa_fields(reports["nhtsa"])
    if (calls / "repeat").is_dir():
        reports["repeat"] = score_repeat(calls)
        print_repeat(reports["repeat"])
    if not reports:
        raise SystemExit(f"no e1/, nhtsa/ or repeat/ folder of calls under {calls}")
    print("value acc: mean share of gold values returned exactly; failed and rejected records score 0")
    print("$/1M docs: list price for a million documents at each model's mean recorded tokens; batch: vendor batch API")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
