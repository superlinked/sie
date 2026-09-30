#!/usr/bin/env python3
"""Re-derive every published figure of the custom entity types study from the recorded rows.

    python3 fetch.py
    python3 score.py
    python3 score.py --rows run-output      # score your own run.py output as the SIE arm

Standard library only, no network. It reads evidence/ (from fetch.py), scores every arm on exact
(start, end, type) spans over the 652 test sentences, recomputes the paired bootstrap against the SIE
arm, the strings the LLMs returned that are not in the text, and the price of each arm per million
characters. Then it compares every figure with the study's own results.json and exits non-zero on any
difference, a file whose digest does not match the manifest, or a row that is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
SEED = 20260930
BOOTSTRAP = 2000
TOLERANCE = 1e-9
OURS = "sie-gliner-biomed-large"


def fail(message: str) -> None:
    raise SystemExit(f"score.py: {message}")


def load_json(relative: str):  # noqa: ANN201 - JSON of any shape
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_digests(manifest: dict) -> None:
    for relative, digest in manifest["files"].items():
        path = EVIDENCE / relative
        if not path.exists():
            fail(f"{relative} is missing; run fetch.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail(f"{relative} does not match the SHA-256 the manifest pins")


def occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Every place a returned string occurs, starting and ending on a token boundary."""
    spans = []
    start = text.find(needle) if needle else -1
    while start != -1:
        end = start + len(needle)
        if (start == 0 or text[start - 1] == " ") and (end == len(text) or text[end] == " "):
            spans.append((start, end))
        start = text.find(needle, start + 1)
    return spans


def predictions(text: str, output: str, by_label: dict, type_map: dict | None) -> tuple[set, int]:
    """(predicted spans as gold types, strings not found in the text) for one sentence."""
    try:
        payload = json.loads(output) if output else {"entities": []}
    except json.JSONDecodeError:
        payload = {"entities": []}
    spans: set[tuple[int, int, str]] = set()
    ungrounded = 0
    for ent in payload.get("entities") or []:
        if "start" in ent:  # an arm that returns offsets
            gold_type = type_map.get(ent["label"]) if type_map is not None else by_label.get(ent["label"])
            if gold_type is not None:
                spans.add((ent["start"], ent["end"], gold_type))
            continue
        gold_type = by_label.get(str(ent.get("type", "")))
        found = occurrences(text, str(ent.get("text", "")))
        if not found:
            ungrounded += 1
        if gold_type is not None:
            spans.update((s, e, gold_type) for s, e in found)
    return spans, ungrounded


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def counts(arm: dict, data: dict, labels: dict, rows_dir: Path, type_maps: dict | None) -> dict:
    """Per set, per sentence: (tp, fp, fn, ungrounded); an ungrounded string counts as a false positive."""
    out = {}
    for set_name, sentences in data.items():
        path = rows_dir / f"{set_name}.jsonl"
        if not path.exists():
            fail(f"{arm['id']}: no rows for {set_name}")
        records = {r["id"]: r for r in load_jsonl(path)}
        by_label = {v: k for k, v in labels[set_name].items()}
        rows = []
        for sentence in sentences:
            record = records.get(sentence["id"])
            if record is None:
                fail(f"{arm['id']}: no row for {sentence['id']}")
            type_map = type_maps.get(set_name) if type_maps is not None else None
            preds, ungrounded = predictions(sentence["text"], record["output"], by_label, type_map)
            gold = {tuple(g) for g in sentence["entities"]}
            tp = len(preds & gold)
            rows.append((tp, len(preds) - tp + ungrounded, len(gold) - tp, ungrounded))
        out[set_name] = rows
    return out


def pooled_f1(sample: list[tuple[str, int]], c: dict) -> float:
    tp = sum(c[s][i][0] for s, i in sample)
    fp = sum(c[s][i][1] for s, i in sample)
    fn = sum(c[s][i][2] for s, i in sample)
    return prf(tp, fp, fn)["f1"]


def paired(a: dict, b: dict) -> dict:
    """a - b in pooled F1, with a sentence bootstrap inside each set, the same draws for both arms."""
    rng = random.Random(SEED)
    keys = [(s, i) for s in a for i in range(len(a[s]))]
    by_set = {s: list(range(len(a[s]))) for s in a}
    diffs = []
    for _ in range(BOOTSTRAP):
        sample = [(s, idx[rng.randrange(len(idx))]) for s, idx in by_set.items() for _ in idx]
        diffs.append(pooled_f1(sample, a) - pooled_f1(sample, b))
    diffs.sort()
    return {"diff": pooled_f1(keys, a) - pooled_f1(keys, b),
            "ci95": [diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]]}  # fmt: skip


def usd_per_1m_chars(arm: dict, data: dict, rows_dir: Path, chars: int, tokens: dict) -> dict | None:
    """Real-time price per million characters of the sample, and the results.json figure it must equal.

    Every arm is priced at a rate a synchronous request pays. The LLMs are at their list price per token;
    their batch tiers are not used. Comprehend is at its cheapest volume tier, which is a real-time rate.
    """
    price = arm.get("price")
    if price is None:
        return None
    if price["unit"] == "sie_input_tokens":
        usd = tokens[arm["model"]] * price["usd_per_1m"] / 1e6
        return {"realtime": usd / chars * 1e6, "published_key": "list"}
    if price["unit"] == "characters":
        return {"realtime": price["cheapest_usd_per_100_chars"] / 100 * 1e6, "published_key": "cheapest"}
    tin = tout = 0
    for set_name in data:
        for record in load_jsonl(rows_dir / f"{set_name}.jsonl"):
            tin += record["tokens_in"]
            tout += record["tokens_out"]
    usd = (tin * price["usd_per_1m_in"] + tout * price["usd_per_1m_out"]) / 1e6
    output_only = tout * price["usd_per_1m_out"] / 1e6
    return {"realtime": usd / chars * 1e6, "published_key": "list", "output_only": output_only / chars * 1e6}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, help="score this directory of <set>.jsonl rows as the SIE arm")
    args = parser.parse_args()
    if not EVIDENCE.exists():
        fail("no evidence/; run fetch.py first")
    manifest = load_json("manifest.json")
    check_digests(manifest)
    results = load_json("results.json")
    labels = results["label_strings"]
    sets = list(results["sets"])
    data = {s: load_jsonl(EVIDENCE / "data" / "main" / f"{s}.jsonl") for s in sets}
    chars = sum(len(r["text"]) for rows in data.values() for r in rows)
    gold = sum(len(r["entities"]) for rows in data.values() for r in rows)
    tokens = load_json("sie_tokens.json")
    if tokens["chars"] != chars:
        fail(f"the sample has {chars} characters, sie_tokens.json counted {tokens['chars']}")

    arms = manifest["arms"]
    scored: dict[str, dict] = {}
    costs: dict[str, dict | None] = {}
    mismatches = []
    print(f"{len(sets)} sets, {sum(len(v) for v in data.values())} sentences, {gold} gold entities\n")
    print(f"{'arm':34} {'F1':>6} {'P':>6} {'R':>6} {'macro':>6} {'not in text':>11} {'$ per 1M chars, real-time':>26}")
    for arm in arms:
        rows_dir = EVIDENCE / "rows" / arm["id"]
        if args.rows is not None and arm["id"] == OURS:
            rows_dir = args.rows
        type_maps = load_json(arm["type_map"]) if arm.get("type_map") else None
        if arm.get("type_map"):
            digest = (EVIDENCE / (arm["type_map"] + ".sha256")).read_text().strip()
            if hashlib.sha256((EVIDENCE / arm["type_map"]).read_bytes()).hexdigest() != digest:
                fail(f"{arm['type_map']} changed after it was frozen")
        c = counts(arm, data, labels, rows_dir, type_maps)
        scored[arm["id"]] = c
        pooled = prf(*(sum(x[i] for rows in c.values() for x in rows) for i in range(3)))
        per_set = {s: prf(*(sum(x[i] for x in rows) for i in range(3)))["f1"] for s, rows in c.items()}
        macro = sum(per_set.values()) / len(per_set)
        ungrounded = sum(x[3] for rows in c.values() for x in rows)
        cost = usd_per_1m_chars(arm, data, rows_dir, chars, tokens)
        cost_text = f"{cost['realtime']:.4f}" if cost else "self-hosted"
        costs[arm["id"]] = cost
        print(f"{arm['name']:34} {pooled['f1']:6.3f} {pooled['precision']:6.3f} {pooled['recall']:6.3f} "
              f"{macro:6.3f} {ungrounded:11d} {cost_text:>26}")  # fmt: skip
        if args.rows is not None and arm["id"] == OURS:
            continue
        published = results["arms"][arm["study_arm"]]
        for key, value in (("f1", pooled["f1"]), ("precision", pooled["precision"]),
                           ("recall", pooled["recall"]), ("macro_f1", macro)):  # fmt: skip
            if not close(value, published[key]):
                mismatches.append(f"{arm['name']} {key}: {value} here, {published[key]} published")
        if ungrounded != published["ungrounded"]:
            mismatches.append(
                f"{arm['name']} strings not in the text: {ungrounded}, published {published['ungrounded']}"
            )
        if cost is not None:
            want = published["usd_per_1m_chars"][cost["published_key"]]
            if not close(cost["realtime"], want):
                mismatches.append(f"{arm['name']} $ per 1M chars: {cost['realtime']} here, {want} published")

    base = costs[OURS]["realtime"]
    print("\nReal-time price as a multiple of SIE GLiNER BioMed Large's:")
    for arm in arms:
        cost = costs[arm["id"]]
        if arm["id"] == OURS or cost is None:
            continue
        line = f"  {arm['name']:32} {cost['realtime'] / base:7.1f}x"
        if "output_only" in cost:
            line += f"   output tokens alone {cost['output_only'] / base:.1f}x"
        print(line)

    ours = scored[OURS]
    print(f"\nSIE GLiNER BioMed Large minus each arm, pooled F1 points, 95% interval ({BOOTSTRAP} paired resamples):")
    by_study = {arm["id"]: arm["study_arm"] for arm in arms}
    published_pairs = results[f"paired_vs_{by_study[OURS]}"]
    for arm in arms:
        if arm["id"] == OURS:
            continue
        pair = paired(ours, scored[arm["id"]])
        print(
            f"  {arm['name']:32} {100 * pair['diff']:+6.1f}  [{100 * pair['ci95'][0]:+.1f}, {100 * pair['ci95'][1]:+.1f}]"
        )
        if args.rows is None:
            want = published_pairs[arm["study_arm"]]
            if not (close(pair["diff"], want["diff"]) and all(close(x, y) for x, y in zip(pair["ci95"], want["ci95"]))):
                mismatches.append(f"paired vs {arm['name']}: {pair} here, {want} published")

    if args.rows is not None:
        print("\nScored your rows as the SIE arm; nothing is compared with the published figures.")
        return 0
    if mismatches:
        print("\nDoes not match the published figures:", file=sys.stderr)
        for line in mismatches:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nEvery figure matches the study's results.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
