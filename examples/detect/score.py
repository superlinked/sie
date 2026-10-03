#!/usr/bin/env python3
"""Replay the recorded RF100-VL detection study without sending model requests."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
ARMS = {
    "sie-owlv2-base": ("sie-owlv2-base", 0.1),
    "gpt-6-luna": ("gpt-6-luna@pixels", 0.0),
    "gpt-5.4-mini": ("gpt-5.4-mini@pixels", 0.0),
    "claude-haiku-4-5": ("claude-haiku-4-5@pixels", 0.0),
    "owlv2-large": ("owlv2-large-fast", 0.1),
    "owlv2-base-transformers": ("owlv2-base-fast", 0.1),
    "grounding-dino-base": ("grounding-dino-base", 0.25),
    "llmdet-large": ("llmdet-large", 0.25),
}
PUBLISHED_AP = {"sie-owlv2-base": 11.1, "gpt-6-luna": 13.3, "gpt-5.4-mini": 5.4, "claude-haiku-4-5": 0.9}
PUBLISHED_PRICE = {"sie-owlv2-base": 0.24, "gpt-6-luna": 0.15, "gpt-5.4-mini": 1.33, "claude-haiku-4-5": 1.95}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def near(actual: float, expected: float, message: str) -> None:
    require(abs(actual - expected) < 1e-9, f"{message}: {actual} != {expected}")


def read(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def rows(stem: str) -> list[dict]:
    with gzip.open(EVIDENCE / "rows" / f"{stem}.jsonl.gz", "rt") as handle:
        return [json.loads(line) for line in handle]


def coco_score(dataset: str, records: list[dict], threshold: float) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = read(f"annotations/{dataset}.json")
    labels = {row["name"]: row["id"] for row in truth["categories"]}
    detections = [
        {"image_id": row["image_id"], "category_id": labels[d["label"]], "bbox": d["bbox"], "score": d["score"]}
        for row in records
        for d in row["detections"]
        if d["score"] >= threshold and d["label"] in labels
    ]
    if not detections:
        return {"ap": 0.0, "ap50": 0.0, "n_dets": 0, "n_gt": len(truth["annotations"])}
    with contextlib.redirect_stdout(io.StringIO()):
        gold = COCO()
        gold.dataset = truth
        gold.createIndex()
        predicted = gold.loadRes(detections)
        evaluator = COCOeval(gold, predicted, "bbox")
        evaluator.params.imgIds = sorted(row["image_id"] for row in records)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap": max(float(evaluator.stats[0]), 0.0),
        "ap50": max(float(evaluator.stats[1]), 0.0),
        "n_dets": len(detections),
        "n_gt": len(truth["annotations"]),
    }


def bootstrap(left: list[float], right: list[float]) -> dict:
    generator = random.Random(20260930)
    difference = [a - b for a, b in zip(left, right, strict=True)]
    count = len(difference)
    means = sorted(sum(difference[generator.randrange(count)] for _ in range(count)) / count for _ in range(10_000))
    return {"diff": sum(difference) / count, "low": means[250], "high": means[9749]}


def iou(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union else 0.0


def hits(detections: list[dict], truth: list[dict], names: dict[int, str]) -> int:
    used = set()
    for box in sorted(detections, key=lambda row: -row.get("score", 1.0)):
        best, overlap = None, 0.5
        for index, gold in enumerate(truth):
            if index in used or names[gold["category_id"]] != box["label"]:
                continue
            score = iou(box["bbox"], gold["bbox"])
            if score >= overlap:
                best, overlap = index, score
        if best is not None:
            used.add(best)
    return len(used)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Check manifest, recorded score means, usage and published figures; skip COCO and bootstrap replay",
    )
    args = parser.parse_args()
    manifest = read("manifest.json")
    require(
        hashlib.sha256((EVIDENCE / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA256,
        "Manifest differs from the example pin; fetch its immutable revision again",
    )
    for name, metadata in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe manifest path: {name}")
        data = (EVIDENCE / relative).read_bytes()
        require(
            len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"],
            f"File digest mismatch: {name}",
        )
    samples = read("samples.json")
    identities = {(r["dataset"], r["image_id"]) for r in samples}
    require(len(samples) == len(identities) == 2895, "Sample set changed or contains duplicate identities")
    datasets = sorted({r["dataset"] for r in samples})
    require(len(datasets) == 100, "Dataset count changed")
    page, scores, protocol = read("page.json"), read("scores.json"), read("protocol.json")
    for arm, (stem, threshold) in ARMS.items():
        records = rows(stem)
        actual_ids = {(r["dataset"], r["image_id"]) for r in records}
        require(
            len(records) == len(actual_ids) == 2895 and actual_ids == identities, f"{arm}: incomplete or duplicate rows"
        )
        require(not any(r.get("error") for r in records), f"{arm}: recorded transport failures")
        for dataset in datasets:
            expected = scores[stem]["shipped"][dataset]
            if not args.summary_only:
                actual = coco_score(dataset, [r for r in records if r["dataset"] == dataset], threshold)
                for metric in ["ap", "ap50", "n_dets", "n_gt"]:
                    near(actual[metric], expected[metric], f"{arm}/{dataset}/{metric}")
            for metric in ["ap", "ap50"]:
                near(
                    expected[metric], page["arms"][arm]["perDataset"][dataset][metric], f"{arm}/{dataset}/{metric} page"
                )
        for metric in ["ap", "ap50"]:
            mean = sum(scores[stem]["shipped"][d][metric] for d in datasets) / 100
            near(mean, page["arms"][arm][metric], f"{arm}/{metric} mean")
        if arm in PUBLISHED_AP:
            near(round(page["arms"][arm]["ap"] * 100, 1), PUBLISHED_AP[arm], f"{arm} published AP")
        if arm in protocol["llm_prices_per_million_tokens"]:
            pin, pout = protocol["llm_prices_per_million_tokens"][arm]
            for record in records:
                for field in ("tokens_in", "tokens_out"):
                    value = record.get(field)
                    require(
                        type(value) is int and value > 0,
                        f"{arm}: missing or invalid {field} for {record['dataset']}/{record['image_id']}",
                    )
            tin, tout = sum(r["tokens_in"] for r in records), sum(r["tokens_out"] for r in records)
            near(tin, scores[stem]["tokens_in"], f"{arm} input tokens")
            near(tout, scores[stem]["tokens_out"], f"{arm} output tokens")
            price = (tin * pin + tout * pout) / 1e6 / 2895 * 1000
            near(price, page["arms"][arm]["usdPer1k"]["standard"], f"{arm} recorded price")
            near(price / 2, page["arms"][arm]["usdPer1k"]["batch"], f"{arm} recorded Batch price")
        else:
            price = protocol["sie_usd_per_1000_images"] if arm == "sie-owlv2-base" else None
        if arm in PUBLISHED_PRICE:
            rounded = (math.ceil(price * 100) if arm == "sie-owlv2-base" else math.floor(price * 100)) / 100
            near(rounded, PUBLISHED_PRICE[arm], f"{arm} published price")
        print(f"{arm}: AP {page['arms'][arm]['ap'] * 100:.1f}%, AP50 {page['arms'][arm]['ap50'] * 100:.1f}%")
    for shown in page["shown"]:
        truth = read(f"annotations/{shown['dataset']}.json")
        gold = [r for r in truth["annotations"] if r["image_id"] == shown["imageId"]]
        labels = {r["id"]: r["name"] for r in truth["categories"]}
        for key, arm in [("ours", "sie-owlv2-base"), ("detections", shown["arm"])]:
            stem, threshold = ARMS[arm]
            record = next(
                r for r in rows(stem) if r["dataset"] == shown["dataset"] and r["image_id"] == shown["imageId"]
            )
            boxes = [r for r in record["detections"] if r["score"] >= threshold]
            displayed = [
                {
                    "label": b["label"],
                    "bbox": [round(v, 1) for v in b["bbox"]],
                    **({"score": round(b["score"], 3)} if arm == "sie-owlv2-base" else {}),
                }
                for b in boxes
            ]
            require(displayed == shown[key], f"Displayed {key} boxes changed: {shown['dataset']}/{shown['imageId']}")
            count = hits(boxes, gold, labels)
            near(count, shown["oursHits" if key == "ours" else "hits"], f"Displayed {key} hits")
        near(len(gold), shown["truth"], "Displayed annotated-object count")
    for comparison, expected in page["comparisons"].items():
        left, right = comparison.split(" - ")
        for metric in ["ap", "ap50"]:
            a = [page["arms"][left]["perDataset"][d][metric] for d in datasets]
            b = [page["arms"][right]["perDataset"][d][metric] for d in datasets]
            if not args.summary_only:
                actual = bootstrap(a, b)
                for name in ["diff", "low", "high"]:
                    near(actual[name], expected[metric][name], f"{comparison}/{metric}/{name}")
    for rival in ["gpt-5.4-mini", "claude-haiku-4-5"]:
        for metric in ["ap", "ap50"]:
            require(
                page["comparisons"][f"sie-owlv2-base - {rival}"][metric]["low"] > 0,
                f"Unsupported superiority over {rival}",
            )
    require(page["arms"]["gpt-6-luna"]["ap"] > page["arms"]["sie-owlv2-base"]["ap"], "Luna limitation changed")
    with gzip.open(EVIDENCE / "latency-rows.jsonl.gz", "rt") as handle:
        latency_rows = [json.loads(line) for line in handle]
    selected = random.Random(20260930).sample(samples, 205)
    expected_ids = [(r["dataset"], r["image_id"]) for r in selected]
    summary = read("latency-summary.json")
    paired = {}
    for arm in ["sie-owlv2-base", "gpt-6-luna", "gpt-5.4-mini"]:
        records = [r for r in latency_rows if r["arm"] == arm]
        require([(r["dataset"], r["image_id"]) for r in records] == expected_ids, f"{arm}: latency sample differs")
        require([r["warmup"] for r in records] == [True] * 5 + [False] * 200, "Latency warmups changed")
        require(all(r["attempts"] == 1 for r in records), "Latency failure count changed")
        if arm != "sie-owlv2-base":
            for record in records:
                for field in ("tokens_in", "tokens_out"):
                    require(type(record.get(field)) is int and record[field] > 0, f"{arm}: invalid latency {field}")
        measured = records[5:]
        seconds = sorted(r["seconds"] for r in measured)
        near(statistics.median(seconds), summary["arms"][arm]["p50_seconds"], f"{arm} latency median")
        near(seconds[179], summary["arms"][arm]["p90_seconds"], f"{arm} latency p90")
        near(sum(r["billed_usd"] for r in measured), summary["arms"][arm]["billed_usd"], f"{arm} latency cost")
        paired[arm] = seconds if args.summary_only else [r["seconds"] for r in measured]
    ratio = statistics.median(paired["sie-owlv2-base"]) / statistics.median(paired["gpt-6-luna"])
    near(ratio, summary["sie_to_luna_p50_ratio"], "Latency median ratio")
    if not args.summary_only:
        rng = random.Random(20260930)
        ratios = []
        for _ in range(10000):
            positions = rng.choices(range(200), k=200)
            ratios.append(
                statistics.median(paired["sie-owlv2-base"][i] for i in positions)
                / statistics.median(paired["gpt-6-luna"][i] for i in positions)
            )
        ratios.sort()
        for actual, expected in zip([ratios[250], ratios[9749]], summary["paired_ratio_ci95"]):
            near(actual, expected, "Paired latency ratio interval")
    passed = ratio <= 1 / 3 and summary["paired_ratio_ci95"][1] < 0.5
    require(
        not passed and summary["strong_speed_bar_passed"] is False and page["latency"]["passed"] is False,
        "Registered speed claim changed",
    )
    print("Verified paired latency records; registered threefold speed bar did not pass.")
    print("Verified 2,895 images across 100 datasets; every published accuracy and price matches.")
    return 0


MANIFEST_SHA256 = "4ea377268746f0b19cb29ab1407f4ccdeb120c77c9218db6a7a0ed7528a312b4"
if __name__ == "__main__":
    raise SystemExit(main())
