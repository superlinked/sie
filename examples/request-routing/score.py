#!/usr/bin/env python3
"""Verify the fixed paired 3,000-request recording offline.

    uv run python score.py --evidence evidence

Checks manifest hashes first, reconstructs the frozen front and conditional
answers, and compares quality, paired intervals, OOS and listed cost bounds to
results.json. No keys, network, model fitting or fresh-run scoring. The recording
is one serving draw; it does not prove repeatability, latency, actual invoices,
managed settlement or self-hosted margins. See METHOD.md for the fixed method.
The embedding upper bound rounds each item as an individual API request;
generation rounds input/output together per attempt, including failure bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ("clinc150", "banking77", "massive")
STRATA = ("clinc150_in_scope", "clinc150_oos", "banking77", "massive")
COUNTS = dict(zip(STRATA, (818, 182, 1000, 1000)))
POPULATION = dict(zip(STRATA, (4500, 1000, 3080, 2974)))
ARMS = ("C", "E5", "E6")
NONE = "none of these"
THRESHOLD = 0.29
FILES = {
    "assets.json",
    "inputs.jsonl",
    "front-clinc150.npz",
    "front-banking77.npz",
    "front-massive.npz",
    "vectors.npz",
    "records.jsonl",
    "results.json",
    "prices.json",
    "METHOD.md",
}


class EvidenceError(ValueError):
    """A fixed, content-free verification failure."""


def require(ok, code):
    if not ok:
        raise EvidenceError(code)


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identifier(row):
    require(isinstance(row, dict) and row.get("dataset") in DATASETS and isinstance(row.get("id"), str), "row_id")
    require(bool(row["id"]), "row_id")
    return row["dataset"] + "|" + row["id"]


def token(value):
    require(type(value) is int and value >= 0, "token_count")
    return value


def verify_manifest(directory):
    require(directory.is_dir() and not directory.is_symlink(), "evidence_directory")
    marker = directory / "manifest.json"
    require(marker.is_file() and not marker.is_symlink(), "manifest_required")
    manifest = load(marker)
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "manifest_version")
    require(set(manifest["files"]) == FILES, "manifest_files")
    require({path.name for path in directory.iterdir()} == FILES | {"manifest.json"}, "evidence_files")
    for name, expected in manifest["files"].items():
        require(set(expected) == {"sha256", "size"}, "manifest_entry")
        require(
            isinstance(expected["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]), "manifest_digest"
        )
        require(type(expected["size"]) is int and expected["size"] >= 0, "manifest_size")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "evidence_file")
        content = path.read_bytes()
        require(
            len(content) == expected["size"] and hashlib.sha256(content).hexdigest() == expected["sha256"],
            "file_digest",
        )


def front_scores(vectors, model, labels):
    routes = [label for label in labels if label != NONE]
    require(len(routes) >= 10 and model["labels"].tolist() == routes, "front_labels")
    classes = model["classes"]
    require(classes.ndim == 1 and np.issubdtype(classes.dtype, np.integer), "front_classes")
    require(
        len(set(classes)) == len(classes) and bool(((classes >= 0) & (classes < len(routes))).all()), "front_classes"
    )
    require(model["coef"].shape == (len(classes), 2560) and model["intercept"].shape == (len(classes),), "front_shape")
    require(bool(np.isfinite(model["coef"]).all() and np.isfinite(model["intercept"]).all()), "front_nonfinite")
    query = vectors.astype(np.float16).astype(np.float32)
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    require(bool(np.isfinite(query).all() and (norm > 0).all()), "vectors_invalid")
    query /= norm
    logits = query @ model["coef"].T + model["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    scores = np.zeros((len(query), len(routes)), np.float32)
    scores[:, classes] = exp / exp.sum(axis=1, keepdims=True)
    require(
        bool(np.isfinite(scores).all() and np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0)), "front_probabilities"
    )
    return scores


def weighted(values, indices, target="route3"):
    means = [np.asarray(values)[indices[stratum]].mean(axis=0) for stratum in STRATA]
    weights = (
        (Fraction(3, 11), Fraction(2, 33), Fraction(1, 3), Fraction(1, 3))
        if target == "route3"
        else tuple(Fraction(POPULATION[s], 11554) for s in STRATA)
    )
    return sum(float(weight) * mean for weight, mean in zip(weights, means))


def statistics(rows, gold, predictions, kept, costs):
    indices = {s: np.array([i for i, row in enumerate(rows) if row["stratum"] == s]) for s in STRATA}
    require({s: len(indices[s]) for s in STRATA} == COUNTS, "analysis_strata")
    correct = np.array([[predictions[arm][i] == gold[i] for arm in ARMS] for i in range(len(rows))], dtype=float)
    point = weighted(correct, indices)
    rng = np.random.default_rng(20260930)
    boots, per_set_boots = np.empty((2000, 3)), np.empty((2000, 3, 3))
    oos_precision, oos_recall = [], []

    def oos(ix):
        ins, out = ix["clinc150_in_scope"], ix["clinc150_oos"]
        tp = np.mean([predictions["C"][i] == NONE for i in out])
        fp = np.mean([predictions["C"][i] == NONE for i in ins])
        denominator = 2 / 11 * tp + 9 / 11 * fp
        return tp, (2 / 11 * tp / denominator) if denominator else None

    for b in range(2000):
        resample = {s: rng.choice(indices[s], size=len(indices[s]), replace=True) for s in STRATA}
        boots[b] = weighted(correct, resample)
        per_set_boots[b, 0] = 9 / 11 * correct[resample[STRATA[0]]].mean(axis=0) + 2 / 11 * correct[
            resample[STRATA[1]]
        ].mean(axis=0)
        per_set_boots[b, 1] = correct[resample[STRATA[2]]].mean(axis=0)
        per_set_boots[b, 2] = correct[resample[STRATA[3]]].mean(axis=0)
        recall, precision = oos(resample)
        oos_recall.append(recall)
        if precision is not None:
            oos_precision.append(precision)

    def interval(values):
        return [float(x) for x in np.percentile(values, (2.5, 97.5))]

    share = float(weighted(np.asarray(kept, dtype=float), indices, "population"))
    per_set = {
        arm: {
            "clinc150": float(
                9 / 11 * correct[indices[STRATA[0]], i].mean() + 2 / 11 * correct[indices[STRATA[1]], i].mean()
            ),
            "banking77": float(correct[indices[STRATA[2]], i].mean()),
            "massive": float(correct[indices[STRATA[3]], i].mean()),
        }
        for i, arm in enumerate(ARMS)
    }
    paired = {}
    for j, rival in enumerate(ARMS[1:], 1):
        bounds = interval(boots[:, 0] - boots[:, j])
        paired[rival] = {
            "difference": float(point[0] - point[j]),
            "ci95": bounds,
            "R5_point_pass": all(per_set["C"][dataset] - per_set[rival][dataset] >= -0.02 for dataset in DATASETS),
            "R2_accuracy_front_pass": bounds[0] >= -0.01 and share >= 0.5,
        }
    both = all(value["R2_accuracy_front_pass"] and value["R5_point_pass"] for value in paired.values())
    for value in paired.values():
        value["wording"] = "no more than one percentage point worse" if both else "descriptive-only"
    recall, precision = oos(indices)
    estimates = {arm: float(weighted(np.array([float(v) for v in costs[arm]]), indices, "population")) for arm in ARMS}
    selected = [{"dataset": row["dataset"], "stratum": row["stratum"], "id": row["id"]} for row in rows]
    return {
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "sample_n": len(rows),
        "sample_counts": COUNTS,
        "original_population_counts": POPULATION,
        "route3": {
            arm: {
                "estimate": float(point[i]),
                "ci95": interval(boots[:, i]),
                "per_set": per_set[arm],
                "per_set_ci95": {dataset: interval(per_set_boots[:, j, i]) for j, dataset in enumerate(DATASETS)},
            }
            for i, arm in enumerate(ARMS)
        },
        "paired": paired,
        "frontshare_population_weighted": share,
        "R2_both_pass": both,
        "oos": {
            "recall": recall,
            "recall_ci95": interval(oos_recall),
            "precision_population_weighted": precision,
            "precision_ci95": interval(oos_precision) if oos_precision else None,
            "gold_oos_n": len(indices[STRATA[1]]),
            "predicted_oos_sample_n": sum(
                predictions["C"][i] == NONE for i, row in enumerate(rows) if row["dataset"] == "clinc150"
            ),
            "undefined_precision_bootstraps": 2000 - len(oos_precision),
            "R3_point_pass": bool(recall >= 0.8 and precision is not None and precision >= 0.8),
        },
        "listed_cost_bounds_usd_per_request": {
            "C_uncached_upper": estimates["C"],
            "E5_cheapest_theoretical_lower": estimates["E5"],
            "E6_cheapest_theoretical_lower": estimates["E6"],
        },
        "conservative_savings_pass": {rival: estimates["C"] < estimates[rival] for rival in ARMS[1:]},
    }


def listed_upper(embedding, prompt=0, output=0):
    return Fraction(
        math.ceil(Fraction(3 * token(embedding), 500))
        + math.ceil(Fraction(token(prompt), 40) + Fraction(token(output), 5)),
        100000,
    )


def rival_lower(row, model, rates):
    incoming, outgoing, cached = token(row["in"]), token(row["out"]), token(row["cached"])
    written = token(row["cache_write"])
    if model == "claude-sonnet-5":
        incoming += cached + written
    else:
        require(cached <= incoming, "rival_cached_gt_input")
    price_in, price_out, price_cache, _ = [Fraction(str(value)) for value in rates[model]]
    return (
        min((incoming * price_in + outgoing * price_out) / 2, incoming * price_cache + outgoing * price_out) / 1000000
    )


def compare(actual, recorded):
    if isinstance(actual, dict):
        require(isinstance(recorded, dict) and set(recorded) == set(actual), "results_fields")
        for key, value in actual.items():
            compare(value, recorded[key])
    elif isinstance(actual, list):
        require(isinstance(recorded, list) and len(recorded) == len(actual), "results_fields")
        for value, expected in zip(actual, recorded):
            compare(value, expected)
    elif isinstance(actual, (float, np.floating)):
        require(
            type(recorded) in (int, float)
            and math.isfinite(recorded)
            and math.isclose(float(actual), recorded, rel_tol=1e-9, abs_tol=1e-12),
            "results_drift",
        )
    else:
        require(type(recorded) is type(actual) and recorded == actual, "results_drift")


def recorded_inputs(directory):
    assets, inputs, rows = (
        load(directory / "assets.json"),
        jsonl(directory / "inputs.jsonl"),
        jsonl(directory / "records.jsonl"),
    )
    require(set(assets) == set(DATASETS), "asset_datasets")
    require(len(inputs) == len(rows) == 3000, "paired_coverage")
    ids = [identifier(row) for row in inputs]
    require(len(set(ids)) == 3000 and [identifier(row) for row in rows] == ids, "paired_order")
    require(
        all(set(row) == {"dataset", "id", "text"} and isinstance(row["text"], str) for row in inputs), "input_fields"
    )
    require(all(sum(row["dataset"] == dataset for row in inputs) == 1000 for dataset in DATASETS), "set_counts")
    with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"ids", "vectors"}, "vector_fields")
        require(archive["ids"].tolist() == ids, "vector_order")
        vectors = archive["vectors"].copy()
    require(vectors.dtype == np.float32 and vectors.shape == (3000, 2560), "vector_shape")
    require(
        bool(np.isfinite(vectors).all() and (np.abs(np.linalg.norm(vectors, axis=1) - 1) <= 0.01).all()),
        "vectors_invalid",
    )
    for dataset in DATASETS:
        data = assets[dataset]
        require(set(data) == {"labels", "examples"}, "asset_fields")
        labels = data["labels"]
        require(
            isinstance(labels, list)
            and all(isinstance(label, str) and label for label in labels)
            and len(set(labels)) == len(labels),
            "asset_labels",
        )
        require(set(data["examples"]) == set(labels) - {NONE}, "asset_examples")
        require(
            all(
                isinstance(examples, list)
                and 1 <= len(examples) <= 10
                and all(isinstance(text, str) for text in examples)
                for examples in data["examples"].values()
            ),
            "asset_examples",
        )
        indices = [i for i, row in enumerate(inputs) if row["dataset"] == dataset]
        with np.load(directory / f"front-{dataset}.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"coef", "intercept", "classes", "labels"}, "front_fields")
            model = dict(archive)
        scores = front_scores(vectors[indices], model, labels)
        order = np.argsort(-scores, axis=1)[:, :10]
        for index, score, ranks in zip(indices, scores, order):
            recorded = rows[index]["front"]
            top10 = [str(model["labels"][i]) for i in ranks]
            confidence = float(score.max())
            require(set(recorded) == {"pred", "conf", "top10"}, "recorded_front_fields")
            require(recorded["pred"] == top10[0] and recorded["top10"] == top10, "recorded_front_order")
            require(
                type(recorded["conf"]) in (int, float)
                and math.isfinite(recorded["conf"])
                and math.isclose(confidence, recorded["conf"], rel_tol=1e-6, abs_tol=1e-7),
                "recorded_front_confidence",
            )
            require((confidence >= THRESHOLD) == (recorded["conf"] >= THRESHOLD), "front_threshold")
    return rows, assets


def paired_values(rows, assets, prices):
    require(set(prices) == {"as_of", "credit_per_usd", "sie", "rival_rates", "sources", "limitations"}, "price_fields")
    require(type(prices["credit_per_usd"]) is int and prices["credit_per_usd"] == 100000, "credit_denomination")
    require(
        prices["sie"]
        == {
            "embedding_credits_per_token": "3/500",
            "generation_input_credits_per_token": "1/40",
            "generation_output_credits_per_token": "1/5",
        },
        "sie_prices",
    )
    rates = prices["rival_rates"]
    require(set(rates) == {"gpt-6-luna", "claude-sonnet-5"}, "rival_rates")
    require(
        all(
            isinstance(rate, list)
            and len(rate) == 4
            and all(isinstance(value, str) and Fraction(value) >= 0 for value in rate)
            for rate in rates.values()
        ),
        "rival_rates",
    )
    require(isinstance(prices["as_of"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", prices["as_of"]), "price_date")
    require(
        isinstance(prices["sources"], list)
        and prices["sources"]
        and isinstance(prices["limitations"], str)
        and prices["limitations"],
        "price_context",
    )
    predictions, costs, gold, kept = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, [], []
    for row in rows:
        require(
            set(row)
            == {
                "dataset",
                "id",
                "stratum",
                "gold",
                "front",
                "cascade",
                "embedding_input_tokens",
                "embedding_additional_attempt_input_token_bounds",
                "generation_attempts",
                "rivals",
            },
            "record_fields",
        )
        dataset, stratum = row["dataset"], row["stratum"]
        require(
            stratum in STRATA and dataset == ("clinc150" if stratum.startswith("clinc150_") else stratum),
            "stratum_membership",
        )
        labels = assets[dataset]["labels"]
        require(
            row["gold"] in labels and (dataset != "clinc150" or (row["gold"] == NONE) == (stratum == "clinc150_oos")),
            "gold_membership",
        )
        cascade = row["cascade"]
        require(set(cascade) == {"answer", "error"}, "cascade_fields")
        require(
            (cascade["error"] is None and isinstance(cascade["answer"], str) and cascade["answer"] in labels)
            or (isinstance(cascade["error"], str) and cascade["error"] and cascade["answer"] is None),
            "cascade_outcome",
        )
        use_front = row["front"]["conf"] >= THRESHOLD
        attempts = row["generation_attempts"]
        require(isinstance(attempts, list) and len(attempts) <= 2, "generation_attempts")
        if use_front:
            require(not attempts and cascade == {"answer": row["front"]["pred"], "error": None}, "front_branch")
        else:
            require(
                bool(attempts) and all(attempt["status"] == "failed" for attempt in attempts[:-1]),
                "conditional_generation",
            )
            require((attempts[-1]["status"] == "success") == (cascade["error"] is None), "generation_outcome")
        upper = listed_upper(token(row["embedding_input_tokens"]))
        additional_bounds = row["embedding_additional_attempt_input_token_bounds"]
        require(isinstance(additional_bounds, list), "embedding_additional_bounds")
        for bound in additional_bounds:
            upper += listed_upper(token(bound))
        for attempt in attempts:
            require(
                set(attempt)
                == {"status", "prompt_tokens", "completion_tokens", "prompt_token_bound", "completion_token_bound"}
                and attempt["status"] in {"success", "failed"},
                "generation_attempt_fields",
            )
            prompt, output = token(attempt["prompt_token_bound"]), token(attempt["completion_token_bound"])
            require(prompt + 64 <= 8192 and output == 64, "generation_bounds")
            for name, bound in (("prompt_tokens", prompt), ("completion_tokens", output)):
                if attempt[name] is not None:
                    require(token(attempt[name]) <= bound, "generation_usage_bound")
            if attempt["status"] == "success":
                prompt, output = token(attempt["prompt_tokens"]), token(attempt["completion_tokens"])
            upper += listed_upper(0, prompt, output)
        costs["C"].append(upper)
        predictions["C"].append(cascade["answer"] if cascade["error"] is None else None)
        gold.append(row["gold"])
        kept.append(use_front)
        require(set(row["rivals"]) == {"E5", "E6"}, "rival_coverage")
        for arm, model in (("E5", "gpt-6-luna"), ("E6", "claude-sonnet-5")):
            rival = row["rivals"][arm]
            require(set(rival) == {"answer", "error", "in", "out", "cached", "cache_write"}, "rival_fields")
            require(
                (rival["error"] is None and isinstance(rival["answer"], str) and rival["answer"] in labels)
                or (isinstance(rival["error"], str) and rival["error"] and rival["answer"] is None),
                "rival_outcome",
            )
            predictions[arm].append(rival["answer"] if rival["error"] is None else None)
            costs[arm].append(rival_lower(rival, model, rates))
    return gold, predictions, kept, costs


def verify(directory):
    require(np.__version__ == "2.4.6", "numpy_version")
    directory = Path(directory)
    verify_manifest(directory)
    rows, assets = recorded_inputs(directory)
    gold, predictions, kept, costs = paired_values(rows, assets, load(directory / "prices.json"))
    result = statistics(rows, gold, predictions, kept, costs)
    recorded = load(directory / "results.json")
    for field, value in result.items():
        require(field in recorded, "results_fields")
        compare(value, recorded[field])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    try:
        print(json.dumps({"status": "verified", **verify(args.evidence)}, indent=2, allow_nan=False))
        return 0
    except Exception as error:  # noqa: BLE001 — Corrupt recordings produce a content-free verification failure.
        print(
            json.dumps(
                {"status": "blocked", "reason": str(error) if isinstance(error, EvidenceError) else "invalid_evidence"}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
