"""Train-only shared dollar prediction; saved quality probabilities never change."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from embedding import development_budget, recipes as core
from evaluation import frontier, utility_router as utility

PROTOCOL_SHA = "be1a9641d79474b70e7013c7ffaf9a390c66627ca98b6c5cf46adee46a15f872"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
FEATURE_MODELS = ("length_log_ridge", "text_length_log_ridge")
QUALITY_METHODS = {"knn", "embedding", "text_length", "text_topic", "task_metadata"}


def positive_costs(values, n=None):
    value = np.asarray(values, dtype=float)
    core.require(
        value.ndim == 2
        and value.shape[1] == 2
        and (n is None or len(value) == n)
        and np.isfinite(value).all()
        and (value > 0).all(),
        "Expected complete finite strictly positive cheap/strong USD",
    )
    return value


def error_metrics(actual, predicted, train_means):
    actual, predicted = positive_costs(actual), positive_costs(predicted)
    means = np.asarray(train_means, dtype=float)
    core.require(
        actual.shape == predicted.shape
        and means.shape == (2,)
        and np.isfinite(means).all()
        and (means > 0).all(),
        "Cost metric alignment",
    )
    residual = predicted - actual
    normalized = np.mean((residual / means) ** 2, axis=0)
    return {
        "mean_normalized_squared_dollar_error": float(normalized.mean()),
        "per_arm_normalized_squared_dollar_error": normalized.tolist(),
        "per_arm_mae_usd": np.mean(abs(residual), axis=0).tolist(),
        "per_arm_bias_usd": residual.mean(axis=0).tolist(),
        "per_arm_actual_mean_usd": actual.mean(axis=0).tolist(),
        "per_arm_predicted_mean_usd": predicted.mean(axis=0).tolist(),
        "incremental_cost_rmse_usd": float(
            np.sqrt(np.mean((residual[:, 1] - residual[:, 0]) ** 2))
        ),
        "train_mean_usd_normalizer": means.tolist(),
    }


def fit_ridge(matrix, costs, alpha):
    from sklearn.linear_model import Ridge

    costs = positive_costs(costs, matrix.shape[0])
    log_cost = np.log(costs)
    model = Ridge(
        alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-8, max_iter=10000
    )
    model.fit(matrix, log_cost)
    core.require(
        np.all(np.asarray(model.n_iter_) < 10000), "Ridge iteration limit reached"
    )
    residual = log_cost - model.predict(matrix)
    smearing = np.exp(residual).mean(axis=0)
    core.require(
        np.isfinite(smearing).all() and (smearing > 0).all(), "Nonfinite TRAIN smearing"
    )
    saved = {
        "alpha": alpha,
        "weights": model.coef_.tolist(),
        "bias": model.intercept_.tolist(),
        "train_residual_smearing": smearing.tolist(),
        "iterations": np.asarray(model.n_iter_).tolist(),
    }
    positive_costs(predict_ridge(matrix, saved), len(costs))
    return saved


def predict_ridge(matrix, saved):
    log_cost = np.asarray(matrix @ np.asarray(saved["weights"]).T) + np.asarray(
        saved["bias"]
    )
    result = np.exp(log_cost) * np.asarray(saved["train_residual_smearing"])
    return positive_costs(result, matrix.shape[0])


def select_candidate(candidates):
    core.require(bool(candidates), "No cost candidates")
    return min(
        candidates,
        key=lambda row: (
            row["development_error"]["mean_normalized_squared_dollar_error"],
            -row["alpha"],
        ),
    )


def fit_costs(train_rows, train_costs, dev_rows, dev_costs):
    core.require(
        all(r["role"] == "train" for r in train_rows)
        and all(r["role"] == "development" for r in dev_rows),
        "Wrong fit/tuning roles",
    )
    core.require(
        not ({r["family"] for r in train_rows} & {r["family"] for r in dev_rows}),
        "Family crosses cost fitting and tuning",
    )
    train_costs, dev_costs = (
        positive_costs(train_costs, len(train_rows)),
        positive_costs(dev_costs, len(dev_rows)),
    )
    started = time.perf_counter()
    _, train_text, text = core.text_matrix(train_rows)
    _, dev_text, _ = core.text_matrix(dev_rows, text)
    scale = np.asarray(text["length_scale"])
    matrices = {
        "length_log_ridge": (
            core.lengths(train_rows) / scale,
            core.lengths(dev_rows) / scale,
        ),
        "text_length_log_ridge": (train_text, dev_text),
    }
    means = train_costs.mean(axis=0)
    predictions = {"mean": np.tile(means, (len(dev_rows), 1))}
    models = {"mean": {"train_mean_usd": means.tolist()}}
    errors = {"mean": error_metrics(dev_costs, predictions["mean"], means)}
    tuning = {}
    for name, (train_matrix, dev_matrix) in matrices.items():
        candidates = []
        for alpha in ALPHAS:
            before = time.perf_counter()
            fitted = fit_ridge(train_matrix, train_costs, alpha)
            predicted = predict_ridge(dev_matrix, fitted)
            candidates.append(
                {
                    "alpha": alpha,
                    "model": fitted,
                    "development_error": error_metrics(dev_costs, predicted, means),
                    "fit_and_development_prediction_seconds": time.perf_counter()
                    - before,
                }
            )
        chosen = select_candidate(candidates)
        models[name] = chosen["model"]
        predictions[name] = predict_ridge(dev_matrix, chosen["model"])
        errors[name] = chosen["development_error"]
        tuning[name] = [
            {k: v for k, v in row.items() if k != "model"} for row in candidates
        ]
    return {
        "format": "bbeh-shared-input-dollar-model-v1",
        "fit_role": "train",
        "tuning_role": "development",
        "refit_on_development": False,
        "text_features": text,
        "models": models,
        "tuning": tuning,
        "train_mean_usd": means.tolist(),
        "development_errors": errors,
        "training_and_tuning_seconds": time.perf_counter() - started,
        "inference_dependencies": "Request text only; no embeddings, topic metadata, labels or generated tokens",
    }, predictions


def predict_rows(model, rows, name):
    """Reusable input-only cost scorer, also suitable for native correctness heads."""
    core.require(name in {"mean", *FEATURE_MODELS}, "Unknown cost model")
    if name == "mean":
        return np.tile(np.asarray(model["train_mean_usd"]), (len(rows), 1))
    if name == "length_log_ridge":
        matrix = core.lengths(rows) / np.asarray(model["text_features"]["length_scale"])
    else:
        _, matrix, _ = core.text_matrix(rows, model["text_features"])
    return predict_ridge(matrix, model["models"][name])


def selected_quality(model, y):
    selected = {}
    for candidate in model["tuning"]:
        name = candidate["method"]
        if candidate["setting"] != model["selected"].get(name, "fixed"):
            continue
        core.require(
            name in QUALITY_METHODS and name not in selected,
            "Unexpected/duplicate quality method",
        )
        p = utility._matrix(
            candidate["probabilities"],
            "saved development probabilities",
            probability=True,
        )
        positive_costs(candidate["expected_usd"], len(y))
        core.require(
            len(p) == len(y) and abs(core.nll(y, p) - candidate["nll"]) <= 1e-12,
            "Stored quality NLL/population changed",
        )
        selected[name] = candidate
    core.require(set(selected) == QUALITY_METHODS, "Missing frozen quality method")
    return selected


def make_curve(candidate, expected, rows, y, costs, bootstrap):
    p = np.asarray(candidate["probabilities"])
    curve = frontier.enumerate_frontier(
        {"probabilities": p, "expected_usd": expected}, y, costs, None
    )
    frontier_ids = {node["candidate_id"] for node in curve["frontier"]}
    points = []
    for point in curve["candidates"]:
        node = {
            "candidate_id": point["candidate_id"],
            "rule": point["policy"],
            "development": point["calibration"],
            "development_matched_random_strong_probability": point[
                "calibration_matched_random_strong_probability"
            ],
        }
        if point["candidate_id"] in frontier_ids:
            routes = utility.replay(point["policy"], p, expected)
            fees = (
                np.zeros(len(rows))
                if point["policy"]["mode"].startswith("always_")
                else None
            )
            intervals = frontier.uncertainty(
                bootstrap,
                routes,
                y,
                costs,
                fees,
                point["calibration_matched_random_strong_probability"],
            )
            node["paired_ci95"] = {
                k.replace("calibration_budget", "development_budget"): v
                for k, v in intervals.items()
            }
        points.append(node)
    return {
        "setting": candidate["setting"],
        "nll": candidate["nll"],
        "selected_by_development_nll": True,
        "probabilities": candidate["probabilities"],
        "expected_usd": np.asarray(expected).tolist(),
        "points": points,
        "frontier_candidate_ids": sorted(frontier_ids),
        "randomized_hull_candidate_ids": curve["randomized_hull_candidate_ids"],
    }


def compare(quality, predictions, rows, y, costs, train_means, *, draws=2000):
    bootstrap = frontier.Bootstrap(rows, draws)
    methods, original_errors = {}, {}
    for name, candidate in quality.items():
        original_errors[name] = error_metrics(
            costs, candidate["expected_usd"], train_means
        )
        for cost_name, expected in {
            "original": np.asarray(candidate["expected_usd"]),
            **predictions,
        }.items():
            methods[f"{name}__{cost_name}"] = {
                "quality_method": name,
                "cost_method": cost_name,
                "router_kind": "embedding" if name in {"knn", "embedding"} else "cpu",
                "metadata_assisted": name == "task_metadata",
                "settings": [
                    make_curve(candidate, expected, rows, y, costs, bootstrap)
                ],
            }
    return methods, original_errors


def load_inputs(run, train_path, dev_path, protocol_path):
    run = Path(run)
    core.require(core.sha(protocol_path) == PROTOCOL_SHA, "Cost protocol changed")
    protocol, plan = core.read(protocol_path), core.checked_plan(run)
    core.require(
        plan["source_plan_sha256"] == protocol["source_plan_sha256"],
        "Source plan changed",
    )
    core.require(
        core.sha(run / "model.json") == protocol["quality_model_sha256"],
        "Quality model changed",
    )
    model = core.read(run / "model.json")
    core.require(
        model["plan_sha256"] == core.sha(run / "plan.json")
        and model["source_plan_sha256"] == plan["source_plan_sha256"]
        and model["pair_sha256"] == plan["pair_sha256"]
        and model["calibration_or_final_outcomes_accessed"] is False,
        "Quality lineage changed",
    )
    rows, data = {}, {}
    for role, path in (("train", train_path), ("development", dev_path)):
        expected = protocol[f"{role}_grades_sha256"]
        core.require(
            core.sha(path) == expected == model[f"{role}_grades_sha256"],
            "Grades changed",
        )
        rows[role] = core.role_rows(run, plan, role)
        data[role] = core.grades(plan, rows[role], path, role)
    core.require(
        data["train"][2]["grader_sha256"] == data["development"][2]["grader_sha256"],
        "Grader changed between roles",
    )
    return plan, model, rows, data


def build(run, train_path, dev_path, output, *, protocol_path=None, draws=2000):
    protocol_path = Path(protocol_path or Path(__file__).with_name("protocol.json"))
    output = Path(output)
    core.require(not output.exists(), "Preserve prior cost iteration")
    plan, quality_model, rows, grades = load_inputs(
        run, train_path, dev_path, protocol_path
    )
    train_costs = grades["train"][1]
    y, costs = grades["development"][:2]
    quality = selected_quality(quality_model, y)
    quality_hashes = {
        name: core.digest(value["probabilities"]) for name, value in quality.items()
    }
    cost_model, predictions = fit_costs(
        rows["train"], train_costs, rows["development"], costs
    )
    output.mkdir(parents=True)
    lineage = {
        "protocol_sha256": core.sha(protocol_path),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "quality_model_sha256": core.sha(Path(run) / "model.json"),
        "train_grades_sha256": core.sha(train_path),
        "development_grades_sha256": core.sha(dev_path),
        "quality_probabilities_sha256": quality_hashes,
        "study_source_sha256": core.sha(__file__),
        "calibration_or_final_outcomes_accessed": False,
    }
    cost_model.update(lineage)
    core.write(output / "cost-model.json", cost_model)
    core.write(
        output / "development-cost-predictions.json",
        {
            "format": "bbeh-shared-development-cost-predictions-v1",
            "role": "development",
            **lineage,
            "cost_model_sha256": core.sha(output / "cost-model.json"),
            "ids": [r["id"] for r in rows["development"]],
            "expected_usd": {
                name: value.tolist() for name, value in predictions.items()
            },
            "arms": ["cheap", "strong"],
            "actual_outcomes_used_as_prediction_inputs": False,
        },
    )
    before = time.perf_counter()
    methods, original_errors = compare(
        quality,
        predictions,
        rows["development"],
        y,
        costs,
        train_costs.mean(axis=0),
        draws=draws,
    )
    core.require(
        quality_hashes
        == {
            name: core.digest(value["probabilities"]) for name, value in quality.items()
        },
        "Quality probabilities mutated",
    )
    report = {
        "format": "bbeh-development-candidate-frontiers-v1",
        "role": "development",
        **lineage,
        "plan_sha256": core.sha(Path(run) / "plan.json"),
        "model_sha256": core.sha(Path(run) / "model.json"),
        "grades_sha256": core.sha(dev_path),
        "cost_model_sha256": core.sha(output / "cost-model.json"),
        "ids": [r["id"] for r in rows["development"]],
        "n": len(y),
        "families": len({r["family"] for r in rows["development"]}),
        "methods": methods,
        "random": {
            "analytic_expectation": True,
            "strong_probability_range": [0, 1],
            "accuracy_endpoints_percent": (100 * y.mean(axis=0)).tolist(),
            "answer_usd_per_10000_endpoints": (10000 * costs.mean(axis=0)).tolist(),
        },
        "cost_scope": grades["development"][2]["cost_scope"],
        "fees_sha256": None,
        "router_cost_scope": "Answer-model costs only; encoder and CPU/native hosting are unpriced, not free.",
        "refit_used_for_development_predictions": False,
        "bootstrap": {
            "draws": draws,
            "seed": 89,
            "unit": "whole families within task, fixed observed task weights",
        },
        "interpretation": "Development diagnostic after fixed cost-NMSE selection. Quality models/NLL choices unchanged. Best budget points and pointwise intervals are neither unbiased final estimates nor simultaneous curve guarantees.",
    }
    core.write(output / "development-frontiers.json", report)
    summary = development_budget.summarize(report)
    summary.update(
        {
            "cost_iteration": lineage,
            "selected_alpha": {
                name: cost_model["models"][name]["alpha"] for name in FEATURE_MODELS
            },
            "shared_cost_errors": cost_model["development_errors"],
            "original_cost_errors": original_errors,
            "cost_tuning": cost_model["tuning"],
            "cost_fit_seconds": cost_model["training_and_tuning_seconds"],
            "frontier_seconds": time.perf_counter() - before,
        }
    )
    core.write(output / "budget-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--train-grades", required=True)
    parser.add_argument("--development-grades", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.run, args.train_grades, args.development_grades, args.output)


if __name__ == "__main__":
    main()
