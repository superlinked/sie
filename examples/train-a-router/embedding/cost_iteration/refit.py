"""Promoted fixed dollar model: train+development refit, input-only future scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from embedding import recipes as core
from embedding.cost_iteration import study
from evaluation import utility_router as utility

PROMOTION_SHA = "13a500178d56ed1f08b17b8ad15be5f6a5f1644ea768e7aacf878e44519d48da"


def file_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def augment(score, prediction):
    """Return augmented scores + combined-model binding; works for native or CPU heads."""
    core.require(
        score["format"] == "bbeh-input-only-scores-v1"
        and prediction["format"] == "bbeh-shared-input-cost-scores-v1"
        and score["role"] in {"calibration", "final"}
        and score["outcomes_accessed"] is False
        and prediction["outcomes_accessed"] is False,
        "Input-only future scores required",
    )
    for key in ("role", "ids", "source_plan_sha256", "pair_sha256"):
        core.require(
            score[key] == prediction[key], "Cost/quality source-role-order mismatch"
        )
    core.require(
        len(set(score["ids"])) == len(score["ids"]) > 0, "Duplicate/empty score IDs"
    )
    for key in ("model_sha256",):
        core.require_hash(score[key])
    core.require_hash(prediction["cost_model_sha256"])
    core.require(
        prediction["promotion_sha256"] == PROMOTION_SHA, "Cost decision changed"
    )
    expected = study.positive_costs(prediction["expected_usd"], len(score["ids"]))
    binding = {
        "format": "bbeh-quality-and-shared-cost-binding-v1",
        "quality_model_sha256": score["model_sha256"],
        "cost_model_sha256": prediction["cost_model_sha256"],
        "source_plan_sha256": score["source_plan_sha256"],
        "pair_sha256": score["pair_sha256"],
        "promotion_sha256": PROMOTION_SHA,
        "methods": sorted(score["methods"]),
        "original_methods_preserved": True,
    }
    out = copy.deepcopy(score)
    for name, value in score["methods"].items():
        core.require(
            not name.endswith("__textcost")
            and name + "__textcost" not in score["methods"],
            "Scores already augmented or method-name collision",
        )
        p = utility._matrix(
            value["probabilities"], "quality probabilities", probability=True
        )
        core.require(len(p) == len(expected), "Quality probability population changed")
        core.require(
            value["router_kind"] in {"native", "embedding", "cpu"}
            and type(value["metadata_assisted"]) is bool,
            "Invalid quality method metadata",
        )
        out["methods"][name + "__textcost"] = {
            **copy.deepcopy(value),
            "expected_usd": expected.tolist(),
            "quality_method": name,
            "cost_variant": "shared_text_length",
            "cost_model_sha256": prediction["cost_model_sha256"],
            "cost_prediction_inputs": "Question TFIDF and length/structure only",
            "additional_cpu_cost_predictor_hosting": "unpriced; not assumed zero",
        }
    out.update(
        {
            "model_sha256": hashlib.sha256(file_bytes(binding)).hexdigest(),
            "quality_model_sha256": score["model_sha256"],
            "cost_model_sha256": prediction["cost_model_sha256"],
            "promotion_sha256": PROMOTION_SHA,
            "variant_scope": "Original quality and cost methods retained; added shared text-cost variants only",
        }
    )
    return out, binding


def augment_files(score_path, cost_path, output, binding_path):
    score, binding = augment(core.read(score_path), core.read(cost_path))
    score["parent_scores_sha256"] = core.sha(score_path)
    score["cost_scores_sha256"] = core.sha(cost_path)
    if Path(binding_path).exists():
        core.require(
            Path(binding_path).read_bytes() == file_bytes(binding),
            "Combined model binding changed",
        )
    else:
        core.write(binding_path, binding)
    core.require(
        core.sha(binding_path) == score["model_sha256"], "Binding encoding differs"
    )
    core.write(output, score)
    return score


def fit_score(run, train_path, dev_path, scores_root, output, *, promotion_path=None):
    run, scores_root, output = Path(run), Path(scores_root), Path(output)
    promotion_path = Path(promotion_path or Path(__file__).with_name("promotion.json"))
    core.require(
        core.sha(promotion_path) == PROMOTION_SHA, "Prospective refit decision changed"
    )
    core.require(not output.exists(), "Preserve cost refit")
    promotion = core.read(promotion_path)
    plan, quality, rows, grades = study.load_inputs(
        run, train_path, dev_path, Path(__file__).with_name("protocol.json")
    )
    # No fitting function receives calibration/final labels or their file paths.
    fit_rows = rows["train"] + rows["development"]
    fit_costs = np.concatenate((grades["train"][1], grades["development"][1]))
    started = time.perf_counter()
    _, matrix, text = core.text_matrix(fit_rows)
    cost_model = {
        "format": "bbeh-shared-input-dollar-model-v1",
        "fit_role": "train+development",
        "tuning_role": "development_previous_fixed_choice",
        "refit_on_development": True,
        "text_features": text,
        "models": {
            "text_length_log_ridge": study.fit_ridge(
                matrix, fit_costs, promotion["alpha"]
            )
        },
        "train_mean_usd": fit_costs.mean(axis=0).tolist(),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "promotion_sha256": PROMOTION_SHA,
        "protocol_sha256": study.PROTOCOL_SHA,
        "train_grades_sha256": core.sha(train_path),
        "development_grades_sha256": core.sha(dev_path),
        "refit_ids": [row["id"] for row in fit_rows],
        "fit_seconds": time.perf_counter() - started,
        "quality_model_sha256": core.sha(run / "model.json"),
        "study_source_sha256": core.sha(study.__file__),
        "refit_source_sha256": core.sha(__file__),
        "calibration_or_final_outcomes_accessed": False,
    }
    output.mkdir(parents=True)
    core.write(output / "cost-model.json", cost_model)
    results = {}
    for role in ("calibration", "final"):
        role_rows = core.role_rows(run, plan, role)
        expected = study.predict_rows(cost_model, role_rows, "text_length_log_ridge")
        prediction = {
            "format": "bbeh-shared-input-cost-scores-v1",
            "role": role,
            "ids": [row["id"] for row in role_rows],
            "arms": ["cheap", "strong"],
            "source_plan_sha256": plan["source_plan_sha256"],
            "pair_sha256": plan["pair_sha256"],
            "promotion_sha256": PROMOTION_SHA,
            "cost_model_sha256": core.sha(output / "cost-model.json"),
            "expected_usd": expected.tolist(),
            "outcomes_accessed": False,
        }
        cost_path = output / f"costs-{role}.json"
        core.write(cost_path, prediction)
        old_score_path = scores_root / f"scores-{role}.json"
        old = core.read(old_score_path)
        core.require(
            old["model_sha256"] == core.sha(run / "model.json")
            and set(old["methods"]) == study.QUALITY_METHODS,
            "Original quality score/model changed",
        )
        combined = augment_files(
            old_score_path,
            cost_path,
            output / f"scores-{role}.json",
            output / "combined-model.json",
        )
        for name in old["methods"]:
            core.require(
                combined["methods"][name] == old["methods"][name],
                "Original method mutated",
            )
        results[role] = {
            "n": len(role_rows),
            "cost_scores_sha256": core.sha(cost_path),
            "scores_sha256": core.sha(output / f"scores-{role}.json"),
            "parent_scores_sha256": core.sha(old_score_path),
            "methods": list(combined["methods"]),
        }
    receipt = {
        "format": "bbeh-shared-cost-refit-receipt-v1",
        "promotion_sha256": PROMOTION_SHA,
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "cost_model_sha256": core.sha(output / "cost-model.json"),
        "combined_model_sha256": core.sha(output / "combined-model.json"),
        "roles": results,
        "fit_n": len(fit_rows),
        "alpha": promotion["alpha"],
        "quality_selected": quality["selected"],
        "quality_heads_refitted": False,
        "calibration_or_final_outcomes_accessed": False,
    }
    core.write(output / "refit.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "augment", help="Add the same refit cost arrays to native or CPU quality scores"
    )
    p.add_argument("--scores", required=True)
    p.add_argument("--costs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--binding", required=True)
    args = parser.parse_args()
    augment_files(args.scores, args.costs, args.output, args.binding)


if __name__ == "__main__":
    main()
