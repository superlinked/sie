"""Repeat the selected cost recipe without binding new fits to original timing bytes."""

from pathlib import Path
import time

import numpy as np

from embedding import recipes as core
from embedding.cost_iteration import refit, study


def fit_score(run, train_path, development_path, output):
    run, output = Path(run), Path(output)
    core.require(not output.exists(), "Preserve the previous cost fit")
    plan = core.checked_plan(run)
    quality = core.read(run / "model.json")
    core.require(quality["plan_sha256"] == core.sha(run / "plan.json")
        and quality["train_grades_sha256"] == core.sha(train_path)
        and quality["development_grades_sha256"] == core.sha(development_path)
        and quality["calibration_or_final_outcomes_accessed"] is False, "Quality fit/input lineage changed")
    rows, dollars = [], []
    families = set()
    for role, path in (("train", train_path), ("development", development_path)):
        selected = core.role_rows(run, plan, role)
        current = {row["family"] for row in selected}
        core.require(not families.intersection(current), "Training/development families overlap")
        families.update(current)
        _, usd, _ = core.grades(plan, selected, path, role)
        rows.extend(selected)
        dollars.append(usd)
    promotion = Path(refit.__file__).with_name("promotion.json")
    core.require(core.sha(promotion) == refit.PROMOTION_SHA, "Published cost-recipe decision changed")
    alpha = core.read(promotion)["alpha"]
    started = time.perf_counter()
    _, matrix, text = core.text_matrix(rows)
    model = {"format": "bbeh-shared-input-dollar-model-v1", "fit_role": "train+development",
        "text_features": text, "models": {"text_length_log_ridge": study.fit_ridge(matrix, np.concatenate(dollars), alpha)},
        "source_plan_sha256": plan["source_plan_sha256"], "pair_sha256": plan["pair_sha256"],
        "promotion_sha256": refit.PROMOTION_SHA, "protocol_sha256": study.PROTOCOL_SHA,
        "train_grades_sha256": core.sha(train_path), "development_grades_sha256": core.sha(development_path),
        "refit_ids": [row["id"] for row in rows], "quality_model_sha256": core.sha(run / "model.json"),
        "study_source_sha256": core.sha(study.__file__), "adapter_sha256": core.sha(__file__),
        "calibration_or_final_outcomes_accessed": False, "fit_seconds": time.perf_counter() - started,
        "reproduction": True, "new_blind_evaluation": False}
    output.mkdir(parents=True)
    core.write(output / "cost-model.json", model)
    for role in ("calibration", "final"):
        held = core.role_rows(run, plan, role)
        expected = study.predict_rows(model, held, "text_length_log_ridge")
        prediction = {"format": "bbeh-shared-input-cost-scores-v1", "role": role,
            "ids": [row["id"] for row in held], "arms": ["cheap", "strong"],
            "source_plan_sha256": plan["source_plan_sha256"], "pair_sha256": plan["pair_sha256"],
            "promotion_sha256": refit.PROMOTION_SHA, "cost_model_sha256": core.sha(output / "cost-model.json"),
            "expected_usd": expected.tolist(), "outcomes_accessed": False}
        original = core.read(run / f"scores-{role}.json")
        core.require(original["model_sha256"] == core.sha(run / "model.json"), "Quality scores use another fit")
        core.write(output / f"costs-{role}.json", prediction)
        refit.augment_files(run / f"scores-{role}.json", output / f"costs-{role}.json",
                            output / f"scores-{role}.json", output / "combined-model.json")
    return {"alpha": alpha, "fit_n": len(rows), "model_sha256": core.sha(output / "cost-model.json")}
