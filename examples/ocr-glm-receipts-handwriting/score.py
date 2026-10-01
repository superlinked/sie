#!/usr/bin/env python3
"""Recompute the recorded word metrics and frozen paired test from public counts."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from fetch import EVIDENCE, MANIFEST_SHA256, REQUIRED

COLUMNS = ["id", "set", "reference_words", "matched_words", "value_tokens", "matched_values", "output_words"]
SEED = 20261001
MARGIN = 3


def load_evidence() -> dict:
    body = (EVIDENCE / "manifest.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != MANIFEST_SHA256:
        raise ValueError("The evidence manifest is not the pinned comparison.")
    manifest = json.loads(body)
    if set(manifest["files"]) != REQUIRED:
        raise ValueError("The comparison file set differs from the pinned evidence.")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name}: evidence checksum failed")
    return manifest


def summarize(c: np.ndarray) -> dict:
    reference, matched, values, matched_values, output = c.sum(axis=0)
    f1 = 200 * c[:, 1] / (c[:, 0] + c[:, 4])
    return {
        "n": len(c),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(200 * matched / (reference + output)),
        "recall": float(100 * matched / reference),
        "precision": float(100 * matched / output) if output else 0.0,
        "value_recall": float(100 * matched_values / values) if values else None,
    }


def main() -> int:
    manifest = load_evidence()
    data = json.loads((EVIDENCE / "per_image.json").read_text())
    recorded = json.loads((EVIDENCE / "results.json").read_text())
    if data["columns"] != COLUMNS or set(data["arms"]) != {"glm", "mini"}:
        raise ValueError("Unexpected count schema or model arms.")
    rows = {arm: sorted(values, key=lambda row: row[0]) for arm, values in data["arms"].items()}
    ids = [row[0] for row in rows["glm"]]
    sets = np.asarray([row[1] for row in rows["glm"]])
    if len(ids) != 96 or len(set(ids)) != 96:
        raise ValueError("The comparison must contain exactly 96 unique images.")
    if [(r[0], r[1]) for r in rows["mini"]] != [(r[0], r[1]) for r in rows["glm"]]:
        raise ValueError("Model arms do not contain the same images.")
    if {s: int((sets == s).sum()) for s in ("cord", "gnhk")} != manifest["by_set"]:
        raise ValueError("The balanced comparison selection differs.")
    matrices = {}
    for arm, values in rows.items():
        c = np.asarray([row[2:] for row in values], dtype=float)
        if not np.all(np.isfinite(c)) or np.any(c < 0) or np.any(c != np.floor(c)):
            raise ValueError("Word counts must be finite nonnegative integers.")
        if np.any(c[:, 0] < 5) or np.any(c[:, 1] > np.minimum(c[:, 0], c[:, 4])):
            raise ValueError("Matched/reference/output counts are inconsistent.")
        if np.any(c[:, 2] > c[:, 0]) or np.any(c[:, 3] > np.minimum(c[:, 1], c[:, 2])):
            raise ValueError("Value-token counts are inconsistent.")
        matrices[arm] = c
        for scope, selected in [("overall", c), *[(s, c[sets == s]) for s in ("cord", "gnhk")]]:
            computed = summarize(selected)
            for key, value in computed.items():
                if not np.isclose(value, recorded["arms"][arm][scope][key], atol=1e-10, rtol=0):
                    raise ValueError(f"{arm}/{scope}/{key}: recorded metric differs from public counts")
    a, b = matrices["glm"], matrices["mini"]
    delta = 200 * a[:, 1] / (a[:, 0] + a[:, 4]) - 200 * b[:, 1] / (b[:, 0] + b[:, 4])
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(100):
        indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(sets == s), size=(100, int((sets == s).sum())), replace=True)
                for s in ("cord", "gnhk")
            ],
            axis=1,
        )
        ca, cb = a[indices].sum(axis=1), b[indices].sum(axis=1)
        boot.append(
            np.column_stack((delta[indices].mean(axis=1), 100 * ca[:, 1] / ca[:, 4] - 100 * cb[:, 1] / cb[:, 4]))
        )
    bootstrap = np.concatenate(boot)
    f_lower, p_lower = float(np.quantile(bootstrap[:, 0], 0.05)), float(np.quantile(bootstrap[:, 1], 0.05))
    domain = {s: float(delta[sets == s].mean()) for s in ("cord", "gnhk")}
    value_diff = summarize(a)["value_recall"] - summarize(b)["value_recall"]
    paired = {
        "macro_f1_diff_pp": float(delta.mean()),
        "macro_f1_ci95": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
        "macro_f1_one_sided_lower95": f_lower,
        "precision_one_sided_lower95": p_lower,
        "domain_f1_diff_pp": domain,
        "value_recall_diff_pp": value_diff,
    }
    for key, value in paired.items():
        if isinstance(value, dict):
            matches = all(np.isclose(v, recorded["paired"][key][name], atol=1e-10, rtol=0) for name, v in value.items())
        else:
            matches = np.all(np.isclose(value, recorded["paired"][key], atol=1e-10, rtol=0))
        if not matches:
            raise ValueError(f"{key}: the recorded paired statistic differs from public counts")
    gates = {
        "f1_noninferior": f_lower > -MARGIN,
        "each_domain_point": all(value >= -MARGIN for value in domain.values()),
        "value_recall": value_diff >= -MARGIN,
        "precision_noninferior": p_lower > -MARGIN,
        "reliable": all(value == 0 for arm in recorded["arms"].values() for value in arm["reliability"].values()),
    }
    if gates != recorded["gates"] or all(gates.values()) != recorded["publication_pass"]:
        raise ValueError("The frozen comparison gates do not match the public evidence.")
    usage = json.loads((EVIDENCE / "mini_usage.json").read_text())
    total = 0.0
    rates = usage["rates_usd_per_million"]
    for row in usage["per_image"].values():
        usd = (
            (row["input_tokens"] - row["cached_input_tokens"]) * rates["input"]
            + row["cached_input_tokens"] * rates["cached_input"]
            + row["output_tokens"] * rates["output"]
        ) / 1e6
        if not np.isclose(usd, row["usd"], atol=1e-12, rtol=0):
            raise ValueError("Recorded mini token cost is inconsistent.")
        total += usd
    if set(usage["per_image"]) != set(ids) or not np.isclose(total, usage["usd_total"], atol=1e-12, rtol=0):
        raise ValueError("Recorded mini cost does not cover the full comparison.")
    print(
        json.dumps(
            {
                "glm": summarize(a),
                "mini": summarize(b),
                "paired_f1_diff_pp": float(delta.mean()),
                "f1_ci95": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
                "gates": gates,
                "mini_usd_per_1k": total / len(ids) * 1000,
                "reliability": "Recorded checks; count-only evidence does not independently re-audit raw response text.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
