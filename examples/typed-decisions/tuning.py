"""The settings chosen on the dev slice, and how they apply to a recorded answer.

Standard library only. tune.py writes tuning.json from dev recordings; run.py
reads it to build the `tuned` requests and score.py reads it to turn a tuned
recording's probabilities into answers. Nothing here is fitted on test.

Per backend and question, tuning.json holds:

    phrasing   which phrasing (short, described, concrete) the question is sent in
    rule       how a probability becomes an answer:
                 {"kind": "argmax"}                             the top option
                 {"kind": "cutoff", "cutoff": t}                yes when P(true) >= t; P(true)
                                                                is shifted by t's log-odds
                 {"kind": "prior", "weights": {option: w}}      the top option after
                                                                multiplying each option's
                                                                probability by its weight
                                                                and renormalising
    dev        the dev figures the choice was made on
    page       whether the question passes the page rule on dev, and why not
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from questions import Phrasing

TUNED = "tuned"
TUNING_PATH = Path(__file__).resolve().with_name("tuning.json")


def load(path: Path | None = None) -> dict[str, Any]:
    path = TUNING_PATH if path is None else path
    if not path.exists():
        return {"backends": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def entry(set_name: str, backend: str, tuning: dict[str, Any] | None = None) -> dict[str, Any] | None:
    tuning = load() if tuning is None else tuning
    return tuning.get("backends", {}).get(set_name, {}).get(backend)


def phrasing(set_name: str, backend: str, variant: str, tuning: dict[str, Any] | None = None) -> Phrasing:
    """The phrasing a recorded variant stands for: itself, or the tuned per-question mix."""
    if variant != TUNED:
        return variant
    found = entry(set_name, backend, tuning)
    if found is None or "phrasing" not in found:
        raise SystemExit(f"tuning.json has no tuned phrasing for {set_name}/{backend}; run tune.py first")
    return {qid: settings["phrasing"] for qid, settings in found["phrasing"].items()}


def rules(set_name: str, backend: str, variant: str, tuning: dict[str, Any] | None = None) -> dict[str, dict]:
    """The decision rule per question for a recorded variant; argmax everywhere unless it is `tuned`."""
    if variant != TUNED:
        return {}
    found = entry(set_name, backend, tuning) or {}
    return {qid: settings["rule"] for qid, settings in found.get("questions", {}).items()}


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def apply_rule(distribution: dict[str, float], rule: dict[str, Any] | None) -> tuple[dict[str, float], str]:
    """The distribution after a rule, and the answer it gives."""
    kind = (rule or {}).get("kind", "argmax")
    if kind == "cutoff":
        # Shift P(true) by the cut-off's log-odds: the answer is yes exactly when
        # P(true) >= cut-off, and the shifted probability says how far past the
        # cut-off it is, which is what a composition or a confidence check reads.
        p_true = min(max(distribution["true"], 1e-6), 1 - 1e-6)
        shifted = 1 / (1 + math.exp(-(logit(p_true) - logit(rule["cutoff"]))))
        return {"true": shifted, "false": 1 - shifted}, "true" if shifted >= 0.5 else "false"
    if kind == "prior":
        weighted = {key: p * rule["weights"][key] for key, p in distribution.items()}
        total = sum(weighted.values())
        distribution = {key: value / total for key, value in weighted.items()} if total > 0 else distribution
    elif kind != "argmax":
        raise SystemExit(f"unknown rule kind {kind!r}")
    return distribution, max(distribution, key=distribution.get)
