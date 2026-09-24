#!/usr/bin/env python3
"""Choose phrasing, decision rules and lane cut-offs on the dev slice, and write tuning.json.

    python3 tune.py phrasing --calls <dev recordings in short, described and concrete>
    python3 tune.py rules    --calls <dev recordings in the tuned variant>
    python3 tune.py lanes    --calls <dev recordings in the tuned variant>

Standard library only, no model. Only dev calls are read; test calls in the
input are skipped unread, so the published calls.json can be passed whole. The procedure is written down in PREREGISTRATION.md before
any test record is sent, and applies to every backend and question the same way.

A decision rule turns returned probabilities into an answer (see tuning.py):

    yes or no   a cut-off on P(true), from 0.05 to 0.95 in steps of 0.01. Among
                cut-offs whose dev accuracy is at least the majority answer's,
                the one with the highest balanced accuracy wins; if none reaches
                the majority's accuracy, the highest balanced accuracy wins
                anyway. Ties go to the cut-off nearest 0.5.
    pick one    either the plain top option, or the top option after dividing
                each option's probability by its mean over the dev records (a
                correction for the options a model favours regardless of input;
                it uses no labels). The correction is kept only if it raises dev
                balanced accuracy without taking accuracy below the majority's
                and without lowering the worst recall among options with at
                least 10 dev records.

A backend that returns only its top option (GLiNER2 on pick-one questions)
keeps the plain top option.

`phrasing` fits a rule for each phrasing of each question and keeps the
phrasing that passes the page rule below, preferring the highest 5th-percentile
balanced accuracy; when none passes, the highest balanced accuracy (ties:
described, then short, then concrete). `rules` refits the rule on the tuned recording,
where every question is sent in its chosen phrasing.

The page rule, with margin. A question passes for a backend or a lane when, on
the dev records with its settings fixed, in at least 95% of 1,000 bootstrap
resamples (seed SEED) its balanced accuracy is at least 0.70 and its accuracy is
at least the resample's majority answer's; and every option with at least 10
dev records is answered right at least 40% of the time. The composed severity
is held to the same rule on exact level.

`lanes` judges the board's lanes (Fast, Smart and the LLM, each one model on
its own) by the page rule, and fits the cascade: Fast first, Smart for the
questions Fast is not asked and for the answers whose top probability is below
a cut-off. The cut-off per question is the lowest (least escalated) at which,
on dev, the cascade's accuracy and balanced accuracy are at least Smart's alone
and no record Smart answers right is answered wrong. Escalating everything
always meets that. The cascade then passes its control only if it also costs
less than Smart alone, in calls per record; a cascade that fails is reported in
the README and does not go on the page.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lanes
import score
import tuning
from questions import CHOICE, CONCRETE, DESCRIBED, NOUL, SHORT, VULNERABILITY_QUESTIONS, VULNERABILITY_TRIAGE, asked
from run import LLM_BACKENDS, PAGE_LLM

PHRASING_ORDER = (DESCRIBED, SHORT, CONCRETE)
CUTOFFS = tuple(round(step / 100, 2) for step in range(5, 96))
PAGE_MIN_BALANCED = 0.70
PAGE_MIN_CLASS_RECALL = 0.40
PAGE_MIN_CLASS_RECORDS = 10
BOOTSTRAP = 1000
BOOTSTRAP_QUANTILE = 0.05
SEED = 20260924
LANE_CUTOFFS = (None, *(round(step / 100, 2) for step in range(50, 100, 5)), 1.01)

# The lanes the page's board compares, each one model on its own.
LANES = {
    "fast": {"backend": "gliformer-large"},
    "smart": {"backend": "gliclass-large-v1"},
    "llm": {"backend": PAGE_LLM},
}
# Fast first, and Smart for what Fast is unsure of. Judged against Smart alone.
CASCADE = {"backend": "gliformer-large", "escalate_to": "gliclass-large-v1"}


def metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    right = sum(g == p for g, p in zip(gold, predicted))
    return {
        "n": len(gold),
        "accuracy": right / len(gold),
        "balanced_accuracy": score.balanced_accuracy(gold, predicted),
        "macro_f1": score.macro_f1(gold, predicted),
        "recall": score.class_recall(gold, predicted),
    }


def majority_accuracy(gold: list[str]) -> float:
    return max(gold.count(option) for option in set(gold)) / len(gold)


def worst_recall(figures: dict[str, Any]) -> float:
    """The lowest recall among options with at least PAGE_MIN_CLASS_RECORDS records."""
    common = [right / total for right, total in figures["recall"].values() if total >= PAGE_MIN_CLASS_RECORDS]
    return min(common) if common else 1.0


def fit_rule(rows: list[tuple[dict[str, Any], str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The decision rule for one question on dev rows of (answer, gold), and its dev figures."""
    gold = [g for _, g in rows]
    floor = majority_accuracy(gold)
    qtype = rows[0][0]["type"]
    have_distributions = all(a["distribution"] is not None for a, _ in rows)
    plain = metrics(gold, [a["top"] for a, _ in rows])
    if qtype == NOUL and have_distributions:
        scored = []
        for cutoff in CUTOFFS:
            rule = {"kind": "cutoff", "cutoff": cutoff}
            figures = metrics(gold, [tuning.apply_rule(a["distribution"], rule)[1] for a, _ in rows])
            scored.append(
                (figures["accuracy"] >= floor, figures["balanced_accuracy"], -abs(cutoff - 0.5), rule, figures)
            )
        allowed = [entry for entry in scored if entry[0]] or scored
        _, _, _, rule, figures = max(allowed, key=lambda entry: (entry[1], entry[2]))
        return rule, figures
    if qtype == CHOICE and have_distributions:
        options = list(rows[0][0]["distribution"])
        means = {key: sum(a["distribution"][key] for a, _ in rows) / len(rows) for key in options}
        if all(mean > 0 for mean in means.values()):
            rule = {"kind": "prior", "weights": {key: 1 / means[key] for key in options}}
            corrected = metrics(gold, [tuning.apply_rule(a["distribution"], rule)[1] for a, _ in rows])
            if (
                corrected["balanced_accuracy"] > plain["balanced_accuracy"]
                and corrected["accuracy"] >= floor
                and worst_recall(corrected) >= worst_recall(plain)
            ):
                return rule, corrected
    return {"kind": "argmax"}, plain


def decide(sample: list[tuple[dict[str, Any], str]], rule: dict[str, Any]) -> list[str]:
    return [
        tuning.apply_rule(a["distribution"], rule)[1] if a["distribution"] is not None else a["top"] for a, _ in sample
    ]


def verdict(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    """The page rule with margin, on fixed answers."""
    rng = random.Random(SEED)
    n = len(gold)
    balanced, margins = [], []
    for _ in range(BOOTSTRAP):
        picks = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in picks]
        p = [predicted[i] for i in picks]
        balanced.append(score.balanced_accuracy(g, p))
        margins.append(sum(a == b for a, b in zip(g, p)) / n - majority_accuracy(g))
    at = int(BOOTSTRAP * BOOTSTRAP_QUANTILE)
    low_balanced, low_margin = sorted(balanced)[at], sorted(margins)[at]
    figures = metrics(gold, predicted)
    reasons = []
    if low_balanced < PAGE_MIN_BALANCED:
        reasons.append(f"balanced accuracy {figures['balanced_accuracy']:.3f}, 5th percentile {low_balanced:.3f}")
    if low_margin < 0:
        reasons.append(
            f"accuracy over the majority answer {figures['accuracy'] - majority_accuracy(gold):+.3f}, 5th percentile {low_margin:+.3f}"
        )
    for option, (right, total) in figures["recall"].items():
        if total >= PAGE_MIN_CLASS_RECORDS and right / total < PAGE_MIN_CLASS_RECALL:
            reasons.append(f"{option} right {right} of {total}")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "bootstrap_p05": {"balanced_accuracy": low_balanced, "accuracy_over_majority": low_margin},
        **figures,
        "majority_accuracy": majority_accuracy(gold),
    }


def dev_calls(paths: list[str]) -> list[dict[str, Any]]:
    calls = []
    for path in paths:
        for call in score.usable(score.load(Path(path))):
            if call["set"] != VULNERABILITY_TRIAGE:
                continue
            # Only dev calls are read; a test call is never seen.
            if call["split"] == "dev":
                calls.append(call)
    return calls


def answers_by(calls: list[dict[str, Any]], data_dir: Path, decided: bool) -> dict[tuple[str, str], dict]:
    """(backend, variant) -> case -> question -> answer."""
    cases = score.load(data_dir / f"inputs/{VULNERABILITY_TRIAGE}/cases.json")
    by_slug = {case["slug"]: case for case in cases["cases"]}
    out: dict[tuple[str, str], dict] = defaultdict(dict)
    for call in calls:
        case = by_slug[call["case"]]
        answers = score.NORMALISERS[call["transport"]](
            call, asked(VULNERABILITY_TRIAGE, call["backend"], score.questions_for(cases, case))
        )
        out[(call["backend"], call["variant"])][call["case"]] = score.decided(call, answers) if decided else answers
    return out


def gold_of(data_dir: Path) -> dict[str, dict[str, str]]:
    cases = score.load(data_dir / f"inputs/{VULNERABILITY_TRIAGE}/cases.json")
    return {case["slug"]: case["gold"] for case in cases["cases"]}


def rows_for(per_case: dict[str, dict], gold: dict[str, dict[str, str]], qid: str) -> list[tuple[dict, str]]:
    return [(answers[qid], gold[case][qid]) for case, answers in sorted(per_case.items()) if qid in answers]


def best_phrasing(tried: dict[str, dict[str, Any]]) -> str:
    """A phrasing that passes the page rule over one that does not, then the higher 5th percentile."""
    return max(
        tried,
        key=lambda p: (
            tried[p]["passes"],
            tried[p]["p05_balanced_accuracy"],
            tried[p]["balanced_accuracy"],
            -PHRASING_ORDER.index(p),
        ),
    )


def lane_call(call: dict[str, Any]) -> bool:
    """A call the lanes read: a tuned recording, or an LLM's described one."""
    return call["variant"] == tuning.TUNED or (call["backend"] in LLM_BACKENDS and call["variant"] == DESCRIBED)


def choose_phrasing(paths: list[str], data_dir: Path, tuned: dict[str, Any]) -> None:
    calls = [call for call in dev_calls(paths) if call["variant"] in PHRASING_ORDER]
    by = answers_by(calls, data_dir, decided=False)
    gold = gold_of(data_dir)
    target = tuned.setdefault("backends", {}).setdefault(VULNERABILITY_TRIAGE, {})
    for backend in sorted({backend for backend, _ in by} - set(LLM_BACKENDS)):
        phrasings: dict[str, Any] = {}
        for qid in asked(VULNERABILITY_TRIAGE, backend, VULNERABILITY_QUESTIONS):
            tried = {}
            for phrasing in PHRASING_ORDER:
                sample = rows_for(by.get((backend, phrasing), {}), gold, qid)
                if not sample:
                    continue
                rule, _ = fit_rule(sample)
                judged = verdict([g for _, g in sample], decide(sample, rule))
                tried[phrasing] = {
                    "rule": rule,
                    "passes": judged["eligible"],
                    "p05_balanced_accuracy": judged["bootstrap_p05"]["balanced_accuracy"],
                    **{k: judged[k] for k in ("accuracy", "balanced_accuracy")},
                }
            phrasings[qid] = {"phrasing": best_phrasing(tried), "dev": tried}
        target[backend] = {"phrasing": phrasings}
        print(f"{backend}: " + ", ".join(f"{qid}={p['phrasing']}" for qid, p in phrasings.items()))


def fit_rules(paths: list[str], data_dir: Path, tuned: dict[str, Any]) -> None:
    calls = [call for call in dev_calls(paths) if call["variant"] == tuning.TUNED]
    by = answers_by(calls, data_dir, decided=False)
    gold = gold_of(data_dir)
    target = tuned["backends"][VULNERABILITY_TRIAGE]
    for (backend, variant), per_case in sorted(by.items()):
        if variant != tuning.TUNED:
            continue
        settings: dict[str, Any] = {}
        for qid in asked(VULNERABILITY_TRIAGE, backend, VULNERABILITY_QUESTIONS):
            sample = rows_for(per_case, gold, qid)
            rule, _ = fit_rule(sample)
            settings[qid] = {"rule": rule, "dev": verdict([g for _, g in sample], decide(sample, rule))}
            report(backend, qid, rule, settings[qid]["dev"])
        target[backend]["questions"] = settings


def report(name: str, qid: str, rule: dict[str, Any] | None, figures: dict[str, Any]) -> None:
    mark = "PASS" if figures["eligible"] else "fail"
    kind = "" if rule is None else rule["kind"] + (f" {rule['cutoff']}" if rule["kind"] == "cutoff" else "")
    print(
        f"{name:<34}{qid:<24}{mark} {kind:<12} acc {figures['accuracy']:.3f} (maj {figures['majority_accuracy']:.3f})"
        f" bal {figures['balanced_accuracy']:.3f} p05 {figures['bootstrap_p05']['balanced_accuracy']:.3f}"
        + ("" if figures["eligible"] else f"  [{'; '.join(figures['reasons'])}]")
    )


def lane_inputs(calls: list[dict[str, Any]], data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """backend -> case -> question -> answer after its rules: the tuned recording, or an LLM's described one."""
    by = answers_by(calls, data_dir, decided=True)
    out = {backend: per_case for (backend, variant), per_case in by.items() if variant == tuning.TUNED}
    for (backend, variant), per_case in by.items():
        if backend in LLM_BACKENDS and variant == DESCRIBED:
            out[backend] = per_case
    return out


def judge_lane(
    lane: dict[str, Any], per_backend: dict, gold: dict[str, dict[str, str]], latency: dict, name: str
) -> dict[str, Any]:
    answers, escalated = lanes.lane_answers(lane, per_backend)
    cases = sorted(answers)
    out: dict[str, Any] = {**lane, "questions": {}}
    for qid in sorted({qid for a in answers.values() for qid in a}):
        rows = [(answers[c][qid]["top"], gold[c][qid]) for c in cases if qid in answers[c]]
        figures = verdict([g for _, g in rows], [p for p, _ in rows])
        if qid != lanes.SEVERITY and "escalate_to" in lane:
            figures["escalated"] = statistics.fmean(escalated[c][qid] for c in cases)
        out["questions"][qid] = figures
        report(f"lane {name}", qid, None, figures)
    out["latency_ms_median"] = statistics.median(record_latency(lane, c, escalated[c], latency) for c in cases)
    out["calls_per_record"] = statistics.fmean(record_calls(lane, c, escalated[c], latency) for c in cases)
    print(f"lane {name}: median {out['latency_ms_median']:.0f} ms per record, {out['calls_per_record']:.2f} calls")
    return out


def fit_cascade(per_backend: dict, gold: dict[str, dict[str, str]]) -> dict[str, Any]:
    """The cascade's cut-off per question, found on dev against Smart alone."""
    fast_answers = per_backend[CASCADE["backend"]]
    smart_answers = per_backend[CASCADE["escalate_to"]]
    cases = sorted(set(smart_answers) & set(fast_answers))
    cutoffs: dict[str, float] = {}
    for qid in sorted(next(iter(fast_answers.values()))):
        g = [gold[c][qid] for c in cases]
        s_pred = [smart_answers[c][qid]["top"] for c in cases]
        s_fig = metrics(g, s_pred)
        for cutoff in LANE_CUTOFFS:
            pred = [
                smart_answers[c][qid]["top"]
                if cutoff is not None and fast_answers[c][qid]["top_p"] < cutoff
                else fast_answers[c][qid]["top"]
                for c in cases
            ]
            fig = metrics(g, pred)
            regressions = sum(sp == gg and p != gg for sp, p, gg in zip(s_pred, pred, g))
            if (
                fig["accuracy"] >= s_fig["accuracy"]
                and fig["balanced_accuracy"] >= s_fig["balanced_accuracy"]
                and regressions == 0
            ):
                if cutoff is not None:
                    cutoffs[qid] = cutoff
                break
    return {**CASCADE, "cutoffs": cutoffs}


def fit_lanes(paths: list[str], data_dir: Path, tuned: dict[str, Any]) -> None:
    calls = [call for call in dev_calls(paths) if lane_call(call)]
    per_backend = lane_inputs(calls, data_dir)
    latency = request_latencies(calls)
    gold = gold_of(data_dir)
    out = {name: judge_lane(lane, per_backend, gold, latency, name) for name, lane in LANES.items()}
    cascade = judge_lane(fit_cascade(per_backend, gold), per_backend, gold, latency, "cascade")
    smart = out["smart"]
    shared = [qid for qid in smart["questions"] if qid in cascade["questions"] and qid != lanes.SEVERITY]
    answers, _ = lanes.lane_answers(cascade, per_backend)
    alone, _ = lanes.lane_answers(LANES["smart"], per_backend)
    regressions = sum(
        alone[c][qid]["top"] == gold[c][qid] and answers[c][qid]["top"] != gold[c][qid]
        for c in answers
        for qid in shared
    )
    control = {
        "matches_smart": all(
            cascade["questions"][qid][k] >= smart["questions"][qid][k]
            for qid in shared
            for k in ("accuracy", "balanced_accuracy")
        ),
        "regressions": regressions,
        "costs_less": cascade["calls_per_record"] < smart["calls_per_record"],
    }
    control["passes"] = control["matches_smart"] and regressions == 0 and control["costs_less"]
    cascade["control"] = control
    print(
        f"cascade against Smart alone: matches {control['matches_smart']}, regressions {regressions}, "
        f"{cascade['calls_per_record']:.2f} calls per record against {smart['calls_per_record']:.2f}: "
        + ("PASS" if control["passes"] else "fail")
    )
    tuned["lanes"] = out
    tuned["cascade"] = cascade


def request_latencies(calls: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(backend, case) -> {"total": ms, "requests": n, question id: ms} from the recordings lanes read."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for call in calls:
        if not lane_call(call):
            continue
        entry = {"total": call["timing"]["latency_ms"], "requests": len(call["requests"])}
        for request, ms in zip(call["requests"], call["timing"]["per_request_ms"]):
            if "question" in request:
                entry[request["question"]] = ms
        out[(call["backend"], call["case"])] = entry
    return out


def record_latency(lane: dict[str, Any], case: str, escalated: dict[str, bool], latency: dict) -> float:
    total = latency[(lane["backend"], case)]["total"]
    if "escalate_to" in lane:
        second = latency[(lane["escalate_to"], case)]
        total += sum(second[qid] for qid, up in escalated.items() if up)
    return total


def record_calls(lane: dict[str, Any], case: str, escalated: dict[str, bool], latency: dict) -> int:
    calls = latency[(lane["backend"], case)]["requests"]
    if "escalate_to" in lane:
        calls += sum(escalated.values())
    return calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("phrasing", "rules", "lanes"))
    parser.add_argument("--calls", nargs="+", required=True, help="dev recordings")
    parser.add_argument("--data", default="data", help="directory holding inputs/ (default: data)")
    args = parser.parse_args()
    tuned = tuning.load()
    if args.stage == "phrasing":
        choose_phrasing(args.calls, Path(args.data), tuned)
    elif args.stage == "rules":
        fit_rules(args.calls, Path(args.data), tuned)
    else:
        fit_lanes(args.calls, Path(args.data), tuned)
    tuning.TUNING_PATH.write_text(json.dumps(tuned, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {tuning.TUNING_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
