#!/usr/bin/env python3
"""Score the recorded answers against each case's gold, per backend and question type.

    python3 fetch.py
    python3 score.py                                  # reads data/calls.json
    python3 score.py --calls run-output/calls.json    # or any recordings
    python3 score.py --json summary.json              # also write every figure as JSON

Standard library only. No API key, no model, no inference spend.

For each set, backend and question type it reports:

    accuracy     the answer's top option equals the gold option
    macro F1     F1 per option, averaged over the options that occur in the gold
                 or the answers of that question, then over the questions
    Brier        mean squared error of the returned probabilities against the
                 gold option; only where the backend returns a probability for
                 every option (GLiNER2 in single-label mode returns the top one)
    ECE          expected calibration error of the top option's probability,
                 10 equal-width bins
    score MAE    for rubric questions, |expected level - gold level|, where the
                 expected level is the probability-weighted mean when every
                 level has a probability and the top level otherwise

and two references computed on the same cases: the majority option of each
question (what a model that ignores the input scores at best), and for yes-or-no
questions the Brier score of always answering the base rate.

On vulnerability triage a backend is scored in its `tuned` configuration: the
phrasing and decision rule per question that tune.py chose on the dev slice
(PREREGISTRATION.md). Each backend's severity is computed from its answers
(cvss.compose_severity) and scored on the exact level. Rubric answers are never
scored within one level.

Latency is the recorded wall time per case, all of that case's requests, on the
recording machine, one case at a time. It is printed for orientation only.

PUBLISHED holds every figure the page and the README publish from this
recording. The script exits nonzero whenever the recording stops reproducing
one of them. page.py asserts the page's cards, speed claims and withdrawals.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cvss
import lanes
import tuning
from questions import (
    CHOICE,
    DESCRIBED,
    NOUL,
    SCORE,
    SHORT,
    TYPES,
    VULNERABILITY_QUESTIONS,
    VULNERABILITY_TRIAGE,
    WORKFLOWS,
    asked,
    from_source_question,
    label_options,
    phrasing_of,
    typed_questions,
)

HTTP_OK = "ok"
BINS = 10
# Balanced accuracy averages recall over options with at least this many records.
BALANCED_MIN_RECORDS = 10
# The confidence cut-offs selective accuracy is reported at, for pick-one questions.
SELECTIVE_CUTOFFS = (0.5, 0.7, 0.8, 0.9, 0.95)
SUM_TOLERANCE = 0.02

# "<set>/<backend>/<scope>/<metric>" -> a figure the page or the README
# publishes. The scope is a question type (choice, noul, score), "all", or a
# question id. `correct` is a count; every other metric is rounded to 3.
# All test slice, in each backend's tuned configuration (the LLM's described).
VT, WF = VULNERABILITY_TRIAGE, WORKFLOWS
PUBLISHED: dict[str, float] = {
    # The board: Smart (knowledgator/gliclass-large-v1.0, one call per record)
    f"{VT}/gliclass-large-v1-one-call/weakness/correct": 142,
    f"{VT}/gliclass-large-v1-one-call/weakness/balanced_accuracy": 0.887,
    f"{VT}/gliclass-large-v1-one-call/attack_vector/correct": 147,
    f"{VT}/gliclass-large-v1-one-call/attack_vector/balanced_accuracy": 0.796,
    f"{VT}/gliclass-large-v1-one-call/remote_unauthenticated/correct": 115,
    f"{VT}/gliclass-large-v1-one-call/remote_unauthenticated/balanced_accuracy": 0.725,
    # The figures row: Smart's pick-one and yes-or-no answers
    f"{VT}/gliclass-large-v1-one-call/choice/accuracy": 0.903,
    f"{VT}/gliclass-large-v1-one-call/noul/accuracy": 0.719,
    # The board: Fast (knowledgator/gliformer-large-v1)
    f"{VT}/gliformer-large/attack_vector/correct": 144,
    f"{VT}/gliformer-large/attack_vector/balanced_accuracy": 0.873,
    # The board: the LLM (Qwen/Qwen3-4B-Instruct-2507)
    f"{VT}/qwen3-4b-instruct/attack_vector/correct": 150,
    f"{VT}/qwen3-4b-instruct/attack_vector/balanced_accuracy": 0.867,
    f"{VT}/qwen3-4b-instruct/remote_unauthenticated/correct": 129,
    f"{VT}/qwen3-4b-instruct/remote_unauthenticated/balanced_accuracy": 0.815,
    # The catalog
    f"{VT}/gliclass-instruct-large/weakness/correct": 142,
    f"{VT}/gliclass-instruct-large/attack_vector/correct": 149,
    f"{VT}/laya/weakness/correct": 134,
    f"{VT}/laya-typed-decisions/weakness/correct": 145,
    f"{VT}/gliner2.5-decide/weakness/correct": 146,
    f"{VT}/gliner2.5-multi-decide/weakness/correct": 137,
    f"{VT}/gliner2.5-decide-1b/weakness/correct": 139,
    f"{VT}/gliner2.5-decide-1b/attack_vector/correct": 150,
    f"{VT}/gliner2.5-decide-1b/attack_vector/balanced_accuracy": 0.852,
    # The README: computed severity, reported and not shown
    f"{VT}/gliclass-large-v1-one-call/severity/accuracy": 0.606,
    f"{VT}/gliclass-large-v1-one-call/severity/balanced_accuracy": 0.612,
    f"{VT}/qwen3-4b-instruct/severity/accuracy": 0.675,
    f"{VT}/qwen3-4b-instruct/severity/balanced_accuracy": 0.613,
    # The README: the fine-tuned Laya checkpoint on the workflow benchmark's teacher labels
    f"{WF}/laya-typed-decisions/all/accuracy": 0.769,
    f"{WF}/laya-typed-decisions/all/macro_f1": 0.663,
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def probability(value: Any, where: str) -> float:
    """A returned probability, refusing anything that is not a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"{where}: non-numeric probability {value!r}")
    if not -1e-6 <= value <= 1 + 1e-6:
        raise SystemExit(f"{where}: probability {value!r} outside [0, 1]")
    return min(1.0, max(0.0, float(value)))


def full_distribution(raw: dict[str, float], keys: list[str], where: str) -> dict[str, float]:
    if set(raw) != set(keys):
        raise SystemExit(f"{where}: answered options {sorted(raw)}, expected {sorted(keys)}")
    total = sum(raw.values())
    if total <= 0 or abs(total - 1) > SUM_TOLERANCE:
        raise SystemExit(f"{where}: probabilities sum to {total}")
    return {key: value / total for key, value in raw.items()}


def answer(distribution: dict[str, float] | None, top: str, top_p: float, qtype: str) -> dict[str, Any]:
    return {"distribution": distribution, "top": top, "top_p": top_p, "type": qtype}


def phrasing_for(call: dict[str, Any]) -> Any:
    """The phrasing a call was sent in: its variant, or the tuned per-question mix it stands for."""
    return tuning.phrasing(call["set"], call["backend"], call["variant"])


def from_typed(call: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers from a typed decision model's response: one response holding every question."""
    _, label_maps = typed_questions(questions, phrasing_for(call))
    response = call["responses"][0]
    # The package returns `answers`; the SIE adapter returns the same dict as `data`.
    returned = response["answers"] if call["transport"] == "laya-python-package" else response["data"]
    answers = {}
    for qid, question in questions.items():
        where = f"{call['id']} {qid}"
        if qid not in returned:
            raise SystemExit(f"{where}: no answer")
        entry = returned[qid]
        option_keys = [option["key"] for option in question["options"]]
        if question["type"] == NOUL:
            p_true = probability(entry["noul"], where)
            distribution = {"true": p_true, "false": 1.0 - p_true}
        else:
            raw: dict[str, float] = {}
            for label, value in entry["probabilities"].items():
                if label not in label_maps[qid]:
                    raise SystemExit(f"{where}: returned option {label!r} that was not sent")
                raw[label_maps[qid][label]] = probability(value, where)
            distribution = full_distribution(raw, option_keys, where)
        top = max(option_keys, key=lambda key: distribution[key])
        answers[qid] = answer(distribution, top, distribution[top], question["type"])
    return answers


def from_labels(call: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers from a label scorer: one response per question, in request order."""
    answers = {}
    if len(call["responses"]) != len(call["requests"]):
        raise SystemExit(f"{call['id']}: {len(call['responses'])} responses to {len(call['requests'])} requests")
    for request, response in zip(call["requests"], call["responses"]):
        qid = request["question"]
        question = questions[qid]
        where = f"{call['id']} {qid}"
        key_of = {label: key for key, label in label_options(question, phrasing_of(phrasing_for(call), qid))}
        raw: dict[str, float] = {}
        for entry in response["classifications"]:
            if entry["label"] not in key_of:
                raise SystemExit(f"{where}: returned label {entry['label']!r} that was not sent")
            if key_of[entry["label"]] in raw:
                raise SystemExit(f"{where}: label {entry['label']!r} returned twice")
            raw[key_of[entry["label"]]] = probability(entry["score"], where)
        if not raw:
            raise SystemExit(f"{where}: no classification returned")
        option_keys = [option["key"] for option in question["options"]]
        if len(raw) == len(option_keys):
            distribution = full_distribution(raw, option_keys, where)
        elif len(raw) == 1 and len(option_keys) == 2:  # noqa: PLR2004
            # A single-label answer over two options fixes the other one.
            (top_key, top_p), other = next(iter(raw.items())), next(k for k in option_keys if k not in raw)
            distribution = {top_key: top_p, other: 1.0 - top_p}
        elif len(raw) == 1:
            top_key, top_p = next(iter(raw.items()))
            answers[qid] = answer(None, top_key, top_p, question["type"])
            continue
        else:
            raise SystemExit(f"{where}: {len(raw)} of {len(option_keys)} options returned")
        top = max(option_keys, key=lambda key: distribution[key])
        answers[qid] = answer(distribution, top, distribution[top], question["type"])
    return answers


def distribution_over(
    raw: dict[str, float], question: dict[str, Any], variant: str, where: str, *, missing_is_zero: bool = False
) -> dict[str, float]:
    """Map `{label string: score}` onto option keys and normalise it to sum to one."""
    key_of = {label: key for key, label in label_options(question, variant)}
    keyed: dict[str, float] = {}
    for label, value in raw.items():
        if label not in key_of:
            raise SystemExit(f"{where}: returned label {label!r} that was not sent")
        keyed[key_of[label]] = probability(value, where)
    option_keys = [option["key"] for option in question["options"]]
    if missing_is_zero:
        keyed = {key: keyed.get(key, 0.0) for key in option_keys}
    if set(keyed) != set(option_keys):
        raise SystemExit(f"{where}: answered options {sorted(keyed)}, expected {sorted(option_keys)}")
    total = sum(keyed.values())
    if total <= 0:
        raise SystemExit(f"{where}: every option scored zero")
    return {key: value / total for key, value in keyed.items()}


def from_group_answers(call: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers from one grouped GLiClass call: `data[group]` is a choice answer per question.

    Each group was scored with its own softmax, so its probabilities already
    sum to one; they are still checked and normalised the same way.
    """
    request = call["requests"][0]
    data = call["responses"][0]["data"]
    answers = {}
    for group, qid in request["groups"].items():
        where = f"{call['id']} {qid}"
        if group not in data:
            raise SystemExit(f"{where}: no answer for group {group!r}")
        question = questions[qid]
        distribution = distribution_over(
            data[group]["probabilities"], question, phrasing_of(phrasing_for(call), qid), where
        )
        option_keys = [option["key"] for option in question["options"]]
        top = max(option_keys, key=lambda key: distribution[key])
        answers[qid] = answer(distribution, top, distribution[top], question["type"])
    if set(answers) != set(questions):
        raise SystemExit(f"{call['id']}: groups cover {sorted(answers)}, questions are {sorted(questions)}")
    return answers


def from_group_labels(call: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers from one grouped GLiFormer call: flat `"group.label"` classifications.

    GLiFormer scores each label with its own sigmoid and only reports labels
    above the threshold (sent as 0, which the model reads as 1e-6), so a label
    that is absent counts as zero. A question's scores are divided by their sum;
    that makes the top option and Brier and ECE computable, not calibrated.
    Labels can contain dots, so each one is matched against the group names
    that were sent rather than split.
    """
    request = call["requests"][0]
    groups = request["groups"]
    raw: dict[str, dict[str, float]] = {group: {} for group in groups}
    for entry in call["responses"][0]["classifications"]:
        owner = next((group for group in groups if entry["label"].startswith(f"{group}.")), None)
        if owner is None:
            raise SystemExit(f"{call['id']}: classification {entry['label']!r} belongs to no group that was sent")
        label = entry["label"][len(owner) + 1 :]
        if label in raw[owner]:
            raise SystemExit(f"{call['id']}: {entry['label']!r} returned twice")
        raw[owner][label] = entry["score"]
    answers = {}
    for group, qid in groups.items():
        where = f"{call['id']} {qid}"
        question = questions[qid]
        distribution = distribution_over(
            raw[group], question, phrasing_of(phrasing_for(call), qid), where, missing_is_zero=True
        )
        option_keys = [option["key"] for option in question["options"]]
        top = max(option_keys, key=lambda key: distribution[key])
        answers[qid] = answer(distribution, top, distribution[top], question["type"])
    return answers


LLM_CODES = {
    "attack_vector": {"network": "N", "adjacent": "A", "local": "L", "physical": "P"},
    "attack_complexity": {"low": "L", "high": "H"},
    "privileges_required": {"none": "N", "low": "L", "high": "H"},
    "user_interaction": {"none": "N", "required": "R"},
    "scope": {"unchanged": "U", "changed": "C"},
    "confidentiality": {"none": "N", "low": "L", "high": "H"},
    "integrity": {"none": "N", "low": "L", "high": "H"},
    "availability": {"none": "N", "low": "L", "high": "H"},
}


def from_cvss_llm(call: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers read off an LLM's CVSS v3.1 base metrics and weakness class.

    Every answer is the one the metrics imply, with probability 1: the LLM
    returns a value, not a distribution, so it has no probability to branch on.
    `severity` is computed from the full vector, attack complexity included.
    """
    parsed = call["responses"][0]["parsed"]
    where = call["id"]
    try:
        metrics = {
            code: LLM_CODES[field][parsed[field]]
            for field, code in (
                ("attack_vector", "AV"),
                ("attack_complexity", "AC"),
                ("privileges_required", "PR"),
                ("user_interaction", "UI"),
                ("scope", "S"),
                ("confidentiality", "C"),
                ("integrity", "I"),
                ("availability", "A"),
            )
        }
    except KeyError as error:
        raise SystemExit(f"{where}: CVSS field missing or out of range: {error}") from error
    weakness_key = {option["name"]: option["key"] for option in questions["weakness"]["options"]}.get(
        parsed.get("weakness")
    )
    if weakness_key is None:
        raise SystemExit(f"{where}: weakness {parsed.get('weakness')!r} is not an option")
    implied = {
        "weakness": weakness_key,
        "attack_vector": parsed["attack_vector"],
        "user_interaction": "true" if metrics["UI"] == "R" else "false",
        **cvss.derived_gold("CVSS:3.1/" + "/".join(f"{k}:{v}" for k, v in metrics.items())),
    }
    answers = {}
    for qid, question in questions.items():
        keys = [option["key"] for option in question["options"]]
        distribution = {key: 1.0 if key == implied[qid] else 0.0 for key in keys}
        answers[qid] = answer(distribution, implied[qid], 1.0, question["type"])
    level = cvss.severity(cvss.base_score(metrics))
    if level is not None:
        answers[lanes.SEVERITY] = answer(
            {lvl: float(lvl == level) for lvl in cvss.SEVERITY_LEVELS}, level, 1.0, "score"
        )
    return answers


NORMALISERS = {
    "laya-python-package": from_typed,
    "sie-extract-typed": from_typed,
    "sie-extract-groups": from_group_answers,
    "sie-extract-group-labels": from_group_labels,
    "sie-extract": from_labels,
    "sie-chat-cvss": from_cvss_llm,
}


def questions_for(cases: dict[str, Any], case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if cases["set"] == VULNERABILITY_TRIAGE:
        if cases["questions"] != VULNERABILITY_QUESTIONS:
            raise SystemExit("the case file's questions differ from questions.py; rebuild or re-fetch the inputs")
        return cases["questions"]
    source = cases["questions"][case["workflow"]]
    return {qid: from_source_question(question) for qid, question in source.items()}


def ece(pairs: list[tuple[float, bool]]) -> float:
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for confidence, correct in pairs:
        bins[min(BINS - 1, int(confidence * BINS))].append((confidence, correct))
    total = len(pairs)
    return sum(
        len(members)
        / total
        * abs(sum(c for c, _ in members) / len(members) - sum(1 for _, ok in members if ok) / len(members))
        for members in bins.values()
    )


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def class_recall(gold: list[str], predicted: list[str]) -> dict[str, list[int]]:
    """Per gold option: [answered right, records]."""
    counts: dict[str, list[int]] = {}
    for g, p in zip(gold, predicted):
        counts.setdefault(g, [0, 0])
        counts[g][1] += 1
        counts[g][0] += g == p
    return dict(sorted(counts.items()))


def balanced_accuracy(gold: list[str], predicted: list[str]) -> float:
    """Mean recall over the options with at least BALANCED_MIN_RECORDS records in the gold.

    An option with one or two records swings an unweighted mean by a third or
    more on a single answer, so rarer options count in accuracy only. When no
    option has that many records, every option counts.
    """
    recalls = class_recall(gold, predicted)
    common = [right / total for right, total in recalls.values() if total >= BALANCED_MIN_RECORDS]
    return statistics.fmean(common or [right / total for right, total in recalls.values()])


def selective(rows: list[tuple[dict[str, Any], str]], cutoffs: tuple[float, ...] = SELECTIVE_CUTOFFS) -> dict:
    """For each cut-off, how many answers were at least that sure, and how many of those were right."""
    out = {}
    for cutoff in cutoffs:
        sure = [(a, g) for a, g in rows if a["top_p"] >= cutoff]
        out[f"{cutoff:.2f}"] = {"answered": len(sure), "right": sum(a["top"] == g for a, g in sure)}
    return out


def brier(distribution: dict[str, float], gold: str) -> float:
    return sum((p - (1.0 if key == gold else 0.0)) ** 2 for key, p in distribution.items())


def expected_level(item: dict[str, Any]) -> float:
    if item["distribution"] is None:
        return float(item["top"])
    return sum(int(key) * p for key, p in item["distribution"].items())


def question_metrics(rows: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    """Figures for one question over its cases. `rows` pairs each answer with its gold key."""
    gold = [g for _, g in rows]
    predicted = [a["top"] for a, _ in rows]
    correct = [a["top"] == g for a, g in rows]
    qtype = rows[0][0]["type"]
    out: dict[str, Any] = {
        "type": qtype,
        "n": len(rows),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(rows),
        "balanced_accuracy": balanced_accuracy(gold, predicted),
        "recall": class_recall(gold, predicted),
        "macro_f1": macro_f1(gold, predicted),
        "ece": ece([(a["top_p"], ok) for (a, _), ok in zip(rows, correct)]),
        "brier": None,
    }
    if all(a["distribution"] is not None for a, _ in rows):
        if qtype == NOUL:
            out["brier"] = statistics.fmean((a["distribution"]["true"] - (g == "true")) ** 2 for a, g in rows)
        else:
            out["brier"] = statistics.fmean(brier(a["distribution"], g) for a, g in rows)
    if qtype == CHOICE:
        out["selective"] = selective(rows)
    if qtype == SCORE:
        out["mae"] = statistics.fmean(abs(expected_level(a) - int(g)) for a, g in rows)
    majority, count = Counter(gold).most_common(1)[0]
    out["majority"] = {
        "option": majority,
        "accuracy": count / len(rows),
        "macro_f1": macro_f1(gold, [majority] * len(rows)),
    }
    if qtype == NOUL:
        rate = sum(g == "true" for g in gold) / len(gold)
        out["majority"]["base_rate_brier"] = rate * (1 - rate)
    return out


def pooled(per_question: dict[str, dict[str, Any]], qtype: str | None) -> dict[str, Any] | None:
    chosen = [m for m in per_question.values() if qtype is None or m["type"] == qtype]
    if not chosen:
        return None
    n = sum(m["n"] for m in chosen)
    out = {
        "questions": len(chosen),
        "decisions": n,
        "accuracy": sum(m["correct"] for m in chosen) / n,
        "macro_f1": statistics.fmean(m["macro_f1"] for m in chosen),
        "ece": sum(m["ece"] * m["n"] for m in chosen) / n,
        "brier": None,
        "majority_accuracy": sum(m["majority"]["accuracy"] * m["n"] for m in chosen) / n,
        "majority_macro_f1": statistics.fmean(m["majority"]["macro_f1"] for m in chosen),
    }
    if all(m["brier"] is not None for m in chosen):
        out["brier"] = sum(m["brier"] * m["n"] for m in chosen) / n
    if qtype == SCORE:
        out["mae"] = sum(m["mae"] * m["n"] for m in chosen) / n
    if qtype == NOUL:
        out["base_rate_brier"] = sum(m["majority"]["base_rate_brier"] * m["n"] for m in chosen) / n
    return out


def decided(call: dict[str, Any], answers: dict[str, dict[str, Any]], rules: dict | None = None) -> dict:
    """Answers after the call's decision rules (tuning.json); unchanged for an untuned variant.

    A rule needs every option's probability, so it leaves an answer that only
    carries its top option (GLiNER2's pick-one) as it is.
    """
    rules = tuning.rules(call["set"], call["backend"], call["variant"]) if rules is None else rules
    out = {}
    for qid, item in answers.items():
        rule = rules.get(qid)
        if rule is None or item["distribution"] is None:
            out[qid] = item
            continue
        distribution, top = tuning.apply_rule(item["distribution"], rule)
        out[qid] = answer(distribution, top, distribution[top], item["type"])
    return out


def score_slice(calls: list[dict[str, Any]], cases: dict[str, Any]) -> dict[str, Any]:
    by_slug = {case["slug"]: case for case in cases["cases"]}
    rows: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    latencies = []
    for call in calls:
        case = by_slug.get(call["case"])
        if case is None:
            raise SystemExit(f"{call['id']}: no such case in the inputs")
        questions = asked(cases["set"], call["backend"], questions_for(cases, case))
        answers = decided(call, NORMALISERS[call["transport"]](call, questions))
        prefix = f"{case['workflow']}/" if cases["set"] == WORKFLOWS else ""
        for qid in questions:
            rows[prefix + qid].append((answers[qid], case["gold"][qid]))
        if cases["set"] == VULNERABILITY_TRIAGE:
            composed = lanes.severity_answer(answers)
            if composed is not None:
                rows[lanes.SEVERITY].append((composed, case["gold"]["severity"]))
        latencies.append(call["timing"]["latency_ms"])
    per_question = {qid: question_metrics(pairs) for qid, pairs in sorted(rows.items())}
    return {
        "cases": len(calls),
        "per_question": per_question,
        "by_type": {qtype: pooled(per_question, qtype) for qtype in TYPES},
        "overall": pooled(per_question, None),
        "latency_ms": {
            "median": statistics.median(latencies),
            "p90": sorted(latencies)[max(0, math.ceil(0.9 * len(latencies)) - 1)],
            "requests_per_case": statistics.fmean(len(call["requests"]) for call in calls),
        },
    }


def usable(payload: dict[str, Any]) -> list[dict[str, Any]]:
    failed = [call["id"] for call in payload["calls"] if call.get("status") != HTTP_OK]
    if failed:
        raise SystemExit("refusing to score calls that failed: " + ", ".join(sorted(failed)[:10]))
    return payload["calls"]


def choose_variant(dev: dict[str, dict[str, Any]]) -> str:
    """The phrasing with the higher pooled dev accuracy, `described` on a tie."""
    if SHORT not in dev:
        return DESCRIBED
    if DESCRIBED not in dev:
        return SHORT
    return SHORT if dev[SHORT]["overall"]["accuracy"] > dev[DESCRIBED]["overall"]["accuracy"] else DESCRIBED


def summarise(calls: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[(call["set"], call["backend"], call["variant"], call["split"])].append(call)
    summary: dict[str, Any] = {}
    for set_name in sorted({key[0] for key in grouped}):
        cases = load(data_dir / f"inputs/{set_name}/cases.json")
        summary[set_name] = {}
        for backend in sorted({key[1] for key in grouped if key[0] == set_name}):
            variants = sorted({key[2] for key in grouped if key[:2] == (set_name, backend)})
            dev = {
                v: score_slice(grouped[(set_name, backend, v, "dev")], cases)
                for v in variants
                if grouped.get((set_name, backend, v, "dev"))
            }
            test = {
                v: score_slice(grouped[(set_name, backend, v, "test")], cases)
                for v in variants
                if grouped.get((set_name, backend, v, "test"))
            }
            if tuning.TUNED in variants:
                chosen = tuning.TUNED
            else:
                chosen = choose_variant(dev) if dev else (DESCRIBED if DESCRIBED in test else variants[0])
            any_call = next(calls for key, calls in grouped.items() if key[:2] == (set_name, backend) and calls)
            summary[set_name][backend] = {
                "model": any_call[0]["model"],
                "chosen_variant": chosen,
                "dev": dev,
                "test": test,
            }
    return summary


def fmt(value: float | None, digits: int = 3) -> str:
    return "  n/a" if value is None else f"{value:.{digits}f}"


def report(summary: dict[str, Any], split: str = "test") -> None:
    for set_name, all_backends in summary.items():
        backends = {b: r for b, r in all_backends.items() if r["chosen_variant"] in r[split]}
        for backend in sorted(set(all_backends) - set(backends)):
            print(f"\nnote: {set_name}/{backend} has no test calls in its chosen phrasing; not reported")
        if not backends:
            continue
        print(f"\n== {set_name} ({split} slice) ==")
        first = next(iter(backends.values()))
        test = first[split][first["chosen_variant"]]
        print(f"{test['cases']} cases; majority-option reference per type:")
        for qtype in (*TYPES, None):
            ref = test["overall"] if qtype is None else test["by_type"][qtype]
            if ref:
                line = f"  {qtype or 'all':<7} acc {fmt(ref['majority_accuracy'])}  macro-F1 {fmt(ref['majority_macro_f1'])}"
                if qtype == NOUL:
                    line += f"  base-rate Brier {fmt(ref['base_rate_brier'])}"
                print(line)
        header = f"  {'backend':<22}{'phrasing':<11}{'type':<8}{'acc':>7}{'mF1':>7}{'Brier':>7}{'ECE':>7}{'MAE':>7}"
        print(header)
        for backend, result in backends.items():
            for variant in (result["chosen_variant"], *[v for v in result[split] if v != result["chosen_variant"]]):
                metrics = result[split][variant]
                tag = variant if variant == result["chosen_variant"] else f"({variant})"
                for qtype in (*TYPES, None):
                    m = metrics["overall"] if qtype is None else metrics["by_type"][qtype]
                    if m is None:
                        continue
                    print(
                        f"  {backend:<22}{tag:<11}{qtype or 'all':<8}{fmt(m['accuracy']):>7}{fmt(m['macro_f1']):>7}"
                        f"{fmt(m['brier']):>7}{fmt(m['ece']):>7}{fmt(m.get('mae')):>7}"
                    )
            latency = result[split][result["chosen_variant"]]["latency_ms"]
            dev_line = ""
            if result["dev"]:
                dev_line = "  dev acc " + ", ".join(
                    f"{v} {fmt(d['overall']['accuracy'])}" for v, d in sorted(result["dev"].items())
                )
            print(
                f"  {backend:<22}latency per case: median {latency['median']:.0f} ms, p90 {latency['p90']:.0f} ms, "
                f"{latency['requests_per_case']:.0f} request(s) per case{dev_line}"
            )
        print("\n  per question, chosen phrasing: accuracy / balanced accuracy; majority in brackets;")
        print("  for pick-one questions, right of answered when the top option is at least 0.9")
        per_backend = {b: r[split][r["chosen_variant"]]["per_question"] for b, r in backends.items()}
        qids = sorted({qid for per_question in per_backend.values() for qid in per_question})
        for qid in qids:
            majority = next(pq[qid] for pq in per_backend.values() if qid in pq)["majority"]["accuracy"]
            print(f"    {qid} [{majority:.2f}]")
            for backend, per_question in per_backend.items():
                if qid not in per_question:
                    print(f"      {backend:<32}  not asked")
                    continue
                m = per_question[qid]
                sure = m.get("selective", {}).get("0.90")
                extra = f"   >=0.9: {sure['right']}/{sure['answered']}" if sure else ""
                print(f"      {backend:<32}{fmt(m['accuracy'], 3)} / {fmt(m['balanced_accuracy'], 3)}{extra}")


def check_published(summary: dict[str, Any]) -> list[str]:
    failures = []
    for key, expected in PUBLISHED.items():
        set_name, backend, scope, metric = key.split("/")
        result = summary.get(set_name, {}).get(backend)
        if result is None or result["chosen_variant"] not in result["test"]:
            failures.append(f"{key}: not in this recording")
            continue
        metrics = result["test"][result["chosen_variant"]]
        if scope == "all":
            block = metrics["overall"]
        elif scope in TYPES:
            block = metrics["by_type"][scope]
        else:
            block = metrics["per_question"].get(scope)
        if block is None or metric not in block:
            failures.append(f"{key}: not in this recording")
            continue
        actual = block[metric] if metric == "correct" else round(block[metric], 3)
        if actual != expected:
            failures.append(f"{key}: got {actual}, published {expected}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="directory holding inputs/ (default: data)")
    parser.add_argument("--calls", nargs="+", default=["data/calls.json"], help="recordings to score")
    parser.add_argument("--json", help="also write the full summary to this path")
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="which slice to report")
    args = parser.parse_args()
    calls: list[dict[str, Any]] = []
    for path in args.calls:
        payload = load(Path(path))
        if payload.get("complete") is False:
            print(f"note: {path} is not a complete recording", file=sys.stderr)
        calls.extend(usable(payload))
    summary = summarise(calls, Path(args.data))
    report(summary, args.split)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    failures = check_published(summary)
    if failures:
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    if PUBLISHED:
        print(f"\nAll {len(PUBLISHED)} published figures reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
