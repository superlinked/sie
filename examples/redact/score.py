#!/usr/bin/env python3
"""Re-derive every figure the /redact page publishes from the recorded run.

    python3 fetch.py
    python3 score.py                          # every figure, checked against the recording
    python3 score.py --no-bootstrap           # skip the 95% intervals
    python3 score.py --sie-rows run-output    # score your own SIE run from run.py --all

Standard library only. No API key, no network, no inference spend.

The run: 660 documents from the English test split of
gretelai/synthetic_pii_finance_multilingual, 1,792 in-scope personal-data
spans, five arms. The figure is coverage recall: the share of in-scope gold
spans whose every non-space character sits under the union of an arm's masks,
whatever label the arm gave it.

SIE's arm is a composition of two models on https://api.superlinked.com:

1. `urchade/gliner_multi_pii-v1` and `numind/NuNER_Zero`, the same 36 labels,
   each span kept at score >= 0.6. A document over 300 words is sent as
   300-word windows with a 50-word overlap, and each window's offset is added
   back to its spans.
2. The two models' spans unioned.
3. Every other whole-word, case-sensitive occurrence of any three-letter or
   longer token of a returned `person` span masked, in the caller's code.

An LLM returns strings, not offsets, so every occurrence of a returned string
is masked. Presidio and OpenAI Privacy Filter return offsets; every span is
masked.

Every figure is compared with the recorded results file and with the value the
page publishes. A mismatch is a failure and the script exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

SEED = 20260930
RESAMPLES = 2000
FIRST_MODEL = "urchade/gliner_multi_pii-v1"
SECOND_MODEL = "numind/NuNER_Zero"
SCORE_FLOOR = 0.6
WINDOW_WORDS = 300
WINDOW_OVERLAP = 50
# GLiNER's WhitespaceTokenSplitter: a run of word characters, or one other
# character. This is the unit its 384-word input window counts.
GLINER_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")

# The 36 labels every request named, SIE and LLM alike: 34 for HIPAA Safe
# Harbor identifiers and credentials, plus two for Gretel types the 34 have no
# word for.
REQUEST_LABELS = [
    "person", "street address", "city", "county", "postal code", "gps coordinates", "date of birth",
    "phone number", "fax number", "email address", "url", "ip address", "mac address", "social security number",
    "national id number", "tax id number", "medical record number", "health plan beneficiary number",
    "account number", "credit card number", "cvv", "pin", "password", "api key", "http cookie", "username",
    "customer id", "employee id", "certificate or license number", "license plate",
    "vehicle identification number", "device identifier", "biometric identifier", "unique identifier",
    "passport number", "driver's license number",
]  # fmt: skip
PERSON_LABEL = "person"
LLM_PROMPT = (
    "Find every piece of personally identifiable information in the text. Use these types: {labels}. "
    "Copy each mention exactly as it appears in the text. Return an empty list if there is none."
)

# Gretel labels that count: the HIPAA Safe Harbor identifiers the set labels,
# plus credentials. Company names, dates, times, routing numbers and SWIFT
# codes do not.
IN_SCOPE = frozenset(
    {
        "name", "first_name", "last_name", "street_address", "local_latlng", "date_of_birth", "phone_number",
        "email", "ssn", "bban", "iban", "credit_card_number", "driver_license_number", "passport_number",
        "employee_id", "customer_id", "user_name", "ipv4", "ipv6", "password", "account_pin",
        "credit_card_security_code", "api_key",
    }
)  # fmt: skip
# In-scope Gretel labels with no AWS Comprehend DetectPiiEntities type in AWS's
# documentation (read 2026-09-30).
NO_COMPREHEND_TYPE = frozenset({"local_latlng", "employee_id", "customer_id"})

# A name token is three letters or more, matched whole-word and case-sensitively.
NAME_TOKEN = re.compile(r"[^\W\d_]{3,}")
# Currency amounts, for the "what stays readable" figure. It also matches some IP addresses.
AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d.,]*\s?EUR|(?<![\w-])\d+\.\d{2}(?![\d-])")

COMPOSITION = "sie"
# (arm id, display name, rows file). The arm id is the key in results/gretel-main_results.json.
ARMS = [
    (COMPOSITION, "SIE (GLiNER PII + NuNER Zero, composed)", None),
    ("llm:claude-haiku-4-5", "Claude Haiku 4.5", "llm__claude-haiku-4-5.jsonl"),
    ("llm:gpt-6-luna", "GPT-6 Luna", "llm__gpt-6-luna.jsonl"),
    ("privacy-filter", "OpenAI Privacy Filter", "privacy-filter.jsonl"),
    ("presidio", "Microsoft Presidio", "presidio.jsonl"),
    (f"sie:{FIRST_MODEL}", "SIE GLiNER PII alone, one request", "sie__urchade__gliner_multi_pii-v1.jsonl"),
    (f"sie:{SECOND_MODEL}", "SIE NuNER Zero alone, one request", "sie__numind__NuNER_Zero.jsonl"),
]
RECORDED_KEY = {COMPOSITION: "sie-composition"}

# What the page publishes: spans masked, of 1,792, per arm.
PUBLISHED_MASKED = {
    COMPOSITION: 1591,
    "llm:claude-haiku-4-5": 1490,
    "llm:gpt-6-luna": 1448,
    "privacy-filter": 1214,
    "presidio": 859,
}
PUBLISHED_GOLD = 1792
PUBLISHED_DOCUMENTS = 660

# --- price inputs, list prices read on 2026-09-30 ------------------------------------------------------------------
MONTHLY_DOCUMENTS = 1_000_000
# SIE: $ per 1M input tokens, and the tokens each model's tokenizer counts in the 660 documents.
SIE_PRICE_PER_1M_TOKENS = {FIRST_MODEL: 0.04, SECOND_MODEL: 0.05}
SIE_TOKENS = {FIRST_MODEL: 238_605, SECOND_MODEL: 227_727}
# AWS Comprehend DetectPiiEntities: cheapest tier, per 100-character unit, 3-unit minimum per request,
# units counted as characters over 100 with no rounding up.
COMPREHEND_PER_UNIT = 0.000025
COMPREHEND_MIN_UNITS = 3
COMPREHEND_CHARS_PER_UNIT = 100
# LLMs: $ per 1M input and output tokens, times the tokens each provider reported for the run, at each
# vendor's Batch API price, half of list and its cheapest (list: GPT-6 Luna $0.10/$0.50, Claude Haiku 4.5 $1/$5).
LLM_PRICES = {"llm:gpt-6-luna": (0.05, 0.25), "llm:claude-haiku-4-5": (0.50, 2.50)}
# Self-hosted arms: Modal list price per second of the container, at the throughput measured in
# results/e2_results.json, divided by 75% utilisation and times 1.75 for region.
L4_PER_S = 0.000222
CORE_PER_S = 0.0000131
GIB_PER_S = 0.00000222
UTILISATION = 0.75
REGION_MULTIPLIER = 1.75
CONTAINERS = {
    "presidio": {"gpus": 0, "cores": 8, "gib": 16},
    "privacy-filter": {"gpus": 1, "cores": 8, "gib": 32},
}
COMPREHEND = "aws-comprehend"
PUBLISHED_MONTHLY_USD = {
    COMPOSITION: 32,
    "presidio": 3,
    "privacy-filter": 23,
    "llm:gpt-6-luna": 57,
    "llm:claude-haiku-4-5": 725,
    COMPREHEND: 351,
}

# --- the composition -----------------------------------------------------------------------------------------------


def windows(text: str) -> list[tuple[int, int]]:
    """300-word windows with a 50-word overlap as (start, end) character ranges; one range for a short text."""
    words = [m.span() for m in GLINER_WORD.finditer(text)]
    if len(words) <= WINDOW_WORDS:
        return [(0, len(text))]
    out = []
    step = WINDOW_WORDS - WINDOW_OVERLAP
    for first in range(0, len(words), step):
        last = min(first + WINDOW_WORDS, len(words)) - 1
        out.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return out


def sie_spans(output: dict[str, Any] | None, *, windowed: bool, floor: float | None) -> list[dict[str, Any]]:
    """One SIE model's spans for a document: the whole-document request, or its windows when it has them."""
    output = output or {}
    if windowed and output.get("windows"):
        entities = [e for w in output["windows"] for e in w["entities"]]
    else:
        entities = output.get("whole") or []
    if floor is not None:
        entities = [e for e in entities if (e.get("score") or 0) >= floor]
    return entities


def propagated(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every other whole-word, case-sensitive occurrence of a name token not already inside a span."""
    tokens = sorted(
        {t for s in spans if s["label"] == PERSON_LABEL for t in NAME_TOKEN.findall(text[s["start"] : s["end"]])}
    )
    added = []
    for token in tokens:
        pattern = re.compile(r"(?<![^\W\d_])" + re.escape(token) + r"(?![^\W\d_])")
        for m in pattern.finditer(text):
            if any(s["start"] <= m.start() and s["end"] >= m.end() for s in spans):
                continue
            added.append({"start": m.start(), "end": m.end(), "label": PERSON_LABEL, "derived": True})
    return added


def compose(text: str, first: dict[str, Any] | None, second: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Both models' spans at score >= 0.6, windowed when windows were sent, unioned, then name propagation."""
    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for output in (first, second):
        for e in sie_spans(output, windowed=True, floor=SCORE_FLOOR):
            key = (e["start"], e["end"], e["label"])
            if key not in merged or (e.get("score") or 0) > (merged[key].get("score") or 0):
                merged[key] = dict(e)
    spans = sorted(merged.values(), key=lambda s: (s["start"], s["end"]))
    return sorted(spans + propagated(text, spans), key=lambda s: (s["start"], s["end"]))


def occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not needle:
        return out
    at = text.find(needle)
    while at != -1:
        out.append((at, at + len(needle)))
        at = text.find(needle, at + 1)
    return out


def llm_spans(text: str, output: dict[str, Any] | None) -> tuple[list[dict[str, Any]], int, bool]:
    """Masks from an LLM's returned strings (every occurrence), the count of strings found nowhere in the text,
    and whether the output parsed."""
    output = output or {}
    try:
        entities = list(json.loads(output.get("text") or "").get("entities") or [])
        parsed = True
    except (json.JSONDecodeError, AttributeError):
        entities, parsed = [], False
    spans, seen, unmatched = [], set(), 0
    for e in entities:
        found = occurrences(text, str(e.get("text", "")))
        unmatched += not found
        for s, t in found:
            if (s, t) not in seen:
                seen.add((s, t))
                spans.append({"start": s, "end": t, "label": str(e.get("type", ""))})
    return spans, unmatched, parsed


# --- scoring -------------------------------------------------------------------------------------------------------


def merge(spans: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Overlapping spans become one mask covering all of them."""
    out: list[list[int]] = []
    for s in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if out and s["start"] < out[-1][1]:
            out[-1][1] = max(out[-1][1], s["end"])
        else:
            out.append([s["start"], s["end"]])
    return [(a, b) for a, b in out]


def hidden_in(masks: list[tuple[int, int]], gold: dict[str, Any]) -> set[int]:
    hidden: set[int] = set()
    for a, b in masks:
        if a < gold["end"] and b > gold["start"]:
            hidden.update(range(max(a, gold["start"]), min(b, gold["end"])))
    return hidden


def covered(text: str, masks: list[tuple[int, int]], gold: dict[str, Any]) -> bool:
    """Every non-space character of the gold span is under a mask."""
    hidden = hidden_in(masks, gold)
    return all(i in hidden or text[i].isspace() for i in range(gold["start"], gold["end"]))


_ONE_WORD_STATES = (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois "
    "Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri "
    "Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah Vermont Virginia Washington "
    "Wisconsin Wyoming Alberta Manitoba Ontario Quebec Saskatchewan Yukon Nunavut"
)
_STATES = [
    *_ONE_WORD_STATES.split(),
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Rhode Island",
    "South Carolina", "South Dakota", "West Virginia", "District of Columbia", "British Columbia", "New Brunswick",
    "Nova Scotia", "Prince Edward Island", "Northwest Territories", "Newfoundland and Labrador",
]  # fmt: skip
_CODE_WORDS = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR "
    "PA RI SC SD TN TX UT VT VA WA WV WI WY DC AB BC MB NB NL NS NT NU ON PE QC SK YT USA US UK"
)
_CODES = _CODE_WORDS.split()
_COUNTRIES = (
    "United States", "United States of America", "United Kingdom", "Canada", "Australia", "Germany", "France",
    "Spain", "Italy", "Netherlands", "Sweden", "Ireland", "India", "China", "Japan", "Mexico", "Brazil",
    "New Zealand", "Singapore", "Switzerland", "Belgium", "Austria", "Norway", "Denmark", "Finland", "Poland",
    "Portugal", "South Africa", "Nigeria", "Kenya", "Philippines", "Malaysia", "Indonesia", "South Korea",
)  # fmt: skip
# The words a span may leave readable and still count under the state-and-country-excused figure.
EXCUSED = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted([*_STATES, *_COUNTRIES, *_CODES], key=len, reverse=True)) + r")\b"
)


def covered_excusing_state(text: str, masks: list[tuple[int, int]], gold: dict[str, Any]) -> bool:
    """As covered(), except a U.S. state, Canadian province, country or punctuation may stay readable."""
    hidden = hidden_in(masks, gold)
    for m in EXCUSED.finditer(text, gold["start"], gold["end"]):
        hidden.update(range(m.start(), m.end()))
    return all(i in hidden or not text[i].isalnum() for i in range(gold["start"], gold["end"]))


def doc_stats(doc: dict[str, Any], spans: list[dict[str, Any]], unmatched: int = 0) -> dict[str, int]:
    """Per-document counts. `unmatched` is an LLM's strings found nowhere in the text: each counts as one
    mask and one exact false positive, since the LLM claimed personal data that is not there."""
    text = doc["text"]
    masks = merge(spans)
    scoped = doc["scoped"]
    amounts = [m.span() for m in AMOUNT.finditer(text)]
    exact_pred = {(s["start"], s["end"]) for s in spans}
    exact_gold = {(g["start"], g["end"]) for g in scoped}
    tp = len(exact_pred & exact_gold)
    return {
        "gold": len(scoped),
        "covered": sum(covered(text, masks, g) for g in scoped),
        "covered_excused": sum(covered_excusing_state(text, masks, g) for g in scoped),
        "masks": len(masks) + unmatched,
        # A mask counts toward overlap precision when it touches any gold span, in scope or not.
        "masks_on_gold": sum(any(a < g["end"] and b > g["start"] for g in doc["gold"]) for a, b in masks),
        "amounts": len(amounts),
        "amounts_masked": sum(any(a < e and b > s for a, b in masks) for s, e in amounts),
        "exact_tp": tp,
        "exact_fp": len(exact_pred) - tp + unmatched,
        "exact_fn": len(exact_gold) - tp,
    }


def summarise(stats: list[dict[str, int]]) -> dict[str, Any]:
    total = {k: sum(d[k] for d in stats) for k in stats[0]}
    tp, fp, fn = total["exact_tp"], total["exact_fp"], total["exact_fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        **total,
        "coverage_recall": total["covered"] / total["gold"],
        "coverage_recall_state_excused": total["covered_excused"] / total["gold"],
        "overlap_precision": total["masks_on_gold"] / total["masks"] if total["masks"] else 0.0,
        "exact_f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def bootstrap(per_arm: dict[str, list[dict[str, int]]]) -> dict[str, dict[str, list[float]]]:
    """Paired document-cluster bootstrap: the same resampled documents for every arm in each draw."""
    n = len(per_arm[COMPOSITION])
    rng = random.Random(SEED)
    gold = [d["gold"] for d in per_arm[COMPOSITION]]
    hits = {arm: [d["covered"] for d in stats] for arm, stats in per_arm.items()}
    hits_excused = {arm: [d["covered_excused"] for d in stats] for arm, stats in per_arm.items()}
    draws: dict[str, list[float]] = {arm: [] for arm in per_arm}
    diffs: dict[str, list[float]] = {arm: [] for arm in per_arm if arm != COMPOSITION}
    diffs_excused: dict[str, list[float]] = {arm: [] for arm in diffs}
    for _ in range(RESAMPLES):
        pick = [rng.randrange(n) for _ in range(n)]
        denominator = sum(gold[i] for i in pick)
        value = {arm: sum(h[i] for i in pick) / denominator for arm, h in hits.items()}
        excused = {arm: sum(h[i] for i in pick) / denominator for arm, h in hits_excused.items()}
        for arm in per_arm:
            draws[arm].append(value[arm])
        for arm, series in diffs.items():
            series.append(value[COMPOSITION] - value[arm])
            diffs_excused[arm].append(excused[COMPOSITION] - excused[arm])

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]

    return {
        "coverage_ci95": {a: ci(v) for a, v in draws.items()},
        "composition_minus_ci95": {a: ci(v) for a, v in diffs.items()},
        "composition_minus_state_excused_ci95": {a: ci(v) for a, v in diffs_excused.items()},
    }


# --- price ---------------------------------------------------------------------------------------------------------


def monthly_usd(
    documents: list[dict[str, Any]], rows: dict[str, dict[int, dict[str, Any]]], e2: dict[str, Any]
) -> dict[str, float]:
    """$ a month for MONTHLY_DOCUMENTS documents like the study's, per arm."""
    n = len(documents)
    per_document = {COMPOSITION: sum(SIE_TOKENS[m] * SIE_PRICE_PER_1M_TOKENS[m] / 1e6 for m in SIE_TOKENS) / n}
    units = sum(max(COMPREHEND_MIN_UNITS, len(d["text"]) / COMPREHEND_CHARS_PER_UNIT) for d in documents)
    per_document[COMPREHEND] = units * COMPREHEND_PER_UNIT / n
    for arm, (price_in, price_out) in LLM_PRICES.items():
        tokens_in = sum((r["output"] or {}).get("tokens_in", 0) for r in rows[arm].values())
        tokens_out = sum((r["output"] or {}).get("tokens_out", 0) for r in rows[arm].values())
        per_document[arm] = (tokens_in * price_in + tokens_out * price_out) / 1e6 / n
    for arm, box in CONTAINERS.items():
        per_second = box["gpus"] * L4_PER_S + box["cores"] * CORE_PER_S + box["gib"] * GIB_PER_S
        per_second = per_second / UTILISATION * REGION_MULTIPLIER
        per_document[arm] = per_second / e2["self_host"][arm]["documents_per_second"]
    return {arm: usd * MONTHLY_DOCUMENTS for arm, usd in per_document.items()}


# --- io ------------------------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents(evidence: Path) -> list[dict[str, Any]]:
    documents = read_jsonl(evidence / "inputs" / "gretel-main.jsonl")
    for d in documents:
        d["scoped"] = [g for g in d["gold"] if g["label"] in IN_SCOPE]
    return documents


def load_rows(path: Path, documents: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows = {r["index"]: r for r in read_jsonl(path)}
    if set(rows) != {d["index"] for d in documents}:
        raise SystemExit(f"{path}: rows do not match the {len(documents)} documents")
    return rows


def arm_spans(
    arm: str, doc: dict[str, Any], rows: dict[str, dict[int, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], int]:
    """An arm's spans for one document, and the count of LLM strings found nowhere in it."""
    if arm == COMPOSITION:
        first = rows[f"sie:{FIRST_MODEL}"][doc["index"]]["output"]
        second = rows[f"sie:{SECOND_MODEL}"][doc["index"]]["output"]
        return compose(doc["text"], first, second), 0
    record = rows[arm][doc["index"]]
    if record.get("error") is not None or record.get("output") is None:
        return [], 0
    if arm.startswith("sie:"):
        return sie_spans(record["output"], windowed=False, floor=None), 0
    if arm.startswith("llm:"):
        spans, unmatched, _ = llm_spans(doc["text"], record["output"])
        return spans, unmatched
    return list(record["output"]["spans"]), 0


# --- output --------------------------------------------------------------------------------------------------------


class Checks:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def equal(self, what: str, got: Any, expected: Any) -> None:
        if got != expected:
            self.failed.append(f"{what}: got {got}, expected {expected}")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def points(ci: list[float]) -> str:
    return f"{100 * ci[0]:+.1f} to {100 * ci[1]:+.1f} points"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE, help="the directory fetch.py wrote")
    parser.add_argument("--no-bootstrap", action="store_true", help="skip the 95%% intervals")
    parser.add_argument(
        "--sie-rows",
        type=Path,
        help="score the two SIE rows files run.py --all wrote in this directory instead of the recorded ones; "
        "figures are printed, not checked",
    )
    args = parser.parse_args()
    replay = args.sie_rows is None

    documents = load_documents(args.evidence)
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    for arm, _, name in ARMS:
        if name is not None:
            source = args.sie_rows if not replay and arm.startswith("sie:") else args.evidence / "rows"
            rows[arm] = load_rows(source / name, documents)
    recorded = read_json(args.evidence / "results" / "gretel-main_results.json")
    tokens = read_json(args.evidence / "results" / "gretel-main_tokens.json")
    e2 = read_json(args.evidence / "results" / "e2_results.json")
    check = Checks()

    per_arm = {arm: [doc_stats(d, *arm_spans(arm, d, rows)) for d in documents] for arm, _, _ in ARMS}
    summary = {arm: summarise(stats) for arm, stats in per_arm.items()}
    gold = summary[COMPOSITION]["gold"]
    check.equal("documents", len(documents), PUBLISHED_DOCUMENTS)
    check.equal("in-scope gold spans", gold, PUBLISHED_GOLD)
    check.equal("in-scope gold spans (recorded)", gold, recorded["gold_spans"])

    print(f"{len(documents)} documents, {gold:,} in-scope personal-data spans")
    print()
    print(f"{'Arm':<42} {'Masked':>16} {'Coverage':>9} {'State excused':>14} {'Precision':>10} {'Exact F1':>9}")
    for arm, name, _ in ARMS:
        s = summary[arm]
        print(
            f"{name:<42} {s['covered']:>7,} of {gold:,} {pct(s['coverage_recall']):>9} "
            f"{pct(s['coverage_recall_state_excused']):>14} {s['overlap_precision']:>10.3f} {s['exact_f1']:>9.3f}"
        )
        if not replay:
            continue
        want = recorded["arms"][RECORDED_KEY.get(arm, arm)]
        for key in ("covered", "masks", "amounts_masked"):
            check.equal(f"{name} {key}", s[key], want[key])
        for key in ("coverage_recall", "coverage_recall_state_excused", "overlap_precision", "exact_f1"):
            check.equal(f"{name} {key}", round(s[key], 12), round(want[key], 12))
        if arm in PUBLISHED_MASKED:
            check.equal(f"{name} masked (published)", s["covered"], PUBLISHED_MASKED[arm])
    for arm in LLM_PRICES:
        unparsed = sum(
            not llm_spans(d["text"], rows[arm][d["index"]]["output"])[2]
            for d in documents
            if rows[arm][d["index"]]["output"]
        )
        print(f"  {arm}: {unparsed} of {len(documents)} replies did not parse as JSON and count as no masks")
        check.equal(f"{arm} unparsed replies", unparsed, recorded["arms"][arm]["unparsed"])

    s = summary[COMPOSITION]
    print()
    print(
        f"Currency amounts SIE's masks touch: {s['amounts_masked']} of {s['amounts']:,} "
        "(the pattern also matches IP addresses)"
    )
    scoped = [g for d in documents for g in d["scoped"]]
    bound = sum(g["label"] not in NO_COMPREHEND_TYPE for g in scoped) / len(scoped)
    print(f"In-scope spans with a documented AWS Comprehend type: {pct(bound)} (Comprehend was not measured)")
    check.equal(
        "Comprehend documented-type bound", round(bound, 12), round(recorded["comprehend_documented_type_bound"], 12)
    )

    if not args.no_bootstrap:
        intervals = bootstrap(per_arm)
        print()
        print(f"95% intervals: paired document-cluster bootstrap, {RESAMPLES:,} resamples, seed {SEED}")
        for arm, name, _ in ARMS:
            ci = intervals["coverage_ci95"][arm]
            print(f"  {name:<42} coverage {pct(ci[0])} to {pct(ci[1])}")
        print("SIE minus each arm, paired: coverage recall / with states and countries excused")
        for arm, name, _ in ARMS[1:]:
            a = intervals["composition_minus_ci95"][arm]
            b = intervals["composition_minus_state_excused_ci95"][arm]
            print(f"  {name:<42} {points(a)} / {points(b)}")
        if replay:
            for block, values in intervals.items():
                for arm, ci in values.items():
                    want = recorded[block][RECORDED_KEY.get(arm, arm)]
                    check.equal(f"{block} {arm}", [round(x, 12) for x in ci], [round(x, 12) for x in want])

    for model, count in SIE_TOKENS.items():
        check.equal(f"{model} tokens", tokens[model], count)
        check.equal(f"{model} tokens (e2)", e2["sie_tokens"][model], count)
    usd = monthly_usd(documents, rows, e2)
    print()
    print(f"$ a month for {MONTHLY_DOCUMENTS:,} documents like these:")
    names = {arm: name for arm, name, _ in ARMS} | {COMPOSITION: "SIE", COMPREHEND: "AWS Comprehend"}
    for arm in (COMPOSITION, "presidio", "privacy-filter", "llm:gpt-6-luna", "llm:claude-haiku-4-5", COMPREHEND):
        masked = pct(summary[arm]["coverage_recall"]) if arm in summary else "not measured"
        print(f"  {names[arm]:<24} ${usd[arm]:>9,.2f}  (${round(usd[arm]):,})  masked {masked}")
        check.equal(f"{arm} $ a month (published)", round(usd[arm]), PUBLISHED_MONTHLY_USD[arm])
    priced = e2["usd_per_1m_documents"]
    recorded_usd = {
        COMPOSITION: next(v for k, v in priced.items() if k.startswith("sie-composition (")),
        COMPREHEND: next(v for k, v in priced.items() if k.startswith("aws-comprehend")),
        "llm:gpt-6-luna": priced["llm:gpt-6-luna"],
        "llm:claude-haiku-4-5": priced["llm:claude-haiku-4-5"],
        "presidio": e2["self_host"]["presidio"]["usd_per_1m_documents"],
        "privacy-filter": e2["self_host"]["privacy-filter"]["usd_per_1m_documents"],
    }
    for arm, value in recorded_usd.items():
        check.equal(f"{arm} $ per 1M documents (recorded)", round(usd[arm], 6), round(value, 6))

    print()
    if not replay:
        print(f"Scored the SIE rows in {args.sie_rows}; figures are not checked against the recording.")
        return 0
    if check.failed:
        print("FAILED: these figures do not match the recording or the page:", file=sys.stderr)
        for line in check.failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("Every figure matches the recorded results and the published page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
