#!/usr/bin/env python3
"""Build the task page's surfaces from the recordings, and refuse any surface that would show a miss.

    python3 page.py --calls <recordings> [--split test] [--json page.json]

Standard library only, no model. Every setting comes from tuning.json, which
was fixed on the dev slice and committed before any test record was sent
(PREREGISTRATION.md); this script never changes one.

What the page shows:

    board     one row per lane in tuning.json (Fast, Smart and the LLM, each one
              model on its own), with a cell per question the lane passed the
              page rule for on dev. A question a lane was not asked, or did not
              pass on dev, is left out, and the foot counts it. Each cell holds
              the slice's right-of-answered and balanced accuracy. The median
              per-record time is a round trip, not a service level.
    figures   one row per question type, for Smart: right of answered, summed
              over its shown questions of that type.
    cards     CARDS below, in order. A card is the first record, in the order of
              sha256(CARD_SALT + CVE id), that meets its condition, has a
              weakness class no earlier card has, and on which every answer the
              card shows (each CARD_LANES lane's shown questions) is right and
              is the model's own top option, so the decision rule changed
              none of them. A card with no such record is dropped. A card
              cell holds only the returned answer and the probability the
              model returned for it.
    catalog   CATALOG below, each backend with the questions it passed on dev.
    totals    records shown on cards, of records recorded, and each shown
              question's figure over the whole slice.
    credits   per lane, from a copy of the published SIE Cloud prices (--prices):
              only for a model the prices name, whose recorded responses
              carry the units it is billed in. Otherwise no credit figure.
    speed     a speed claim, only when the slower median is at least
              SPEED_MARGIN times the faster, comparing the same answers.
    cascade   not shown: the cascade with its dev cut-offs against Smart
              alone, under its control (PREREGISTRATION.md).

On the slice, a shown question whose point figures fail the page rule
(balanced accuracy under 0.70, accuracy under the majority answer's, or an
option with at least 10 records answered right under 40% of the time) is
withdrawn: it is left off every surface and listed under `withdrawn`.
"""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lanes
import score
import tune
import tuning
from questions import DESCRIBED, VULNERABILITY_QUESTIONS, VULNERABILITY_TRIAGE, asked
from run import LLM_BACKENDS

CARD_SALT = "sie-typed-decisions-cards-v1:"
CARD_LANES = ("fast", "smart")
# (name, the argument the card makes, condition on the record's gold answers)
CARDS = (
    (
        "remote-no-login",
        "a record anyone on the network can exploit without an account",
        lambda gold: gold["remote_unauthenticated"] == "true",
    ),
    (
        "network-needs-login",
        "reachable over the network, but only with an account: the yes-or-no answer is not the attack vector restated",
        lambda gold: gold["attack_vector"] == "network" and gold["remote_unauthenticated"] == "false",
    ),
    (
        "local-access",
        "a record exploitable only with local or physical access, the attack vector most records are not",
        lambda gold: gold["attack_vector"] in ("local", "physical"),
    ),
)
CATALOG = ("gliclass-instruct-large", "laya", "laya-typed-decisions")
SPEED_MARGIN = 1.5

# What the page publishes from the test slice beyond score.py's figures. page.py
# exits nonzero when the recording stops reproducing any of it.
PUBLISHED = {
    "cards": ["CVE-2023-49378", "CVE-2023-6826", "CVE-2023-44278"],
    "median_ms_per_record": {"fast": 69, "smart": 205, "llm": 2671},
    "speed_claims": [["attack vector", False], ["a record's three answers", True]],
    "withdrawn": [["laya-typed-decisions", "remote_unauthenticated"]],
}


def published_failures(page: dict[str, Any]) -> list[str]:
    actual = {
        "cards": [card["case"] for card in page["cards"]],
        "median_ms_per_record": {name: round(lane["median_ms_per_record"]) for name, lane in page["board"].items()},
        "speed_claims": [[claim["compares"].split(":")[0], claim["claimed"]] for claim in page["speed"]],
        "withdrawn": [[item.get("lane") or item.get("catalog"), item["question"]] for item in page["withdrawn"]],
    }
    return [
        f"{key}: got {actual[key]}, published {expected}"
        for key, expected in PUBLISHED.items()
        if actual[key] != expected
    ]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_calls(paths: list[str], split: str) -> list[dict[str, Any]]:
    calls = []
    for path in paths:
        for call in score.usable(score.load(Path(path))):
            if call["set"] == VULNERABILITY_TRIAGE and call["split"] == split:
                calls.append(call)
    return calls


def per_backend(calls: list[dict[str, Any]], data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """backend -> case -> question -> answer after its rules: the tuned call, or an LLM's described one."""
    cases = score.load(data_dir / f"inputs/{VULNERABILITY_TRIAGE}/cases.json")
    by_slug = {case["slug"]: case for case in cases["cases"]}
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for call in calls:
        wanted = DESCRIBED if call["backend"] in LLM_BACKENDS else tuning.TUNED
        if call["variant"] != wanted:
            continue
        case = by_slug[call["case"]]
        questions = asked(VULNERABILITY_TRIAGE, call["backend"], score.questions_for(cases, case))
        returned = score.NORMALISERS[call["transport"]](call, questions)
        answers = score.decided(call, returned)
        for qid, item in answers.items():
            # What the model itself returned: its own top option and that option's probability.
            item["returned"] = {"top": returned[qid]["top"], "top_p": returned[qid]["top_p"]}
        out[call["backend"]][call["case"]] = answers
    return out


def point_verdict(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    """The page rule on the slice's point figures, without the dev margin."""
    figures = tune.metrics(gold, predicted)
    majority = tune.majority_accuracy(gold)
    reasons = []
    if figures["balanced_accuracy"] < tune.PAGE_MIN_BALANCED:
        reasons.append(f"balanced accuracy {figures['balanced_accuracy']:.3f}")
    if figures["accuracy"] < majority:
        reasons.append(f"accuracy {figures['accuracy']:.3f} under the majority answer's {majority:.3f}")
    for option, (right, total) in figures["recall"].items():
        if total >= tune.PAGE_MIN_CLASS_RECORDS and right / total < tune.PAGE_MIN_CLASS_RECALL:
            reasons.append(f"{option} right {right} of {total}")
    right = sum(g == p for g, p in zip(gold, predicted))
    return {
        "right": right,
        "n": len(gold),
        "accuracy": figures["accuracy"],
        "balanced_accuracy": figures["balanced_accuracy"],
        "majority_accuracy": majority,
        "recall": figures["recall"],
        "holds": not reasons,
        "reasons": reasons,
    }


def option_label(qid: str, key: str) -> str:
    if qid == lanes.SEVERITY:
        return key
    return next(option["name"] for option in VULNERABILITY_QUESTIONS[qid]["options"] if option["key"] == key)


def build(
    calls: list[dict[str, Any]], data_dir: Path, settings: dict[str, Any], prices: dict[str, Any] | None = None
) -> dict[str, Any]:
    answers_by_backend = per_backend(calls, data_dir)
    cases = score.load(data_dir / f"inputs/{VULNERABILITY_TRIAGE}/cases.json")
    split = {call["split"] for call in calls}
    gold = {case["slug"]: case["gold"] for case in cases["cases"] if case["split"] in split}
    latency = tune.request_latencies(calls)
    withdrawn: list[dict[str, Any]] = []

    board: dict[str, Any] = {}
    lane_answers: dict[str, dict[str, dict[str, Any]]] = {}
    for name, lane_settings in settings["lanes"].items():
        lane = {key: value for key, value in lane_settings.items() if key in ("backend", "escalate_to", "cutoffs")}
        answers, escalated = lanes.lane_answers(lane, answers_by_backend)
        lane_answers[name] = answers
        records = sorted(answers)
        cells, left_out = {}, []
        for qid, dev in lane_settings["questions"].items():
            if not dev["eligible"]:
                left_out.append(qid)
                continue
            rows = [(answers[c][qid]["top"], gold[c][qid]) for c in records if qid in answers[c]]
            cell = point_verdict([g for _, g in rows], [p for p, _ in rows])
            if not cell["holds"]:
                withdrawn.append({"lane": name, "question": qid, "reasons": cell["reasons"]})
                continue
            cells[qid] = cell
        not_asked = [qid for qid in VULNERABILITY_QUESTIONS if qid not in lane_settings["questions"]]
        board[name] = {
            "model": next(call["model"] for call in calls if call["backend"] == lane["backend"]),
            "cells": cells,
            "left_out": {"not_passed_on_dev": left_out, "not_asked": not_asked},
            "records": len(records),
            "median_ms_per_record": statistics.median(
                tune.record_latency(lane, c, escalated[c], latency) for c in records
            ),
            "calls_per_record": statistics.fmean(tune.record_calls(lane, c, escalated[c], latency) for c in records),
            "credits": credits(lane, calls, prices),
        }

    figures: dict[str, dict[str, int]] = {}
    for qid, cell in board["smart"]["cells"].items():
        qtype = "score" if qid == lanes.SEVERITY else VULNERABILITY_QUESTIONS[qid]["type"]
        entry = figures.setdefault(qtype, {"right": 0, "answered": 0, "questions": 0})
        entry["right"] += cell["right"]
        entry["answered"] += cell["n"]
        entry["questions"] += 1

    cards = choose_cards(gold, lane_answers, board)

    catalog = {}
    for backend in CATALOG:
        dev = tuning.entry(VULNERABILITY_TRIAGE, backend, settings) or {}
        shown = {}
        for qid, entry in dev.get("questions", {}).items():
            if not entry["dev"]["eligible"] or backend not in answers_by_backend:
                continue
            per_case = answers_by_backend[backend]
            rows = [(per_case[c][qid]["top"], gold[c][qid]) for c in sorted(per_case)]
            cell = point_verdict([g for _, g in rows], [p for p, _ in rows])
            if not cell["holds"]:
                withdrawn.append({"catalog": backend, "question": qid, "reasons": cell["reasons"]})
                continue
            shown[qid] = cell
        catalog[backend] = shown

    return {
        "split": sorted(split),
        "records": len(gold),
        "board": board,
        "figures": figures,
        "cards": cards,
        "catalog": catalog,
        "totals": {
            "records_on_cards": len(cards),
            "records_recorded": len(gold),
            "whole_slice": {
                name: {qid: f"{cell['right']} of {cell['n']}" for qid, cell in lane["cells"].items()}
                for name, lane in board.items()
            },
        },
        "speed": speed_claims(board, latency, gold),
        "cascade": cascade_control(settings, answers_by_backend, gold, latency, board),
        "withdrawn": withdrawn,
    }


def credits(lane: dict[str, Any], calls: list[dict[str, Any]], prices: dict[str, Any] | None) -> dict[str, Any] | None:
    """Credits per record from the published SIE Cloud prices, where they can be computed.

    Only a lane of one model the price list names, whose recorded responses
    carry the units it bills (a chat completion's prompt and completion
    tokens), gets a figure. Anything else gets None: no price, no figure.
    """
    if prices is None or "escalate_to" in lane:
        return None
    lane_calls = [call for call in calls if call["backend"] == lane["backend"]]
    model = lane_calls[0]["model"].split(":", 1)[0]
    unit_price = {
        item["unit"]: Fraction(int(item["usd_per_unit"]["numerator"]), int(item["usd_per_unit"]["denominator"]))
        for item in prices["items"]
        if item["model"] == model and item["operation"] == "generate" and item["profile"] == "default"
    }
    if not {"input_tokens", "output_tokens"} <= set(unit_price):
        return None
    pack = prices["stripe_credit_pack"]
    credits_per_usd = Fraction(pack["credits"]) / (Fraction(pack["usd_cents"]) / 100)
    per_record = []
    for call in lane_calls:
        usage = call["responses"][0].get("usage") or {}
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            return None
        usd = (
            usage["prompt_tokens"] * unit_price["input_tokens"]
            + usage["completion_tokens"] * unit_price["output_tokens"]
        )
        per_record.append(float(usd * credits_per_usd))
    return {
        "pricing_version": prices["pricing_version"],
        "median_credits_per_record": statistics.median(per_record),
        "usd_per_1000_records": float(Fraction(statistics.fmean(per_record)) * 1000 / credits_per_usd),
    }


def choose_cards(
    gold: dict[str, dict[str, str]], lane_answers: dict[str, dict[str, dict[str, Any]]], board: dict[str, Any]
) -> list[dict[str, Any]]:
    order = sorted(gold, key=lambda slug: sha256(CARD_SALT + slug))
    cards: list[dict[str, Any]] = []
    for name, argument, condition in CARDS:
        taken = {card["gold"]["weakness"] for card in cards}
        for slug in order:
            if not condition(gold[slug]) or gold[slug]["weakness"] in taken:
                continue
            shown = {
                lane: {
                    qid: lane_answers[lane][slug][qid]
                    for qid in board[lane]["cells"]
                    if slug in lane_answers[lane] and qid in lane_answers[lane][slug]
                }
                for lane in CARD_LANES
            }
            complete = all(set(shown[lane]) == set(board[lane]["cells"]) and bool(shown[lane]) for lane in CARD_LANES)
            if not complete or any(
                a["top"] != gold[slug][qid] or a["returned"]["top"] != a["top"]
                for cells in shown.values()
                for qid, a in cells.items()
            ):
                continue
            cards.append(
                {
                    "card": name,
                    "argument": argument,
                    "case": slug,
                    "gold": gold[slug],
                    "answers": {
                        lane: {
                            qid: {
                                "answer": a["top"],
                                "label": option_label(qid, a["top"]),
                                "probability": a["returned"]["top_p"],
                            }
                            for qid, a in cells.items()
                        }
                        for lane, cells in shown.items()
                    },
                }
            )
            break
    for card in cards:
        for lane, cells in card["answers"].items():
            for qid, cell in cells.items():
                if cell["answer"] != card["gold"][qid]:
                    raise SystemExit(f"card {card['card']} would show a miss: {lane} {qid}")
    return cards


def cascade_control(
    settings: dict[str, Any], answers_by_backend: dict, gold: dict[str, dict[str, str]], latency: dict, board: dict
) -> dict[str, Any]:
    """The cascade, with its dev cut-offs, against Smart alone on this slice. Reported, never shown."""
    lane = {key: settings["cascade"][key] for key in ("backend", "escalate_to", "cutoffs")}
    cascade, escalated = lanes.lane_answers(lane, answers_by_backend)
    alone, _ = lanes.lane_answers(dict(settings["lanes"]["smart"], cutoffs={}), answers_by_backend)
    records = sorted(cascade)
    questions: dict[str, Any] = {}
    regressions = 0
    for qid in [q for q in VULNERABILITY_QUESTIONS if q in cascade[records[0]]]:
        g = [gold[c][qid] for c in records]
        mine = tune.metrics(g, [cascade[c][qid]["top"] for c in records])
        theirs = tune.metrics(g, [alone[c][qid]["top"] for c in records])
        lost = sum(alone[c][qid]["top"] == gold[c][qid] != cascade[c][qid]["top"] for c in records)
        regressions += lost
        questions[qid] = {
            "cascade": {k: mine[k] for k in ("accuracy", "balanced_accuracy")},
            "smart": {k: theirs[k] for k in ("accuracy", "balanced_accuracy")},
            "escalated": statistics.fmean(escalated[c][qid] for c in records),
            "regressions": lost,
        }
    calls = statistics.fmean(tune.record_calls(lane, c, escalated[c], latency) for c in records)
    matches = all(q["cascade"][k] >= q["smart"][k] for q in questions.values() for k in q["cascade"])
    return {
        "cutoffs": lane["cutoffs"],
        "questions": questions,
        "calls_per_record": calls,
        "smart_calls_per_record": board["smart"]["calls_per_record"],
        "matches_smart": matches,
        "regressions": regressions,
        "passes": matches and regressions == 0 and calls < board["smart"]["calls_per_record"],
    }


def speed_claims(board: dict[str, Any], latency: dict, gold: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Claims that clear SPEED_MARGIN, each comparing the same answers."""
    claims = []
    fast, smart = board["fast"], board["smart"]
    if "attack_vector" in fast["cells"] and "attack_vector" in smart["cells"]:
        records = sorted(gold)
        fast_ms = statistics.median(latency[("gliformer-large", c)]["total"] for c in records)
        smart_ms = statistics.median(latency[("gliclass-large-v1", c)]["attack_vector"] for c in records)
        claims.append(("attack vector: Fast's one call against Smart's attack-vector request", fast_ms, smart_ms))
    if "llm" in board:
        claims.append(
            (
                "a record's three answers: Smart's three requests against the LLM's one",
                smart["median_ms_per_record"],
                board["llm"]["median_ms_per_record"],
            )
        )
    out = []
    for what, first, second in claims:
        ratio = max(first, second) / min(first, second)
        out.append(
            {
                "compares": what,
                "median_ms": [first, second],
                "ratio": ratio,
                "claimed": ratio >= SPEED_MARGIN,
            }
        )
    return out


def report(page: dict[str, Any]) -> None:
    print(f"== page surfaces, {', '.join(page['split'])} slice, {page['records']} records")
    for name, lane in page["board"].items():
        cells = ", ".join(
            f"{qid} {c['right']}/{c['n']} (bal {c['balanced_accuracy']:.3f})" for qid, c in lane["cells"].items()
        )
        foot = lane["left_out"]
        print(
            f"  {name:<6}{lane['model']:<42}{cells or '(nothing shown)'}"
            f"\n        left out: not passed on dev {foot['not_passed_on_dev']}, not asked {foot['not_asked']};"
            f" median {lane['median_ms_per_record']:.0f} ms per record (a round trip, not a service level),"
            f" {lane['calls_per_record']:.2f} calls; "
            + (
                f"{lane['credits']['median_credits_per_record']:.1f} credits per record (median), "
                f"${lane['credits']['usd_per_1000_records']:.3f} per 1,000 records"
                if lane["credits"]
                else "no credit figure"
            )
        )
    for qtype, entry in page["figures"].items():
        print(f"  figures  {qtype}: {entry['right']} of {entry['answered']} over {entry['questions']} question(s)")
    for card in page["cards"]:
        print(f"  card {card['card']}: {card['case']} ({card['gold']['weakness']})")
    for backend, cells in page["catalog"].items():
        print(f"  catalog  {backend}: " + ", ".join(f"{q} {c['right']}/{c['n']}" for q, c in cells.items()))
    for claim in page["speed"]:
        mark = "claimed" if claim["claimed"] else "not claimed"
        print(
            f"  speed    {claim['compares']}: {claim['median_ms'][0]:.0f} vs {claim['median_ms'][1]:.0f} ms, "
            f"{claim['ratio']:.2f}x, {mark}"
        )
    cascade = page["cascade"]
    print(
        f"  cascade  against Smart alone: {cascade['calls_per_record']:.2f} calls per record against "
        f"{cascade['smart_calls_per_record']:.2f}, regressions {cascade['regressions']}, "
        + ("passes its control" if cascade["passes"] else "fails its control (README only)")
    )
    for qid, q in cascade["questions"].items():
        print(
            f"           {qid}: escalated {q['escalated']:.3f}, accuracy {q['cascade']['accuracy']:.3f} "
            f"against {q['smart']['accuracy']:.3f}, balanced {q['cascade']['balanced_accuracy']:.3f} "
            f"against {q['smart']['balanced_accuracy']:.3f}"
        )
    for item in page["withdrawn"]:
        print(f"  WITHDRAWN {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", nargs="+", default=["data/calls.json"], help="recordings (default: data/calls.json)")
    parser.add_argument("--data", default="data", help="directory holding inputs/ (default: data)")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--json", help="also write the surfaces to this path")
    parser.add_argument(
        "--prices", help="a copy of the published SIE Cloud prices (superlinked.com/pricing/sie-cloud-prices.json)"
    )
    args = parser.parse_args()
    settings = tuning.load()
    if "lanes" not in settings:
        raise SystemExit("tuning.json has no lanes; run tune.py lanes on dev first")
    prices = json.loads(Path(args.prices).read_text(encoding="utf-8")) if args.prices else None
    page = build(split_calls(args.calls, args.split), Path(args.data), settings, prices)
    report(page)
    if args.json:
        Path(args.json).write_text(json.dumps(page, indent=1) + "\n", encoding="utf-8")
    if args.split == "test":
        failures = published_failures(page)
        if failures:
            print("\nFAILED to reproduce what the page publishes:", file=sys.stderr)
            for line in failures:
                print(f"  {line}", file=sys.stderr)
            return 1
        print("\nThe page's cards, time medians, speed claims and withdrawals reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
