#!/usr/bin/env python3
"""Reproduce the /sparse-embeddings page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

The page has no single headline number. It publishes concrete figures per card,
and this script re-derives every one of them offline, with no API key and no
inference spend, exiting nonzero if any fails.

Hero, the shopper query "color switching led lights" against the ILC bulb
listing, under prithivida/Splade_PP_en_v2:

    Sparse match score    13.998
    Words in both texts    5.342
    Terms SPLADE added     8.656
    7 shared terms listed, "11 more shared terms"   (18 shared terms)

Six cards, each "SPLADE added A of its N terms" and "bge-m3 sparse added 0 of
its M":

    b77-card-arrival             46 of 56    0 of 15
    b77-transfer-not-received    54 of 65    0 of 16
    b77-compromised-card         42 of 56    0 of 21
    b77-atm-short-cash           44 of 54    0 of 18
    esci-query-chrome-notebook   23 of 25    0 of  3
    cpsc-25437-power-bank        96 of 113   0 of 27

How each number is derived:

  active terms      the number of nonzero weights in the recorded response
  added terms       terms whose token is not in the model's own tokenization of
                    the input text. That mapping needs the tokenizer, so it was
                    recorded at run time in derived/decoded/ and is re-checked
                    here against the response
  match score       dot product of the two recorded sparse vectors
  words in both     the part of that dot product contributed by terms that
                    appear literally in BOTH texts
  terms added       the rest of the score, which is what the model contributed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HTTP_OK = 200

SPLADE_DIR = "splade"

# Recomputing a dot product in a different order moves the last bits.
TOLERANCE = 1e-9

HERO_PAIR = "color-switching-to-color-changing-bulb"
HERO_EXPECTED = {"score": 13.998, "in_both": 5.342, "added": 8.656, "shared_terms": 18}

CARDS = {
    "b77-card-arrival": {"splade": (46, 56), "bge": (0, 15)},
    "b77-transfer-not-received": {"splade": (54, 65), "bge": (0, 16)},
    "b77-compromised-card": {"splade": (42, 56), "bge": (0, 21)},
    "b77-atm-short-cash": {"splade": (44, 54), "bge": (0, 18)},
    "esci-query-chrome-notebook": {"splade": (23, 25), "bge": (0, 3)},
    "cpsc-25437-power-bank": {"splade": (96, 113), "bge": (0, 27)},
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def keep_all(_call: dict[str, Any]) -> bool:
    """Every recorded call counts toward this task's figures."""
    return True


def scored_calls(payload: dict[str, Any], keep: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """The calls the figure is computed from, refusing anything that failed.

    A recorder that hit an error writes the call with status "error" and sets
    `complete` to false. Scoring such a file would turn a failed run into a
    published number, so it stops here instead.
    """
    if payload.get("complete") is False:
        failed = payload.get("failed_calls", "some")
        raise SystemExit(f"refusing to score: {failed} calls in this calls.json failed, so it is not a complete run")
    calls = [call for call in payload["calls"] if keep(call)]
    broken = [call["id"] for call in calls if call.get("status") != HTTP_OK]
    if broken:
        raise SystemExit("refusing to score calls that did not return 200: " + ", ".join(sorted(broken)))
    return calls


def vector(call: dict[str, Any]) -> dict[str, Any]:
    return call["response"]["body"]["items"][0]["sparse"]


def thousandths(value: float) -> int:
    """The displayed figure as an integer, so a sum is exact."""
    return round(value * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    data_dir = Path(args.data)

    calls = {call["id"]: call for call in scored_calls(load(data_dir / "calls.json"), keep_all)}
    failures: list[str] = []

    # --- per input: active terms and added terms --------------------------
    print("input                              model           added  active")
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for call_id, call in sorted(calls.items()):
        folder, input_id = call_id.split("/", 1)
        decoded_path = data_dir / f"derived/decoded/{folder}/{input_id}.json"
        decoded = load(decoded_path)
        sparse = vector(call)
        active_from_response = len(sparse["indices"])
        if len(sparse["values"]) != active_from_response:
            failures.append(f"{call_id}: indices and values differ in length")
        if decoded["active_terms"] != active_from_response:
            failures.append(
                f"{call_id}: decoded active_terms {decoded['active_terms']} "
                f"but the response holds {active_from_response} nonzero weights"
            )
        if len(decoded["terms"]) != active_from_response:
            failures.append(f"{call_id}: decoded terms list is not the same length as the response")
        added = sum(1 for term in decoded["terms"] if not term["in_input"])
        if added != decoded["added_terms"]:
            failures.append(f"{call_id}: recounted added terms {added} but decoded says {decoded['added_terms']}")
        table.setdefault(input_id, {})["splade" if folder == SPLADE_DIR else "bge"] = (
            added,
            active_from_response,
        )
        print(f"{input_id:<34} {call['model']:<26} {added:>4}  {active_from_response:>5}")

    print("\nCards published on the page")
    for input_id, expected in CARDS.items():
        got = table.get(input_id)
        if got is None:
            failures.append(f"{input_id}: no recorded calls")
            continue
        for model_key, want in expected.items():
            have = got.get(model_key)
            label = "SPLADE" if model_key == "splade" else "bge-m3 sparse"
            print(f"  {input_id:<32} {label:<14} added {have[0]} of its {have[1]} terms")
            if have != want:
                failures.append(
                    f"{input_id}/{model_key}: got {have[0]} of {have[1]}, page publishes {want[0]} of {want[1]}"
                )

    # --- hero pair ---------------------------------------------------------
    pairs = load(data_dir / f"derived/decoded/{SPLADE_DIR}/pairs.json")
    pair = next((entry for entry in pairs if entry["id"] == HERO_PAIR), None)
    if pair is None:
        print(f"no recorded pair {HERO_PAIR}", file=sys.stderr)
        return 1

    query = vector(calls[f"{SPLADE_DIR}/{pair['query']}"])
    document = vector(calls[f"{SPLADE_DIR}/{pair['document']}"])
    query_weights = dict(zip(query["indices"], query["values"], strict=True))
    document_weights = dict(zip(document["indices"], document["values"], strict=True))
    shared = sorted(set(query_weights) & set(document_weights))
    score = sum(query_weights[index] * document_weights[index] for index in shared)
    in_both = sum(
        term["contribution"] for term in pair["overlap"] if term["in_query_text"] and term["in_document_text"]
    )
    added = score - in_both

    if len(pair["overlap"]) != len(shared):
        failures.append(
            f"hero pair: {len(shared)} shared terms in the responses, {len(pair['overlap'])} in the recorded overlap"
        )
    if abs(score - pair["score"]) > TOLERANCE:
        failures.append(f"hero pair: recomputed score {score} but the recorded score is {pair['score']}")
    if abs(in_both - pair["score_from_terms_in_both_texts"]) > TOLERANCE:
        failures.append("hero pair: recomputed in-both contribution differs from the recorded one")

    print(f"\nHero pair {HERO_PAIR}, model prithivida/Splade_PP_en_v2")
    print(f"  sparse match score   {score:.3f}   ({score!r})")
    print(f"  words in both texts  {in_both:.3f}")
    print(f"  terms SPLADE added   {added:.3f}")
    print(f"  shared terms         {len(shared)}   (7 listed on the page, 11 more)")

    for key, want in HERO_EXPECTED.items():
        got = {
            "score": round(score, 3),
            "in_both": round(in_both, 3),
            "added": round(added, 3),
            "shared_terms": len(shared),
        }[key]
        if got != want:
            failures.append(f"hero {key}: got {got}, page publishes {want}")
    # Check the decomposition as integer thousandths, the way a reader adds up
    # the three displayed figures. Adding the rounded floats reintroduces the
    # binary representation error the display has already dropped.
    if thousandths(in_both) + thousandths(added) != thousandths(score):
        failures.append(f"hero: {in_both:.3f} + {added:.3f} does not add up to {score:.3f}")

    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: the hero pair 13.998 = 5.342 + 8.656 over 18 shared terms,")
    print("and the added-of-active counts on all six published cards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
