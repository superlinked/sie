"""Route a request to a cheap or a strong LLM with the study's knn20 router.

    export SIE_API_KEY=...
    uv run --frozen python run.py "Solve for x: x/2 + x/3 = 5"
    uv run --frozen python run.py --pair anthropic "Write a limerick about a lighthouse keeper"
    uv run --frozen python run.py --item math500/test/prealgebra/505.json
    uv run --frozen python run.py --item math500/test/prealgebra/505.json --recorded
    OPENAI_API_KEY=... uv run --frozen python run.py --answer "Solve for x: x/2 + x/3 = 5"

One call to SIE Serverless embeds the request with BAAI/bge-m3; the study's
calls were billed 1.3 credits on average, at USD 0.00001 a credit. The request
is compared with the study's 729 calibration requests and their labels for the
chosen pair, and goes to the cheap model if its score reaches the pair's
threshold from results.json. Run fetch.py first.

--item routes one of the study's requests by id and prints its recorded score
beside the new one; --recorded uses its recorded vector instead of calling SIE,
so it needs no key. --answer also sends the request to the chosen OpenAI model,
with the study's settings, and needs OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from sie_sdk import SIEClient, SIEError
from sie_sdk.types import Item

import study

OPENAI_RESPONSES = "https://api.openai.com/v1/responses"


class Encoder(Protocol):
    def encode(self, model: str, items: Item) -> Any: ...


def embed(client: Encoder, text: str) -> np.ndarray:
    return np.asarray(client.encode(study.EMBEDDER, Item(text=text))["dense"], dtype=np.float64)


def answer_openai(model: str, text: str, api_key: str) -> str:
    """One OpenAI Responses call with the study's generation settings (code/arms.py `gen_body`)."""
    body = {
        "model": model,
        "input": [{"role": "user", "content": text}],
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 32000,
        "store": False,
    }
    response = httpx.post(OPENAI_RESPONSES, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
    response.raise_for_status()
    return "".join(
        part.get("text", "")
        for item in response.json().get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", help="the request text to route")
    parser.add_argument("--pair", choices=sorted(study.PAIRS), default="openai")
    parser.add_argument("--item", help="route a study request by id instead of REQUEST")
    parser.add_argument("--recorded", action="store_true", help="with --item: use its recorded vector, no SIE call")
    parser.add_argument("--answer", action="store_true", help="send the request to the chosen OpenAI model")
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE, help="directory written by fetch.py")
    args = parser.parse_args()
    if (args.request is None) == (args.item is None):
        parser.error("give either REQUEST or --item")
    if args.recorded and args.item is None:
        parser.error("--recorded needs --item")
    if args.answer and args.pair != "openai":
        parser.error("--answer calls OpenAI models only; use --pair openai")
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if args.answer and not openai_key:
        parser.error("--answer needs OPENAI_API_KEY")

    try:
        s = study.load_study(args.evidence)
    except (OSError, ValueError) as error:
        print(f"Evidence check failed: {error}", file=sys.stderr)
        return 1
    bank = study.bank(s, args.pair)

    index = None
    text = args.request
    if args.item is not None:
        try:
            index = s.index(args.item)
        except KeyError as error:
            print(error.args[0], file=sys.stderr)
            return 1
        text = s.prompts[index]

    if args.recorded:
        vector = s.raw_vectors[index]
    else:
        try:
            with SIEClient() as client:  # SIE Serverless when SIE_API_KEY is set; SIE_BASE_URL selects another server
                vector = embed(client, text)
        except ValueError as error:
            print(f"Cannot reach SIE: {error}", file=sys.stderr)
            return 1
        except SIEError as error:  # Messages are printed without the request, which may be private.
            print(f"The {study.EMBEDDER} call failed: {type(error).__name__}", file=sys.stderr)
            return 1

    # A study calibration request is scored without itself, as the study scored it.
    decision = bank.route(vector, exclude=args.item)
    names = study.PAIRS[args.pair]
    chosen = names.cheap_name if decision["route"] == "cheap" else names.strong_name
    relation = ">=" if decision["route"] == "cheap" else "<"
    print(f"pair   {args.pair}: {names.cheap_name} (cheap) or {names.strong_name} (strong)")
    print(f"score  {decision['score']:.4f} {relation} tau {decision['tau']:.4f}")
    print(f"route  {decision['model']} ({chosen})")
    if index is not None:
        recorded = s.pairs[args.pair]
        print(f"study  {s.ids[index]} ({'calibration' if s.calibration[index] else 'held-out'})")
        print(f"       recorded score {recorded.score[index]:.4f}, routed {recorded.route[index]}")

    if args.answer:
        try:
            reply = answer_openai(decision["model"], text, openai_key)
        except (httpx.HTTPError, ValueError) as error:
            print(f"The OpenAI call failed: {type(error).__name__}", file=sys.stderr)
            return 1
        print(f"\n{reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
