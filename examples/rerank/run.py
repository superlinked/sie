"""Inspect recorded cases offline, or explicitly run a bounded SIE trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from score import MODEL_REVISION, load, score

MODEL = "Qwen/Qwen3-Reranker-4B"
RULES = {
    "in-force": "Return the final rule the agency has adopted on this subject, not a proposed rule. A proposal that is still open for comment is not relevant.",
    "proposed": "Return the proposed rule the agency has published for comment on this subject, not a final rule. A rule already adopted is not relevant.",
}


def envelope(case: dict, arm: str) -> dict:
    return {
        "query": {"text": case["query"]},
        "items": [{"id": item["id"], "text": item["text"]} for item in case["candidates"]],
        "instruction": RULES[arm],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--check", action="store_true", help="validate every recorded model offline")
    parser.add_argument("--show", help="print both request envelopes for this case ID")
    parser.add_argument("--record", action="store_true", help="run inference; may incur API charges")
    parser.add_argument("--limit", type=int, default=1, help="case limit for an explicit trial (default: one)")
    parser.add_argument("--base-url", default=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"))
    parser.add_argument("--out", type=Path, default=Path("trial.json"))
    args = parser.parse_args()
    cases = load(args.data / "inputs/cases_test.json")["cases"]
    if args.show:
        found = next((case for case in cases if case["id"] == args.show), None)
        if found is None:
            raise SystemExit("Unknown case ID")
        print(json.dumps({arm: envelope(found, arm) for arm in RULES}, indent=2))
        return
    if not args.record:
        result = score(args.data)
        print(
            f"Verified {result['cases']} questions, {result['candidatesPerQuery']} candidates and all seven recorded models."
        )
        return
    if not 1 <= args.limit <= len(cases):
        raise SystemExit("--limit must be between 1 and 499")
    if args.out.exists():
        raise SystemExit("--out already exists; choose a new file to preserve previous trials")
    from sie_sdk import Item, SIEClient

    client = SIEClient(args.base_url, api_key=os.environ.get("SIE_API_KEY", ""), timeout_s=300)
    models = client.list_models()
    listed = models if isinstance(models, list) else models.get("models", [])
    served = next(
        (model.get("revision") for model in listed if isinstance(model, dict) and model.get("name") == MODEL), None
    )
    if served != MODEL_REVISION:
        raise SystemExit("Endpoint does not report the recorded model revision; no inference sent")
    results = []
    for case in cases[: args.limit]:
        for arm, instruction in RULES.items():
            response = client.score(
                MODEL,
                Item(text=case["query"]),
                [Item(id=item["id"], text=item["text"]) for item in case["candidates"]],
                instruction=instruction,
                wait_for_capacity=True,
                provision_timeout_s=900,
            )
            ids = {item["id"] for item in case["candidates"]}
            returned = response["scores"]
            if len(returned) != len(ids) or {item["item_id"] for item in returned} != ids:
                raise SystemExit("Response candidate set differs; trial not written")
            results.append(
                {"case": case["id"], "arm": arm, "model_revision": client.last_model_revision, "response": response}
            )
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
        output.write("\n")
    print(f"Recorded {len(results)} trial calls in {args.out}. These do not replace the published study.")


if __name__ == "__main__":
    main()
