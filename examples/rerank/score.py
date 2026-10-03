"""Reproduce recorded top-first quality; standard library, no inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
EXPECTED = {
    "sie-qwen3-reranker-4b": (483, 469, 455),
    "cohere-rerank-4-fast": (389, 447, 357),
    "cohere-rerank-4-pro": (435, 477, 420),
    "claude-haiku-4-5": (487, 490, 478),
    "gpt-6-luna": (492, 491, 485),
    "voyage-rerank-2.5": (484, 478, 464),
    "voyage-rerank-3": (475, 485, 462),
}
ARMS = {"in-force": "asked-final", "proposed": "asked-proposal"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score(data: Path) -> dict:
    manifest = load(data / "manifest.json")
    require(manifest["model_revision"] == MODEL_REVISION, "Model revision differs")
    for name, digest in manifest["files_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe manifest path")
        require(hashlib.sha256((data / path).read_bytes()).hexdigest() == digest, f"Digest differs: {name}")
    cases_list = load(data / "inputs/cases_test.json")["cases"]
    cases = {case["id"]: case for case in cases_list}
    require(len(cases) == len(cases_list) == 499, "Expected 499 distinct questions")
    for case in cases.values():
        require(
            len({item["id"] for item in case["candidates"]}) == len(case["candidates"]) == 20,
            "Expected 20 distinct candidates",
        )
    results = {}
    for model, published in EXPECTED.items():
        recorded = rows(data / "calls" / f"{model}.jsonl")
        repeats = 2 if model in {"claude-haiku-4-5", "gpt-6-luna"} else 1
        expected = {(case, arm, rep) for case in cases for arm in ("none", *ARMS) for rep in range(repeats)}
        indexed = {(row["case"], row["arm"], row["rep"]): row for row in recorded}
        require(
            len(indexed) == len(recorded) and set(indexed) == expected,
            f"Missing, duplicate or unexpected call: {model}",
        )
        partial = 0
        for row in recorded:
            candidates = {item["id"]: item for item in cases[row["case"]]["candidates"]}
            ranking = row["ranked_ids"]
            require(not row.get("error") and bool(ranking), f"Unusable result: {model}")
            require(len(set(ranking)) == len(ranking) and set(ranking) <= set(candidates), f"Invalid ranking: {model}")
            require(
                row["top1"] == ranking[0] and row["top1_role"] == candidates[ranking[0]]["role"],
                f"Winner differs: {model}",
            )
            partial += len(ranking) != 20
        hits = [
            sum(all(indexed[(case, arm, rep)]["top1_role"] == role for rep in range(repeats)) for case in cases)
            for arm, role in ARMS.items()
        ]
        reliable = sum(
            all(indexed[(case, arm, rep)]["top1_role"] == role for arm, role in ARMS.items() for rep in range(repeats))
            for case in cases
        )
        require((*hits, reliable) == published, f"Published quality differs: {model}")
        results[model] = {
            "adopted": hits[0],
            "proposal": hits[1],
            "bothRulesAllRepetitions": reliable,
            "repetitions": repeats,
            "partialRankings": partial,
            "calls": len(recorded),
        }
    original = rows(data / "history/first-attempt.jsonl")
    repaired = rows(data / "history/transport-repair.jsonl")
    require(len(original) == 1497 and len(repaired) == 1515 and repaired[:1497] == original, "Original history changed")
    failed = {(row["case"], row["arm"], row["rep"]) for row in original if row.get("error")}
    retries = {(row["case"], row["arm"], row["rep"]) for row in repaired[1497:]}
    require(len(failed) == 18 and retries == failed, "Retries must only repair original transport failures")
    analysis = load(data / "analysis.json")
    return {"cases": 499, "candidatesPerQuery": 20, "transportRetries": 18, "models": results, "analysis": analysis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--emit", type=Path)
    args = parser.parse_args()
    result = score(args.data)
    output = json.dumps(result, indent=2) + "\n"
    if args.emit:
        args.emit.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
