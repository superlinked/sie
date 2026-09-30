#!/usr/bin/env python3
"""Encode the look-alike search test yourself on SIE Cloud, then score it.

    SIE_API_KEY=sk-sie-... uv run python run.py --smoke     # 20 questions, a fraction of a cent
    SIE_API_KEY=sk-sie-... uv run python run.py             # the whole test, about 1.2M tokens
    uv run python score.py --vectors run-output/qwen3-embedding-4b

It sends the 5,658 passages as documents and the 440 questions with
is_query=True, which applies the model's own query instruction, the way the
recorded run did. Passages go in batches of 32. Needs evidence/ from fetch.py.

--smoke encodes the first 20 questions with only their answer and look-alike
passages and reports, per question, which of the two scored higher: enough to
check the key, the endpoint and the query flag without paying for the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MODEL = "Qwen/Qwen3-Embedding-4B"
BASE_URL = "https://api.superlinked.com"
BATCH = 32


def encode(client: SIEClient, rows: list[dict], *, is_query: bool) -> np.ndarray:
    out = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        # A blank passage is not sent; it gets a zero vector, which scores 0 for every question.
        sent = [r for r in chunk if r["text"].strip()]
        vectors = {}
        if sent:
            result = client.encode(MODEL, [{"text": r["text"]} for r in sent], is_query=is_query)
            vectors = {r["id"]: item["dense"] for r, item in zip(sent, result, strict=True)}
        width = len(next(iter(vectors.values()))) if vectors else 2560
        out.extend(vectors.get(r["id"], np.zeros(width)) for r in chunk)
        print(
            f"  {'questions' if is_query else 'passages'}: {min(start + BATCH, len(rows))}/{len(rows)}", file=sys.stderr
        )
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "run-output")
    args = parser.parse_args()
    key = os.environ.get("SIE_API_KEY")
    if not key:
        print("Set SIE_API_KEY.", file=sys.stderr)
        return 1
    if not EVIDENCE.exists():
        print("No evidence yet: run `python3 fetch.py` first.", file=sys.stderr)
        return 1
    client = SIEClient(api_key=key, base_url=BASE_URL)
    passages = json.loads((EVIDENCE / "inputs" / "corpus.json").read_text())["passages"]
    questions = json.loads((EVIDENCE / "inputs" / "questions.json").read_text())["questions"]
    by_id = {p["id"]: p for p in passages}

    if args.smoke:
        sample = questions[:20]
        q = encode(client, [{"id": s["id"], "text": s["question"]} for s in sample], is_query=True)
        pairs = [by_id[s[k]] for s in sample for k in ("gold", "lookalike")]
        d = encode(client, pairs, is_query=False)
        wins = 0
        for i, s in enumerate(sample):
            gold, look = d[2 * i], d[2 * i + 1]
            cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            ahead = cos(q[i], gold) > cos(q[i], look)
            wins += ahead
            print(f"{s['id']}  answer {'above' if ahead else 'below'} its look-alike  {s['question']}")
        print(f"\nthe answer outscored its look-alike on {wins} of {len(sample)}")
        return 0

    target = args.output / "qwen3-embedding-4b"
    # Write all four files to a staging directory and swap it in only once both phases succeed, so a failed run
    # never leaves corpus vectors from one run beside question vectors from another.
    args.output.mkdir(parents=True, exist_ok=True)
    # A fresh staging directory per invocation, so two runs never write into the same one.
    staging = Path(tempfile.mkdtemp(prefix="qwen3-embedding-4b.", dir=args.output))
    corpus_vectors = encode(client, passages, is_query=False)
    question_vectors = encode(client, [{"id": q["id"], "text": q["question"]} for q in questions], is_query=True)
    np.save(staging / "corpus.npy", corpus_vectors)
    (staging / "corpus.ids.json").write_text(json.dumps([p["id"] for p in passages]))
    np.save(staging / "queries.npy", question_vectors)
    (staging / "queries.ids.json").write_text(json.dumps([q["id"] for q in questions]))
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    print(f"wrote {target}; now: uv run python score.py --vectors {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
