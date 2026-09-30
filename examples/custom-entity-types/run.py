#!/usr/bin/env python3
"""Run the SIE arm of the custom entity types study yourself, then score it.

    uv run python run.py --show ai-main-0                     # the request, without sending it
    SIE_URL=http://localhost:8080 uv run python run.py         # all 652 test sentences
    python3 score.py --rows run-output

Each sentence goes to `Ihor/gliner-biomed-large-v1.0` with its set's entity types as `labels` and
`threshold` 0.8, one sentence per item, 16 items per request, as the recorded run sent them. The
responses land in run-output/<set>.jsonl in the recorded rows' format. Needs evidence/ from fetch.py.

The model is in the open SIE catalog. Point SIE_URL at any SIE server that serves it, for example one
started locally with `sie-server serve --models Ihor/gliner-biomed-large-v1.0`; set SIE_API_KEY if the
server needs one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MODEL = "Ihor/gliner-biomed-large-v1.0"
THRESHOLD = 0.8
BATCH = 16


def load() -> tuple[dict, dict]:
    if not EVIDENCE.exists():
        raise SystemExit("no evidence/; run fetch.py first")
    results = json.loads((EVIDENCE / "results.json").read_text(encoding="utf-8"))
    data = {}
    for name in results["sets"]:
        path = EVIDENCE / "data" / "main" / f"{name}.jsonl"
        data[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return results["label_strings"], data


def request(labels: dict, set_name: str, texts: list[str]) -> dict:
    return {
        "model": MODEL,
        "items": [{"text": text} for text in texts],
        "labels": list(labels[set_name].values()),
        "options": {"threshold": THRESHOLD},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", metavar="SENTENCE_ID", help="print the request for one sentence and exit")
    parser.add_argument("--output", type=Path, default=HERE / "run-output")
    args = parser.parse_args()
    labels, data = load()

    if args.show:
        for set_name, rows in data.items():
            for row in rows:
                if row["id"] == args.show:
                    print(json.dumps(request(labels, set_name, [row["text"]]), indent=2, ensure_ascii=False))
                    return 0
        raise SystemExit(f"no sentence {args.show}")

    url = os.environ.get("SIE_URL")
    if not url:
        raise SystemExit("set SIE_URL to an SIE server that serves " + MODEL)
    from sie_sdk import SIEClient
    from sie_sdk.types import Item

    client = SIEClient(url, api_key=os.environ.get("SIE_API_KEY"))
    args.output.mkdir(parents=True, exist_ok=True)
    for set_name, rows in data.items():
        out = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            body = request(labels, set_name, [r["text"] for r in chunk])
            results = client.extract(
                MODEL, [Item(text=r["text"]) for r in chunk], labels=body["labels"], options=body["options"]
            )
            for row, result in zip(chunk, results, strict=True):
                entities = [
                    {
                        "start": e["start"],
                        "end": e["end"],
                        "label": e["label"],
                        "score": e.get("score"),
                        "text": e.get("text"),
                    }
                    for e in result.get("entities") or []
                ]
                for e in entities:
                    if row["text"][e["start"] : e["end"]] != e["text"]:
                        raise SystemExit(f"{row['id']}: the span at {e['start']}:{e['end']} is not the text it names")
                out.append({"id": row["id"], "output": json.dumps({"entities": entities}, ensure_ascii=False)})
        (args.output / f"{set_name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8"
        )
        print(f"  {set_name}: {len(out)} sentences", file=sys.stderr)
    print(f"Wrote {args.output}. Now run: python3 score.py --rows {args.output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
