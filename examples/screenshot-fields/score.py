"""Validate and score the saved answer with the original, hash-verified scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evidence import DEFAULT_EVIDENCE, verify


def score(directory: Path, response_path: Path | None = None) -> dict:
    inputs = verify(directory)
    response_path = response_path or directory / "native/KB01-response.json"
    response = json.loads(response_path.read_text())
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Expected one complete, non-truncated answer")
    text = choices[0].get("message", {}).get("content")
    if not isinstance(text, str):
        raise ValueError("Missing JSON answer")
    value = json.loads(text)
    Draft202012Validator(inputs["schema"]).validate(value)
    # verify() authenticates this exact public scorer before any of its code executes.
    spec = importlib.util.spec_from_file_location("frozen_screenshot_scorer", directory / "source/code/score.py")
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    result = scorer.score_screen(inputs["screen"], {"text": text, "finish_reason": "stop", "http_status": 200})
    return {"screen": "KB01", "schema_valid": True, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--response", type=Path, help="Optional fresh response; pinned inputs remain unchanged")
    args = parser.parse_args()
    result = score(args.evidence, args.response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["fully_right"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
