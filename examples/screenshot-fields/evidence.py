"""Fetch and authenticate the immutable inputs and recorded response."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence"
PINS = json.loads((HERE / "pins.json").read_text())["files"]
MODEL = "google/gemma-4-31B-it:h100-96k-hires-thinking-budget512-no-spec"


def check_bytes(name: str, data: bytes) -> None:
    pin = PINS[name]
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError(f"Evidence digest mismatch: {name}")


def fetch(directory: Path) -> None:
    for name, pin in PINS.items():
        target = directory / name
        if target.exists():
            check_bytes(name, target.read_bytes())
            continue
        request = urllib.request.Request(pin["url"], headers={"User-Agent": "sie-screenshot-fields-example"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        check_bytes(name, data)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    verify(directory)


def verify(directory: Path) -> dict:
    for name in PINS:
        check_bytes(name, (directory / name).read_bytes())
    case = directory / "source/cases/KB01"
    screens = [json.loads(line) for line in (directory / "source/screens.jsonl").read_text().splitlines()]
    selected = [screen for screen in screens if screen["screen"] == "KB01"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one KB01 source record")
    screen = selected[0]
    gold = json.loads((case / "gold.json").read_text())
    schema = json.loads((case / "schema.json").read_text())
    prompt = (case / "prompt.txt").read_bytes().decode("utf-8")
    image = (case / "image.png").read_bytes()
    case_files = {"image": "image.png", "prompt": "prompt.txt", "schema": "schema.json", "gold": "gold.json"}
    for name, filename in case_files.items():
        relative = screen["gold_file"] if name == "gold" else screen[name]
        if relative != f"cases/KB01/{filename}":
            raise ValueError(f"Unexpected source path: {name}")
        if screen[f"{name}_sha256"] != PINS[f"source/{relative}"]["sha256"]:
            raise ValueError(f"Inconsistent source digest: {name}")
    if screen["gold"] != gold or screen["base_gold_leaves"] != 26:
        raise ValueError("Inconsistent original gold")
    expected = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")},
                    }
                ],
            },
        ],
        "max_tokens": 32768,
        "temperature": 0,
        "presence_penalty": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fixture_kb01", "strict": True, "schema": schema},
        },
    }
    recorded = json.loads((directory / "native/KB01-request.json").read_text())
    if recorded != expected:
        raise ValueError("Recorded request differs from the complete verified inputs")
    return {"screen": screen, "schema": schema, "request": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.evidence)
    else:
        verify(args.evidence)
    print(f"Verified {len(PINS)} immutable evidence files")


if __name__ == "__main__":
    main()
