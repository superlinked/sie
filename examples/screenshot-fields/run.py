"""Send the complete KB01 screenshot once to an explicitly configured SIE endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sie_sdk import SIEClient

from evidence import DEFAULT_EVIDENCE, MODEL, verify


def extract(client: SIEClient, directory: Path) -> dict:
    request = verify(directory)["request"]
    return client.chat_completions(
        MODEL,
        request["messages"],
        max_completion_tokens=32768,
        extra_body={"temperature": 0, "presence_penalty": 0},
        response_format=request["response_format"],
        wait_for_capacity=False,
        max_oom_retries=0,
        read_timeout_s=900,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="SIE base URL configured with the recorded model profile")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=Path("run-output/response.json"))
    args = parser.parse_args()
    verify(args.evidence)
    # Reserve before dispatch so an existing response cannot be overwritten or cause a wasted call.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            with SIEClient(args.endpoint, api_key=os.environ.get("SIE_API_KEY", "")) as client:
                result = extract(client, args.evidence)
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        except Exception:
            args.output.unlink(missing_ok=True)
            raise
    print(f"Saved complete response to {args.output}")


if __name__ == "__main__":
    main()
