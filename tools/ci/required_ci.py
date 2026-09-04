"""Fail the required check unless every mandatory lane succeeded."""

from __future__ import annotations

import json
import os
import sys

MANDATORY_JOBS = (
    "policy",
    "bootstrap",
    "python",
    "typescript",
    "rust",
    "contracts",
    "helm",
    "live-sdk",
    "cpu-stack",
    "python-distributions",
    "npm-distributions",
)


def failures(needs: dict[str, dict[str, object]]) -> list[str]:
    return [
        f"{name}: {needs.get(name, {}).get('result', 'missing')}"
        for name in MANDATORY_JOBS
        if needs.get(name, {}).get("result") != "success"
    ]


def main() -> int:
    failed = failures(json.loads(os.environ["NEEDS"]))
    if failed:
        print("Mandatory CI lanes did not succeed:\n" + "\n".join(failed), file=sys.stderr)
        return 1
    print("Every mandatory CI lane succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
