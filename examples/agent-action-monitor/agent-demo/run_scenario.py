"""Run clean and poisoned agent scenarios through the HTTP gate."""

from __future__ import annotations

import argparse
import json
import sys

from harness import run_scenario


def _print_result(scenario: str, result: dict[str, object]) -> None:
    print(f"\n=== {scenario} ===")
    print(f"verdict:  {result['verdict']}")
    print(f"applied:  {result['applied']}")
    if result.get("reasons"):
        print(f"reasons:  {', '.join(result['reasons'])}")  # type: ignore[arg-type]
    print(f"action:   {json.dumps(result['action'], indent=2)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DUSK agent-demo scenarios over HTTP")
    parser.add_argument(
        "--scenario",
        choices=["clean", "poisoned", "both"],
        default="both",
        help="Which scenario to run (default: both).",
    )
    parser.add_argument("--agent-id", default="netops-agent", help="Agent identity to use.")
    args = parser.parse_args()

    scenarios = ["clean", "poisoned"] if args.scenario == "both" else [args.scenario]
    exit_code = 0
    for scenario in scenarios:
        try:
            result = run_scenario(args.agent_id, scenario)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {scenario} ===\nerror: {exc}", file=sys.stderr)
            print("Is the gate (stub or real) and mock-prod running?", file=sys.stderr)
            exit_code = 1
            continue
        _print_result(scenario, result)
        if scenario == "poisoned" and result["verdict"] == "ALLOW":
            # WOULD-BLOCK is valid in watch mode; ALLOW means detection failed.
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
