"""Reproduce saved decisions or report a fully source-reviewed native run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fidelity import load_packet, native_report, recorded_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--native-calls", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    manifest, projection = load_packet(args.packet)
    report = recorded_report(projection)
    if args.native_calls:
        calls = [json.loads(line) for line in args.native_calls.read_text().splitlines() if line]
        decisions = json.loads(args.decisions.read_text()) if args.decisions else []
        native = native_report(manifest, calls, decisions)
        report["native"] = native
        for arm in ("luna", "sol"):
            baseline = {
                row["case_id"]: row["source_grounded"]["source_grounded_primary_outcome"] == "pass"
                for row in projection["provider_records"]
                if row["arm"] == arm
            }
            pairs = {key: value for key, value in native["outcomes"].items() if value in {"pass", "fail"}}
            report["native"][f"paired_{arm}"] = {
                "n": len(pairs),
                "both_pass": sum(value == "pass" and baseline[key] for key, value in pairs.items()),
                "native_only": sum(value == "pass" and not baseline[key] for key, value in pairs.items()),
                "rival_only": sum(value == "fail" and baseline[key] for key, value in pairs.items()),
                "neither_pass": sum(value == "fail" and not baseline[key] for key, value in pairs.items()),
            }
    elif args.decisions:
        parser.error("--decisions requires --native-calls")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
