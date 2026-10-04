#!/usr/bin/env python3
"""Collect existing direct-worker proofs; this does not enable fleet routing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from sie_server.config.equivalence import read_equivalence_record
from sie_server.config.fleet_equivalence import (
    MAX_EVIDENCE_AGE_S,
    MAX_FLEET_BYTES,
    MAX_FLEET_RECORDS,
    FleetEquivalenceRecord,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True, help="direct-worker v2 record; repeat")
    parser.add_argument("--output", type=Path, required=True, help="new inventory file (never overwritten)")
    parser.add_argument("--max-age-s", type=int, default=3600, help="freshness window, 1..86400 seconds")
    args = parser.parse_args(argv)
    if not 1 <= args.max_age_s <= MAX_EVIDENCE_AGE_S or len(args.record) > MAX_FLEET_RECORDS:
        parser.error("max age must be 1..86400 seconds and at most 256 records are allowed")
    try:
        records = []
        total = 0
        for path in args.record:
            record = read_equivalence_record(path)
            total += len(record.model_dump_json().encode())
            if total > MAX_FLEET_BYTES:
                raise ValueError("fleet evidence exceeds the byte limit")
            records.append(record)
        fleet = FleetEquivalenceRecord(records=tuple(records))
        encoded = fleet.model_dump_json(indent=2).encode() + b"\n"
        if len(encoded) > MAX_FLEET_BYTES:
            raise ValueError("fleet evidence exceeds the byte limit")
        # Publish only closed, complete bytes. A same-filesystem hard link
        # atomically refuses an existing file or symlink; failure cleans staging.
        with TemporaryDirectory(prefix=".sie-fleet-", dir=args.output.parent) as directory:
            staged = Path(directory) / "inventory.json"
            staged.write_bytes(encoded)
            os.link(staged, args.output)
    except (OSError, ValueError, RecursionError):
        # Paths and parser errors can contain sensitive operator metadata.
        print("fleet evidence could not be collected", file=sys.stderr)
        return 2
    return 0 if fleet.passed and fleet.is_fresh(max_age_s=args.max_age_s) else 1


if __name__ == "__main__":
    raise SystemExit(main())
