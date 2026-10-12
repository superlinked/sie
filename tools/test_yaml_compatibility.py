"""Compare gateway YAML behavior with a pre-migration Git revision.

Run with Python's standard library only:
    python tools/test_yaml_compatibility.py --baseline-ref <commit>

Both builds read the current catalog. The legacy parser is confined to a
temporary checkout and never enters either production dependency graph.
"""

# ruff: noqa: INP001, S603, S607

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline_ref}^{{commit}}"], cwd=root, text=True
    ).strip()
    test_path = Path("packages/sie_gateway/tests/yaml_compatibility.rs")
    environment = dict(os.environ)
    environment["SIE_YAML_CATALOG_ROOT"] = str(root / "packages/sie_server")
    environment["CARGO_TARGET_DIR"] = str(root / "target")
    with tempfile.TemporaryDirectory(prefix="sie-yaml-compat-") as temporary:
        scratch = Path(temporary)
        baseline = scratch / "baseline"
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(root), str(baseline)], check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=baseline, check=True)
        source = (root / test_path).read_text()
        (baseline / test_path).parent.mkdir(parents=True, exist_ok=True)
        (baseline / test_path).write_text(source.replace("use serde_saphyr as yaml;", "use serde_yaml as yaml;"))
        snapshots = []
        for name, checkout in [("baseline", baseline), ("candidate", root)]:
            destination = scratch / f"{name}.json"
            environment["SIE_YAML_SNAPSHOT"] = str(destination)
            subprocess.run(
                [
                    "mise",
                    "exec",
                    "--",
                    "cargo",
                    "test",
                    "--locked",
                    "--manifest-path",
                    str(checkout / "packages/sie_gateway/Cargo.toml"),
                    "--test",
                    "yaml_compatibility",
                    "--",
                    "--nocapture",
                ],
                cwd=root,
                env=environment,
                check=True,
            )
            snapshots.append(json.loads(destination.read_text()))
        for section in snapshots[0]:
            if snapshots[0][section] != snapshots[1][section]:
                changed = [
                    key
                    for key in snapshots[0][section].keys() | snapshots[1][section].keys()
                    if snapshots[0][section].get(key) != snapshots[1][section].get(key)
                ]
                raise SystemExit(f"YAML compatibility mismatch in {section}: {changed}")
        print(
            f"YAML compatibility matches {revision}: {len(snapshots[0]['documents'])} files, per-file and combined registry states."
        )


if __name__ == "__main__":
    main()
