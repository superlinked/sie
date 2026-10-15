"""Check a built server image's engine launch contract without a GPU.

Runs ``engine_launch_probe.py`` inside the image, then requires that every
routed profile's argv parsed cleanly with the installed engine's own parser,
and that the engine's declared flag set equals the snapshot committed under
``conformance/engine-flags/``. An engine version without a snapshot fails with
the flags it removed and added relative to the nearest older snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools/ci/engine_launch_probe.py"
SNAPSHOT_DIR = ROOT / "conformance/engine-flags"
PROBE_TIMEOUT_S = 900


def probe_command(image: str, bundle: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--pull",
        "never",
        "--network",
        "none",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "--entrypoint",
        "python",
        image,
        "-",
        "--bundle",
        bundle,
    ]


def run_probe(image: str, bundle: str, *, timeout_s: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    try:
        result = subprocess.run(  # noqa: S603
            probe_command(image, bundle),
            input=PROBE.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"engine launch probe in {image} did not finish within {timeout_s:g}s") from error
    if result.returncode:
        raise RuntimeError(f"engine launch probe failed in {image}:\n{result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def snapshot_path(engine: str, version: str, snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    return snapshot_dir / f"{engine}-{version}.txt"


def render_snapshot(flags: list[str]) -> str:
    return "".join(f"{flag}\n" for flag in flags)


def read_snapshot(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _version_key(version: str) -> tuple[tuple[int, str], ...]:
    return tuple((int(part), "") if part.isdigit() else (-1, part) for part in re.split(r"[.+-]", version))


def nearest_older_snapshot(engine: str, version: str, snapshot_dir: Path = SNAPSHOT_DIR) -> Path | None:
    candidates = []
    for path in snapshot_dir.glob(f"{engine}-*.txt"):
        other = path.stem.removeprefix(f"{engine}-")
        if _version_key(other) < _version_key(version):
            candidates.append((_version_key(other), path))
    return max(candidates)[1] if candidates else None


def flag_diff(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    return sorted(set(old) - set(new)), sorted(set(new) - set(old))


def evaluate(report: dict[str, Any], snapshot_dir: Path = SNAPSHOT_DIR) -> list[str]:
    """Every contract violation in one probe report, as readable lines."""
    engine = report.get("engine")
    if engine is None:
        return []
    version = report["engine_version"]
    flags = report["flags"]
    problems = [f"{route['route']}: {route['error']}" for route in report["routes"] if route["error"]]
    expected = snapshot_path(engine, version, snapshot_dir)
    if expected.is_file():
        removed, added = flag_diff(read_snapshot(expected), flags)
        if removed or added:
            problems.append(
                f"{engine} {version} declares a different flag set than {expected.name}: "
                f"removed {removed or 'none'}, added {added or 'none'}"
            )
        return problems
    previous = nearest_older_snapshot(engine, version, snapshot_dir)
    if previous is None:
        problems.append(f"no snapshot of {engine} {version}'s declared flags; commit {expected.name}")
        return problems
    removed, added = flag_diff(read_snapshot(previous), flags)
    problems.append(
        f"no snapshot of {engine} {version}'s declared flags; commit {expected.name}. "
        f"Since {previous.name}: removed {removed or 'none'}, added {added or 'none'}"
    )
    return problems


def check(
    image: str,
    bundle: str,
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
    report_dir: Path | None = None,
    probe: Callable[[str, str], dict[str, Any]] = run_probe,
) -> dict[str, Any]:
    report = probe(image, bundle)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{bundle}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        if report.get("engine") is not None:
            name = snapshot_path(report["engine"], report["engine_version"], report_dir).name
            (report_dir / name).write_text(render_snapshot(report["flags"]), encoding="utf-8")
    problems = evaluate(report, snapshot_dir)
    if report.get("engine") is not None:
        expected = snapshot_path(report["engine"], report["engine_version"], snapshot_dir)
        if not expected.is_file():
            print(f"::group::{expected.relative_to(ROOT) if expected.is_relative_to(ROOT) else expected}")
            print(render_snapshot(report["flags"]), end="")
            print("::endgroup::")
    if problems:
        raise ValueError(f"engine launch contract failed for {bundle}:\n" + "\n".join(problems))
    if report.get("engine") is None:
        print(f"{bundle}: no profile launches an engine with a launch parser")
    else:
        print(
            f"{bundle}: {len(report['routes'])} routed profiles parsed by {report['engine']} "
            f"{report['engine_version']} ({len(report['flags'])} declared flags, sha256 {report['flags_sha256']})"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("image")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    try:
        check(args.image, args.bundle, report_dir=args.report_dir)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
