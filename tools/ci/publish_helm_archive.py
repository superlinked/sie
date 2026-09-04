#!/usr/bin/env python3
"""Publish the retained chart without replacing a different immutable version."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from tools.ci.release_artifact import file_digest, validate_manifest
from tools.ci.release_guard import stable_version

REGISTRY = "oci://ghcr.io/superlinked/charts"


def publish(directory: Path, *, version: str, source_revision: str, run_id: str) -> None:
    stable_version(version)
    manifest = validate_manifest(
        directory,
        kind="helm",
        version=version,
        tag_name=f"v{version}",
        source_revision=source_revision,
        run_id=run_id,
    )
    filename = f"sie-cluster-{version}.tgz"
    if [item["name"] for item in manifest["files"]] != [filename]:
        raise ValueError("chart archive must contain exactly the versioned package")
    chart = directory / filename
    with tempfile.TemporaryDirectory(prefix="sie-chart-verify-") as temporary:
        pull = ["helm", "pull", f"{REGISTRY}/sie-cluster", "--version", version, "--destination", temporary]
        result = subprocess.run(pull, check=False, capture_output=True, text=True)  # noqa: S603
        remote = Path(temporary) / filename
        if result.returncode:
            if not any(marker in result.stderr.lower() for marker in ("not found", "manifest unknown")):
                raise RuntimeError(f"cannot inspect existing chart: {result.stderr.strip()}")
            subprocess.run(["helm", "push", str(chart), REGISTRY], check=True)  # noqa: S603, S607
            subprocess.run(pull, check=True)  # noqa: S603
        if file_digest(chart) != file_digest(remote):
            raise ValueError("remote chart differs from the retained tested chart; refusing to replace it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    publish(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
