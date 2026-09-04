#!/usr/bin/env python3
"""Reuse completed outputs from the original run when a release job is retried."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tools.ci.release_artifact import validate_manifest


def restore(
    directory: Path,
    *,
    name: str,
    kind: str,
    version: str,
    tag_name: str,
    source_revision: str,
    run_id: str,
    evidence_of: Path | None = None,
) -> bool:
    pages = json.loads(
        subprocess.check_output(  # noqa: S603
            ["gh", "api", "--paginate", "--slurp", f"repos/superlinked/sie/actions/runs/{run_id}/artifacts"],  # noqa: S607
            text=True,
        )
    )
    matches = [item for page in pages for item in page["artifacts"] if item["name"] == name]
    if not matches:
        return False
    if len(matches) != 1 or matches[0].get("expired") is not False:
        raise ValueError("original release artifact is duplicated or expired")
    original = matches[0].get("workflow_run", {})
    if str(original.get("id")) != str(run_id) or original.get("head_sha") != source_revision:
        raise ValueError("original release artifact source/run mismatch")
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "gh",
            "run",
            "download",
            run_id,
            "--repo",
            "superlinked/sie",
            "--name",
            name,
            "--dir",
            str(directory),
        ],
        check=True,
    )
    identity = {
        "kind": kind,
        "version": version,
        "tag_name": tag_name,
        "source_revision": source_revision,
        "run_id": run_id,
    }
    if evidence_of is None:
        validate_manifest(directory, **identity)
    else:
        validate_manifest(evidence_of, **identity)
        if (
            sorted(path.name for path in directory.iterdir()) != ["provenance.json"]
            or (directory / "provenance.json").is_symlink()
            or (directory / "provenance.json").read_bytes() != (evidence_of / "provenance.json").read_bytes()
        ):
            raise ValueError("retained image evidence differs from its retained archive")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-of", type=Path)
    restored = restore(**vars(parser.parse_args()))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"restored={str(restored).lower()}\n")


if __name__ == "__main__":
    main()
