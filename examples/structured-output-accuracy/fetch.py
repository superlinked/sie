#!/usr/bin/env python3
"""Download the recorded run, the SOB test split and SOB's scorer, each at a pinned version.

    python3 fetch.py
    python3 fetch.py --from-dir path/to/structured-output-accuracy   # a local copy of the dataset folder

Standard library only. No token, no account, no API key.

Three sources land in evidence/, each checked before it is kept:

    evidence/                 this task's folder of superlinked/sie-task-evidence at REVISION, every file
                              checked against the id the dataset lists for it
    evidence/sob/test.parquet the Structured Output Benchmark test split, interfaze-ai/sob at SOB_REVISION,
                              checked against SOB_PARQUET_SHA256
    evidence/sob/code/        SOB's scorer, prompt and schema code from JigsawStack/sob at SOB_COMMIT,
                              each file checked against its git blob id

score.py and run.py import SOB's code from there unmodified, so every model is scored by SOB's own
evaluate.py. REVISION, SOB_REVISION and SOB_COMMIT are commits, never branches, so a later upload cannot
change what this example scores.

`--from-dir` takes the task folder from disk instead of the dataset. It exists to test a folder before it
is uploaded; score.py still checks every file against the manifest it pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
# TODO(before merge): the dataset commit that holds structured-output-accuracy/. Never `main`.
REVISION = "0e1fb325cf25bd403ff678dd6031f649981eb434"
TASK = "structured-output-accuracy"

SOB_DATASET = "interfaze-ai/sob"
SOB_REVISION = "c118e38abdef6a8e1beba183405c70b28ff7d5a8"
SOB_PARQUET = "data/test-00000-of-00001.parquet"
SOB_PARQUET_SHA256 = "b86a475557aa65028a0d2c71aac780c8258a2a8cf15cd76e1300d9cc241c81a5"

SOB_REPO = "JigsawStack/sob"
SOB_COMMIT = "da785a8521c8954283b2989d01e54d80c4e023c6"
# Every file `import evaluate` and `from sob.common import prompts, schema_utils` need, with the git blob id
# GitHub lists for it at SOB_COMMIT. The empty __init__.py files are part of the package layout.
SOB_CODE = {
    "LICENSE": "74aa2554edf465eb5749e54097ad0bbbc3a96e46",
    "evaluate.py": "b4ea3aee1a4997b9975b1024347b6a5349ccad19",
    "sob/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "sob/common/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "sob/common/prompts.py": "a9b9c2c6ed999c8f49523b1d73996b7b13e0cd53",
    "sob/common/schema_utils.py": "ec9740413962e8f50a2a0549f31f38019228765b",
    "utils/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "utils/utils.py": "64c939edff68f68292549bac36776b54e8debcfd",
}

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
SOB_PARQUET_URL = f"https://huggingface.co/datasets/{SOB_DATASET}/resolve/{SOB_REVISION}/{SOB_PARQUET}"
SOB_CODE_URL = f"https://raw.githubusercontent.com/{SOB_REPO}/{SOB_COMMIT}"

# Every file score.py reads besides the calls, which the manifest lists. A download missing one is a
# failure, not a short download.
REQUIRED = ("manifest.json", "inputs/sob_sample.json", "inputs/nhtsa_records.json")


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes: what HuggingFace and GitHub list for an ordinary file.

    SHA-1 is not chosen here for its strength; it is the identifier both hosts already publish, so each
    value can be checked against them by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    # No Authorization header: every source is public and this must work for a reader with no account.
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def task_files_from_dataset() -> dict[str, bytes]:
    if REVISION.startswith("REPLACE"):
        raise SystemExit("fetch.py has no dataset revision pinned yet; pass --from-dir to use a local copy")
    print(f"{DATASET} at {REVISION}")
    entries = json.loads(get(f"{API}/{TASK}?recursive=true"))
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    out = {}
    for entry in sorted(files, key=lambda item: item["path"]):
        remote = entry["path"]
        body = get(f"{FILES}/{remote}")
        if entry.get("size") is not None and len(body) != entry["size"]:
            raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
        # Checked by whichever id the dataset publishes: the SHA-256 for a Git LFS file, the git object id
        # for an ordinary one. A file with neither is a failure, not a skip.
        lfs_oid = (entry.get("lfs") or {}).get("oid")
        if lfs_oid:
            if hashlib.sha256(body).hexdigest() != lfs_oid:
                raise SystemExit(f"{remote}: the bytes do not hash to the LFS digest the dataset lists")
        elif entry.get("oid"):
            if git_blob_oid(body) != entry["oid"]:
                raise SystemExit(f"{remote}: the bytes do not match the object id the dataset lists")
        else:
            raise SystemExit(f"{remote}: the dataset lists no id for this file, so it cannot be checked")
        out[remote[len(TASK) + 1 :]] = body
    return out


def task_files_from_dir(root: Path) -> dict[str, bytes]:
    print(f"{TASK} from {root} (a local copy; score.py checks it against the manifest)")
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def sob_files() -> dict[str, bytes]:
    print(f"{SOB_DATASET} at {SOB_REVISION}, {SOB_REPO} at {SOB_COMMIT}")
    parquet = get(SOB_PARQUET_URL)
    if hashlib.sha256(parquet).hexdigest() != SOB_PARQUET_SHA256:
        raise SystemExit(f"{SOB_PARQUET}: the bytes do not hash to {SOB_PARQUET_SHA256}")
    out = {"sob/test.parquet": parquet}
    for name, oid in SOB_CODE.items():
        body = get(f"{SOB_CODE_URL}/{name}")
        if git_blob_oid(body) != oid:
            raise SystemExit(f"{SOB_REPO}/{name}: the bytes do not match blob {oid} at {SOB_COMMIT}")
        out[f"sob/code/{name}"] = body
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", type=Path, help="a local copy of the dataset's task folder, for testing")
    args = parser.parse_args()

    files = task_files_from_dir(args.from_dir) if args.from_dir else task_files_from_dataset()
    missing = [name for name in REQUIRED if name not in files]
    if missing:
        print("The download is incomplete; score.py would not be scoring the recorded run:", file=sys.stderr)
        for name in missing:
            print(f"  missing {name}", file=sys.stderr)
        return 1
    files |= sob_files()

    # Written to a staging directory and swapped in only once every file has arrived and passed its
    # check, so a failure never leaves a mix of two revisions in evidence/.
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    try:
        for relative, body in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        backup = EVIDENCE.with_name(EVIDENCE.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        if EVIDENCE.exists():
            EVIDENCE.rename(backup)
        try:
            staging.rename(EVIDENCE)
        except OSError:
            if backup.exists():
                backup.rename(EVIDENCE)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    total = sum(len(body) for body in files.values())
    print(f"Wrote {len(files)} files, {total:,} bytes, to {EVIDENCE}")
    print("Now run: uv run score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
