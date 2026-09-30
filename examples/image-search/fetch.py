#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py             # the catalogue, questions, SIE's vectors and every ranking (about 30 MB)
    python3 fetch.py --photos    # also the 2,573 catalogue photos, for run.py (about 250 MB)

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the frozen catalogue and questions, SIE's
recorded vectors, every product's recorded top 20 for every question, and every
product's rank of the right image on Flickr30k and MS-COCO. score.py reads from
there and never touches the network. The photos are Amazon Berkeley Objects,
CC BY 4.0; evidence/ATTRIBUTION.md carries the credit.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores.
"""

from __future__ import annotations

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
REVISION = "96c541a6a72ecb2ca72ee3644caecdb209ed476c"
TASK = "image-search"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# Every file score.py reads. A listing missing one of these is a failure, not a
# short download: score.py would otherwise report a clean pass over whatever
# happened to arrive.
REQUIRED = (
    "manifest.json",
    "ATTRIBUTION.md",
    "inputs/catalogue.json",
    "inputs/questions.json",
    "rankings/e1.json.gz",
    "rankings/e0.json.gz",
    "vectors/siglip-so400m-384/images.npy",
    "vectors/siglip-so400m-384/images.ids.json",
    "vectors/siglip-so400m-384/texts.npy",
    "vectors/siglip-so400m-384/texts.ids.json",
    "stats/e1.json",
    "stats/e0.json",
)


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the value can be checked against HuggingFace by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(url: str):
    # No Authorization header: the dataset is public and this must work for a
    # reader who has never signed in to HuggingFace.
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    return urllib.request.urlopen(request, timeout=300)


def get(url: str) -> bytes:
    with fetch(url) as response:
        return response.read()


def listing() -> list[dict]:
    """Every file under the task folder. The tree API pages its answer (the photos alone are 2,573 files), so this
    follows each page's `Link: <...>; rel="next"` header until there is none."""
    entries: list[dict] = []
    url: str | None = f"{API}/{TASK}?recursive=true"
    while url:
        with fetch(url) as response:
            entries += json.loads(response.read())
            link = response.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1 : part.index(">")]
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    return files


def main() -> int:
    photos = "--photos" in sys.argv[1:]
    print(f"{DATASET} at {REVISION}")
    # Download into a staging directory and swap it in only once every required
    # file has arrived. Writing straight into evidence/ meant a listing missing
    # a file raised AFTER overwriting some of them, leaving a mixed set from two
    # revisions that score.py can accept whenever it happens to be internally
    # consistent.
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    try:
        total = 0
        written: set[str] = set()
        for entry in sorted(listing(), key=lambda item: item["path"]):
            remote = entry["path"]
            if not photos and remote.startswith(f"{TASK}/photos/"):
                continue
            if not remote.startswith(f"{TASK}/"):
                raise SystemExit(f"{remote}: the listing holds a path outside {TASK}/")
            relative = remote[len(TASK) + 1 :]
            target = staging / relative
            if not target.resolve().is_relative_to(staging.resolve()):
                raise SystemExit(f"{remote}: the listing holds a path that leaves the download directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            body = get(f"{FILES}/{remote}")
            if entry.get("size") is not None and len(body) != entry["size"]:
                raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
            # Every file is checked, by whichever id the dataset publishes for
            # it: the SHA-256 of the content for a Git LFS file, the git object
            # id for an ordinary blob. A file the listing gives neither id for
            # is a failure, not a skip.
            lfs_oid = (entry.get("lfs") or {}).get("oid")
            blob_oid = entry.get("oid")
            if lfs_oid:
                if hashlib.sha256(body).hexdigest() != lfs_oid:
                    raise SystemExit(f"{remote}: the bytes do not hash to the LFS digest the dataset lists")
            elif blob_oid:
                if git_blob_oid(body) != blob_oid:
                    raise SystemExit(f"{remote}: the bytes do not match the object id the dataset lists")
            else:
                raise SystemExit(f"{remote}: the dataset lists no id for this file, so it cannot be checked")
            target.write_bytes(body)
            written.add(relative)
            total += len(body)
            if not relative.startswith("photos/"):
                print(f"  {relative} ({len(body)} bytes)")

        missing = [name for name in REQUIRED if name not in written]
        if missing:
            print(
                "The download is incomplete; score.py would not be scoring the recorded run:",
                file=sys.stderr,
            )
            for name in missing:
                print(f"  missing {name}", file=sys.stderr)
            return 1

        # Move the old evidence aside and delete it only once the new set is in place,
        # so a failed rename never leaves no evidence at all.
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
        # A failure leaves the previous evidence/ untouched and removes the
        # half-downloaded staging directory rather than leaving it to be found.
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Wrote {total} bytes to {EVIDENCE}")
    print("Now run: uv run python score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
