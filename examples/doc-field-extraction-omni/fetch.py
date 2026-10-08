#!/usr/bin/env python3
"""Download the pinned Omni documents and the published replies of one set, checking every hash.

    python3 fetch.py --set confirm
    python3 fetch.py --set pilot

Standard library only, except that the pilot needs Pillow for one re-encoded page (see below). No token,
no account, no API key: both datasets are public and are read anonymously at pinned revisions, never `main`.

Everything lands in data/ (or --cache):

  data/omni/metadata.jsonl     the Omni rows: each document's JSON schema and gold JSON
  data/omni/images/            the set's page images, each checked against sets/<set>.image_manifest.json
  data/packet/<set>/...        the published replies, per-document scores and readable request bodies

run.py and score.py call the same functions, so they also download whatever is missing.

Pilot document omni-407 is a PNG over 5 MB as base64, so the study re-encoded it once to JPEG quality 90 and
sent the re-encoded bytes. Pillow 12.2.0 reproduces those bytes exactly; other Pillow or libjpeg builds may not,
and the fetch stops if the hash differs. The confirmation set has no such page.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
SETS = ("pilot", "confirm")
# A sanity bound on one page image; the largest pinned source is 4.3 MB, and every file is also hash-checked.
MAX_IMAGE_BYTES = 16 * 1024 * 1024
PUBLISHED = {
    "pilot": {
        "replies": "pilot/grader/A1.jsonl",
        "scores": "pilot/grader/omni_scores.json",
        "show": "pilot/requests/A1.bodies.show.jsonl",
    },
    "confirm": {
        "replies": "confirm/grader/A1.jsonl",
        "scores": "confirm/grader/omni_scores.json",
        "show": "confirm/requests/A1.confirm.bodies.show.jsonl",
        "bodies_sha256": "confirm/requests/A1.confirm.bodies.jsonl.sha256",
    },
}


def http_get(url: str, max_bytes: int | None = None, attempts: int = 5) -> bytes:
    """GET a pinned public file, retrying transient failures."""
    request = urllib.request.Request(url, headers={"User-Agent": "sie-doc-field-extraction-omni/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # pinned HTTPS origins
                data = response.read() if max_bytes is None else response.read(max_bytes + 1)
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 + 3 * attempt)
            continue
        if max_bytes is not None and len(data) > max_bytes:
            raise SystemExit(f"{url}: larger than its pinned size")
        return data
    raise AssertionError("unreachable")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Write a whole file or nothing, so an interrupted or concurrent run never leaves a truncated cache entry."""
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    partial = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def cached_pinned(url: str, path: Path, size: int, digest: str) -> bytes:
    """Return a pinned file, downloading it once into the cache and checking its size and sha256."""
    if path.exists():
        data = path.read_bytes()
        if len(data) == size and sha256(data) == digest:
            return data
    data = http_get(url, max_bytes=size)
    if len(data) != size or sha256(data) != digest:
        raise SystemExit(f"{url}: size or sha256 does not match the pin")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, data)
    return data


def load_omni_rows(cache: Path) -> dict[str, dict[str, Any]]:
    """The pinned Omni metadata rows, keyed as omni-<id>."""
    pin = PINS["omni"]["metadata"]
    data = cached_pinned(
        f"{PINS['omni']['base_url']}/{pin['path']}", cache / "omni" / pin["path"], pin["bytes"], pin["sha256"]
    )
    rows = {}
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[f"omni-{row['id']}"] = row
    return rows


def load_set(name: str) -> list[dict[str, Any]]:
    """The published image manifest of a set, verified against its pin, in request order."""
    spec = PINS["sets"][name]
    data = (HERE / spec["image_manifest"]).read_bytes()
    if sha256(data) != spec["image_manifest_sha256"]:
        raise SystemExit(f"{spec['image_manifest']}: sha256 does not match the published manifest")
    rows = json.loads(data)
    for row in rows:
        row.setdefault("slot", "study")
    study = sorted(row["id"] for row in rows if row["slot"] == "study")
    if len(study) != spec["study_documents"] or sha256("\n".join(study).encode()) != spec["study_ids_sha256"]:
        raise SystemExit(f"{name}: study id list does not match its pinned hash")
    return rows


def study_ids(name: str) -> list[str]:
    """The scored documents of a set; the pilot also has six unscored display rows."""
    return [row["id"] for row in load_set(name) if row["slot"] == "study"]


def derive_image(doc_id: str, source: bytes, spec: dict[str, Any]) -> bytes:
    """Re-encode a page the study re-encoded (pilot omni-407 only)."""
    try:
        from PIL import Image  # optional: only the pilot needs it
    except ImportError as exc:
        raise SystemExit(f"{doc_id} is a re-encoded page; install Pillow ({spec['reproduced_with']} matched)") from exc
    out = io.BytesIO()
    Image.open(io.BytesIO(source)).convert("RGB").save(out, "JPEG", quality=90)
    return out.getvalue()


def prepare_images(set_name: str, cache: Path) -> dict[str, Path]:
    """Download every image of a set once, check it against the published manifest, return id -> file."""
    rows = load_set(set_name)
    omni = load_omni_rows(cache)
    derived = PINS["sets"][set_name]["derived_images"]
    folder = cache / "omni" / "images"
    folder.mkdir(parents=True, exist_ok=True)

    def one(row: dict[str, Any]) -> tuple[str, Path]:
        doc_id = row["id"]
        name = omni[doc_id]["file_name"]
        spec = derived.get(doc_id)
        if spec is None and name.split("/")[-1] != row["image_file"]:
            raise SystemExit(f"{doc_id}: Omni file {name} is not the manifest's {row['image_file']}")
        if spec is not None and name != spec["source_file"]:
            raise SystemExit(f"{doc_id}: Omni file {name} is not the pinned source {spec['source_file']}")
        source_sha = spec["source_sha256"] if spec else row["image_sha256"]
        path = folder / name.split("/")[-1]
        data = path.read_bytes() if path.exists() else b""
        if sha256(data) != source_sha:
            data = http_get(f"{PINS['omni']['base_url']}/{name}", max_bytes=MAX_IMAGE_BYTES)
            if sha256(data) != source_sha:
                raise SystemExit(f"{doc_id}: downloaded {name} does not match its pinned sha256")
            write_atomic(path, data)
        if spec is not None:
            image = derive_image(doc_id, data, spec)
            if sha256(image) != row["image_sha256"]:
                raise SystemExit(
                    f"{doc_id}: the JPEG q90 re-encode gives {sha256(image)}, not the published "
                    f"{row['image_sha256']}. The study used {spec['reproduced_with']}; other Pillow or libjpeg "
                    "builds can encode differently."
                )
            path = cache / "omni" / "derived" / row["image_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(path, image)
        return doc_id, path

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(one, rows))


def fetch_published(set_name: str, cache: Path) -> dict[str, Path]:
    """The published replies, per-document scores and readable bodies of a set, verified against their pins."""
    paths = {}
    for role, rel in PUBLISHED[set_name].items():
        pin = PINS["packet"]["files"][rel]
        path = cache / "packet" / rel
        data = cached_pinned(f"{PINS['packet']['base_url']}/{rel}", path, pin["bytes"], pin["sha256"])
        if role == "bodies_sha256" and data.decode().split()[0] != PINS["sets"][set_name]["bodies_sha256"]:
            raise SystemExit(f"{rel} does not hold the pinned bodies sha256")
        paths[role] = path
    return paths


def main() -> None:
    """Fetch and verify one set."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--cache", type=Path, default=DATA, help="download folder (default data/)")
    args = parser.parse_args()
    images = prepare_images(args.set, args.cache)
    published = fetch_published(args.set, args.cache)
    report = {
        "set": args.set,
        "omni_revision": PINS["omni"]["revision"],
        "evidence_revision": PINS["packet"]["revision"],
        "images_verified": len(images),
        "published_files_verified": sorted(published),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
