"""Download the pinned public BBEH source; verify bytes before any source use."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

REVISION = "80d12ca916b7158f22293fcf3144f4d3d854d4be"
PREFIX = f"https://raw.githubusercontent.com/google-deepmind/bbeh/{REVISION}/"


def fetch(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve existing source snapshots; use a new output directory")
    pins = Path(__file__).parent / "pins"
    manifests, files = {}, {}
    for name in ("source-download.json", "full-source-download.json"):
        data = (pins / name).read_bytes()
        manifest = json.loads(data)
        if manifest["revision"] != REVISION:
            raise ValueError("Source revision changed")
        manifests[name] = data
        for relative, receipt in manifest["files"].items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or receipt["url"] != PREFIX + relative:
                raise ValueError("Unpinned source URL or unsafe output path")
            if relative in files and files[relative] != receipt:
                raise ValueError("Conflicting source inventory")
            files[relative] = receipt
    output.mkdir(parents=True)

    def download(item):
        relative, receipt = item
        with urlopen(receipt["url"], timeout=60) as response:
            data = response.read(receipt["bytes"] + 1)
        if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError("Source download failed exact size/hash validation: " + relative)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, files.items()))
        for name, data in manifests.items():
            (output / name).write_bytes(data)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {"revision": REVISION, "files": len(files), "bytes": sum(r["bytes"] for r in files.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(fetch(parser.parse_args().output)))
