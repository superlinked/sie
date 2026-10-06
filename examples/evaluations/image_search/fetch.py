"""Download the hash-pinned public recording, without credentials or inference."""

import argparse
import io
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from evidence import ARCHIVE_SHA256, COST_SHA256, DATASET, DEFAULT_ROOT, PREFIX, REVISION, ROOT_NAME, digest, verify


def download(name: str, expected_sha256: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PREFIX}/{name}"
    with urlopen(url, timeout=60) as response:
        data = response.read(10_000_001)
    if len(data) > 10_000_000 or digest(data) != expected_sha256:
        raise ValueError(f"Download failed its pinned SHA256 check: {name}")
    return data


def unpack(data: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 2
                or path.parts[0] != ROOT_NAME
                or member.name in names
            ):
                raise ValueError("Archive contains an unsafe or duplicate member")
            names.add(member.name)
            total += member.size
        if len(members) != 57 or total > 9_000_000:
            raise ValueError("Archive inventory differs from the bounded 57-file recording")
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.name).parts[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member)
            if content is None:
                raise ValueError("Archive member has no data")
            path.write_bytes(content.read())


def fetch(root: Path) -> None:
    if root.exists():
        if root.is_symlink():
            raise ValueError("Output may not be a symbolic link")
        verify(root)
        return
    archive = download("image20-8-semantic-audit.tar.gz", ARCHIVE_SHA256)
    cost = download("cost-projection.json", COST_SHA256)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="image-search-", dir=root.parent) as temporary:
        stage = Path(temporary) / ROOT_NAME
        stage.mkdir()
        unpack(archive, stage)
        (stage / "cost-projection.json").write_bytes(cost)
        verify(stage)
        shutil.move(str(stage), str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    fetch(args.output)
    print(f"Verified public recording at {args.output}")


if __name__ == "__main__":
    main()
