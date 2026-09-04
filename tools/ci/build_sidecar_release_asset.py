#!/usr/bin/env python3
"""Extract the Linux sidecar asset from its tested release image archive."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

from tools.ci.release_artifact import create_manifest, file_digest
from tools.mise_tasks.docker_task import capture, load_image_archive, run, singleton_image

ELF_HEADER_PREFIX_SIZE = 20
ELF_MACHINE_X86_64 = 62


def inspect_abi(binary: Path) -> dict[str, object]:
    with binary.open("rb") as stream:
        header = stream.read(ELF_HEADER_PREFIX_SIZE)
    if (
        len(header) != ELF_HEADER_PREFIX_SIZE
        or header[:6] != b"\x7fELF\x02\x01"
        or struct.unpack("<H", header[18:20])[0] != ELF_MACHINE_X86_64
    ):
        raise ValueError("sidecar asset must be an ELF64 little-endian x86_64 executable")
    versions = capture(["readelf", "--version-info", str(binary)])
    glibc = sorted({tuple(map(int, value.split("."))) for value in re.findall(r"GLIBC_([0-9.]+)", versions)})
    if not glibc or glibc[-1] > (2, 36):
        raise ValueError("sidecar glibc requirement must be present and no newer than Debian 12 glibc 2.36")
    dynamic = capture(["readelf", "--dynamic", str(binary)])
    libraries = sorted(re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic))
    return {
        "format": "ELF64",
        "os": "linux",
        "architecture": "amd64",
        "glibc_minimum": ".".join(map(str, glibc[-1])),
        "needed_libraries": libraries,
        "runtime_baseline": "Debian 12 (glibc 2.36)",
    }


def build(directory: Path, out: Path, *, version: str, source_revision: str, run_id: str) -> None:
    image = singleton_image("ghcr.io/superlinked", version, "sie-server-sidecar")
    source = load_image_archive(image, directory, version=version, source_revision=source_revision, run_id=run_id)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("refusing to replace an existing native asset directory")
    binary = out / f"sie-server-sidecar-v{version}-linux-amd64"
    container = capture(["docker", "create", image])
    try:
        run(["docker", "cp", f"{container}:/sie-server-sidecar", str(binary)])
    finally:
        run(["docker", "rm", container])
    binary.chmod(0o755)
    abi = inspect_abi(binary)
    run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={binary.resolve()},dst=/asset,readonly",
            "--entrypoint",
            "/asset",
            image,
            "--help",
        ]
    )
    (out / f"{binary.name}.sha256").write_text(f"{file_digest(binary)}  {binary.name}\n")
    metadata = {
        **abi,
        "source_image": image,
        "source_image_id": source["metadata"]["image_id"],
        "source_revision": source_revision,
        "version": version,
        "tag_name": f"v{version}",
        "run_id": run_id,
        "sha256": file_digest(binary),
    }
    (out / f"{binary.name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    create_manifest(
        out,
        kind="native-sidecar",
        version=version,
        tag_name=f"v{version}",
        source_revision=source_revision,
        run_id=run_id,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
