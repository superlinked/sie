"""Validate the shipped Candle CUDA image on a runner without an NVIDIA driver.

This checks the executable and its runtime dependencies, not GPU inference. Only
libcuda.so.1 may be supplied by the host's NVIDIA container runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import struct
import subprocess
import tempfile
from pathlib import Path

DRIVER_LIBRARY = "libcuda.so.1"


def validate_config(records: list[dict], source_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("expected a full source revision")
    if len(records) != 1:
        raise ValueError("expected exactly one loaded CUDA image")
    record = records[0]
    if (record.get("Os"), record.get("Architecture")) != ("linux", "amd64"):
        raise ValueError("Candle CUDA image must be linux/amd64")
    config = record.get("Config", {})
    labels = config.get("Labels", {})
    if labels.get("org.opencontainers.image.revision") != source_revision:
        raise ValueError("Candle CUDA image source revision mismatch")
    if labels.get("org.opencontainers.image.source") != "https://github.com/superlinked/sie":
        raise ValueError("Candle CUDA image source repository mismatch")
    if config.get("Entrypoint") != ["/sie-server-rust"] or config.get("User") != "sie:sie":
        raise ValueError("Candle CUDA image entrypoint/user mismatch")
    environment = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    for key, expected in {
        "CUDA_VERSION": "12.4.1",
        "CUDA_COMPUTE_CAP": "89",
        "SIE_BUNDLE": "candle",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }.items():
        if environment.get(key) != expected:
            raise ValueError(f"Candle CUDA image {key} must be {expected}")


def validate_binary(binary: Path) -> None:
    info = binary.lstat()
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111 or info.st_size <= 1_000_000:
        raise ValueError("Candle worker must be a regular executable larger than 1 MB")
    with binary.open("rb") as stream:
        header = stream.read(64)
    if (
        len(header) != 64
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or struct.unpack_from("<H", header, 16)[0] not in (2, 3)
        or struct.unpack_from("<H", header, 18)[0] != 62
        or struct.unpack_from("<I", header, 20)[0] != 1
        or struct.unpack_from("<Q", header, 24)[0] == 0
        or struct.unpack_from("<H", header, 52)[0] != 64
    ):
        raise ValueError("Candle worker must be an ELF64 little-endian x86_64 executable")
    program_offset = struct.unpack_from("<Q", header, 32)[0]
    program_size, program_count = struct.unpack_from("<HH", header, 54)
    if (
        program_offset < 64
        or program_size != 56
        or program_count == 0
        or program_offset + program_size * program_count > info.st_size
    ):
        raise ValueError("Candle worker has an invalid ELF program header table")


def validate_dependencies(output: str) -> None:
    libraries: set[str] = set()
    missing: set[str] = set()
    loader = False
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([\w.+-]+) => (not found|/\S+ \(0x[0-9a-f]+\))", line)
        if match:
            name, destination = match.groups()
            if name in libraries:
                raise ValueError(f"duplicate runtime dependency: {name}")
            libraries.add(name)
            if destination == "not found":
                missing.add(name)
        elif re.fullmatch(r"linux-vdso\.so\.1 \(0x[0-9a-f]+\)", line):
            continue
        elif re.fullmatch(r"/\S*/ld-linux-x86-64\.so\.2 \(0x[0-9a-f]+\)", line):
            loader = True
        else:
            raise ValueError(f"unexpected runtime dependency diagnostic: {line}")
    if missing - {DRIVER_LIBRARY}:
        raise ValueError(f"missing image runtime libraries: {sorted(missing - {DRIVER_LIBRARY})}")
    if not loader or not {DRIVER_LIBRARY, "libc.so.6"} <= libraries:
        raise ValueError("expected CUDA driver, libc and x86_64 loader dependencies")


def capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
    if result.stderr.strip():
        raise ValueError(f"image validation diagnostic: {result.stderr.strip()}")
    return result.stdout.strip()


def validate(image: str, *, source_revision: str) -> None:
    validate_config(json.loads(capture(["docker", "image", "inspect", image])), source_revision)
    with tempfile.TemporaryDirectory(prefix="sie-rust-cuda-smoke-") as temporary:
        container = capture(["docker", "create", "--pull", "never", "--network", "none", image])
        try:
            binary = Path(temporary) / "sie-server-rust"
            capture(["docker", "cp", f"{container}:/sie-server-rust", str(binary)])
            validate_binary(binary)
        finally:
            capture(["docker", "rm", container])
    output = capture(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--env",
            "LC_ALL=C",
            "--entrypoint",
            "/bin/sh",
            image,
            "-ec",
            "test -f /sie-server-rust; test -x /sie-server-rust; ldd /sie-server-rust",
        ]
    )
    validate_dependencies(output)
    print("Candle CUDA executable and driverless runtime dependency checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--source-revision", required=True)
    validate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
