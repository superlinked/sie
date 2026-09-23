from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools.ci import rust_cuda_image_smoke as smoke
from tools.mise_tasks import docker_task

SHA = "a" * 40
IMAGE = "sie-rust-cuda:local"
DEPENDENCIES = """
linux-vdso.so.1 (0x00007fff12340000)
libcuda.so.1 => not found
libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12 (0x00007fff12340000)
libssl.so.3 => /lib/x86_64-linux-gnu/libssl.so.3 (0x00007fff12340000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fff12340000)
/lib64/ld-linux-x86-64.so.2 (0x00007fff12340000)
"""


@pytest.fixture
def config():
    return [
        {
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "Entrypoint": ["/sie-server-rust"],
                "User": "sie:sie",
                "Labels": {
                    "org.opencontainers.image.revision": SHA,
                    "org.opencontainers.image.source": "https://github.com/superlinked/sie",
                },
                "Env": [
                    "CUDA_VERSION=12.4.1",
                    "CUDA_COMPUTE_CAP=89",
                    "SIE_BUNDLE=candle",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                ],
            },
        }
    ]


def executable(path: Path) -> Path:
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHIQQ", header, 16, 3, 62, 1, 4096, 64)
    struct.pack_into("<HHH", header, 52, 64, 56, 1)
    with path.open("wb") as stream:
        stream.write(header)
        stream.truncate(1_000_001)
    path.chmod(0o755)
    return path


def test_valid_loaded_image_and_executable(config, tmp_path):
    smoke.validate_config(config, SHA)
    smoke.validate_binary(executable(tmp_path / "worker"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("Architecture", "arm64"),
        ("Os", "windows"),
        ("Entrypoint", ["/bin/true"]),
        ("User", "root"),
        ("revision", "b" * 40),
        ("source", "https://github.com/other/repo"),
        ("CUDA_VERSION", "13.0.0"),
        ("CUDA_COMPUTE_CAP", "80"),
        ("SIE_BUNDLE", "default"),
    ],
)
def test_wrong_image_contract_fails(config, field, value):
    if field in ("Os", "Architecture"):
        config[0][field] = value
    elif field in ("Entrypoint", "User"):
        config[0]["Config"][field] = value
    elif field in ("revision", "source"):
        config[0]["Config"]["Labels"][f"org.opencontainers.image.{field}"] = value
    else:
        config[0]["Config"]["Env"] = [
            f"{field}={value}" if item.startswith(f"{field}=") else item for item in config[0]["Config"]["Env"]
        ]
    with pytest.raises(ValueError):
        smoke.validate_config(config, SHA)


@pytest.mark.parametrize(
    "change", ["empty", "truncated", "text", "arm64", "elf32", "not-executable", "symlink", "no-entry", "bad-programs"]
)
def test_bad_executable_fails(tmp_path, change):
    binary = executable(tmp_path / "worker")
    if change in ("empty", "truncated", "text"):
        binary.write_bytes({"empty": b"", "truncated": b"\x7fELF", "text": b"shell script"}[change])
    elif change == "not-executable":
        binary.chmod(0o644)
    elif change == "symlink":
        target = binary.rename(tmp_path / "target")
        binary.symlink_to(target)
    else:
        offset, data = {
            "arm64": (18, struct.pack("<H", 183)),
            "elf32": (4, b"\x01"),
            "no-entry": (24, b"\0" * 8),
            "bad-programs": (32, struct.pack("<Q", 2_000_000)),
        }[change]
        with binary.open("r+b") as stream:
            stream.seek(offset)
            stream.write(data)
    with pytest.raises(ValueError):
        smoke.validate_binary(binary)


def test_only_host_driver_may_be_missing():
    smoke.validate_dependencies(DEPENDENCIES)
    smoke.validate_dependencies(
        DEPENDENCIES.replace("libcuda.so.1 => not found", "libcuda.so.1 => /usr/lib/libcuda.so.1 (0x1234)")
    )


@pytest.mark.parametrize(
    "library", ["libc.so.6", "libstdc++.so.6", "libssl.so.3", "libcudart.so.12", "libcublas.so.12", "libnvrtc.so.12"]
)
def test_missing_image_library_is_fatal_even_when_ldd_exits_zero(library):
    lines = [line for line in DEPENDENCIES.splitlines() if not line.startswith(f"{library} =>")]
    with pytest.raises(ValueError, match="missing image runtime libraries"):
        smoke.validate_dependencies("\n".join([*lines, f"{library} => not found"]))


@pytest.mark.parametrize(
    "output",
    [
        "",
        "not a dynamic executable",
        "statically linked",
        "garbage",
        DEPENDENCIES.replace("libcuda.so.1 => not found", ""),
        DEPENDENCIES + "/sie-server-rust: version `GLIBC_2.39' not found\n",
        DEPENDENCIES + "libcuda.so.1 => not found\n",
    ],
)
def test_absent_cuda_or_loader_diagnostics_fail(output):
    with pytest.raises(ValueError):
        smoke.validate_dependencies(output)


def test_actual_image_binary_is_checked_without_running_it(config, monkeypatch):
    commands = []

    def capture(command):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return json.dumps(config)
        if command[1] == "create":
            return "container"
        if command[1] == "cp":
            executable(Path(command[-1]))
        if command[1] == "run":
            return DEPENDENCIES
        return ""

    monkeypatch.setattr(smoke, "capture", capture)
    smoke.validate(IMAGE, source_revision=SHA)
    assert [command[1] for command in commands] == ["image", "create", "cp", "rm", "run"]
    for command in (commands[1], commands[-1]):
        assert command[command.index("--pull") + 1] == "never"
        assert command[command.index("--network") + 1] == "none"
        assert IMAGE in command
    assert commands[-1][commands[-1].index("--entrypoint") + 1] == "/bin/sh"
    assert "--help" not in commands[-1]


def test_bad_extracted_binary_still_removes_container(config, monkeypatch):
    commands = []

    def capture(command):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return json.dumps(config)
        if command[1] == "create":
            return "container"
        if command[1] == "cp":
            Path(command[-1]).write_bytes(b"bad")
        return ""

    monkeypatch.setattr(smoke, "capture", capture)
    with pytest.raises(ValueError):
        smoke.validate(IMAGE, source_revision=SHA)
    assert commands[-1] == ["docker", "rm", "container"]
    assert not any(command[1] == "run" for command in commands)


@pytest.mark.parametrize("failure", [ValueError("bad dependencies"), subprocess.CalledProcessError(1, ["ldd"])])
def test_failed_cuda_worker_validation_prevents_release_export(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(docker_task, "run", Mock())
    validator = Mock(side_effect=failure)
    monkeypatch.setattr(smoke, "validate", validator)
    export = Mock()
    monkeypatch.setattr(docker_task, "export_image", export)
    monkeypatch.setattr(
        docker_task.sys,
        "argv",
        [
            "docker_task.py",
            "build-service",
            "--registry",
            "ghcr.io/superlinked",
            "--version",
            "0.8.2",
            "--service",
            "sie-server-rust",
            "--source-revision",
            SHA,
            "--run-id",
            "1234",
            "--archive-dir",
            str(tmp_path / "artifact"),
        ],
    )
    assert docker_task.main() == 1
    validator.assert_called_once_with("ghcr.io/superlinked/sie-server-rust:v0.8.2-cuda12-sm89", source_revision=SHA)
    export.assert_not_called()
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize(
    "service", ["sie-server-rust-cpu", "sie-server-sidecar", "sie-gateway", "sie-config", "sie-mcp"]
)
def test_other_services_keep_cli_smoke_before_export(service, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(docker_task, "run", commands.append)
    validator = Mock()
    monkeypatch.setattr(smoke, "validate", validator)
    monkeypatch.setattr(docker_task, "export_image", Mock())
    monkeypatch.setattr(
        docker_task.sys,
        "argv",
        [
            "docker_task.py",
            "build-service",
            "--registry",
            "local/test",
            "--version",
            "0.8.2",
            "--service",
            service,
            "--source-revision",
            SHA,
            "--run-id",
            "1234",
            "--archive-dir",
            str(tmp_path / "artifact"),
        ],
    )
    assert docker_task.main() == 0
    validator.assert_not_called()
    assert commands[-1] == [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        docker_task.singleton_image("local/test", "0.8.2", service),
        "--help",
    ]


def test_probe_nonzero_and_stderr_are_fatal(monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, ["ldd"])))
    with pytest.raises(subprocess.CalledProcessError):
        smoke.capture(["ldd"])
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess(["ldd"], 0, DEPENDENCIES, "loader warning")),
    )
    with pytest.raises(ValueError, match="loader warning"):
        smoke.capture(["ldd"])
