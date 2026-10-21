"""Composite bundle lanes in the worker StatefulSet.

The chart reads a bundle's ``components`` from the staged bundle files (the
same files the gateway ConfigMap carries), so these renders stage them under
the chart's config lock and remove them afterwards.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from tools.mise_tasks import helm

ROOT = Path(__file__).resolve().parents[3]
RESOURCES = {"requests": {"cpu": "8", "memory": "64Gi"}, "limits": {"memory": "96Gi"}}


@pytest.fixture
def staged_chart() -> Iterator[None]:
    with helm._helm_config_sync_lock():
        helm._sync_configs_to_helm()
        try:
            yield
        finally:
            helm._cleanup_helm_configs()


def _render(tmp_path: Path, values: dict) -> subprocess.CompletedProcess[str]:
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values), encoding="utf-8")
    return subprocess.run(  # noqa: S603 - fixed local chart render
        [  # noqa: S607 - mise-managed Helm
            "mise",
            "exec",
            "--",
            "helm",
            "template",
            "sie",
            str(helm.CHART_DIR),
            "--namespace",
            "sie",
            *helm.validation_args(["-f", str(values_file), "--show-only", "templates/worker-statefulset.yaml"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _pool(bundle: str, gpus: int = 1, **gpu: object) -> dict:
    return {
        "enabled": True,
        "gpuType": "h100",
        "gpu": {"count": gpus, "product": "NVIDIA-H100-80GB-HBM3", **gpu},
        "resources": RESOURCES,
        "bundles": {bundle: {"minReplicas": 0, "maxReplicas": 1}},
    }


def _statefulset(tmp_path: Path, pool: dict) -> dict:
    result = _render(tmp_path, {"workers": {"pools": {"h100": pool}}})
    assert result.returncode == 0, result.stderr
    (statefulset,) = [doc for doc in yaml.safe_load_all(result.stdout) if doc and doc["kind"] == "StatefulSet"]
    return statefulset


def _containers(statefulset: dict) -> dict[str, dict]:
    return {container["name"]: container for container in statefulset["spec"]["template"]["spec"]["containers"]}


def _env(container: dict) -> dict[str, str]:
    return {entry["name"]: entry.get("value") for entry in container["env"]}


@pytest.mark.usefixtures("staged_chart")
def test_composite_lane_runs_every_component_on_each_gpu(tmp_path: Path) -> None:
    containers = _containers(_statefulset(tmp_path, _pool("default-sglang")))

    worker = containers["worker"]
    assert worker["image"].endswith("-cuda12-default-sglang")
    assert "--bundle=default-sglang" in worker["args"]
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert _env(worker)["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc.sock"

    sidecar = _env(containers["worker-sidecar"])
    assert sidecar["SIE_IPC_SOCKET_PATHS"] == "/var/run/sie/ipc.default.sock,/var/run/sie/ipc.sglang.sock"
    assert sidecar["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc.default.sock"
    assert sidecar["SIE_IPC_SOCKETS_PER_SLOT"] == "2"
    assert sidecar["SIE_GPU_COUNT"] == "1"


@pytest.mark.usefixtures("staged_chart")
def test_multi_gpu_composite_lane_groups_component_sockets_by_gpu(tmp_path: Path) -> None:
    containers = _containers(_statefulset(tmp_path, _pool("default-sglang", gpus=2)))

    assert {name for name in containers if name.startswith("worker-")} == {"worker-0", "worker-1", "worker-sidecar"}
    sidecar = _env(containers["worker-sidecar"])
    assert sidecar["SIE_IPC_SOCKET_PATHS"].split(",") == [
        "/var/run/sie/ipc-0.default.sock",
        "/var/run/sie/ipc-0.sglang.sock",
        "/var/run/sie/ipc-1.default.sock",
        "/var/run/sie/ipc-1.sglang.sock",
    ]
    assert sidecar["SIE_IPC_SOCKETS_PER_SLOT"] == "2"
    assert sidecar["SIE_GPU_COUNT"] == "2"


@pytest.mark.usefixtures("staged_chart")
def test_ordinary_lane_is_unchanged(tmp_path: Path) -> None:
    sidecar = _env(_containers(_statefulset(tmp_path, _pool("default")))["worker-sidecar"])

    assert sidecar["SIE_IPC_SOCKET_PATH"] == "/var/run/sie/ipc.sock"
    assert "SIE_IPC_SOCKET_PATHS" not in sidecar
    assert "SIE_IPC_SOCKETS_PER_SLOT" not in sidecar


@pytest.mark.usefixtures("staged_chart")
def test_composite_lane_refuses_a_device_group(tmp_path: Path) -> None:
    result = _render(tmp_path, {"workers": {"pools": {"h100": _pool("default-sglang", gpus=2, deviceGroup=True)}}})

    assert result.returncode != 0
    assert "cannot use gpu.deviceGroup" in result.stderr
