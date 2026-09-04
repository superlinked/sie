from __future__ import annotations

import subprocess

from tools.mise_tasks.common import device


def test_nvidia_detection_has_a_finite_timeout(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(device.platform, "system", lambda: "Linux")

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(device.subprocess, "run", run)
    assert device.detect_gpu() == "cuda"
    assert calls[0][1]["timeout"] == 5


def test_nvidia_detection_timeout_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(device.platform, "system", lambda: "Linux")

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(device.subprocess, "run", run)
    assert device.detect_gpu() is None
    assert device.default_device() == "cpu"
