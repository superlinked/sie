from __future__ import annotations

import os
import signal
import subprocess
from unittest.mock import Mock

import pytest


def test_wait_ready_rejects_missing_log_handles(sie_server_process_factory, tmp_path) -> None:
    server = sie_server_process_factory(port=8090, models_dir=tmp_path)
    server._process = Mock()

    with pytest.raises(RuntimeError, match="log is not initialized"):
        server.wait_ready(timeout_s=0)


def test_server_url_matches_ipv4_bind_address(sie_server_process_factory, tmp_path) -> None:
    server = sie_server_process_factory(port=8090, models_dir=tmp_path)
    server._process = Mock()

    assert server.get_url() == "http://127.0.0.1:8090"


def test_stop_cleans_files_when_post_kill_wait_times_out(
    monkeypatch,
    sie_server_process_factory,
    tmp_path,
) -> None:
    server = sie_server_process_factory(port=8090, models_dir=tmp_path)
    process = Mock(pid=4321)
    process.wait.side_effect = (
        subprocess.TimeoutExpired(cmd="server", timeout=30),
        subprocess.TimeoutExpired(cmd="server", timeout=5),
    )
    server._process = process

    log_path = tmp_path / "server.log"
    log_path.write_text("startup output", encoding="utf-8")
    server._log_file = log_path.open("r+", encoding="utf-8")
    server._log_path = log_path
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("example==1", encoding="utf-8")
    server._requirements_path = requirements_path

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal)))

    server.stop()

    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert process.wait.call_count == 2
    assert server._process is None
    assert server._log_file is None
    assert server._log_path is None
    assert server._requirements_path is None
    assert not log_path.exists()
    assert not requirements_path.exists()
