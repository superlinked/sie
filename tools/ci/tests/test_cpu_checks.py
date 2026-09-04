from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from tools.ci import cpu_stack_smoke, live_sdk, rust_tests


def test_rust_fails_without_nats(monkeypatch):
    monkeypatch.setattr(rust_tests.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="nats-server is required"):
        rust_tests.main()


def test_rust_fails_when_broker_exits():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="NATS exited"):
        rust_tests.wait_for_jetstream("http://127.0.0.1:1/jsz", process)


def test_rust_runs_real_nats_opt_in_and_cloud_feature_without_benchmarks(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIE_RUN_TELEMETRY_BENCHMARK", "1")
    monkeypatch.setattr(rust_tests.shutil, "which", lambda _: "/test/bin")
    monkeypatch.setattr(rust_tests, "wait_for_jetstream", Mock())
    process = Mock()
    monkeypatch.setattr(rust_tests.subprocess, "Popen", Mock(return_value=process))
    run = Mock()
    monkeypatch.setattr(rust_tests.subprocess, "run", run)
    rust_tests.main()
    commands = [call.args[0] for call in run.call_args_list]
    assert commands == [
        ["cargo", "test", "--locked", "--workspace"],
        ["cargo", "test", "--locked", "-p", "sie-server-sidecar", "--features", "cloud-storage"],
        ["cargo", "test", "--locked", "--manifest-path", "packages/sie_server_rust/Cargo.toml"],
    ]
    for call in run.call_args_list:
        assert call.kwargs["env"]["NATS_URL"].startswith("nats://127.0.0.1:")
        assert call.kwargs["env"]["SIE_RUN_NATS_PUBLISHER_TEST"] == "1"
        assert not any("BENCHMARK" in key for key in call.kwargs["env"])
    process.terminate.assert_called_once()


def test_live_sdk_fails_when_server_exits():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="SIE server exited"):
        live_sdk.wait_for_api("http://127.0.0.1:1", process)


@pytest.mark.parametrize("endpoint", ["tcp://remote.example:2375", "ssh://remote.example"])
def test_cpu_smoke_rejects_remote_docker(monkeypatch, endpoint):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(
        cpu_stack_smoke, "docker", lambda *args: json.dumps([{"Endpoints": {"docker": {"Host": endpoint}}}])
    )
    with pytest.raises(RuntimeError, match="local Unix"):
        cpu_stack_smoke.require_local_docker()


def test_docker_host_override_cannot_hide_remote_endpoint(monkeypatch):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2375")
    monkeypatch.setattr(
        cpu_stack_smoke,
        "docker",
        lambda *args: json.dumps([{"Endpoints": {"docker": {"Host": "unix:///var/run/docker.sock"}}}]),
    )
    with pytest.raises(RuntimeError, match="local Unix"):
        cpu_stack_smoke.require_local_docker()


def test_cpu_builds_all_six_images_without_publish(monkeypatch):
    run = Mock()
    monkeypatch.setattr(cpu_stack_smoke.subprocess, "run", run)
    cpu_stack_smoke.build_images("local/test", "a" * 40)
    assert len(run.call_args_list) == 6
    commands = [call.args[0] for call in run.call_args_list]
    assert all(command[:4] == ["mise", "run", "docker", "--"] for command in commands)
    assert all("--push" not in command for command in commands)
    assert [command[command.index("--service") + 1] for command in commands[1:]] == list(cpu_stack_smoke.SERVICES)
