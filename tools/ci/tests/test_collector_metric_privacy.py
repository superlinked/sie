"""Raw OTLP duplicate-key regression against the rendered pinned collector."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest
import yaml
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.metrics.v1.metrics_pb2 import AGGREGATION_TEMPORALITY_DELTA

from tools.ci.tests.test_collector_trace_privacy import SENTINEL, run, wait_for
from tools.ci.tests.test_helm_render import render_collector_config

pytestmark = pytest.mark.docker


def rendered_config():
    values = {
        "payloadStore.enabled": "false",
        "observability.otel.metrics.enabled": "true",
        "observability.otel.collector.install": "true",
        "observability.otel.collector.betterStack.enabled": "true",
        "observability.otel.collector.betterStack.endpoint": "https://example.invalid",
        "observability.otel.collector.betterStack.existingSecret": "synthetic",
        "observability.otel.resource.deploymentEnvironment": "dev",
        "observability.otel.resource.cloudRegion": "test-region",
    }
    documents = list(
        yaml.safe_load_all(
            run(
                "mise",
                "exec",
                "--",
                "helm",
                "template",
                "sie",
                "deploy/helm/sie-cluster",
                "--namespace",
                "sie",
                "--show-only",
                "templates/otel-collector.yaml",
                *(arg for key, value in values.items() for arg in ["--set", f"{key}={value}"]),
            )
        )
    )
    config = next(d for d in documents if d and d.get("kind") == "ConfigMap")
    deployment = next(d for d in documents if d and d.get("kind") == "Deployment")
    return yaml.safe_load(config["data"]["collector.yaml"]), deployment["spec"]["template"]["spec"]["containers"][0][
        "image"
    ]


def add(attributes, key, value):
    attr = attributes.add(key=key)
    if isinstance(value, str):
        attr.value.string_value = value
    elif isinstance(value, int):
        attr.value.int_value = value
    else:
        attr.value.kvlist_value.values.add(key="payload").value.string_value = SENTINEL


def payload(receiver):
    wire = ExportMetricsServiceRequest()
    service = "sie-gateway" if receiver == "gateway" else "sie-worker"
    prefix = "sie.gateway" if receiver == "gateway" else "sie.worker"
    for reverse_identity in [False, True]:
        rs = wire.resource_metrics.add()
        for key, value in {
            "service.name": service,
            "service.instance.id": "instance",
            "service.version": "version",
            "deployment.environment": "producer-env",
            "cloud.region": "producer-region",
        }.items():
            if key == "service.name" and reverse_identity:
                add(rs.resource.attributes, key, SENTINEL)
            add(rs.resource.attributes, key, value)
            add(rs.resource.attributes, key, SENTINEL)
            add(rs.resource.attributes, key, None)
        scope = rs.scope_metrics.add()
        specs = [(f"{prefix}.requests", "sum"), (f"{prefix}.request.duration", "histogram")]
        # Gateway-only diagnostics are also sent to the application receiver:
        # its allowlist must drop them rather than authorize their service.
        specs.extend(
            [
                ("sie.gateway.remote.fallbacks", "sum"),
                ("sie.gateway.remote.serving.duration", "gauge"),
            ]
        )
        for name, kind in specs:
            metric = scope.metrics.add(name=name)
            data = getattr(metric, kind)
            if kind != "gauge":
                data.aggregation_temporality = AGGREGATION_TEMPORALITY_DELTA
            if kind == "sum":
                data.is_monotonic = True
            for case in range(3):
                point = data.data_points.add(start_time_unix_nano=1_000, time_unix_nano=2_000 + case)
                if kind == "sum":
                    point.as_int = 10 + case
                elif kind == "gauge":
                    point.as_double = 10.0 + case
                else:
                    point.count = 2
                    point.sum = 0.5 + case
                    point.explicit_bounds.extend([1.0])
                    point.bucket_counts.extend([1, 1])
                add(point.attributes, "outcome", "committed" if name == "sie.gateway.remote.fallbacks" else "success")
                add(point.attributes, "outcome", None)
                if case == 0:
                    add(point.attributes, "operation", "encode")
                    add(point.attributes, "operation", SENTINEL)
                    add(point.attributes, "operation", None)
                elif case == 2:
                    add(point.attributes, "operation", None)
                    add(point.attributes, "operation", "encode")
                if receiver == "gateway":
                    add(point.attributes, "http.status_code", 200)
                    add(point.attributes, "http.status_code", None)
                add(point.attributes, "cloud.region", SENTINEL)
                if name.startswith("sie.gateway.remote."):
                    add(point.attributes, "model", "acme/model")
                    add(point.attributes, "model", SENTINEL)
                    add(point.attributes, "model", None)
                    add(point.attributes, "fallback.reason", "provisioning")
                    add(point.attributes, "fallback.reason", None)
    return wire.SerializeToString()


def batches(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def model_loading_payload():
    wire = ExportMetricsServiceRequest()
    resource = wire.resource_metrics.add()
    for key, value in {
        "service.name": "sie-gateway",
        "service.instance.id": "model-loading-test",
        "service.version": "test-version",
    }.items():
        add(resource.resource.attributes, key, value)
    metric = resource.scope_metrics.add().metrics.add(name="sie.gateway.model_loads_in_progress", unit="1")
    for profile, bundle, value in [("cpu", "default", 0), ("l4", "default", 1), ("cpu", "candle", 0)]:
        point = metric.gauge.data_points.add(time_unix_nano=time.time_ns(), as_double=value)
        for key, label in {"pool": "default", "machine_profile": profile, "bundle": bundle}.items():
            add(point.attributes, key, label)
    return wire.SerializeToString()


def require_cached_collector_image(image):
    try:
        run("docker", "image", "inspect", image)
    except subprocess.CalledProcessError:
        pytest.fail(
            f"Docker must be available with {image} already cached; this test never pulls images.", pytrace=False
        )


def model_loading_container_user(directory):
    owner = directory.stat()
    return f"--user={owner.st_uid}:{owner.st_gid}"


def model_loading_startup_diagnostics(container):
    results = []
    commands = (
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}",
            container,
        ],
        ["docker", "logs", "--tail", "20", container],
    )
    for command in commands:
        try:
            result = subprocess.run(  # noqa: S603 - fixed diagnostics for the exact test-owned container
                command, check=False, capture_output=True, text=True, errors="replace", timeout=5
            )
            results.append(f"{command[1]} exit={result.returncode}: {(result.stdout + result.stderr)[-4096:]}")
        except (OSError, subprocess.SubprocessError) as error:
            results.append(f"{command[1]} unavailable: {type(error).__name__}")
    return "Collector startup diagnostics:\n" + "\n".join(results)


def wait_for_model_loading_collector(container, health_check):
    try:
        wait_for(health_check)
    except AssertionError as error:
        error.add_note(model_loading_startup_diagnostics(container))
        raise


def cleanup_model_loading_collector(container, original_error):
    try:
        removed = subprocess.run(  # noqa: S603 - exact test-owned container
            ["docker", "rm", "--force", container],  # noqa: S607 - local Docker CLI
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        remaining = subprocess.run(  # noqa: S603 - exact test-owned container
            ["docker", "ps", "--all", "--filter", f"name=^/{container}$", "--format", "{{.Names}}"],  # noqa: S607 - local Docker CLI
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if remaining.stdout.strip():
            raise RuntimeError(f"test-owned collector was not removed: {container}")
        if original_error is None:
            removed.check_returncode()
    except Exception as cleanup_error:
        if original_error is not None:
            raise BaseExceptionGroup("collector test and cleanup failed", [original_error, cleanup_error]) from None
        raise


def test_model_loading_image_prerequisite_never_pulls(monkeypatch):
    calls = []

    def missing(*args):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setitem(require_cached_collector_image.__globals__, "run", missing)
    image = "otel/opentelemetry-collector-contrib:0.119.0"
    with pytest.raises(pytest.fail.Exception, match="already cached; this test never pulls"):
        require_cached_collector_image(image)
    assert calls == [("docker", "image", "inspect", image)]


def test_model_loading_container_uses_private_fixture_owner(tmp_path):
    tmp_path.chmod(0o700)
    owner = tmp_path.stat()
    assert model_loading_container_user(tmp_path) == f"--user={owner.st_uid}:{owner.st_gid}"
    assert owner.st_mode & 0o777 == 0o700


@pytest.mark.parametrize("diagnostics_fail", [False, True])
def test_model_loading_startup_failure_retains_bounded_diagnostics(monkeypatch, diagnostics_fail):
    original_error = AssertionError("collector not ready")
    calls = []

    def failed_wait(_predicate):
        raise original_error

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if diagnostics_fail:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        raw_output = b"exited exit=1 oom=false" if args[1] == "inspect" else b"x" * 8192 + b"permission denied\xff"
        output = raw_output.decode("utf-8", errors=kwargs.get("errors", "strict"))
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setitem(wait_for_model_loading_collector.__globals__, "wait_for", failed_wait)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AssertionError) as caught:
        wait_for_model_loading_collector("sie-model-loading-test", lambda: True)
    assert caught.value is original_error
    (note,) = caught.value.__notes__
    assert len(note) < 8300
    assert "TimeoutExpired" in note if diagnostics_fail else "exited exit=1 oom=false" in note
    if not diagnostics_fail:
        assert "permission denied" in note
        assert "\ufffd" in note
    assert [args[1] for args, _ in calls] == ["inspect", "logs"]
    assert calls[1][0] == ["docker", "logs", "--tail", "20", "sie-model-loading-test"]
    assert all(args[-1] == "sie-model-loading-test" and kwargs["timeout"] == 5 for args, kwargs in calls)
    assert all(kwargs["errors"] == "replace" for _, kwargs in calls)


@pytest.mark.parametrize("removed", [False, True])
@pytest.mark.parametrize("original_error", [None, RuntimeError("startup or test failure")])
def test_model_loading_cleanup_preserves_failures(monkeypatch, removed, original_error):
    container = "sie-model-loading-test"
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0 if removed or args[1] == "ps" else 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    expected = type(original_error) if original_error else None if removed else subprocess.CalledProcessError
    if expected:
        with pytest.raises(expected) as caught:
            try:
                if original_error:
                    raise original_error
            finally:
                cleanup_model_loading_collector(container, sys.exception())
        if original_error:
            assert caught.value is original_error
    else:
        cleanup_model_loading_collector(container, None)
    assert calls[0][0] == ["docker", "rm", "--force", container]
    assert calls[1][0] == ["docker", "ps", "--all", "--filter", f"name=^/{container}$", "--format", "{{.Names}}"]
    assert [kwargs["timeout"] for _, kwargs in calls] == [20, 10]


@pytest.mark.parametrize("verification_fails", [False, True])
def test_model_loading_cleanup_failure_reports_both_errors(monkeypatch, verification_fails):
    original_error = RuntimeError("original failure")

    def fake_run(args, **_kwargs):
        if args[1] == "ps" and verification_fails:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0, "sie-model-loading-test" if args[1] == "ps" else "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExceptionGroup) as caught:
        cleanup_model_loading_collector("sie-model-loading-test", original_error)
    assert caught.value.exceptions[0] is original_error
    assert isinstance(caught.value.exceptions[1], subprocess.CalledProcessError if verification_fails else RuntimeError)


def test_model_loading_gauge_preserves_prometheus_name_and_remote_unit(tmp_path):
    config, image = render_collector_config(tmp_path)
    assert image == "otel/opentelemetry-collector-contrib:0.119.0"
    require_cached_collector_image(image)
    exporter = config["exporters"]["prometheus"]
    assert exporter["add_metric_suffixes"] is True
    exporter["endpoint"] = "0.0.0.0:9464"
    config["exporters"] = {
        "prometheus": exporter,
        "file/remote": {"path": "/out/remote.json", "flush_interval": "100ms"},
    }
    pipelines = config["service"]["pipelines"]
    config["service"]["pipelines"] = {
        key: pipelines[key] for key in ["metrics/prometheus/gateway", "metrics/betterstack/gateway"]
    }
    pipelines["metrics/betterstack/gateway"]["exporters"] = ["file/remote"]
    config["receivers"]["otlp/gateway"]["protocols"]["http"] = {"endpoint": "0.0.0.0:4338"}
    config["processors"]["batch"]["timeout"] = "100ms"
    (tmp_path / "collector.yaml").write_text(yaml.safe_dump(config))
    container = f"sie-model-loading-{uuid.uuid4().hex}"
    try:
        run(
            "docker",
            "run",
            "--detach",
            "--pull=never",
            "--name",
            container,
            "--cpus=1",
            "--memory=256m",
            "--pids-limit=64",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            model_loading_container_user(tmp_path),
            "-p",
            "127.0.0.1::4338",
            "-p",
            "127.0.0.1::9464",
            "-p",
            "127.0.0.1::13133",
            "-v",
            f"{tmp_path}:/out",
            image,
            "--config=/out/collector.yaml",
        )
        ports = {port: run("docker", "port", container, f"{port}/tcp").split(":")[-1] for port in [4338, 9464, 13133]}

        def get(port, path=""):
            with urllib.request.urlopen(f"http://127.0.0.1:{ports[port]}{path}", timeout=2) as response:
                return response.read(1_048_577).decode()

        wait_for_model_loading_collector(container, lambda: get(13133))
        request = urllib.request.Request(
            f"http://127.0.0.1:{ports[4338]}/v1/metrics",
            data=model_loading_payload(),
            headers={"Content-Type": "application/x-protobuf"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed loopback HTTP
            assert response.status == 200
        remote = wait_for(lambda: batches(tmp_path / "remote.json"))
        text = wait_for(
            lambda: value if "sie_gateway_model_loads_in_progress" in (value := get(9464, "/metrics")) else None
        )
        assert len(text) <= 1_048_576
        assert "sie_gateway_model_loads_in_progress_ratio" not in text
        samples = re.findall(
            r"^sie_gateway_model_loads_in_progress\{([^}]+)\} ([01])(?:\.0)?(?: \d+)?$", text, re.MULTILINE
        )
        assert len(samples) == 3
        observed = {}
        for labels, value in samples:
            attributes = dict(re.findall(r'(\w+)="([^"\\]*)"', labels))
            assert attributes["pool"] == "default"
            assert attributes["producer_service"] == "sie-gateway"
            assert attributes["producer_instance"] == "model-loading-test"
            assert re.fullmatch(r"[0-9a-f-]{36}", attributes["collector_generation"])
            observed[attributes["machine_profile"], attributes["bundle"]] = int(value)
        assert observed == {("cpu", "default"): 0, ("l4", "default"): 1, ("cpu", "candle"): 0}
        metrics = [
            metric
            for batch in remote
            for resource in batch["resourceMetrics"]
            for scope in resource["scopeMetrics"]
            for metric in scope["metrics"]
        ]
        assert len(metrics) == 1
        assert metrics[0]["name"] == "sie.gateway.model_loads_in_progress"
        assert metrics[0]["unit"] == "1"
        assert sorted(point["asDouble"] for point in metrics[0]["gauge"]["dataPoints"]) == [0, 0, 1]
    finally:
        cleanup_model_loading_collector(container, sys.exception())


@pytest.mark.parametrize("receiver", ["gateway", "application"])
def test_remote_metric_maps_have_one_scalar_per_retained_key(tmp_path, receiver):
    config, image = rendered_config()
    assert image.endswith(":0.119.0")
    processors = config["processors"]
    shared = processors["transform/contract_metrics"]["metric_statements"]
    shared_points = next(group["statements"] for group in shared if group["context"] == "datapoint")
    allowed = {
        key
        for statement in shared_points
        for key in json.loads(statement.split("keep_keys(attributes, ", 1)[1].split(") where", 1)[0])
    }
    remote_groups = processors["transform/remote_metric_scalars"]["metric_statements"]
    remote_points = next(group["statements"] for group in remote_groups if group["context"] == "datapoint")
    rebuilt = {
        statement.split('set(attributes["', 1)[1].split('"]', 1)[0]
        for statement in remote_points
        if statement.startswith('set(attributes["')
    }
    assert rebuilt == allowed
    for name, pipeline in config["service"]["pipelines"].items():
        if name.startswith("metrics/prometheus/"):
            assert "transform/remote_metric_scalars" not in pipeline["processors"]
    config["exporters"] = {
        f"file/{name}": {"path": f"/out/{name}.json", "flush_interval": "100ms"} for name in ["local", "remote", "self"]
    }
    for name, pipeline in config["service"]["pipelines"].items():
        sink = "self" if name == "metrics/self" else "remote" if "betterstack" in name else "local"
        pipeline["exporters"] = [f"file/{sink}"]
    config["receivers"][f"otlp/{receiver}"]["protocols"]["http"] = {"endpoint": "0.0.0.0:4338"}
    config["processors"]["batch"]["timeout"] = "100ms"
    (tmp_path / "collector.yaml").write_text(yaml.safe_dump(config))
    container = run(
        "docker",
        "run",
        "-d",
        "--user",
        "0",
        "-p",
        "127.0.0.1::4338",
        "-p",
        "127.0.0.1::13133",
        "-v",
        f"{tmp_path}:/out",
        image,
        "--config=/out/collector.yaml",
    )
    try:
        ports = {p: run("docker", "port", container, f"{p}/tcp").split(":")[-1] for p in [4338, 13133]}
        wait_for(lambda: urllib.request.urlopen(f"http://127.0.0.1:{ports[13133]}", timeout=1).status == 200)
        request = urllib.request.Request(
            f"http://127.0.0.1:{ports[4338]}/v1/metrics",
            data=payload(receiver),
            headers={"Content-Type": "application/x-protobuf"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed loopback HTTP
            assert response.status == 200
        remote = wait_for(lambda: batches(tmp_path / "remote.json"))
        wait_for(lambda: batches(tmp_path / "local.json"))
        assert SENTINEL in (tmp_path / "local.json").read_text()
        assert SENTINEL not in (tmp_path / "remote.json").read_text()
        seen = []
        for batch in remote:
            for rs in batch["resourceMetrics"]:
                attrs = rs["resource"]["attributes"]
                assert len(attrs) == 5
                assert {a["key"]: a["value"] for a in attrs} == {
                    "service.name": {"stringValue": "sie-gateway" if receiver == "gateway" else "sie-worker"},
                    "service.instance.id": {"stringValue": "instance"},
                    "service.version": {"stringValue": "version"},
                    "deployment.environment": {"stringValue": "dev"},
                    "cloud.region": {"stringValue": "test-region"},
                }
                for scope in rs["scopeMetrics"]:
                    for metric in scope["metrics"]:
                        kind = next(kind for kind in ["sum", "histogram", "gauge"] if kind in metric)
                        data = metric[kind]
                        if kind != "gauge":
                            assert data["aggregationTemporality"] == 1
                        for point in data["dataPoints"]:
                            seen.append(point)
                            case = int(point["timeUnixNano"]) - 2_000
                            expected = {"outcome": {"stringValue": "success"}}
                            if case == 0:
                                expected["operation"] = {"stringValue": "encode"}
                            if receiver == "gateway":
                                expected["http.status_code"] = {"intValue": "200"}
                            if metric["name"].startswith("sie.gateway.remote."):
                                expected.pop("http.status_code", None)
                                expected["model"] = {"stringValue": "acme/model"}
                                if kind == "gauge":
                                    expected = {"model": expected["model"]}
                                else:
                                    expected["outcome"] = {"stringValue": "committed"}
                                    expected["fallback.reason"] = {"stringValue": "provisioning"}
                            assert len(point["attributes"]) == len(expected)
                            assert {a["key"]: a["value"] for a in point["attributes"]} == expected
                            assert int(point["startTimeUnixNano"]) == 1_000
                            if kind == "sum":
                                assert int(point["asInt"]) == 10 + case
                            elif kind == "gauge":
                                assert point["asDouble"] == 10.0 + case
                            else:
                                assert int(point["count"]) == 2
                                assert point["sum"] == 0.5 + case
                                assert point["bucketCounts"] == ["1", "1"]
                                assert point["explicitBounds"] == [1]
        assert len(seen) == (12 if receiver == "gateway" else 6)
    finally:
        try:
            run("docker", "stop", "--time", "10", container)
            logs = run("docker", "logs", container, include_stderr=True)
            assert "Error: " not in logs, logs
        finally:
            run("docker", "rm", "-f", container)
