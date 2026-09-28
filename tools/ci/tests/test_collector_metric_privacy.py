"""Raw OTLP duplicate-key regression against the rendered pinned collector."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest
import yaml
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.metrics.v1.metrics_pb2 import AGGREGATION_TEMPORALITY_DELTA

from tools.ci.tests.test_collector_trace_privacy import SENTINEL, run, wait_for

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
        for suffix, kind in [("requests", "sum"), ("request.duration", "histogram")]:
            metric = scope.metrics.add(name=f"{prefix}.{suffix}")
            data = getattr(metric, kind)
            data.aggregation_temporality = AGGREGATION_TEMPORALITY_DELTA
            if kind == "sum":
                data.is_monotonic = True
            for case in range(3):
                point = data.data_points.add(start_time_unix_nano=1_000, time_unix_nano=2_000 + case)
                if kind == "sum":
                    point.as_int = 10 + case
                else:
                    point.count = 2
                    point.sum = 0.5 + case
                    point.explicit_bounds.extend([1.0])
                    point.bucket_counts.extend([1, 1])
                add(point.attributes, "outcome", "success")
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
    return wire.SerializeToString()


def batches(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


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
                        kind = "sum" if "sum" in metric else "histogram"
                        data = metric[kind]
                        assert data["aggregationTemporality"] == 1
                        for point in data["dataPoints"]:
                            seen.append(point)
                            case = int(point["timeUnixNano"]) - 2_000
                            expected = {"outcome": {"stringValue": "success"}}
                            if case == 0:
                                expected["operation"] = {"stringValue": "encode"}
                            if receiver == "gateway":
                                expected["http.status_code"] = {"intValue": "200"}
                            assert len(point["attributes"]) == len(expected)
                            assert {a["key"]: a["value"] for a in point["attributes"]} == expected
                            assert int(point["startTimeUnixNano"]) == 1_000
                            if kind == "sum":
                                assert int(point["asInt"]) == 10 + case
                            else:
                                assert int(point["count"]) == 2
                                assert point["sum"] == 0.5 + case
                                assert point["bucketCounts"] == ["1", "1"]
                                assert point["explicitBounds"] == [1]
        assert len(seen) == 6
    finally:
        try:
            run("docker", "stop", "--time", "10", container)
            logs = run("docker", "logs", container, include_stderr=True)
            assert "Error: " not in logs, logs
        finally:
            run("docker", "rm", "-f", container)
