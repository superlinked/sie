"""Pinned collector runtime proof; no credentials, vendor traffic or deployment.

Run after `mise run helm -- dependencies` with pytest -m docker. Processor
lists and statements come from the real rendered chart, not a test facsimile.
Only receiver bindings, self-scrape interval and exporter destinations change.
"""

from __future__ import annotations

import copy
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
import yaml
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, Status, StatusCode, get_current_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from sie_server.observability.batch_fanin import BatchFanInSpanProcessor

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.docker
SENTINEL = "private-payload-sentinel"


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=120).strip()  # noqa: S603 - fixed local test commands


def rendered_config():
    rendered = run(
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
        "--set",
        "payloadStore.enabled=false",
        "--set",
        "observability.tracing.enabled=true",
        "--set",
        "observability.otel.collector.install=true",
        "--set",
        "observability.otel.collector.traces.endpoint=local:4317",
        "--set",
        "observability.otel.collector.betterStack.enabled=true",
        "--set",
        "observability.otel.collector.betterStack.endpoint=https://example.invalid",
        "--set",
        "observability.otel.collector.betterStack.existingSecret=synthetic",
        "--set",
        "observability.otel.resource.deploymentEnvironment=dev",
        "--set",
        "observability.otel.resource.cloudRegion=test-region",
    )
    documents = list(yaml.safe_load_all(rendered))
    config = next(d for d in documents if d and d.get("kind") == "ConfigMap")
    deployment = next(d for d in documents if d and d.get("kind") == "Deployment")
    image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
    return yaml.safe_load(config["data"]["collector.yaml"]), image


def context(trace: int, parent: int, *, sampled=True):
    return TraceContextTextMapPropagator().extract(
        {
            "traceparent": f"00-{trace:032x}-{parent:016x}-{int(sampled):02x}",
            "tracestate": f"vendor={SENTINEL}",
        }
    )


def sample_spans() -> list[ReadableSpan]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "sie-worker", "customer": SENTINEL}))
    provider.add_span_processor(BatchFanInSpanProcessor(SimpleSpanProcessor(exporter)))
    tracer = provider.get_tracer(SENTINEL, attributes={"customer": SENTINEL})
    # Distinct traces, same-trace distinct parents, and duplicate parents.
    links = [
        Link(get_current_span(context(t, s)).get_span_context(), {"payload": SENTINEL})
        for t, s in [(3, 4), (1, 5), (3, 4)]
    ]
    links += [Link(get_current_span(context(7, 8, sampled=False)).get_span_context())]
    with tracer.start_as_current_span("worker.run_batch", context=context(1, 2), links=links) as span:
        span.set_attribute("customer", SENTINEL)
        span.add_event(SENTINEL, {"exception.message": SENTINEL})
        span.set_status(Status(StatusCode.ERROR, SENTINEL))
    # An ordinary one-parent batch survives unchanged structurally.
    with tracer.start_as_current_span("sidecar.dispatch", context=context(9, 10)):
        pass
    # Malformed W3C context becomes a root; never invent a contributing parent.
    invalid = TraceContextTextMapPropagator().extract({"traceparent": "not-a-context"})
    with tracer.start_as_current_span("worker.run_batch", context=invalid):
        pass
    provider.force_flush()
    spans = list(exporter.get_finished_spans())
    provider.shutdown()
    return spans


def read_spans(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    try:
        return [
            span
            for line in lines
            for rs in json.loads(line).get("resourceSpans", [])
            for scope in rs["scopeSpans"]
            for span in scope["spans"]
        ]
    except json.JSONDecodeError:
        return []  # the file exporter may be finishing its last line


def wait_for(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError("collector did not become ready or flush the expected output")


@pytest.mark.parametrize("receiver", ["gateway", "application"])
def test_batch_fanin_survives_remote_privacy_with_local_links_intact(tmp_path, receiver):
    config, image = rendered_config()
    assert image.endswith(":0.119.0"), "revalidate the runtime proof when changing the pin"
    # Preserve production processor ordering; redirect only outputs and sockets.
    config["exporters"] = {
        f"file/{name}": {"path": f"/out/{name}.json", "flush_interval": "100ms"}
        for name in ["local", "remote", "self", "unsafe"]
    }
    for name, pipeline in config["service"]["pipelines"].items():
        sink = "self" if name.startswith("metrics/") else "remote" if "betterstack" in name else "local"
        pipeline["exporters"] = [f"file/{sink}"]
    # Prove that merely deleting the linked-span guard would leak on this pin.
    unsafe = copy.deepcopy(config["service"]["pipelines"][f"traces/betterstack/{receiver}"])
    unsafe["processors"].remove("filter/remote_linked_spans")
    unsafe["exporters"] = ["file/unsafe"]
    config["service"]["pipelines"]["traces/unsafe_reproduction"] = unsafe
    config["receivers"][f"otlp/{receiver}"]["protocols"]["http"] = {"endpoint": "0.0.0.0:4338"}
    telemetry_reader = config["service"]["telemetry"]["metrics"]["readers"][0]["pull"]["exporter"]["prometheus"]
    telemetry_reader["host"] = "0.0.0.0"  # noqa: S104 - container-only; published on loopback
    config["processors"]["batch"]["timeout"] = "100ms"
    config["receivers"]["prometheus/self"]["config"]["scrape_configs"][0]["scrape_interval"] = "100ms"
    (tmp_path / "collector.yaml").write_text(yaml.safe_dump(config))
    container = run(
        "docker",
        "run",
        "--rm",
        "-d",
        "--user",
        "0",
        "-p",
        "127.0.0.1::4338",
        "-p",
        "127.0.0.1::8888",
        "-p",
        "127.0.0.1::13133",
        "-v",
        f"{tmp_path}:/out",
        image,
        "--config=/out/collector.yaml",
    )
    try:
        ports = {port: run("docker", "port", container, f"{port}/tcp").split(":")[-1] for port in [4338, 8888, 13133]}
        wait_for(lambda: urllib.request.urlopen(f"http://127.0.0.1:{ports[13133]}", timeout=1).status == 200)
        source = sample_spans()
        wire = encode_spans(source)
        # Inject fields the SDK normally validates: zero IDs and adversarial
        # link attributes/tracestate still must never cross the remote branch.
        linked = next(s for rs in wire.resource_spans for sc in rs.scope_spans for s in sc.spans if s.links)
        invalid_link = linked.links.add(trace_id=bytes(16), span_id=bytes(8), trace_state=SENTINEL)
        invalid_link.attributes.add(key="customer").value.string_value = SENTINEL
        payload = wire.SerializeToString()
        request = urllib.request.Request(
            f"http://127.0.0.1:{ports[4338]}/v1/traces",
            data=payload,
            headers={"Content-Type": "application/x-protobuf"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed loopback HTTP
            assert response.status == 200
        local = wait_for(lambda: s if len(s := read_spans(tmp_path / "local.json")) == len(source) else None)
        remote = wait_for(lambda: s if len(s := read_spans(tmp_path / "remote.json")) == len(source) - 1 else None)
        unsafe_spans = wait_for(lambda: read_spans(tmp_path / "unsafe.json"))
        assert len(unsafe_spans) == len(source)
        assert any(s.get("links") for s in unsafe_spans), "0.119 no-op set(links, []) reproduction changed"
        assert SENTINEL in (tmp_path / "unsafe.json").read_text()
        assert SENTINEL in (tmp_path / "local.json").read_text()
        assert SENTINEL not in (tmp_path / "remote.json").read_text()
        assert not any(
            s.get("links")
            or s.get("events")
            or s.get("attributes")
            or s.get("traceState")
            or s.get("status", {}).get("message")
            for s in remote
        )
        assert len([s for s in remote if s["name"] == "worker.run_batch.request"]) == 3
        assert {s["spanId"] for s in remote} == {s["spanId"] for s in local if not s.get("links")}
        local_by_id = {s["spanId"]: s for s in local}
        for span in remote:
            before = local_by_id[span["spanId"]]
            for key in ["traceId", "spanId", "parentSpanId", "startTimeUnixNano", "endTimeUnixNano", "kind", "flags"]:
                assert span.get(key) == before.get(key)
        # Observe filter loss separately from transport/export failure, using
        # the pinned collector's own numeric counter (no request dimensions).
        with urllib.request.urlopen(f"http://127.0.0.1:{ports[8888]}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        filtered = [
            line
            for line in metrics.splitlines()
            if line.startswith("otelcol_processor_filter_spans_filtered{")
            and 'filter="filter/remote_linked_spans"' in line
        ]
        assert filtered, metrics
        assert sum(float(line.rsplit(" ", 1)[1]) for line in filtered) == 1

        def self_points():
            path = tmp_path / "self.json"
            if not path.exists():
                return []
            points = [
                p
                for line in path.read_text().splitlines()
                for rs in json.loads(line).get("resourceMetrics", [])
                for scope in rs["scopeMetrics"]
                for metric in scope["metrics"]
                if metric["name"] == "otelcol_processor_filter_spans_filtered"
                for p in metric["sum"]["dataPoints"]
            ]
            return points if any(int(p.get("asInt", p.get("asDouble", 0))) == 1 for p in points) else []

        points = wait_for(self_points)
        for point in points:
            assert point["attributes"] == [{"key": "filter", "value": {"stringValue": "filter/remote_linked_spans"}}]
    finally:
        logs = run("docker", "logs", container)
        run("docker", "stop", container)
        assert "Error: " not in logs, logs
