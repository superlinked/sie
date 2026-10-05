"""Real pinned collector privacy regression; run with pytest -m docker.

Fixtures are synthetic. The test renders the production Helm partial and sends
OTLP through both existing receiver boundaries, capturing the actual protobuf.
"""

from __future__ import annotations

# ruff: noqa: S603, S607 - fixed local Docker/Helm fixture commands
import gzip
import queue
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from opentelemetry._logs import LogRecord
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpExporter
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState, set_span_in_context

ROOT = Path(__file__).resolve().parents[3]


def render_config(directory: Path, endpoint: str) -> dict:
    chart = directory / "chart"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text("apiVersion: v2\nname: lifecycle-test\nversion: 0.0.0\n")
    shutil.copyfile(
        ROOT / "deploy/helm/sie-cluster/templates/_otel-collector-config.tpl", chart / "templates/_collector.tpl"
    )
    (chart / "templates/config.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test\ndata:\n  collector.yaml: |\n"
        '    {{- include "sie-cluster.otel.collectorConfig" .Values | nindent 4 }}\n'
    )
    values = {
        "metricsEnabled": False,
        "logsEnabled": True,
        "tracesEnabled": False,
        "prometheusEnabled": False,
        "collector": {},
        "traceEndpoint": "",
        "logEndpoint": endpoint,
        "betterStack": {"enabled": False},
        "deploymentEnvironment": "dev",
        "cloudRegion": "local",
        "localTraceExporters": [],
        "logExporters": ["otlphttp/logs"],
    }
    (chart / "values.yaml").write_text(yaml.safe_dump(values))
    rendered = subprocess.run(
        ["helm", "template", "test", str(chart)], check=True, capture_output=True, text=True
    ).stdout
    return yaml.safe_load(yaml.safe_load(rendered)["data"]["collector.yaml"])


@pytest.mark.docker
def test_lifecycle_allowlist_in_real_collector(tmp_path):
    received = queue.Queue()

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            received.put(data)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    sink = ThreadingHTTPServer(("0.0.0.0", 0), Sink)  # noqa: S104 - local Docker fixture
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    config = render_config(tmp_path, f"http://host.docker.internal:{sink.server_port}")
    config["processors"]["batch"]["timeout"] = "100ms"
    config_path = tmp_path / "collector.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config_path.chmod(0o644)
    name = f"sie-lifecycle-{uuid.uuid4().hex[:10]}"
    providers = []
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                name,
                "--add-host",
                "host.docker.internal:host-gateway",
                "--volume",
                f"{config_path}:/etc/otelcol/config.yaml:ro",
                "--publish",
                "127.0.0.1::4318",
                "--publish",
                "127.0.0.1::4327",
                "--publish",
                "127.0.0.1::13133",
                "otel/opentelemetry-collector-contrib:0.119.0",
                "--config=/etc/otelcol/config.yaml",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        ports = {}
        for internal in (4318, 4327, 13133):
            ports[internal] = int(
                subprocess.run(["docker", "port", name, f"{internal}/tcp"], check=True, capture_output=True, text=True)
                .stdout.strip()
                .rsplit(":", 1)[1]
            )
        # A published port accepts connections before the collector listens on
        # it. The health check answers 200 only once every receiver has started.
        deadline = time.monotonic() + 30
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{ports[13133]}", timeout=1):
                    break
            except OSError:
                assert time.monotonic() < deadline, subprocess.run(
                    ["docker", "logs", name], capture_output=True, text=True, check=False
                ).stderr
                time.sleep(0.1)

        def send(service, phase, *, application=False, changes=None, body="inference.lifecycle.completed"):
            exporter = (
                GrpcExporter(endpoint=f"http://127.0.0.1:{ports[4327]}", insecure=True)
                if application
                else HttpExporter(endpoint=f"http://127.0.0.1:{ports[4318]}/v1/logs")
            )
            provider = LoggerProvider(
                resource=Resource(
                    {"service.name": service, "service.instance.id": "fixture", "secret": "payload-secret"}
                ),
                shutdown_on_exit=False,
            )
            providers.append(provider)
            provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
            attrs = {
                "event.name": "inference.lifecycle.completed",
                "event.schema.version": "1",
                "operation": "generate",
                "phase": phase,
                "outcome": "error",
                "error_class": "worker",
                "duration_ms": 12.0,
                "first_token_ms": 2.0,
                "exception.message": "payload-secret",
            }
            attrs.update(changes or {})
            context = set_span_in_context(
                NonRecordingSpan(SpanContext(0x1234567890, 0x12345678, False, TraceFlags(1), TraceState()))
            )
            provider.get_logger(
                "payload-secret", version="payload-secret", attributes={"secret": "payload-secret"}
            ).emit(
                LogRecord(
                    timestamp=time.time_ns(),
                    context=context,
                    body=body,
                    severity_text="payload-secret",
                    attributes=attrs,
                )
            )

        # Three permitted service/phase pairs; hostile extra fields must vanish.
        send("untrusted-claimed-name", "response_body")
        send("sie-worker", "worker_generation", application=True)
        send("sie-dispatcher", "dispatch_attempt", application=True)
        # Valid elapsed times have no daily ceiling. Preserve both numeric
        # types, very large finite doubles, and zero at each receiver boundary.
        long_timings = [(86400001, 86400000), (86400001.5, 86400000.5), (1e300, 1e299), (0.0, 0.0)]
        for duration, first_token in long_timings:
            for service, phase, application in [
                ("sie-gateway", "response_body", False),
                ("sie-worker", "worker_generation", True),
            ]:
                send(
                    service,
                    phase,
                    application=application,
                    changes={"duration_ms": duration, "first_token_ms": first_token},
                )
        # Malformed protobuf can repeat a key: filters read the first value,
        # while keep_keys alone retains both. Rebuild the validated scalars.
        duplicate = ExportLogsServiceRequest()
        resource = duplicate.resource_logs.add()
        for key, value in [
            ("service.name", "sie-worker"),
            ("service.version", "fixture"),
            ("service.version", "payload-secret"),
        ]:
            attr = resource.resource.attributes.add(key=key)
            attr.value.string_value = value
        scope = resource.scope_logs.add()
        record = scope.log_records.add(
            time_unix_nano=time.time_ns(),
            trace_id=(0x1234567890).to_bytes(16, "big"),
            span_id=(0x12345678).to_bytes(8, "big"),
        )
        record.body.string_value = "inference.lifecycle.completed"
        for key, value in {
            "event.name": "inference.lifecycle.completed",
            "event.schema.version": "1",
            "operation": "generate",
            "phase": "response_body",
            "outcome": "error",
            "error_class": "worker",
            "duration_ms": 12.0,
            "first_token_ms": 2.0,
        }.items():
            attr = record.attributes.add(key=key)
            if isinstance(value, float):
                attr.value.double_value = value
            else:
                attr.value.string_value = value
            record.attributes.add(key=key).value.string_value = "payload-secret"
        raw_request = urllib.request.Request(
            f"http://127.0.0.1:{ports[4318]}/v1/logs",
            data=duplicate.SerializeToString(),
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        with urllib.request.urlopen(raw_request, timeout=5) as response:  # noqa: S310 - fixed loopback HTTP
            assert response.status == 200

        for version in ("1", "2"):
            completion = ExportLogsServiceRequest()
            completion.CopyFrom(duplicate)
            resource = completion.resource_logs[0]
            record = resource.scope_logs[0].log_records[0]
            record.body.string_value = "inference.request.completed"
            del record.attributes[:]
            attrs = {
                "event.name": "inference.request.completed",
                "event.schema.version": version,
                "operation": "generate",
                "outcome": "success",
                "http.status_code": 200,
            }
            if version == "2":
                attrs.update(model="other", machine_profile="other", duration_ms=12.0, admission_outcome="admitted")
            for key, value in attrs.items():
                attr = record.attributes.add(key=key)
                if isinstance(value, float):
                    attr.value.double_value = value
                elif isinstance(value, int):
                    attr.value.int_value = value
                else:
                    attr.value.string_value = value
                record.attributes.add(key=key).value.string_value = "payload-secret"
            request = urllib.request.Request(
                f"http://127.0.0.1:{ports[4318]}/v1/logs",
                data=completion.SerializeToString(),
                headers={"Content-Type": "application/x-protobuf"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed loopback HTTP
                assert response.status == 200

        # Invalid enum/type/body/service/phase fixtures must vanish completely.
        for changes in (
            {"error_class": "payload-secret"},
            {"outcome": "payload-secret"},
            {"operation": "payload-secret"},
            {"duration_ms": "payload-secret"},
            {"duration_ms": -1},
            {"duration_ms": float("nan")},
            {"duration_ms": float("inf")},
            {"duration_ms": float("-inf")},
            {"first_token_ms": float("nan")},
            {"first_token_ms": float("inf")},
            {"first_token_ms": float("-inf")},
            {"first_token_ms": "payload-secret"},
            {"first_token_ms": 100},
            {"event.schema.version": "2"},
        ):
            send("sie-worker", "worker_generation", application=True, changes=changes)
        send("sie-worker", "worker_generation", application=True, body="payload-secret")
        send("sie-worker", "response_body", application=True)
        send("sie-gateway", "request", application=True)
        send("unknown", "worker_generation", application=True)
        send("sie-worker", "worker_generation")
        for provider in providers:
            provider.shutdown()
        providers.clear()
        records = []
        completions = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                data = received.get(timeout=min(0.2, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            request = ExportLogsServiceRequest.FromString(data)
            assert b"payload-secret" not in request.SerializeToString()
            for resource_logs in request.resource_logs:
                resource = {item.key: item.value.string_value for item in resource_logs.resource.attributes}
                assert resource["deployment.environment"] == "dev"
                assert resource["cloud.region"] == "local"
                for scope in resource_logs.scope_logs:
                    assert scope.scope.name in ("", "sie-gateway.request-completion")
                    assert not scope.scope.version
                    assert not scope.scope.attributes
                    for record in scope.log_records:
                        assert record.trace_id == (0x1234567890).to_bytes(16, "big")
                        assert record.span_id == (0x12345678).to_bytes(8, "big")
                        attrs = {a.key: a.value for a in record.attributes}
                        assert len(attrs) == len(record.attributes)
                        if record.body.string_value == "inference.request.completed":
                            completions.append(attrs)
                        else:
                            records.append((resource["service.name"], attrs))
            if len(records) + len(completions) >= 6 + 2 * len(long_timings):
                deadline = min(deadline, time.monotonic() + 1)
        assert {service for service, _attrs in records} == {"sie-gateway", "sie-worker", "sie-dispatcher"}
        assert len(records) == 4 + 2 * len(long_timings)
        timings = []
        for _, attrs in records:
            values = [attrs[key] for key in ("duration_ms", "first_token_ms")]
            assert all(value.WhichOneof("value") in {"double_value", "int_value"} for value in values)
            timings.append(
                tuple(value.double_value if value.HasField("double_value") else value.int_value for value in values)
            )
        for timing in long_timings:
            assert timings.count(timing) == 2
        assert len(completions) == 2
        assert {attrs["event.schema.version"].string_value for attrs in completions} == {"1", "2"}
        allowed = {
            "event.name",
            "event.schema.version",
            "phase",
            "operation",
            "outcome",
            "error_class",
            "duration_ms",
            "first_token_ms",
        }
        assert all(set(attrs) == allowed for _, attrs in records)
    finally:
        for provider in providers:
            provider.shutdown()
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
        sink.shutdown()
        sink.server_close()
        thread.join()
