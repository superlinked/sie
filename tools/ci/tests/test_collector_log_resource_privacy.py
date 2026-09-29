"""Raw protobuf resource validation through unchanged pinned log processors."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest
import yaml
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue

from tools.ci.tests.test_collector_lifecycle_privacy import render_config
from tools.ci.tests.test_collector_trace_privacy import SENTINEL, run, wait_for

pytestmark = pytest.mark.docker


def value(raw):
    result = AnyValue()
    if isinstance(raw, str):
        result.string_value = raw
    elif isinstance(raw, bool):
        result.bool_value = raw
    elif isinstance(raw, int):
        result.int_value = raw
    elif isinstance(raw, float):
        result.double_value = raw
    elif isinstance(raw, bytes):
        result.bytes_value = raw
    elif isinstance(raw, list):
        result.array_value.SetInParent()
        for item in raw:
            result.array_value.values.add().CopyFrom(value(item))
    elif isinstance(raw, dict):
        result.kvlist_value.SetInParent()
        for key, item in raw.items():
            result.kvlist_value.values.add(key=key).value.CopyFrom(value(item))
    else:
        assert raw is None
    return result


def resource_cases(application):
    baseline = {
        "service.name": "sie-worker" if application else "untrusted-gateway-name",
        "service.instance.id": "fixture-instance",
        "service.version": "fixture-version",
        "deployment.environment": "untrusted-environment",
        "cloud.region": "untrusted-region",
    }
    yield "valid", list(baseline.items())
    yield "missing-all", []
    malformed = [
        {"prompt": SENTINEL},
        [SENTINEL],
        7,
        1.5,
        True,
        SENTINEL.encode(),
        None,
        {},
        [],
        [{"nested": [SENTINEL]}],
    ]
    for key, valid in baseline.items():
        others = [(name, item) for name, item in baseline.items() if name != key]
        yield f"{key}-missing", others
        yield f"{key}-empty-string", [*others, (key, "")]
        for index, bad in enumerate(malformed):
            yield f"{key}-type-{index}", [*others, (key, bad)]
        # Validate the first value, reconstruct one attribute, and never fall
        # back to a later duplicate if the first value is malformed.
        yield f"{key}-valid-first", [*others, (key, valid), (key, {"prompt": SENTINEL})]
        yield f"{key}-invalid-first", [*others, (key, [SENTINEL]), (key, valid)]
        yield f"{key}-two-strings", [*others, (key, valid), (key, SENTINEL)]


def expected_resource(pairs, application):
    first = {}
    for key, item in pairs:
        first.setdefault(key, item)
    if application and first.get("service.name") != "sie-worker":
        return None
    expected = {
        "service.name": "sie-worker" if application else "sie-gateway",
        "deployment.environment": "dev",
        "cloud.region": "local",
    }
    for key in ("service.instance.id", "service.version"):
        if isinstance(first.get(key), str):
            expected[key] = first[key]
    return expected


def request_for(kind, pairs, trace_id, changes=None, extras=()):
    request = ExportLogsServiceRequest()
    resource = request.resource_logs.add(schema_url=SENTINEL)
    for key, raw in [*pairs, ("private.payload", {"prompt": SENTINEL})]:
        resource.resource.attributes.add(key=key).value.CopyFrom(value(raw))
    scope = resource.scope_logs.add(schema_url=SENTINEL)
    scope.scope.name = SENTINEL
    scope.scope.version = SENTINEL
    scope.scope.attributes.add(key="prompt").value.string_value = SENTINEL
    record = scope.log_records.add(
        time_unix_nano=time.time_ns(), trace_id=trace_id.to_bytes(16, "big"), span_id=(123).to_bytes(8, "big")
    )
    completion = kind.startswith("completion")
    attrs = {
        "event.name": "inference.request.completed" if completion else "inference.lifecycle.completed",
        "event.schema.version": "2" if kind == "completion-v2" else "1",
        "operation": "generate",
        "outcome": "success",
    }
    if completion:
        attrs["http.status_code"] = 200
        if kind == "completion-v2":
            attrs.update(model="other", machine_profile="other", duration_ms=12.0, admission_outcome="admitted")
    else:
        attrs.update(
            phase="worker_generation" if kind == "application" else "response_body",
            error_class="none",
            duration_ms=12.0,
        )
    attrs.update(changes or {})
    record.body.string_value = attrs["event.name"]
    for key, raw in [*attrs.items(), *extras]:
        record.attributes.add(key=key).value.CopyFrom(value(raw))
    return request.SerializeToString(), attrs


def read_records(path):
    if not path.exists():
        return []
    try:
        return [
            (resource, scope, record)
            for line in path.read_text().splitlines()
            for resource in json.loads(line).get("resourceLogs", [])
            for scope in resource.get("scopeLogs", [])
            for record in scope.get("logRecords", [])
        ]
    except json.JSONDecodeError:
        return []  # the file exporter may still be finishing its last line


def completion_cases(kind):
    if not kind.startswith("completion"):
        return
    wrong_strings = [{"prompt": SENTINEL}, [SENTINEL], 7, 1.5, True, SENTINEL.encode(), None, {}, []]
    for index, bad in enumerate([*wrong_strings, "200", 200.0, 99, 600, float("nan"), float("inf")]):
        yield f"status-type-{index}", {"http.status_code": bad}, (), False
    if kind == "completion-v1":
        for key in ("model", "machine_profile", "duration_ms", "admission_outcome"):
            for index, extra in enumerate([*wrong_strings, SENTINEL]):
                yield f"v1-extra-{key}-{index}", None, [(key, extra)], True
    else:
        for key in ("model", "machine_profile", "admission_outcome"):
            for index, bad in enumerate(wrong_strings):
                yield f"v2-{key}-type-{index}", {key: bad}, (), False
        nonnumeric = [{"prompt": SENTINEL}, [SENTINEL], "12", True, SENTINEL.encode(), None, {}, []]
        for index, bad in enumerate([*nonnumeric, -1, float("nan"), float("inf"), float("-inf")]):
            yield f"v2-duration-type-{index}", {"duration_ms": bad}, (), False
            yield f"v2-duration-invalid-first-{index}", {"duration_ms": bad}, [("duration_ms", 12.0)], False
        for duration in (0, 86400001.5, 1e300):
            yield f"v2-finite-duration-{duration}", {"duration_ms": duration}, (), True
        for key in ("model", "machine_profile", "duration_ms", "admission_outcome"):
            yield f"v2-{key}-valid-first", None, [(key, {"prompt": SENTINEL})], True


@pytest.mark.parametrize("kind", ["gateway", "application", "completion-v1", "completion-v2"])
def test_log_privacy_with_malformed_protobuf(tmp_path, kind):
    config = render_config(tmp_path, "http://unused.invalid")
    config["exporters"] = {"file/probe": {"path": "/out/received.json", "flush_interval": "100ms"}}
    for pipeline in config["service"]["pipelines"].values():
        pipeline["exporters"] = ["file/probe"]
    # Only local sockets, batching latency, and sink change. Use the real
    # production processors and their receiver-specific ordering unchanged.
    config["receivers"]["otlp/application"]["protocols"]["http"] = {"endpoint": "0.0.0.0:4338"}
    config["processors"]["batch"]["timeout"] = "100ms"
    (tmp_path / "collector.yaml").write_text(yaml.safe_dump(config))
    container = run(
        "docker",
        "run",
        "--detach",
        "--user",
        "0",
        "--publish",
        "127.0.0.1::4318",
        "--publish",
        "127.0.0.1::4338",
        "--publish",
        "127.0.0.1::13133",
        "--volume",
        f"{tmp_path}:/out",
        "otel/opentelemetry-collector-contrib:0.119.0",
        "--config=/out/collector.yaml",
    )
    try:
        ports = {
            port: int(run("docker", "port", container, f"{port}/tcp").rsplit(":", 1)[1]) for port in (4318, 4338, 13133)
        }

        def ready():
            with urllib.request.urlopen(f"http://127.0.0.1:{ports[13133]}", timeout=0.2) as response:  # noqa: S310 - fixed loopback
                return response.status == 200

        wait_for(ready)
        application = kind == "application"
        endpoint = f"http://127.0.0.1:{ports[4338 if application else 4318]}/v1/logs"
        expected = {}
        combined = ExportLogsServiceRequest()
        cases = [(case, pairs, None, (), True) for case, pairs in resource_cases(application)]
        baseline = cases[0][1]
        cases.extend(
            (case, baseline, changes, extras, retain) for case, changes, extras, retain in completion_cases(kind)
        )
        for trace_id, (case, pairs, changes, extras, retain) in enumerate(cases, start=1):
            data, attrs = request_for(kind, pairs, trace_id, changes, extras)
            resource = expected_resource(pairs, application)
            if resource is not None and retain:
                expected[f"{trace_id:032x}"] = (case, resource, attrs)
            combined.resource_logs.extend(ExportLogsServiceRequest.FromString(data).resource_logs)
        # Mix valid and malformed resources in one OTLP batch so invalid values
        # cannot suppress neighboring valid records or reuse their identity.
        request = urllib.request.Request(
            endpoint,
            data=combined.SerializeToString(),
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed loopback
            assert response.status == 200
        output = tmp_path / "received.json"
        wait_for(lambda: len(read_records(output)) >= len(expected))
        # Include late records that should have been rejected.
        time.sleep(1)
        records = read_records(output)
        assert SENTINEL not in output.read_text()
        assert len(records) == len(expected)
        seen = set()
        for resource, scope, record in records:
            trace_id = record["traceId"]
            assert trace_id not in seen
            seen.add(trace_id)
            case, wanted, attrs = expected[trace_id]
            actual = resource["resource"]["attributes"]
            assert len(actual) == len({item["key"] for item in actual}), case
            assert {item["key"]: item["value"] for item in actual} == {
                key: {"stringValue": item} for key, item in wanted.items()
            }, case
            assert not resource.get("schemaUrl")
            assert not scope.get("schemaUrl")
            assert not scope.get("scope", {}).get("attributes")
            assert not scope.get("scope", {}).get("version")
            assert record["spanId"] == f"{123:016x}"
            assert record["body"] == {"stringValue": attrs["event.name"]}
            actual_attrs = record["attributes"]
            assert len(actual_attrs) == len({item["key"] for item in actual_attrs}), case
            assert {item["key"]: item["value"] for item in actual_attrs} == {
                key: MessageToDict(value(raw)) for key, raw in attrs.items()
            }, case
        assert seen == set(expected)
    finally:
        run("docker", "rm", "--force", container)
