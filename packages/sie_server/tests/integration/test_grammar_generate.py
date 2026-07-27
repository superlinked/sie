"""Grammar integration tests: SIE-native ``/v1/generate/{model}`` with
``grammar``.

Skipped unless ``SIE_GATEWAY_URL`` is set. Mirrors the acceptance
criteria for structured outputs:

* §5.7 — moderate JSON Schema returns JSON conforming to the schema.
* §5.9 — two identical-schema requests show exactly 1 compile + 1 hit
  in the collector's Prometheus compatibility output.
* §5.10 — a pathologically nested schema returns 400 ``invalid_request``
  *before* any compile metric ticks.

The chat-side OpenAI SDK acceptance test (§5.8) lives in
``test_chat_completions.py`` next to the chat-completions fixtures.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass

import httpx
import pytest

pytestmark = pytest.mark.integration

_GATEWAY_URL_ENV = "SIE_GATEWAY_URL"
_COLLECTOR_METRICS_URL_ENV = "SIE_COLLECTOR_METRICS_URL"
_GEN_MODEL_ENV = "SIE_GEN_MODEL"

_PROMETHEUS_EXPORTER_LABELS = {
    "job",
    "instance",
    "otel_scope_name",
    "otel_scope_version",
    "otel_scope_schema_url",
}
_PRODUCER_LABELS = {
    "producer_service",
    "producer_instance",
}
_GRAMMAR_BASE_LABELS = {
    "grammar_backend",
    "grammar",
    "backend",
    "lane",
    "model",
    "profile",
}
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)


@dataclass(frozen=True)
class _PrometheusSample:
    labels: dict[str, str]
    value: float


def _gateway_url() -> str:
    url = os.environ.get(_GATEWAY_URL_ENV)
    if not url:
        pytest.skip(f"set {_GATEWAY_URL_ENV} to run grammar generate integration tests")
    return url.rstrip("/")


def _gen_model() -> str:
    # SIE-safe path form (double underscore separator, matching the
    # gateway's /v1/generate/{model} contract).
    return os.environ.get(_GEN_MODEL_ENV, "Qwen__Qwen3-4B-Instruct-2507")


def _collector_metrics_url() -> str | None:
    return os.environ.get(_COLLECTOR_METRICS_URL_ENV)


def _telemetry_model() -> str:
    """Translate the gateway's path-safe model spelling back to the catalog ID."""
    return _gen_model().replace("__", "/")


@pytest.fixture(scope="module")
def gateway_url() -> str:
    return _gateway_url()


# -----------------------------------------------------------------------------
# §5.7 — JSON Schema round-trip
# -----------------------------------------------------------------------------


def test_generate_with_json_schema_returns_conforming_json(gateway_url: str) -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "year"],
        "additionalProperties": False,
    }
    body = {
        "prompt": ("Return JSON describing a fictional book with a title, a year, and 1-3 tags."),
        "max_new_tokens": 128,
        "grammar": {"json_schema": schema},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    text = payload.get("text") or payload.get("choices", [{}])[0].get("text", "")
    assert text, f"unexpected response shape: {payload!r}"
    parsed = json.loads(text)
    assert "title" in parsed
    assert isinstance(parsed["title"], str)
    assert "year" in parsed
    assert isinstance(parsed["year"], int)


# -----------------------------------------------------------------------------
# §5.10 — safety cap fast-fails before compile
# -----------------------------------------------------------------------------


def _parse_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    entries: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == "," and not quoted:
            entries.append(text[start:index])
            start = index + 1
    if text:
        entries.append(text[start:])
    for entry in entries:
        key, separator, raw_value = entry.partition("=")
        if not separator:
            raise AssertionError(f"invalid Prometheus label entry: {entry!r}")
        value = raw_value.strip()
        if len(value) < 2 or value[0] != '"' or value[-1] != '"':
            raise AssertionError(f"invalid Prometheus label value: {entry!r}")
        parsed_value = json.loads(value)
        if not isinstance(parsed_value, str):
            raise AssertionError(f"invalid Prometheus string label: {entry!r}")
        labels[key.strip()] = parsed_value
    return labels


def _read_metric_samples(text: str, name: str) -> dict[tuple[tuple[str, str], ...], _PrometheusSample]:
    samples: dict[tuple[tuple[str, str], ...], _PrometheusSample] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = _SAMPLE_RE.fullmatch(line.strip())
        if match is None or match.group("name") != name:
            continue
        labels = _parse_labels(match.group("labels") or "")
        key = tuple(sorted(labels.items()))
        if key in samples:
            raise AssertionError(f"duplicate Prometheus series for {name}: {labels}")
        samples[key] = _PrometheusSample(labels=labels, value=float(match.group("value")))
    return samples


def _metric_deltas(
    before: str,
    after: str,
    name: str,
    labels: dict[str, str],
) -> list[tuple[dict[str, str], float]]:
    before_samples = _read_metric_samples(before, name)
    after_samples = _read_metric_samples(after, name)
    deltas: list[tuple[dict[str, str], float]] = []
    for key, sample in after_samples.items():
        if not all(sample.labels.get(label) == value for label, value in labels.items()):
            continue
        previous = before_samples.get(key)
        deltas.append((sample.labels, sample.value - (previous.value if previous else 0.0)))
    return deltas


def _fetch_collector_metrics(url: str) -> str:
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text


def _wait_for_collector_deltas(
    url: str,
    before: str,
    expectations: list[tuple[str, dict[str, str], float]],
    *,
    timeout_s: float = 30.0,
) -> str:
    deadline = time.monotonic() + timeout_s
    after = ""
    while time.monotonic() < deadline:
        after = _fetch_collector_metrics(url)
        if all(
            sum(max(delta, 0.0) for _, delta in _metric_deltas(before, after, metric, labels)) >= minimum
            for metric, labels, minimum in expectations
        ):
            return after
        time.sleep(1)
    observed = [(metric, labels, _metric_deltas(before, after, metric, labels)) for metric, labels, _ in expectations]
    raise AssertionError(f"collector did not expose expected telemetry deltas: {expectations}; observed={observed}")


def _assert_no_collector_delta(
    url: str,
    before: str,
    name: str,
    labels: dict[str, str],
    *,
    observation_s: float = 8.0,
) -> None:
    deadline = time.monotonic() + observation_s
    while time.monotonic() < deadline:
        after = _fetch_collector_metrics(url)
        deltas = _metric_deltas(before, after, name, labels)
        assert not any(delta > 0 for _, delta in deltas), f"unexpected collector telemetry delta: {deltas}"
        time.sleep(1)


def _assert_exact_worker_target(labels: dict[str, str], point_labels: set[str]) -> None:
    expected_point_labels = point_labels | _PRODUCER_LABELS
    actual_labels = set(labels)
    assert actual_labels - _PROMETHEUS_EXPORTER_LABELS == expected_point_labels, labels
    assert actual_labels <= expected_point_labels | _PROMETHEUS_EXPORTER_LABELS, labels
    assert labels["job"] == "sie-worker", labels
    assert labels["producer_service"] == "sie-worker", labels
    assert labels["instance"] == labels["producer_instance"], labels
    assert labels["producer_instance"], labels
    assert labels["lane"] not in ("", "other"), labels
    assert labels["model"] == _telemetry_model(), labels
    assert labels["profile"] == "default", labels
    assert labels["backend"] != "other", labels


def test_collector_exposition_parser_keeps_exact_worker_scope() -> None:
    labels = {
        "backend": "sglang",
        "grammar": "json_schema",
        "grammar_backend": "outlines",
        "instance": "worker-a/process-a",
        "job": "sie-worker",
        "lane": "generation|l4|default",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "phase": "request",
        "producer_instance": "worker-a/process-a",
        "producer_service": "sie-worker",
        "profile": "default",
        "result": "miss",
    }
    label_text = ",".join(f"{key}={json.dumps(value)}" for key, value in sorted(labels.items()))
    exposition = f"sie_worker_generation_grammar_cache_lookups_total{{{label_text}}} 3\n"

    samples = _read_metric_samples(exposition, "sie_worker_generation_grammar_cache_lookups_total")

    assert len(samples) == 1
    sample = next(iter(samples.values()))
    assert sample.labels == labels
    assert sample.value == 3.0
    _assert_exact_worker_target(sample.labels, _GRAMMAR_BASE_LABELS | {"phase", "result"})


def test_pathological_schema_rejects_before_compile(gateway_url: str) -> None:
    """A deeply-nested schema → 400 ``invalid_request`` and no worker
    compile activity. Confirms the gateway is the authority for safety
    caps.
    """
    metrics_url = _collector_metrics_url()
    before = ""
    if metrics_url:
        before = _fetch_collector_metrics(metrics_url)

    deep: dict = {"type": "string"}
    for _ in range(25):
        deep = {"type": "object", "properties": {"nested": deep}}
    body = {
        "prompt": "Hi",
        "max_new_tokens": 8,
        "grammar": {"json_schema": deep},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=10.0,
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["param"].startswith("grammar.json_schema")

    if metrics_url:
        _assert_no_collector_delta(
            metrics_url,
            before,
            "sie_worker_generation_grammar_compile_duration_seconds_count",
            {"model": _telemetry_model(), "phase": "request"},
        )


# -----------------------------------------------------------------------------
# §5.9 — identical schema → 1 compile + 1 cache hit
# -----------------------------------------------------------------------------


def test_cache_hit_after_first_compile(gateway_url: str) -> None:
    metrics_url = _collector_metrics_url()
    if not metrics_url:
        pytest.skip(f"set {_COLLECTOR_METRICS_URL_ENV} to assert collector-exported grammar telemetry")

    property_name = f"value_{uuid.uuid4().hex[:12]}"
    schema = {
        "type": "object",
        "properties": {property_name: {"type": "string"}},
        "required": [property_name],
    }
    body = {
        "prompt": f'Reply with JSON: {{"{property_name}": "ok"}}',
        "max_new_tokens": 32,
        "grammar": {"json_schema": schema},
    }

    def _send() -> None:
        r = httpx.post(
            f"{gateway_url}/v1/generate/{_gen_model()}",
            json=body,
            timeout=60.0,
        )
        assert r.status_code == 200, r.text

    # Snapshot, send twice, snapshot. This opt-in test runs against one
    # generation worker so one exact producer/lane/model series owns the
    # compile and cache deltas.
    before = _fetch_collector_metrics(metrics_url)

    _send()
    _send()

    after = _wait_for_collector_deltas(
        metrics_url,
        before,
        [
            (
                "sie_worker_generation_grammar_cache_lookups_total",
                {"model": _telemetry_model(), "grammar": "json_schema", "phase": "request", "result": "miss"},
                1.0,
            ),
            (
                "sie_worker_generation_grammar_cache_lookups_total",
                {"model": _telemetry_model(), "grammar": "json_schema", "phase": "request", "result": "hit"},
                1.0,
            ),
            (
                "sie_worker_generation_grammar_compile_duration_seconds_count",
                {"model": _telemetry_model(), "grammar": "json_schema", "phase": "request", "outcome": "success"},
                1.0,
            ),
            (
                "sie_worker_generation_grammar_requests_total",
                {"model": _telemetry_model(), "grammar": "json_schema"},
                2.0,
            ),
        ],
    )
    cache_metric = "sie_worker_generation_grammar_cache_lookups_total"
    miss_deltas = [
        (labels, delta)
        for labels, delta in _metric_deltas(
            before,
            after,
            cache_metric,
            {
                "model": _telemetry_model(),
                "grammar": "json_schema",
                "phase": "request",
                "result": "miss",
            },
        )
        if delta > 0
    ]
    assert len(miss_deltas) == 1, f"expected one changed miss series, observed {miss_deltas}"
    miss_labels, miss_delta = miss_deltas[0]
    _assert_exact_worker_target(miss_labels, _GRAMMAR_BASE_LABELS | {"phase", "result"})
    assert miss_labels["grammar_backend"] == "outlines", miss_labels
    assert miss_delta == 1.0, f"expected exactly one cache miss, observed {miss_delta}"

    producer_scope = {
        label: miss_labels[label]
        for label in (
            "job",
            "instance",
            "producer_service",
            "producer_instance",
            "backend",
            "lane",
            "model",
            "profile",
            "grammar_backend",
            "grammar",
        )
    }
    hit_deltas = _metric_deltas(
        before,
        after,
        cache_metric,
        {**producer_scope, "phase": "request", "result": "hit"},
    )
    assert len(hit_deltas) == 1, f"expected one matching hit series, observed {hit_deltas}"
    hit_labels, hit_delta = hit_deltas[0]
    _assert_exact_worker_target(hit_labels, _GRAMMAR_BASE_LABELS | {"phase", "result"})
    assert hit_delta >= 1.0, f"expected at least one cache hit, observed {hit_delta}"

    compile_deltas = _metric_deltas(
        before,
        after,
        "sie_worker_generation_grammar_compile_duration_seconds_count",
        {**producer_scope, "phase": "request", "outcome": "success"},
    )
    assert len(compile_deltas) == 1, f"expected one matching compile series, observed {compile_deltas}"
    compile_labels, compile_delta = compile_deltas[0]
    _assert_exact_worker_target(compile_labels, _GRAMMAR_BASE_LABELS | {"phase", "outcome"})
    assert compile_delta == 1.0, f"expected exactly one compile, observed {compile_delta}"

    request_scope = {
        label: miss_labels[label]
        for label in (
            "job",
            "instance",
            "producer_service",
            "producer_instance",
            "backend",
            "lane",
            "model",
            "profile",
            "grammar",
        )
    }
    request_deltas = [
        (labels, delta)
        for labels, delta in _metric_deltas(
            before,
            after,
            "sie_worker_generation_grammar_requests_total",
            request_scope,
        )
        if delta > 0
    ]
    assert len(request_deltas) == 1, f"expected one matching grammar request series, observed {request_deltas}"
    request_labels, request_delta = request_deltas[0]
    _assert_exact_worker_target(request_labels, _GRAMMAR_BASE_LABELS)
    assert request_labels["grammar_backend"] != "other", request_labels
    assert request_delta == 2.0, f"expected two grammar requests, observed {request_delta}"


# -----------------------------------------------------------------------------
# Mutual exclusivity (cross-cuts §5.5 unit test but worth an integration probe)
# -----------------------------------------------------------------------------


def test_mutex_violation_rejects_at_gateway(gateway_url: str) -> None:
    body = {
        "prompt": "Hi",
        "max_new_tokens": 8,
        "grammar": {"json_schema": {"type": "object"}, "regex": "[a-z]+"},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=10.0,
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["param"] == "grammar"
    assert "mutually exclusive" in err["message"]
