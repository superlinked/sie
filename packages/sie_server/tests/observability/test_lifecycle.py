"""Lifecycle status, cancellation, head sampling and safe record regression tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON
from sie_server import adapter_call_loop
from sie_server.ipc_types import BatchOutcome, RunBatchItem, RunBatchRequest
from sie_server.observability import lifecycle


@pytest.mark.parametrize(
    ("terminal", "outcome", "status"),
    [
        ({"finish_reason": "stop"}, "success", trace.StatusCode.UNSET),
        ({"error": {"code": "private", "message": "secret payload"}}, "error", trace.StatusCode.ERROR),
        ({"finish_reason": "cancelled"}, "cancelled", trace.StatusCode.UNSET),
        (None, "error", trace.StatusCode.ERROR),
    ],
)
def test_terminal_status_and_bounded_correlated_record(monkeypatch, terminal, outcome, status):
    logs = InMemoryLogRecordExporter()
    log_provider = LoggerProvider()
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    monkeypatch.setattr(lifecycle, "_provider", log_provider)
    spans = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(spans))
    with provider.get_tracer("test").start_as_current_span("worker.streaming_processor") as span:
        with lifecycle.observe_generation():
            if terminal is not None:
                lifecycle.current_lifecycle().published_terminal(terminal)
                # A duplicate success terminal must not erase the first failure.
                lifecycle.current_lifecycle().published_terminal({"finish_reason": "stop"})
    record = logs.get_finished_logs()[0].log_record
    assert record.body == "inference.lifecycle.completed"
    assert record.attributes["outcome"] == outcome
    assert record.attributes["phase"] == "worker_generation"
    assert record.trace_id == span.get_span_context().trace_id
    assert record.span_id == span.get_span_context().span_id
    assert "secret" not in str(record.attributes)
    assert set(record.attributes) == {
        "event.name",
        "event.schema.version",
        "phase",
        "operation",
        "outcome",
        "error_class",
        "duration_ms",
    }
    assert spans.get_finished_spans()[0].status.status_code == status
    assert not spans.get_finished_spans()[0].events
    assert lifecycle.current_lifecycle() is None
    log_provider.shutdown()
    provider.shutdown()


@pytest.mark.parametrize(
    ("error", "outcome"), [(asyncio.CancelledError(), "cancelled"), (RuntimeError("secret"), "error")]
)
def test_exception_and_task_cancellation_are_safe_and_reraised(monkeypatch, error, outcome):
    logs = InMemoryLogRecordExporter()
    log_provider = LoggerProvider()
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    monkeypatch.setattr(lifecycle, "_provider", log_provider)
    provider = TracerProvider()
    with (
        pytest.raises(type(error)),
        provider.get_tracer("test").start_as_current_span(
            "worker.streaming_processor", record_exception=False, set_status_on_exception=False
        ),
        lifecycle.observe_generation(),
    ):
        raise error
    record = logs.get_finished_logs()[0].log_record
    assert record.attributes["outcome"] == outcome
    assert "secret" not in str(record.attributes)
    assert lifecycle.current_lifecycle() is None
    log_provider.shutdown()
    provider.shutdown()


def test_head_unsampled_failure_does_not_emit_log(monkeypatch):
    logs = InMemoryLogRecordExporter()
    log_provider = LoggerProvider()
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    monkeypatch.setattr(lifecycle, "_provider", log_provider)
    provider = TracerProvider(sampler=ALWAYS_OFF)
    with (
        provider.get_tracer("test").start_as_current_span("worker.streaming_processor"),
        lifecycle.observe_generation(),
    ):
        lifecycle.current_lifecycle().published_terminal({"error": {"message": "secret"}})
    assert not logs.get_finished_logs()
    log_provider.shutdown()
    provider.shutdown()


@pytest.mark.asyncio
async def test_batch_error_values_set_structural_status(monkeypatch):
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(adapter_call_loop.trace, "get_tracer", lambda *_args: tracer)
    request = RunBatchRequest(
        model_id="test", batch_id=1, lora_key="", total_cost=0, items=[RunBatchItem(op="invalid")]
    )
    executor = SimpleNamespace(process_encode_batch=AsyncMock(return_value=BatchOutcome(outcomes=[])))
    result = await adapter_call_loop.handle_run_batch(executor, request)
    assert result.outcomes[0].disposition == "publish_error_and_ack"
    assert spans.get_finished_spans()[0].status.status_code == trace.StatusCode.ERROR
    assert spans.get_finished_spans()[0].status.description is None
    provider.shutdown()
