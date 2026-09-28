"""Exercise real SDK span completion and the safe shared-duration representation."""

from unittest.mock import Mock

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import (
    INVALID_SPAN_CONTEXT,
    Link,
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    TraceState,
    set_span_in_context,
)
from sie_server.observability.batch_fanin import BatchFanInSpanProcessor, batch_request_spans


def parent(trace_id: int = 1, span_id: int = 2, sampled: bool = True) -> SpanContext:
    return SpanContext(trace_id, span_id, True, TraceFlags(int(sampled)), TraceState([("vendor", "private")]))


def record(name="worker.run_batch", links=(), *, span_limits=None, sampler=None):
    exporter = InMemorySpanExporter()
    kwargs = {"sampler": sampler} if sampler is not None else {}
    provider = TracerProvider(resource=Resource({"service.name": "sie-worker"}), span_limits=span_limits, **kwargs)
    provider.add_span_processor(BatchFanInSpanProcessor(SimpleSpanProcessor(exporter)))
    tracer = provider.get_tracer("local.scope", attributes={"secret": "private"})
    span = tracer.start_span(name, context=set_span_in_context(NonRecordingSpan(parent())), links=links, start_time=100)
    span.set_attribute("customer", "private")
    span.add_event("private exception", {"payload": "private"})
    span.set_status(Status(StatusCode.ERROR, "private error"))
    span.end(end_time=200)
    assert provider.force_flush()
    spans = exporter.get_finished_spans()
    provider.shutdown()
    return spans


@pytest.mark.parametrize("name", ["worker.run_batch", "sidecar.dispatch"])
def test_projects_distinct_sampled_parents_without_mutating_local_span(name):
    # Two traces and two distinct parents within one trace; duplicates with
    # different tracestate must not make additional duration observations.
    links = [Link(parent(3, 4), {"customer": "private"}), Link(parent(1, 5)), Link(parent(3, 4))]
    original, *views = record(name, links)
    assert original.links == tuple(links)
    assert original.attributes["customer"] == "private"
    assert original.status.description == "private error"
    assert original.context.trace_state.get("vendor") == "private"
    assert len(original.events) == 1
    assert {(s.context.trace_id, s.parent.span_id) for s in views} == {(1, 2), (3, 4), (1, 5)}
    assert len({s.context.span_id for s in (original, *views)}) == 4
    for view in views:
        assert view.name == f"{name}.request"
        assert (view.start_time, view.end_time, view.kind) == (100, 200, original.kind)
        assert view.status.status_code == StatusCode.ERROR
        assert view.status.description is None
        assert not view.attributes
        assert not view.events
        assert not view.links
        assert not view.context.trace_state
        assert not view.parent.trace_state
        assert view.instrumentation_scope.name == "sie.batch_fanin"
        assert not view.instrumentation_scope.attributes
        # A projection is terminal: it cannot recursively generate more spans.
        assert not list(batch_request_spans(view))


def test_one_parent_and_unknown_names_remain_unchanged():
    assert [s.name for s in record()] == ["worker.run_batch"]
    assert [s.name for s in record("arbitrary", [Link(parent(3, 4))])] == ["arbitrary"]


def test_invalid_duplicate_and_unsampled_links_do_not_create_parents():
    spans = record(links=[Link(INVALID_SPAN_CONTEXT), Link(parent()), Link(parent(3, 4, False))])
    assert len(spans) == 2
    assert spans[1].parent.span_id == 2


def test_projection_does_not_override_sampling_or_recover_sdk_dropped_links():
    assert record(links=[Link(parent(3, 4))], sampler=ALWAYS_OFF) == ()
    original, *views = record(links=[Link(parent(3, 4)), Link(parent(5, 6))], span_limits=SpanLimits(max_links=1))
    assert original.dropped_links == 1
    assert len(views) == 2  # primary and the single retained link only


def test_processor_delegates_lifecycle_and_uses_original_queue():
    inner = Mock(spec=SpanProcessor)
    processor = BatchFanInSpanProcessor(inner)
    span = record()[0]
    processor.on_end(span)
    inner.on_end.assert_called_once_with(span)
    assert processor.force_flush(17) is inner.force_flush.return_value
    inner.force_flush.assert_called_once_with(17)
    processor.shutdown()
    inner.shutdown.assert_called_once_with()
