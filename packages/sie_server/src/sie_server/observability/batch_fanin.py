"""Per-request timing views of shared batches, before the bounded export queue.

The original span (including links) remains available to local tracing. The
collector independently removes it from the remote branch. These leaf spans
represent the same shared interval, not additional execution or per-item work.
"""

from __future__ import annotations

from collections.abc import Iterator

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, Status, TraceState

_BATCH_NAMES = {"worker.run_batch": "worker.run_batch.request", "sidecar.dispatch": "sidecar.dispatch.request"}
_IDS = RandomIdGenerator()
_SCOPE = InstrumentationScope("sie.batch_fanin")


def batch_request_spans(span: ReadableSpan) -> Iterator[ReadableSpan]:
    """Project only recorded, sampled batches onto distinct sampled parents.

    Deduplicate by numeric trace/span identity, never by untrusted tracestate.
    No recovery is possible for an unsampled batch or SDK-discarded links.
    """
    name = _BATCH_NAMES.get(span.name)
    context = span.context
    if name is None or not span.links or context is None or not context.trace_flags.sampled:
        return
    seen: set[tuple[int, int]] = set()
    primary = (
        SpanContext(context.trace_id, span.parent.span_id, span.parent.is_remote, context.trace_flags, TraceState())
        if span.parent is not None
        else None
    )
    parents = [primary, *(link.context for link in span.links)]
    for parent in parents:
        if parent is None or not parent.is_valid or not parent.trace_flags.sampled:
            continue
        key = (parent.trace_id, parent.span_id)
        if key in seen:
            continue
        seen.add(key)
        safe_parent = SpanContext(parent.trace_id, parent.span_id, parent.is_remote, parent.trace_flags, TraceState())
        yield ReadableSpan(
            name=name,
            context=SpanContext(parent.trace_id, _IDS.generate_span_id(), False, parent.trace_flags, TraceState()),
            parent=safe_parent,
            resource=span.resource,
            attributes={},
            events=(),
            links=(),
            kind=span.kind,
            status=Status(span.status.status_code),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=_SCOPE,
        )


class BatchFanInSpanProcessor(SpanProcessor):
    """Keep originals and enqueue safe timing leaves through the same processor.

    Wrap the existing BatchSpanProcessor so its queue, batching, shutdown and
    loss behavior remain authoritative; never expand a batch after dequeueing.
    """

    def __init__(self, inner: SpanProcessor) -> None:
        self._inner = inner

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        self._inner.on_end(span)
        for request_span in batch_request_spans(span):
            self._inner.on_end(request_span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
