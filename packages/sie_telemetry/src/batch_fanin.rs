//! Per-request timing leaves for shared batches, ahead of the bounded queue.
//!
//! Originals retain rich local links. Remote privacy filtering still drops
//! linked originals; these views carry only a parent, fresh ID, shared timing,
//! kind and status code. They are not additional execution or detailed parents.

use std::collections::HashSet;
use std::time::Duration;

use opentelemetry::trace::{SpanContext, Status, TraceState};
use opentelemetry::{Context, InstrumentationScope};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator, Span, SpanData, SpanProcessor};
use opentelemetry_sdk::Resource;

/// Add timing leaves without bypassing the wrapped processor's queue limits.
#[derive(Debug)]
pub struct BatchFanInSpanProcessor<P> {
    inner: P,
}

impl<P> BatchFanInSpanProcessor<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: SpanProcessor> SpanProcessor for BatchFanInSpanProcessor<P> {
    fn on_start(&self, span: &mut Span, cx: &Context) {
        self.inner.on_start(span, cx);
    }

    fn on_end(&self, span: SpanData) {
        let projections = batch_request_spans(&span);
        self.inner.on_end(span);
        for projection in projections {
            self.inner.on_end(projection);
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

fn batch_request_spans(span: &SpanData) -> Vec<SpanData> {
    let name = match span.name.as_ref() {
        "worker.run_batch" => "worker.run_batch.request",
        "sidecar.dispatch" => "sidecar.dispatch.request",
        _ => return Vec::new(),
    };
    if span.links.links.is_empty() || !span.span_context.is_sampled() {
        return Vec::new();
    }
    // The recorded batch inherited its primary parent's sampling decision.
    let primary = SpanContext::new(
        span.span_context.trace_id(),
        span.parent_span_id,
        span.span_context.trace_flags(),
        span.parent_span_is_remote,
        TraceState::default(),
    );
    let mut seen = HashSet::new();
    std::iter::once(&primary)
        .chain(span.links.links.iter().map(|link| &link.span_context))
        .filter(|parent| {
            parent.is_valid()
                && parent.is_sampled()
                && seen.insert((parent.trace_id(), parent.span_id()))
        })
        .map(|parent| SpanData {
            span_context: SpanContext::new(
                parent.trace_id(),
                RandomIdGenerator::default().new_span_id(),
                parent.trace_flags(),
                false,
                TraceState::default(),
            ),
            parent_span_id: parent.span_id(),
            parent_span_is_remote: parent.is_remote(),
            span_kind: span.span_kind.clone(),
            name: name.into(),
            start_time: span.start_time,
            end_time: span.end_time,
            attributes: Vec::new(),
            dropped_attributes_count: 0,
            events: Default::default(),
            links: Default::default(),
            status: match span.status {
                Status::Error { .. } => Status::error(""),
                Status::Ok => Status::Ok,
                Status::Unset => Status::Unset,
            },
            instrumentation_scope: InstrumentationScope::builder("sie.batch_fanin").build(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{
        Link, Span as _, SpanId, TraceContextExt, TraceFlags, TraceId, Tracer, TracerProvider,
    };
    use opentelemetry::KeyValue;
    use opentelemetry_sdk::trace::{
        InMemorySpanExporter, Sampler, SdkTracerProvider, SimpleSpanProcessor,
    };

    fn parent(trace: u128, span: u64, sampled: bool) -> SpanContext {
        SpanContext::new(
            TraceId::from(trace),
            SpanId::from(span),
            if sampled {
                TraceFlags::SAMPLED
            } else {
                TraceFlags::default()
            },
            true,
            TraceState::from_key_value([("vendor", "private")]).unwrap(),
        )
    }

    fn record(
        name: &'static str,
        links: Vec<Link>,
        sampler: Sampler,
        max_links: u32,
    ) -> Vec<SpanData> {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_sampler(sampler)
            .with_max_links_per_span(max_links)
            .with_span_processor(BatchFanInSpanProcessor::new(SimpleSpanProcessor::new(
                exporter.clone(),
            )))
            .build();
        let tracer = provider.tracer("local.scope");
        let cx = Context::new().with_remote_span_context(parent(1, 2, true));
        let mut span = tracer
            .span_builder(name)
            .with_links(links)
            .with_start_time(std::time::UNIX_EPOCH)
            .start_with_context(&tracer, &cx);
        span.set_attribute(KeyValue::new("customer", "private"));
        span.add_event(
            "private exception",
            vec![KeyValue::new("payload", "private")],
        );
        span.set_status(Status::error("private error"));
        span.end_with_timestamp(std::time::UNIX_EPOCH + Duration::from_secs(1));
        provider.force_flush().unwrap();
        let spans = exporter.get_finished_spans().unwrap();
        provider.shutdown().unwrap();
        spans
    }

    #[test]
    fn projects_distinct_parents_with_safe_fields_and_exact_timing() {
        for name in ["worker.run_batch", "sidecar.dispatch"] {
            let links = vec![
                Link::new(
                    parent(3, 4, true),
                    vec![KeyValue::new("customer", "private")],
                    0,
                ),
                Link::new(parent(1, 5, true), Vec::new(), 0),
                Link::new(parent(3, 4, true), Vec::new(), 0),
            ];
            let spans = record(name, links.clone(), Sampler::AlwaysOn, 128);
            assert_eq!(spans.len(), 4);
            let original = &spans[0];
            assert_eq!(original.links.links, links);
            assert_eq!(
                original.attributes,
                vec![KeyValue::new("customer", "private")]
            );
            assert_eq!(original.status, Status::error("private error"));
            assert_eq!(original.events.len(), 1);
            assert_eq!(
                original.span_context.trace_state().header(),
                "vendor=private"
            );
            let parents: HashSet<_> = spans[1..]
                .iter()
                .map(|s| (s.span_context.trace_id(), s.parent_span_id))
                .collect();
            assert_eq!(
                parents,
                [(1u128, 2u64), (3, 4), (1, 5)]
                    .map(|(t, s)| (TraceId::from(t), SpanId::from(s)))
                    .into()
            );
            assert_eq!(
                spans
                    .iter()
                    .map(|s| s.span_context.span_id())
                    .collect::<HashSet<_>>()
                    .len(),
                4
            );
            for view in &spans[1..] {
                assert_eq!(view.name, format!("{name}.request"));
                assert_eq!(view.start_time, original.start_time);
                assert_eq!(view.end_time, original.end_time);
                assert_eq!(view.span_kind, original.span_kind);
                assert_eq!(view.status, Status::error(""));
                assert!(
                    view.attributes.is_empty() && view.events.is_empty() && view.links.is_empty()
                );
                assert!(view.span_context.trace_state().header().is_empty());
                assert_eq!(view.instrumentation_scope.name(), "sie.batch_fanin");
                assert!(batch_request_spans(view).is_empty());
            }
        }
    }

    #[test]
    fn ignores_unlinked_unknown_invalid_duplicate_and_unsampled_contexts() {
        assert_eq!(
            record("worker.run_batch", Vec::new(), Sampler::AlwaysOn, 128).len(),
            1
        );
        assert_eq!(
            record(
                "unknown",
                vec![Link::new(parent(3, 4, true), Vec::new(), 0)],
                Sampler::AlwaysOn,
                128
            )
            .len(),
            1
        );
        let spans = record(
            "worker.run_batch",
            vec![
                Link::new(SpanContext::empty_context(), Vec::new(), 0),
                Link::new(parent(1, 2, true), Vec::new(), 0),
                Link::new(parent(3, 4, false), Vec::new(), 0),
            ],
            Sampler::AlwaysOn,
            128,
        );
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[1].parent_span_id, SpanId::from(2u64));
    }

    #[test]
    fn respects_sampling_and_retained_link_limit() {
        let links = vec![
            Link::new(parent(3, 4, true), Vec::new(), 0),
            Link::new(parent(5, 6, true), Vec::new(), 0),
        ];
        assert!(record("worker.run_batch", links.clone(), Sampler::AlwaysOff, 128).is_empty());
        let spans = record("worker.run_batch", links, Sampler::AlwaysOn, 1);
        assert_eq!(spans[0].links.dropped_count, 1);
        assert_eq!(spans.len(), 3);
    }
}
