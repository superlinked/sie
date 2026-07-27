//! W3C Trace Context propagation helpers.
//!
//! Every SIE hop goes through the **same** globally-installed
//! [`opentelemetry::propagation::TextMapPropagator`] — the W3C propagator the
//! consumer installs in its `init_tracing` — so the wire format is identical
//! in both directions on every boundary:
//!
//! - the gateway extracts inbound HTTP headers (feature `http-propagation`)
//!   and injects onto the outbound work envelope;
//! - the sidecar extracts the envelope's W3C strings and injects onto the
//!   IPC wire item;
//! - the Rust worker extracts the IPC strings.

use std::collections::HashMap;

use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanContext, TraceContextExt};
use opentelemetry::Context;

/// Adapter exposing a `HashMap<String, String>` as an OTel [`Extractor`].
struct HashMapExtractor<'a>(&'a HashMap<String, String>);

impl Extractor for HashMapExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }
}

/// Adapter exposing a `HashMap<String, String>` as an OTel [`Injector`]. The
/// hashmap is the propagator-friendly intermediate form: the propagator
/// writes the two W3C headers as `String`s and we lift them out for the
/// typed wire fields.
struct HashMapInjector<'a>(&'a mut HashMap<String, String>);

impl Injector for HashMapInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }
}

/// Adapter exposing an [`http::HeaderMap`] as an OTel [`Extractor`].
/// Borrowed view — no allocations beyond a transient `Vec<&str>` in `keys()`.
#[cfg(feature = "http-propagation")]
struct HeaderMapExtractor<'a>(&'a http::HeaderMap);

#[cfg(feature = "http-propagation")]
impl<'a> Extractor for HeaderMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Extract a parent [`Context`] from inbound HTTP request headers.
///
/// Returns the empty (root) context when no `traceparent` header is present,
/// matching W3C semantics: callers should still open their own span; it will
/// simply not be a child of any external trace.
#[cfg(feature = "http-propagation")]
pub fn extract_context_from_headers(headers: &http::HeaderMap) -> Context {
    global::get_text_map_propagator(|propagator| propagator.extract(&HeaderMapExtractor(headers)))
}

/// Build the W3C carrier map from the optional wire strings.
fn carrier_from_w3c(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> HashMap<String, String> {
    let mut carrier: HashMap<String, String> = HashMap::with_capacity(2);
    if let Some(tp) = traceparent {
        carrier.insert("traceparent".to_string(), tp.to_string());
    }
    if let Some(ts) = tracestate {
        carrier.insert("tracestate".to_string(), ts.to_string());
    }
    carrier
}

/// Extract a parent [`Context`] from the inbound wire's W3C strings.
///
/// Returns the empty (root) context when no `traceparent` is present,
/// matching W3C semantics: callers should still open their own span; it
/// will simply not be a child of any external trace.
pub fn extract_context_from_w3c(traceparent: Option<&str>, tracestate: Option<&str>) -> Context {
    let carrier = carrier_from_w3c(traceparent, tracestate);
    global::get_text_map_propagator(|propagator| propagator.extract(&HashMapExtractor(&carrier)))
}

/// Extract the inbound [`SpanContext`] from the wire's W3C strings, for use
/// as a span *link* (batch coalescing means a single span can parent items
/// from several inbound traces; the non-primary parents are recorded as
/// links).
///
/// Returns `None` when the strings are absent or yield an invalid context.
pub fn remote_span_context(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> Option<SpanContext> {
    let cx = extract_context_from_w3c(traceparent, tracestate);
    let span_cx = cx.span().span_context().clone();
    span_cx.is_valid().then_some(span_cx)
}

/// Serialise the current OTel [`Context`] (active span) back into the two
/// W3C strings.
///
/// Returns `(traceparent, tracestate)`. Both are `None` when no span is
/// currently active or the propagator chose to skip them.
pub fn inject_current_context() -> (Option<String>, Option<String>) {
    inject_context(&Context::current())
}

/// Variant of [`inject_current_context`] taking an explicit context — useful
/// when the caller has already detached / re-attached a span and wants to
/// inject the not-currently-attached parent.
pub fn inject_context(cx: &Context) -> (Option<String>, Option<String>) {
    let mut carrier: HashMap<String, String> = HashMap::with_capacity(2);
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut HashMapInjector(&mut carrier));
    });
    let traceparent = carrier.remove("traceparent");
    let tracestate = carrier.remove("tracestate");
    (traceparent, tracestate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{Span as _, Tracer, TracerProvider as _};
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    /// Install the propagator once per test path. Installing twice is
    /// harmless (the global slot accepts the new value), but it must be live
    /// before extract/inject for the wire format to match.
    fn install_propagator() {
        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    }

    const SAMPLE_TP: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn extract_w3c_inherits_trace_id() {
        install_propagator();
        let cx = extract_context_from_w3c(Some(SAMPLE_TP), None);
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start_with_context("child", &cx);
        let span_cx = span.span_context().clone();
        assert!(span_cx.is_valid(), "child span context must be valid");
        assert_eq!(
            format!("{:032x}", span_cx.trace_id()),
            "0af7651916cd43dd8448eb211c80319c",
            "child must inherit the extracted trace id",
        );
    }

    #[test]
    fn extract_w3c_absent_is_root_context() {
        install_propagator();
        let cx = extract_context_from_w3c(None, None);
        assert!(
            !cx.span().span_context().is_valid(),
            "no traceparent means root invalid span context",
        );
    }

    #[test]
    fn remote_span_context_present_and_absent() {
        install_propagator();
        let sc = remote_span_context(Some(SAMPLE_TP), None).expect("valid traceparent gives Some");
        assert_eq!(
            format!("{:032x}", sc.trace_id()),
            "0af7651916cd43dd8448eb211c80319c",
        );
        assert_eq!(format!("{:016x}", sc.span_id()), "b7ad6b7169203331");
        assert!(remote_span_context(None, None).is_none());
    }

    #[test]
    fn inject_with_no_active_span_returns_none_pair() {
        install_propagator();
        let (tp, ts) = inject_context(&Context::new());
        assert!(tp.is_none(), "no active span means no traceparent");
        assert!(ts.is_none(), "no active span means no tracestate");
    }

    #[test]
    fn inject_with_active_span_yields_w3c_traceparent() {
        install_propagator();
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start("parent");
        let cx = Context::current().with_span(span);
        let (tp, _ts) = inject_context(&cx);
        let tp = tp.expect("active span should inject a traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent must be 4 fields: {tp}");
        assert_eq!(parts[0].len(), 2, "version field 2 hex chars");
        assert_eq!(parts[1].len(), 32, "trace_id field 32 hex chars");
        assert_eq!(parts[2].len(), 16, "span_id field 16 hex chars");
        assert_eq!(parts[3].len(), 2, "flags field 2 hex chars");
    }

    #[cfg(feature = "http-propagation")]
    #[test]
    fn extract_headers_round_trip_returns_same_traceparent() {
        install_propagator();
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::HeaderName::from_static("traceparent"),
            http::HeaderValue::from_static(SAMPLE_TP),
        );
        let cx = extract_context_from_headers(&headers);
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start_with_context("child", &cx);
        let span_cx = span.span_context().clone();
        assert!(span_cx.is_valid(), "child span context must be valid");
        assert_eq!(
            format!("{:032x}", span_cx.trace_id()),
            "0af7651916cd43dd8448eb211c80319c",
            "child must inherit the extracted trace id",
        );
    }
}
