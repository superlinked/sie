//! End-to-end HTTP coverage for the sidecar probe server.
//!
//! This test stands up only the metrics HTTP plane and shared readiness
//! state. It does not require NATS or the Python IPC harness.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use sie_server_sidecar::readiness::Readiness;
use sie_server_sidecar::runtime_state::spawn_probe_server;
use sie_server_sidecar::shutdown::Shutdown;

fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn http_get(port: u16, path: &str) -> std::io::Result<(String, String)> {
    let mut sock = TcpStream::connect(("127.0.0.1", port))?;
    sock.set_read_timeout(Some(Duration::from_secs(5)))?;
    sock.set_write_timeout(Some(Duration::from_secs(5)))?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    sock.write_all(request.as_bytes())?;

    let mut raw = String::new();
    sock.read_to_string(&mut raw)?;
    let status_line = raw.lines().next().unwrap_or("").to_owned();
    let body = match raw.find("\r\n\r\n") {
        Some(idx) => raw[idx + 4..].to_owned(),
        None => String::new(),
    };
    Ok((status_line, body))
}

async fn start_server(readiness: Arc<Readiness>) -> u16 {
    let port = pick_free_port();
    let shutdown = Arc::new(Shutdown::new());
    let _handle = spawn_probe_server(port, readiness, shutdown).expect("spawn server");

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return port;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("probe server did not bind on port {port} within budget");
}

async fn http_get_async(port: u16, path: &str) -> std::io::Result<(String, String)> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || http_get(port, &path))
        .await
        .expect("blocking task did not panic")
}

#[tokio::test]
async fn readyz_handshake_pending_returns_503() {
    let readiness = Arc::new(Readiness::new(2_000, 3));
    let port = start_server(Arc::clone(&readiness)).await;

    let (status, body) = http_get_async(port, "/readyz").await.unwrap();
    assert!(
        status.starts_with("HTTP/1.1 503"),
        "expected 503 status line, got {status:?}"
    );
    assert!(
        body.contains("handshake pending"),
        "body should mention handshake pending, got {body:?}"
    );
}

#[tokio::test]
async fn readyz_after_ping_returns_200() {
    let readiness = Arc::new(Readiness::new(2_000, 3));
    readiness.record_ping_success();
    let port = start_server(Arc::clone(&readiness)).await;

    let (status, body) = http_get_async(port, "/readyz").await.unwrap();
    assert!(
        status.starts_with("HTTP/1.1 200"),
        "expected 200 status line, got {status:?}"
    );
    assert_eq!(body, "ok");
}

#[tokio::test]
async fn readyz_after_drain_returns_503_draining() {
    let readiness = Arc::new(Readiness::new(2_000, 3));
    readiness.record_ping_success();
    let port = start_server(Arc::clone(&readiness)).await;

    let (status_before, _) = http_get_async(port, "/readyz").await.unwrap();
    assert!(status_before.starts_with("HTTP/1.1 200"));

    readiness.mark_draining();

    let (status_after, body_after) = http_get_async(port, "/readyz").await.unwrap();
    assert!(
        status_after.starts_with("HTTP/1.1 503"),
        "expected 503 after drain, got {status_after:?}"
    );
    assert!(
        body_after.contains("draining"),
        "body should mention draining, got {body_after:?}"
    );
}

#[tokio::test]
async fn readyz_after_stale_heartbeat_returns_503() {
    let readiness = Arc::new(Readiness::new(50, 1));
    readiness.record_ping_success();
    let port = start_server(Arc::clone(&readiness)).await;

    tokio::time::sleep(Duration::from_millis(150)).await;

    let (status, body) = http_get_async(port, "/readyz").await.unwrap();
    assert!(
        status.starts_with("HTTP/1.1 503"),
        "expected 503 after staleness, got {status:?}"
    );
    assert!(
        body.contains("heartbeat stale"),
        "body should mention staleness, got {body:?}"
    );
    assert!(
        body.contains("threshold 50 ms"),
        "body should surface the threshold, got {body:?}"
    );
}

#[tokio::test]
async fn healthz_remains_200_regardless_of_readiness() {
    let readiness = Arc::new(Readiness::new(2_000, 3));
    readiness.mark_draining();
    let port = start_server(Arc::clone(&readiness)).await;

    let (status, body) = http_get_async(port, "/healthz").await.unwrap();
    assert!(
        status.starts_with("HTTP/1.1 200"),
        "/healthz must stay green, got {status:?}"
    );
    assert_eq!(body, "ok");
}

#[tokio::test]
async fn unknown_path_returns_404() {
    let readiness = Arc::new(Readiness::new(2_000, 3));
    let port = start_server(Arc::clone(&readiness)).await;

    let (status, body) = http_get_async(port, "/does-not-exist").await.unwrap();
    assert!(
        status.starts_with("HTTP/1.1 404"),
        "expected 404 for unknown path, got {status:?}"
    );
    assert_eq!(body, "not found");
}
