//! Integration test: the sidecar on an authenticated NATS server.
//!
//! Runs `nats-server` with the configuration the Helm chart renders by
//! default (`tools/ci/fixtures/sie-cluster-nats.conf`, which
//! `tools/ci/tests/test_helm_render.py` keeps equal to the chart). Skips when
//! `nats-server` is not on `PATH`, unless `NATS_URL` is set as in CI, where a
//! missing binary fails the test.

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use sie_server_sidecar::config::{NatsCredentials, WorkerConfig};
use sie_server_sidecar::nats_consumer;

const CONFIG_PASSWORD: &str = "ConfigPassword0123456789abcdefghij";
const GATEWAY_PASSWORD: &str = "GatewayPassword0123456789abcdefghi";
const WORKER_PASSWORD: &str = "WorkerPassword0123456789abcdefghij";

struct NatsServer {
    child: Child,
    url: String,
    _dir: tempfile::TempDir,
}

impl Drop for NatsServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn nats_server_binary() -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join("nats-server"))
        .find(|path| path.is_file())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("free port")
        .port()
}

async fn start_nats() -> Option<NatsServer> {
    let Some(binary) = nats_server_binary() else {
        assert!(
            std::env::var_os("NATS_URL").is_none(),
            "nats-server must be on PATH when NATS_URL is set"
        );
        eprintln!("skipping: nats-server not on PATH");
        return None;
    };
    let config =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ci/fixtures/sie-cluster-nats.conf");
    let dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();
    let child = Command::new(binary)
        .arg("-c")
        .arg(&config)
        .args(["-a", "127.0.0.1", "-p", &port.to_string()])
        .args(["-m", &free_port().to_string()])
        .arg("-sd")
        .arg(dir.path().join("jetstream"))
        .arg("-P")
        .arg(dir.path().join("nats.pid"))
        .env("SERVER_NAME", "sie-sidecar-auth-test")
        .env("SIE_NATS_AUTH_CONFIG_PASSWORD", CONFIG_PASSWORD)
        .env("SIE_NATS_AUTH_GATEWAY_PASSWORD", GATEWAY_PASSWORD)
        .env("SIE_NATS_AUTH_WORKER_PASSWORD", WORKER_PASSWORD)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start nats-server");
    let server = NatsServer {
        child,
        url: format!("nats://127.0.0.1:{port}"),
        _dir: dir,
    };
    for _ in 0..200 {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Some(server);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("nats-server did not listen on {port}");
}

async fn connect_as(url: &str, user: &str, password: &str) -> async_nats::Client {
    async_nats::ConnectOptions::new()
        .user_and_password(user.to_string(), password.to_string())
        .connect(url)
        .await
        .unwrap_or_else(|e| panic!("connect as {user}: {e}"))
}

fn worker_credentials() -> NatsCredentials {
    NatsCredentials {
        user: "sie-worker".to_string(),
        password: WORKER_PASSWORD.to_string(),
    }
}

fn worker_config(url: &str) -> WorkerConfig {
    WorkerConfig {
        nats_url: Some(url.to_string()),
        nats_credentials: Some(worker_credentials()),
        local_socket_path: None,
        pool: "authpool".into(),
        bundle: "default".into(),
        ipc_socket_path: PathBuf::from("/tmp/auth-test.sock"),
        ipc_socket_paths: vec![PathBuf::from("/tmp/auth-test.sock")],
        ipc_pool_size: 1,
        ipc_request_timeout_s: 60,
        model_ready_timeout_s: 900,
        payload_store_url: None,
        gateway_url: None,
        gateway_api_key: None,
        pool_admission_enabled: false,
        pool_admission_check_interval_ms: 5_000,
        pool_admission_pause_ms: 1_000,
        pool_admission_stale_after_ms: 30_000,
        probe_port: 9095,
        worker_id: "auth-worker-0".into(),
        ping_interval_ms: 2000,
        ready_stale_mult: 3,
        machine_profile: "l4".into(),
        gpu_count: 1,
        bundle_config_hash: String::new(),
        config_service_url: None,
        config_service_token: None,
        config_poll_interval_ms: 30_000,
        config_full_export_interval_ms: 300_000,
        nats_config_trusted_producers: vec!["sie-config".into()],
        health_publish_interval_ms: 5_000,
    }
}

async fn nothing_arrives(subscriber: &mut async_nats::Subscriber) -> bool {
    tokio::time::timeout(Duration::from_millis(500), subscriber.next())
        .await
        .is_err()
}

#[tokio::test]
async fn worker_user_consumes_work_and_returns_results() {
    let Some(nats) = start_nats().await else {
        return;
    };
    let config = worker_config(&nats.url);
    let (worker, js) = nats_consumer::connect(&nats.url, config.nats_credentials.as_ref())
        .await
        .expect("connect as sie-worker");
    let consumer = nats_consumer::ensure_stream_and_consumer(&js, &config)
        .await
        .expect("the worker user provisions its pool stream and consumer");
    nats_consumer::ensure_worker_stream_and_consumer(&js, &config)
        .await
        .expect("the worker user provisions its direct-dispatch stream and consumer");

    let gateway = connect_as(&nats.url, "sie-gateway", GATEWAY_PASSWORD).await;
    let mut results = gateway
        .subscribe("_INBOX.gateway-auth-test.>")
        .await
        .expect("subscribe");
    gateway.flush().await.expect("gateway flush");
    jetstream::new(gateway.clone())
        .publish("sie.work.authpool.l4.default.model", "work".into())
        .await
        .expect("publish work")
        .await
        .expect("work stored");

    let mut messages = consumer.messages().await.expect("pull");
    let work = tokio::time::timeout(Duration::from_secs(5), messages.next())
        .await
        .expect("work delivered")
        .expect("pull stream open")
        .expect("work message");
    assert_eq!(work.payload.as_ref(), b"work");
    work.ack().await.expect("ack");

    worker
        .publish("_INBOX.gateway-auth-test.req-1", "result".into())
        .await
        .expect("publish result");
    worker.flush().await.expect("worker flush");
    let result = tokio::time::timeout(Duration::from_secs(5), results.next())
        .await
        .expect("result delivered")
        .expect("subscription open");
    assert_eq!(result.payload.as_ref(), b"result");
}

#[tokio::test]
async fn worker_user_is_refused_outside_its_subjects() {
    let Some(nats) = start_nats().await else {
        return;
    };
    let config = worker_config(&nats.url);
    let (worker, js) = nats_consumer::connect(&nats.url, config.nats_credentials.as_ref())
        .await
        .expect("connect as sie-worker");
    nats_consumer::ensure_stream_and_consumer(&js, &config)
        .await
        .expect("pool stream");

    let gateway = connect_as(&nats.url, "sie-gateway", GATEWAY_PASSWORD).await;
    let mut config_deltas = gateway
        .subscribe("sie.config.models._all")
        .await
        .expect("subscribe");
    let mut gateway_results = worker
        .subscribe("_INBOX.gateway-auth-test.>")
        .await
        .expect("subscribe request is sent");
    gateway.flush().await.expect("gateway flush");
    worker.flush().await.expect("worker flush");

    worker
        .publish("sie.config.models._all", "{}".into())
        .await
        .expect("publish is sent");
    worker
        .publish("sie.work.authpool.l4.default.model", "forged".into())
        .await
        .expect("publish is sent");
    worker
        .publish("cancel.gateway-auth-test.req-1", "".into())
        .await
        .expect("publish is sent");
    let other_worker = nats_consumer::connect(&nats.url, Some(&worker_credentials()))
        .await
        .expect("second worker")
        .0;
    other_worker
        .publish("_INBOX.gateway-auth-test.req-2", "result".into())
        .await
        .expect("publish result");
    other_worker.flush().await.expect("flush");
    worker.flush().await.expect("worker flush");

    assert!(
        nothing_arrives(&mut config_deltas).await,
        "config subject is sie-config only"
    );
    assert!(
        nothing_arrives(&mut gateway_results).await,
        "workers cannot read gateway inboxes"
    );
    let stream = js
        .get_stream("WORK_POOL_authpool")
        .await
        .expect("stream info");
    assert_eq!(
        stream.cached_info().state.messages,
        0,
        "workers cannot publish work"
    );
    assert!(js.delete_stream("WORK_POOL_authpool").await.is_err());
    assert!(js
        .get_stream("WORK_POOL_authpool")
        .await
        .expect("stream still exists")
        .purge()
        .await
        .is_err());
}

#[test]
fn help_does_not_print_nats_secrets() {
    let output = Command::new(env!("CARGO_BIN_EXE_sie-server-sidecar"))
        .arg("--help")
        .env("SIE_NATS_URL", "nats://url-user:url-secret@nats:4222")
        .env("SIE_NATS_USER", "sie-worker")
        .env("SIE_NATS_PASSWORD", "nats-password-secret")
        .output()
        .expect("run --help");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for secret in ["url-secret", "nats-password-secret"] {
        assert!(!help.contains(secret), "--help printed {secret}: {help}");
    }
}
