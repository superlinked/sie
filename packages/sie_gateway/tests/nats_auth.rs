//! Integration test: the gateway on an authenticated NATS server.
//!
//! Runs `nats-server` with the configuration the Helm chart renders by
//! default (`tools/ci/fixtures/sie-cluster-nats.conf`, which
//! `tools/ci/tests/test_helm_render.py` keeps equal to the chart). Skips when
//! `nats-server` is not on `PATH`, unless `NATS_URL` is set as in CI, where a
//! missing binary fails the test.

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use sie_gateway::config::NatsCredentials;
use sie_gateway::nats::manager::NatsManager;
use sie_gateway::queue::payload_store::DisabledPayloadStore;
use sie_gateway::queue::publisher::{WorkPublisher, WorkStreamConfig};
use sie_gateway::state::config_epoch::ConfigEpoch;
use sie_gateway::state::model_registry::ModelRegistry;

struct Passwords {
    config: String,
    gateway: String,
    worker: String,
}

fn random_password() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

struct NatsServer {
    child: Child,
    url: String,
    passwords: Passwords,
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
    let passwords = Passwords {
        config: random_password(),
        gateway: random_password(),
        worker: random_password(),
    };
    let child = Command::new(binary)
        .arg("-c")
        .arg(&config)
        .args(["-a", "127.0.0.1", "-p", &port.to_string()])
        .args(["-m", &free_port().to_string()])
        .arg("-sd")
        .arg(dir.path().join("jetstream"))
        .arg("-P")
        .arg(dir.path().join("nats.pid"))
        .env("SERVER_NAME", "sie-gateway-auth-test")
        .env("SIE_NATS_AUTH_CONFIG_PASSWORD", &passwords.config)
        .env("SIE_NATS_AUTH_GATEWAY_PASSWORD", &passwords.gateway)
        .env("SIE_NATS_AUTH_WORKER_PASSWORD", &passwords.worker)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start nats-server");
    let server = NatsServer {
        child,
        url: format!("nats://127.0.0.1:{port}"),
        passwords,
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

async fn connect_worker(nats: &NatsServer) -> async_nats::Client {
    async_nats::ConnectOptions::new()
        .user_and_password("sie-worker".to_string(), nats.passwords.worker.clone())
        .custom_inbox_prefix("_INBOX_WORKER")
        .connect(&nats.url)
        .await
        .expect("connect as sie-worker")
}

fn epoch_bump(epoch: u64) -> bytes::Bytes {
    serde_json::to_vec(&serde_json::json!({
        "router_id": "sie-config",
        "bundle_id": "default",
        "epoch": epoch,
        "bundle_config_hash": "",
    }))
    .expect("encode notification")
    .into()
}

async fn wait_for_epoch(epoch: &ConfigEpoch, want: u64) -> bool {
    for _ in 0..100 {
        if epoch.get() == want {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn anonymous_and_wrong_password_connections_are_refused() {
    let Some(nats) = start_nats().await else {
        return;
    };
    assert!(async_nats::ConnectOptions::new()
        .connect(&nats.url)
        .await
        .is_err());
    assert!(async_nats::ConnectOptions::new()
        .user_and_password("sie-gateway".into(), nats.passwords.worker.clone())
        .connect(&nats.url)
        .await
        .is_err());
}

#[tokio::test]
async fn gateway_takes_config_deltas_only_from_the_sie_config_user() {
    let Some(nats) = start_nats().await else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let epoch = ConfigEpoch::new();
    let manager = Arc::new(
        NatsManager::new_with_trusted_producers(
            "gateway-auth-test".to_string(),
            nats.url.clone(),
            Arc::new(ModelRegistry::new(dir.path(), dir.path(), false)),
            epoch.clone(),
            vec!["sie-config".to_string()],
        )
        .with_credentials(Some(NatsCredentials {
            user: "sie-gateway".to_string(),
            password: nats.passwords.gateway.clone(),
        })),
    );
    manager.connect().await.expect("gateway connect");
    let client = manager.get_client().await.expect("gateway client");
    for _ in 0..100 {
        if client.connection_state() == async_nats::connection::State::Connected {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    manager.start_subscription().await;
    client.flush().await.expect("gateway flush");

    let config = connect_as(&nats.url, "sie-config", &nats.passwords.config).await;
    config
        .publish("sie.config.models._all", epoch_bump(5))
        .await
        .expect("publish");
    config.flush().await.expect("config flush");
    assert!(
        wait_for_epoch(&epoch, 5).await,
        "the gateway applies a delta from sie-config"
    );

    let observer = connect_as(&nats.url, "sie-gateway", &nats.passwords.gateway).await;
    let mut seen = observer
        .subscribe("sie.config.models._all")
        .await
        .expect("subscribe");
    observer.flush().await.expect("observer flush");

    let worker = connect_worker(&nats).await;
    worker
        .publish("sie.config.models._all", epoch_bump(7))
        .await
        .expect("publish");
    worker.flush().await.expect("worker flush");

    jetstream::new(worker.clone())
        .create_stream(jetstream::stream::Config {
            name: "WORK_WORKER_republish".to_string(),
            subjects: vec!["sie.health.republish".to_string()],
            storage: jetstream::stream::StorageType::Memory,
            republish: Some(jetstream::stream::Republish {
                source: "sie.health.republish".to_string(),
                destination: "sie.config.models._all".to_string(),
                headers_only: false,
            }),
            ..Default::default()
        })
        .await
        .expect("the worker user may create streams");
    worker
        .publish("sie.health.republish", epoch_bump(9))
        .await
        .expect("publish");
    worker.flush().await.expect("worker flush");

    let delivered = tokio::time::timeout(Duration::from_secs(5), seen.next())
        .await
        .expect("the server republishes the stored bytes")
        .expect("subscription open");
    assert!(
        delivered
            .headers
            .as_ref()
            .is_some_and(|headers| headers.get("Nats-Stream").is_some()),
        "only the republished copy reaches the subject, with JetStream headers"
    );
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        epoch.get(),
        5,
        "neither the refused publish nor the republished copy is applied"
    );
}

#[tokio::test]
async fn gateway_user_manages_work_streams_but_cannot_delete_them() {
    let Some(nats) = start_nats().await else {
        return;
    };
    let gateway = connect_as(&nats.url, "sie-gateway", &nats.passwords.gateway).await;
    let publisher = WorkPublisher::new(
        jetstream::new(gateway.clone()),
        "gateway-auth-test".to_string(),
        Arc::new(DisabledPayloadStore),
        Duration::from_secs(30),
        1024,
        WorkStreamConfig {
            max_age: Duration::from_secs(300),
            storage: jetstream::stream::StorageType::Memory,
            num_replicas: 1,
        },
    );
    publisher
        .ensure_stream("authpool")
        .await
        .expect("the gateway user creates work streams");

    let context = jetstream::new(gateway);
    context
        .publish("sie.work.authpool.cpu.default.model", "work".into())
        .await
        .expect("publish work")
        .await
        .expect("work stored");
    assert!(context.delete_stream("WORK_POOL_authpool").await.is_err());
}
