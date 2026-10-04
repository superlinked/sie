//! Integration test: the gateway on an authenticated NATS server.
//!
//! Runs `nats-server` with the configuration the Helm chart renders by
//! default (`tools/ci/fixtures/sie-cluster-nats.conf`, which
//! `tools/ci/tests/test_helm_render.py` keeps equal to the chart). Skips when
//! `nats-server` is not on `PATH`, unless `NATS_URL` is set as in CI, where a
//! missing binary fails the test.

use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use sie_gateway::config::NatsCredentials;
use sie_gateway::nats::manager::NatsManager;
use sie_gateway::queue::dlq::DlqListener;
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
    format!("p{}", uuid::Uuid::new_v4().simple())
}

struct NatsServer {
    child: Child,
    url: String,
    passwords: Passwords,
    dir: tempfile::TempDir,
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

async fn start_nats() -> Option<NatsServer> {
    start_nats_fixture("sie-cluster-nats.conf").await
}

async fn start_nats_fixture(fixture: &str) -> Option<NatsServer> {
    let Some(binary) = nats_server_binary() else {
        assert!(
            std::env::var_os("NATS_URL").is_none(),
            "nats-server must be on PATH when NATS_URL is set"
        );
        eprintln!("skipping: nats-server not on PATH");
        return None;
    };
    let config = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/ci/fixtures")
        .join(fixture);
    let dir = tempfile::tempdir().expect("tempdir");
    let log = std::fs::File::create(dir.path().join("nats.log")).expect("log file");
    let passwords = Passwords {
        config: random_password(),
        gateway: random_password(),
        worker: random_password(),
    };
    let child = Command::new(binary)
        .arg("-c")
        .arg(&config)
        .args(["-a", "127.0.0.1", "-p", "-1", "-m", "-1"])
        .arg("--ports_file_dir")
        .arg(dir.path())
        .arg("-sd")
        .arg(dir.path().join("jetstream"))
        .arg("-P")
        .arg(dir.path().join("nats.pid"))
        .env("SERVER_NAME", "sie-gateway-auth-test")
        .env("SIE_NATS_AUTH_CONFIG_PASSWORD", &passwords.config)
        .env("SIE_NATS_AUTH_GATEWAY_PASSWORD", &passwords.gateway)
        .env("SIE_NATS_AUTH_WORKER_PASSWORD", &passwords.worker)
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .expect("start nats-server");
    let mut server = NatsServer {
        child,
        url: String::new(),
        passwords,
        dir,
    };
    for _ in 0..600 {
        if let Some(url) = listening_url(server.dir.path()) {
            server.url = url;
            return Some(server);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let log = std::fs::read_to_string(server.dir.path().join("nats.log")).unwrap_or_default();
    panic!("nats-server did not start:\n{log}");
}

// Broker expiry is asynchronous; elapsed gateway time alone does not prove
// that its leader API has stopped exposing the five-second lease.
async fn wait_for_threshold_lease_expiry(leases: &jetstream::kv::Store) {
    tokio::time::timeout(Duration::from_secs(7), async {
        loop {
            match leases
                .stream
                .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
                .await
            {
                Err(error)
                    if error.kind() == jetstream::stream::RawMessageErrorKind::NoMessageFound =>
                {
                    return;
                }
                Ok(_) => tokio::time::sleep(Duration::from_millis(50)).await,
                Err(error) => panic!("failed to observe threshold lease expiry: {error}"),
            }
        }
    })
    .await
    .expect("threshold broker must expire the lease within five seconds plus scheduling margin");
}

/// The client URL from the `*.ports` file nats-server writes once it listens.
fn listening_url(dir: &Path) -> Option<String> {
    let ports = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .find(|entry| entry.path().extension().is_some_and(|ext| ext == "ports"))?;
    let ports: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ports.path()).ok()?).ok()?;
    let url = ports["nats"][0].as_str()?.to_string();
    let address = url.trim_start_matches("nats://").to_string();
    TcpStream::connect(address).ok().map(|_| url)
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

async fn dead_letters(context: &jetstream::Context) -> u64 {
    context
        .get_stream("DEAD_LETTERS")
        .await
        .expect("DEAD_LETTERS")
        .info()
        .await
        .expect("stream info")
        .state
        .messages
}

#[tokio::test]
async fn dlq_forwards_only_advisories_the_server_emits() {
    let Some(nats) = start_nats().await else {
        return;
    };
    let gateway = connect_as(&nats.url, "sie-gateway", &nats.passwords.gateway).await;
    let context = jetstream::new(gateway.clone());
    DlqListener::start(
        context.clone(),
        gateway.clone(),
        jetstream::stream::StorageType::Memory,
        1,
    )
    .await
    .expect("DLQ listener");
    context
        .create_stream(jetstream::stream::Config {
            name: "WORK_POOL_dlq".into(),
            subjects: vec!["sie.work.dlq.*.*.*".into()],
            retention: jetstream::stream::RetentionPolicy::WorkQueue,
            storage: jetstream::stream::StorageType::Memory,
            ..Default::default()
        })
        .await
        .expect("pool stream");

    let worker = connect_worker(&nats).await;
    let worker_context = jetstream::new(worker.clone());
    worker_context
        .create_stream(jetstream::stream::Config {
            name: "FAKE_ADVISORY".into(),
            subjects: vec!["sie.health.fake-advisory".into()],
            storage: jetstream::stream::StorageType::Memory,
            republish: Some(jetstream::stream::Republish {
                source: "sie.health.fake-advisory".into(),
                destination: "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.WORK_POOL_dlq.lane".into(),
                headers_only: false,
            }),
            ..Default::default()
        })
        .await
        .expect("the worker user may create streams");
    let forged = serde_json::json!({
        "type": "io.nats.jetstream.advisory.v1.max_deliver",
        "id": "forged",
        "stream": "WORK_POOL_dlq",
        "consumer": "lane",
        "stream_seq": 7,
        "deliveries": 3,
        "subject": "sie.work.dlq.l4.default.forged",
    });
    worker
        .publish(
            "sie.health.fake-advisory",
            serde_json::to_vec(&forged).expect("encode").into(),
        )
        .await
        .expect("publish");
    worker.flush().await.expect("worker flush");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let consumer: jetstream::consumer::PullConsumer = worker_context
        .get_stream("WORK_POOL_dlq")
        .await
        .expect("pool stream")
        .create_consumer(jetstream::consumer::pull::Config {
            durable_name: Some("lane".into()),
            filter_subject: "sie.work.dlq.l4.default.*".into(),
            ack_wait: Duration::from_secs(1),
            max_deliver: 1,
            ..Default::default()
        })
        .await
        .expect("the worker user may create consumers");
    context
        .publish("sie.work.dlq.l4.default.model", "work".into())
        .await
        .expect("publish work")
        .await
        .expect("work stored");
    let mut first = consumer
        .fetch()
        .max_messages(1)
        .expires(Duration::from_secs(2))
        .messages()
        .await
        .expect("fetch");
    first
        .next()
        .await
        .expect("delivered")
        .expect("work message");
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let mut again = consumer
        .fetch()
        .max_messages(1)
        .expires(Duration::from_millis(500))
        .messages()
        .await
        .expect("fetch");
    while again.next().await.is_some() {}

    let mut forwarded = 0;
    for _ in 0..50 {
        forwarded = dead_letters(&context).await;
        if forwarded > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(forwarded, 1, "the genuine advisory is forwarded");
    assert_eq!(
        dead_letters(&context).await,
        1,
        "the republished advisory is not forwarded"
    );
}

#[tokio::test]
async fn threshold_decisions_share_demand_and_require_current_sampler_authority() {
    use sha2::{Digest, Sha256};
    use sie_gateway::state::threshold_coordinator::{
        ThresholdCoordinator, ThresholdDecision, ThresholdError, ThresholdSampler, ThresholdTarget,
    };

    let Some(mut nats) = start_nats_fixture("sie-threshold-nats.conf").await else {
        return;
    };
    let gateway = connect_as(&nats.url, "sie-gateway", &nats.passwords.gateway).await;
    let context = jetstream::new(gateway.clone());
    let policy = serde_json::from_value(serde_json::json!({
        "policy":"threshold", "fallback_profile":"remote", "wake_above":1,
        "sleep_below":0.5, "window_s":1, "cooldown_s":1
    }))
    .unwrap();
    let target = ThresholdTarget::new("acme/chat", 1, &"a".repeat(64), &policy).unwrap();
    let first = Arc::new(
        ThresholdCoordinator::connect(&context, 1, vec![target.clone()])
            .await
            .unwrap(),
    );
    let second = Arc::new(
        ThresholdCoordinator::connect(&context, 1, vec![target])
            .await
            .unwrap(),
    );
    let mut owner = ThresholdSampler::default();
    let mut standby = ThresholdSampler::default();
    first.sample(&mut owner).await.unwrap();
    assert_eq!(
        second.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );

    let mut tasks = Vec::new();
    for replica in [first.clone(), second.clone()] {
        for _ in 0..4 {
            let replica = replica.clone();
            tasks.push(tokio::spawn(
                async move { replica.record_request("acme/chat") },
            ));
        }
    }
    for task in tasks {
        task.await.unwrap().unwrap();
    }
    let key = Sha256::digest(b"acme/chat")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let subject = format!("$KV.SIE_THRESHOLD_COUNTS.{key}");
    let stream = context.get_stream("KV_SIE_THRESHOLD_COUNTS").await.unwrap();
    tokio::time::sleep(Duration::from_millis(1050)).await;
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    first.sample(&mut owner).await.unwrap();
    let counter = stream
        .get_last_raw_message_by_subject(&subject)
        .await
        .unwrap();
    let counter: serde_json::Value = serde_json::from_slice(&counter.payload).unwrap();
    assert_eq!(
        counter["total"], 8,
        "standby and owner flush each local ingress once into the aggregate"
    );
    assert_eq!(
        second.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    for replica in [&first, &second] {
        replica.record_request("acme/chat").unwrap();
    }
    tokio::time::sleep(Duration::from_millis(1050)).await;
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    first.sample(&mut owner).await.unwrap();
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        first.decision("acme/chat").unwrap(),
        ThresholdDecision::WakeLocal
    );
    assert_eq!(
        second.decision("acme/chat").unwrap(),
        ThresholdDecision::WakeLocal
    );
    for _ in 0..2 {
        tokio::time::sleep(Duration::from_millis(1050)).await;
        first.sample(&mut owner).await.unwrap();
    }
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        second.decision("acme/chat").unwrap(),
        ThresholdDecision::Remote
    );

    // First sight provides no clock-free bound on the age of a lease.
    let newcomer = ThresholdCoordinator::connect(
        &context,
        1,
        vec![ThresholdTarget::new("acme/chat", 1, &"a".repeat(64), &policy).unwrap()],
    )
    .await
    .unwrap();
    let mut newcomer_sampler = ThresholdSampler::default();
    assert_eq!(
        newcomer.sample(&mut newcomer_sampler).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        newcomer.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    first.sample(&mut owner).await.unwrap();
    assert_eq!(
        newcomer.sample(&mut newcomer_sampler).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        newcomer.decision("acme/chat").unwrap(),
        ThresholdDecision::Remote
    );

    // A normal renewal keeps the acquisition term and accepts its earlier
    // decision even before the sampler publishes the next model decision.
    let leases = context.get_key_value("SIE_THRESHOLD_LEASE").await.unwrap();
    let old = leases
        .stream
        .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
        .await
        .unwrap();
    let old_lease: serde_json::Value = serde_json::from_slice(&old.payload).unwrap();
    leases
        .update("sampler", old.payload, old.sequence)
        .await
        .unwrap();
    let renewed_at = tokio::time::Instant::now();
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        second.decision("acme/chat").unwrap(),
        ThresholdDecision::Remote
    );
    let renewed = leases
        .stream
        .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
        .await
        .unwrap();
    let renewed: serde_json::Value = serde_json::from_slice(&renewed.payload).unwrap();
    assert_eq!(old_lease["term"], renewed["term"]);
    tokio::time::sleep_until(renewed_at + Duration::from_millis(1050)).await;
    assert_eq!(
        second.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );

    // Refreshing near expiry cannot extend the lease with another cache TTL.
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    tokio::time::sleep_until(renewed_at + Duration::from_millis(2500)).await;
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    tokio::time::sleep_until(renewed_at + Duration::from_millis(4250)).await;
    assert!(
        (renewed_at + Duration::from_secs(5))
            .saturating_duration_since(tokio::time::Instant::now())
            > Duration::from_millis(500),
        "near-expiry refresh requires a safe margin before broker lease expiry"
    );
    assert_eq!(
        second.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        second.decision("acme/chat").unwrap(),
        ThresholdDecision::Remote
    );
    tokio::time::sleep_until(renewed_at + Duration::from_millis(5100)).await;
    assert_eq!(
        second.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );

    // The broker, rather than either replica's wall clock, expires ownership.
    assert_eq!(
        first.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    wait_for_threshold_lease_expiry(&leases).await;
    second.sample(&mut standby).await.unwrap();
    assert_eq!(
        first.sample(&mut owner).await,
        Err(ThresholdError::Unavailable)
    );
    assert_eq!(
        first.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );

    let next = ThresholdTarget::new("acme/chat", 2, &"b".repeat(64), &policy).unwrap();
    let changed = ThresholdCoordinator::connect(&context, 1, vec![next])
        .await
        .unwrap();
    // A new generation takes over even when the model receives no request.
    changed.sample(&mut standby).await.unwrap();
    assert_eq!(
        first.sample(&mut owner).await,
        Err(ThresholdError::Generation)
    );
    assert_eq!(
        first.sample(&mut owner).await,
        Err(ThresholdError::Generation)
    );
    // Request recording is local even on a stale replica. Its flush refuses
    // the newer authority and cannot add this demand to the new generation.
    first.record_request("acme/chat").unwrap();
    assert_eq!(
        first.sample(&mut owner).await,
        Err(ThresholdError::Generation)
    );
    assert_eq!(
        first.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    let reset = stream
        .get_last_raw_message_by_subject(&subject)
        .await
        .unwrap();
    let reset: serde_json::Value = serde_json::from_slice(&reset.payload).unwrap();
    assert_eq!(reset["generation"], 2);
    assert_eq!(
        reset["total"], 0,
        "sampling does not manufacture ingress demand"
    );
    for _ in 0..2 {
        tokio::time::sleep(Duration::from_millis(1050)).await;
        changed.sample(&mut standby).await.unwrap();
    }
    assert_eq!(
        changed.decision("acme/chat").unwrap(),
        ThresholdDecision::Remote
    );

    // Broker errors discard sustained-window evidence and prevent renewal.
    let mut newest_target = ThresholdTarget::new("acme/chat", 3, &"c".repeat(64), &policy).unwrap();
    let newest = ThresholdCoordinator::connect(&context, 1, vec![newest_target.clone()])
        .await
        .unwrap();
    newest.sample(&mut standby).await.unwrap();
    for _ in 0..4 {
        newest.record_request("acme/chat").unwrap();
    }
    tokio::time::sleep(Duration::from_millis(1050)).await;
    newest.sample(&mut standby).await.unwrap();
    let counts = context.get_key_value("SIE_THRESHOLD_COUNTS").await.unwrap();
    counts
        .put(&key, bytes::Bytes::from_static(b"invalid"))
        .await
        .unwrap();
    assert_eq!(
        newest.sample(&mut standby).await,
        Err(ThresholdError::Untrusted)
    );
    assert_eq!(
        newest.sample(&mut standby).await,
        Err(ThresholdError::Untrusted)
    );
    let failed_lease = leases
        .stream
        .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
        .await
        .unwrap();
    let restored = serde_json::json!({"generation":2,"contract":"obsolete","total":0,
        "incarnation":uuid::Uuid::new_v4().to_string()});
    counts
        .put(&key, serde_json::to_vec(&restored).unwrap().into())
        .await
        .unwrap();
    // Repairing storage cannot restore the failed sampler's old ownership.
    // Until actual broker expiry it must refuse to renew, even with valid data.
    assert_eq!(
        newest.sample(&mut standby).await,
        Err(ThresholdError::Unavailable)
    );
    let refused_lease = leases
        .stream
        .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
        .await
        .unwrap();
    assert_eq!(
        refused_lease.sequence, failed_lease.sequence,
        "failed sampler must leave the existing lease revision unchanged"
    );
    wait_for_threshold_lease_expiry(&leases).await;
    // A single recovery call must succeed once the leader confirms expiry;
    // timeouts or unavailable storage remain test failures, not retry cases.
    newest.sample(&mut standby).await.unwrap();
    assert_eq!(
        newest.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    for tick in 0..2 {
        for _ in 0..4 {
            newest.record_request("acme/chat").unwrap();
        }
        tokio::time::sleep(Duration::from_millis(1050)).await;
        newest.sample(&mut standby).await.unwrap();
        if tick == 0 {
            assert_eq!(
                newest.decision("acme/chat"),
                Err(ThresholdError::Unavailable)
            );
        }
    }
    assert_eq!(
        newest.decision("acme/chat").unwrap(),
        ThresholdDecision::WakeLocal
    );

    // DEL and PURGE preserve their broker revisions for the next CAS.
    // Lease recreation also changes the UUID term, fencing old decisions.
    let before = leases
        .stream
        .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
        .await
        .unwrap();
    let before: serde_json::Value = serde_json::from_slice(&before.payload).unwrap();
    for purge in [false, true] {
        if purge {
            counts.purge(&key).await.unwrap();
            leases.purge("sampler").await.unwrap();
        } else {
            counts.delete(&key).await.unwrap();
            leases.delete("sampler").await.unwrap();
        }
        newest.sample(&mut standby).await.unwrap();
        let reset = counts
            .stream
            .get_last_raw_message_by_subject(&subject)
            .await
            .unwrap();
        let reset: serde_json::Value = serde_json::from_slice(&reset.payload).unwrap();
        assert_eq!(reset["total"], 0);
        assert_eq!(
            newest.decision("acme/chat"),
            Err(ThresholdError::Unavailable)
        );
        let recreated = leases
            .stream
            .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
            .await
            .unwrap();
        let recreated: serde_json::Value = serde_json::from_slice(&recreated.payload).unwrap();
        assert_ne!(before["term"], recreated["term"]);
    }

    // A long publication gap discards accumulated demand and window evidence.
    for _ in 0..100 {
        newest.record_request("acme/chat").unwrap();
    }
    tokio::time::sleep(Duration::from_millis(2600)).await;
    newest.sample(&mut standby).await.unwrap();
    let reset = counts
        .stream
        .get_last_raw_message_by_subject(&subject)
        .await
        .unwrap();
    let reset: serde_json::Value = serde_json::from_slice(&reset.payload).unwrap();
    assert_eq!(
        reset["total"], 0,
        "late demand is never presented as a fresh burst"
    );
    assert_eq!(
        newest.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );

    // Ingress racing an asynchronous drain stays in exactly one batch.
    let ((), sampled) = tokio::join!(
        async {
            for _ in 0..1000 {
                newest.record_request("acme/chat").unwrap();
                tokio::task::yield_now().await;
            }
        },
        newest.sample(&mut standby)
    );
    sampled.unwrap();
    newest.sample(&mut standby).await.unwrap();
    let total = counts
        .stream
        .get_last_raw_message_by_subject(&subject)
        .await
        .unwrap();
    let total: serde_json::Value = serde_json::from_slice(&total.payload).unwrap();
    assert_eq!(
        total["total"], 1000,
        "racing ingress is neither lost nor doubled"
    );

    // Duplicate, oversized or mixed-generation configurations never reach I/O.
    assert!(matches!(
        ThresholdCoordinator::connect(&context, 1, vec![]).await,
        Err(ThresholdError::Configuration)
    ));
    assert!(matches!(
        ThresholdCoordinator::connect(
            &context,
            1,
            vec![newest_target.clone(), newest_target.clone()]
        )
        .await,
        Err(ThresholdError::Configuration)
    ));
    assert!(matches!(
        ThresholdCoordinator::connect(&context, 1, vec![newest_target.clone(); 257]).await,
        Err(ThresholdError::Configuration)
    ));
    newest_target = ThresholdTarget::new("acme/other", 4, &"c".repeat(64), &policy).unwrap();
    assert!(matches!(
        ThresholdCoordinator::connect(
            &context,
            1,
            vec![
                ThresholdTarget::new("acme/chat", 3, &"c".repeat(64), &policy).unwrap(),
                newest_target
            ]
        )
        .await,
        Err(ThresholdError::Configuration)
    ));

    // Inference-component credentials have no authority on this endpoint.
    for (user, password) in [
        ("sie-worker", &nats.passwords.worker),
        ("sie-config", &nats.passwords.config),
    ] {
        assert!(async_nats::ConnectOptions::new()
            .user_and_password(user.into(), password.clone())
            .connect(&nats.url)
            .await
            .is_err());
    }
    // Local request handling remains available with the control broker down.
    nats.child.kill().unwrap();
    nats.child.wait().unwrap();
    newest.record_request("acme/chat").unwrap();
    assert_eq!(
        newest.decision("acme/chat"),
        Err(ThresholdError::Unavailable)
    );
    assert!(newest.sample(&mut standby).await.is_err());
}

#[tokio::test]
async fn threshold_counter_recreation_discards_evidence_even_when_totals_increase() {
    use sha2::{Digest, Sha256};
    use sie_gateway::state::threshold_coordinator::{
        ThresholdCoordinator, ThresholdDecision, ThresholdError, ThresholdSampler, ThresholdTarget,
    };
    let Some(nats) = start_nats_fixture("sie-threshold-nats.conf").await else {
        return;
    };
    let gateway = connect_as(&nats.url, "sie-gateway", &nats.passwords.gateway).await;
    let context = jetstream::new(gateway);
    let policy = serde_json::from_value(serde_json::json!({
        "policy":"threshold", "fallback_profile":"remote", "wake_above":5,
        "sleep_below":2, "window_s":1, "cooldown_s":1
    }))
    .unwrap();
    let target = ThresholdTarget::new("acme/chat", 1, &"a".repeat(64), &policy).unwrap();
    let owner = ThresholdCoordinator::connect(&context, 1, vec![target.clone()])
        .await
        .unwrap();
    let standby = ThresholdCoordinator::connect(&context, 1, vec![target])
        .await
        .unwrap();
    let mut owner_sampler = ThresholdSampler::default();
    let mut standby_sampler = ThresholdSampler::default();
    owner.sample(&mut owner_sampler).await.unwrap();
    assert_eq!(
        standby.sample(&mut standby_sampler).await,
        Err(ThresholdError::Unavailable)
    );
    let key = Sha256::digest(b"acme/chat")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let subject = format!("$KV.SIE_THRESHOLD_COUNTS.{key}");
    let counts = context.get_key_value("SIE_THRESHOLD_COUNTS").await.unwrap();
    let leases = context.get_key_value("SIE_THRESHOLD_LEASE").await.unwrap();
    for purge in [false, true] {
        for _ in 0..2 {
            for _ in 0..500 {
                owner.record_request("acme/chat").unwrap();
            }
            tokio::time::sleep(Duration::from_millis(1050)).await;
            owner.sample(&mut owner_sampler).await.unwrap();
        }
        assert_eq!(
            owner.decision("acme/chat").unwrap(),
            ThresholdDecision::WakeLocal
        );
        let prior = counts
            .stream
            .get_last_raw_message_by_subject(&subject)
            .await
            .unwrap();
        let prior: serde_json::Value = serde_json::from_slice(&prior.payload).unwrap();
        let lease = leases
            .stream
            .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
            .await
            .unwrap();
        let lease: serde_json::Value = serde_json::from_slice(&lease.payload).unwrap();
        if purge {
            counts.purge(&key).await.unwrap();
        } else {
            counts.delete(&key).await.unwrap();
        }
        // The new counter can already exceed the previous cumulative total.
        // Total rollback and a continuity integer alone cannot detect this.
        for _ in 0..=prior["total"].as_u64().unwrap() {
            standby.record_request("acme/chat").unwrap();
        }
        assert_eq!(
            standby.sample(&mut standby_sampler).await,
            Err(ThresholdError::Unavailable)
        );
        owner.sample(&mut owner_sampler).await.unwrap();
        assert_eq!(
            owner.decision("acme/chat"),
            Err(ThresholdError::Unavailable)
        );
        let recreated = counts
            .stream
            .get_last_raw_message_by_subject(&subject)
            .await
            .unwrap();
        let recreated: serde_json::Value = serde_json::from_slice(&recreated.payload).unwrap();
        assert!(recreated["total"].as_u64().unwrap() > prior["total"].as_u64().unwrap());
        assert_ne!(recreated["incarnation"], prior["incarnation"]);
        let renewed = leases
            .stream
            .get_last_raw_message_by_subject("$KV.SIE_THRESHOLD_LEASE.sampler")
            .await
            .unwrap();
        let renewed: serde_json::Value = serde_json::from_slice(&renewed.payload).unwrap();
        assert_eq!(
            renewed["term"], lease["term"],
            "counter recreation does not replace the lease"
        );
    }
}
