//! Loopback test for the local-ingest mode.
//!
//! What this proves:
//!   * `sie-server-sidecar --ingest local` starts with **no NATS server**
//!     anywhere (the mode's whole point).
//!   * The UDS listener speaks the wire protocol v0.1: u32-LE frame + msgpack
//!     `{id, op, body}` envelopes; `ping`, `publish_work`, unknown-op
//!     error envelope.
//!   * `publish_work` WorkItem bytes flow through the real dispatcher →
//!     scheduler → IPC backend pipeline against the Python IPC harness
//!     (`sie_server._ipc_test_harness`) and come back as an ordered
//!     msgpack `WorkResult` array over the socket.
//!   * The admission re-check and the generate-op rejection answer
//!     typed error results instead of NAK-looping.
//!
//! Not proven here: real inference (the harness returns canned payloads)
//! — that is covered by the E2E verify step.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use sie_server_sidecar::work_types::{WorkItem, WorkResult};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

// ---------------------------------------------------------------------------
// Tool / path helpers (self-contained twin of integration_smoke.rs — Rust
// integration tests are separate crates, so no shared module without a
// common-file refactor that isn't worth it for two files).
// ---------------------------------------------------------------------------

fn which(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let p = dir.join(bin);
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        })
    })
}

fn skip_unless_tools_available() -> bool {
    if which("mise").is_none() && which("uv").is_none() {
        eprintln!("local_ingest_loopback: skipping — missing mise/uv on $PATH");
        return true;
    }
    // The fake engine is `sie_server._ipc_test_harness` from the Python
    // workspace. Rust-only CI jobs check out the repo without syncing the
    // Python venv, so probe importability the same way the harness is
    // spawned and skip (not fail) when the module cannot load.
    let (program, base_args) = harness_launcher();
    let importable = std::process::Command::new(program)
        .current_dir(workspace_root())
        .args(&base_args)
        .args(["python", "-c", "import sie_server._ipc_test_harness"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !importable {
        eprintln!(
            "local_ingest_loopback: skipping — sie_server._ipc_test_harness not \
             importable (Python workspace not synced in this environment)"
        );
        return true;
    }
    false
}

/// Program + base args used to run workspace Python (mise-managed uv when
/// available, bare uv otherwise) — shared by the skip probe and the harness.
fn harness_launcher() -> (&'static str, Vec<String>) {
    if which("mise").is_some() {
        (
            "mise",
            vec![
                "exec".to_string(),
                "--".to_string(),
                "uv".to_string(),
                "run".to_string(),
                "--no-sync".to_string(),
            ],
        )
    } else {
        ("uv", vec!["run".to_string(), "--no-sync".to_string()])
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("canonicalize workspace root")
}

struct ShortSocket {
    path: PathBuf,
    _dir: tempfile::TempDir,
}

impl ShortSocket {
    fn new(name: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("sieli-")
            .tempdir_in("/tmp")
            .expect("create short socket dir");
        let path = dir.path().join(name);
        Self { path, _dir: dir }
    }
}

// ---------------------------------------------------------------------------
// Python IPC harness (fake engine)
// ---------------------------------------------------------------------------

struct PythonHarness {
    child: Child,
    socket_path: PathBuf,
}

impl PythonHarness {
    async fn start(socket_path: PathBuf) -> Self {
        let (program, base_args) = harness_launcher();
        let mut cmd = Command::new(program);
        cmd.current_dir(workspace_root())
            .args(&base_args)
            .args([
                "python",
                "-m",
                "sie_server._ipc_test_harness",
                "--socket",
                socket_path.to_str().unwrap(),
                "--worker-id",
                "loopback-harness",
                "--log-level",
                "INFO",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().expect("spawn python harness");

        let stdout = child.stdout.take().expect("stdout");
        let mut reader = BufReader::new(stdout).lines();
        let ready_deadline = Instant::now() + Duration::from_secs(45);
        let mut ready = false;
        while Instant::now() < ready_deadline {
            match timeout(Duration::from_millis(500), reader.next_line()).await {
                Ok(Ok(Some(line))) => {
                    eprintln!("[python-harness] {line}");
                    if line.contains("HARNESS_READY") {
                        ready = true;
                        break;
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(e)) => panic!("read python stdout: {e}"),
                Err(_) => {
                    if let Some(status) = child.try_wait().expect("try_wait") {
                        panic!("python harness exited early: status={status:?}");
                    }
                }
            }
        }
        assert!(ready, "python harness never printed HARNESS_READY");
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                eprintln!("[python-harness] {line}");
            }
        });
        if let Some(stderr) = child.stderr.take() {
            let mut stderr_reader = BufReader::new(stderr).lines();
            tokio::spawn(async move {
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    eprintln!("[python-harness:stderr] {line}");
                }
            });
        }
        Self { child, socket_path }
    }
}

impl Drop for PythonHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

// ---------------------------------------------------------------------------
// Sidecar in local-ingest mode
// ---------------------------------------------------------------------------

struct LocalWorkerHarness {
    child: Child,
}

impl LocalWorkerHarness {
    fn spawn(ipc_socket: &std::path::Path, local_socket: &std::path::Path) -> Self {
        let exe = env!("CARGO_BIN_EXE_sie-server-sidecar");
        let mut cmd = Command::new(exe);
        cmd.env("SIE_SIDECAR_INGEST", "local")
            .env("SIE_SIDECAR_LOCAL_SOCKET", local_socket)
            .env("SIE_POOL", "default")
            .env("SIE_MACHINE_PROFILE", "l4")
            .env("SIE_BUNDLE", "default")
            .env("SIE_IPC_SOCKET_PATH", ipc_socket)
            .env("SIE_WORKER_PROBE_PORT", free_tcp_port().to_string())
            .env("SIE_WORKER_ID", "loopback-worker")
            .env("SIE_WORKER_PING_INTERVAL_MS", "500")
            .env("RUST_LOG", "info,sie_server_sidecar=debug")
            // Explicitly NO SIE_NATS_URL: local mode must not need it.
            .env_remove("SIE_NATS_URL")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let child = cmd.spawn().expect("spawn sie-server-sidecar (local)");
        Self { child }
    }
}

impl Drop for LocalWorkerHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn free_tcp_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

// ---------------------------------------------------------------------------
// Wire protocol v0.1 client
// ---------------------------------------------------------------------------

async fn connect_with_retry(path: &std::path::Path, budget: Duration) -> UnixStream {
    let deadline = Instant::now() + budget;
    loop {
        if let Ok(stream) = UnixStream::connect(path).await {
            return stream;
        }
        assert!(
            Instant::now() < deadline,
            "local ingest socket never came up: {}",
            path.display()
        );
        sleep(Duration::from_millis(100)).await;
    }
}

async fn send_request(stream: &mut UnixStream, id: u64, op: &str, body: rmpv::Value) {
    let map = rmpv::Value::Map(vec![
        (rmpv::Value::from("id"), rmpv::Value::from(id)),
        (rmpv::Value::from("op"), rmpv::Value::from(op)),
        (rmpv::Value::from("body"), body),
    ]);
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &map).unwrap();
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    stream.write_all(&frame).await.expect("write frame");
}

async fn read_response(stream: &mut UnixStream) -> rmpv::Value {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header).await.expect("read header");
    let len = u32::from_le_bytes(header) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await.expect("read payload");
    rmpv::decode::read_value(&mut payload.as_slice()).expect("decode response")
}

fn map_get<'a>(value: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    let rmpv::Value::Map(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find(|(k, _)| matches!(k, rmpv::Value::String(s) if s.as_str() == Some(key)))
        .map(|(_, v)| v)
}

fn work_item(request_id: &str, idx: u32, total: u32, op: &str, admission_pool: &str) -> WorkItem {
    WorkItem {
        work_item_id: format!("{request_id}.{idx}"),
        request_id: request_id.to_string(),
        item_index: idx,
        total_items: total,
        operation: op.into(),
        model_id: "BAAI/bge-m3".into(),
        profile_id: "default".into(),
        engine: String::new(),
        pool_name: "default".into(),
        admission_pool: admission_pool.into(),
        machine_profile: "l4".into(),
        item: Some(rmpv::Value::Map(vec![(
            rmpv::Value::from("text"),
            rmpv::Value::from(format!("loopback text {idx}")),
        )])),
        payload_ref: None,
        output_types: Some(vec!["dense".into()]),
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "loopback".into(),
        accepts_result_chunks: false,
        reply_subject: String::new(), // NATS-only field; unused on this path
        traceparent: None,
        tracestate: None,
        timestamp: 0.0,
    }
}

fn update_digest_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn publish_work_body(items: &[WorkItem], admission_pool: &str, timeout_ms: i64) -> rmpv::Value {
    let items_bytes = rmp_serde::to_vec_named(items).expect("encode items");
    let request_id = items.first().map(|i| i.request_id.as_str()).unwrap_or("");
    let endpoint = items.first().map(|i| i.operation.as_str()).unwrap_or("");
    let params = rmp_serde::to_vec_named(&serde_json::json!({})).unwrap();
    let dispatch_context = b"loopback-test";
    let fields: [&[u8]; 10] = [
        dispatch_context,
        b"default|l4|default",
        endpoint.as_bytes(),
        b"BAAI/bge-m3",
        b"",
        admission_pool.as_bytes(),
        b"",
        request_id.as_bytes(),
        &params,
        &items_bytes,
    ];
    let mut hasher = Sha256::new();
    hasher.update(b"sie-local-ingest-v1\0");
    for field in fields {
        update_digest_field(&mut hasher, field);
    }
    hasher.update(timeout_ms.to_be_bytes());
    let payload_digest = hasher.finalize().to_vec();

    rmpv::Value::Map(vec![
        (
            rmpv::Value::from("lane"),
            rmpv::Value::from("default|l4|default"),
        ),
        (rmpv::Value::from("endpoint"), rmpv::Value::from(endpoint)),
        (rmpv::Value::from("model"), rmpv::Value::from("BAAI/bge-m3")),
        (rmpv::Value::from("engine"), rmpv::Value::from("")),
        (
            rmpv::Value::from("admission_pool"),
            rmpv::Value::from(admission_pool),
        ),
        (
            rmpv::Value::from("bundle_config_hash"),
            rmpv::Value::from(""),
        ),
        (
            rmpv::Value::from("request_id"),
            rmpv::Value::from(request_id),
        ),
        (rmpv::Value::from("params"), rmpv::Value::Binary(params)),
        (rmpv::Value::from("items"), rmpv::Value::Binary(items_bytes)),
        (
            rmpv::Value::from("dispatch_context"),
            rmpv::Value::Binary(dispatch_context.to_vec()),
        ),
        (
            rmpv::Value::from("payload_digest"),
            rmpv::Value::Binary(payload_digest),
        ),
        (
            rmpv::Value::from("timeout_ms"),
            rmpv::Value::from(timeout_ms),
        ),
    ])
}

fn decode_results(response: &rmpv::Value) -> Vec<WorkResult> {
    assert_eq!(
        map_get(response, "ok").and_then(rmpv::Value::as_bool),
        Some(true),
        "publish_work not ok: {response:?}"
    );
    let body = map_get(response, "body").expect("body present");
    let results = map_get(body, "results").expect("results present");
    let rmpv::Value::Binary(bytes) = results else {
        panic!("results should be msgpack bin, got {results:?}");
    };
    rmp_serde::from_slice(bytes).expect("decode WorkResult array")
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_ingest_round_trips_publish_work_without_nats() {
    if skip_unless_tools_available() {
        return;
    }

    let ipc_sock = ShortSocket::new("ipc.sock");
    let ingest_sock = ShortSocket::new("ingest.sock");
    let _python = PythonHarness::start(ipc_sock.path.clone()).await;
    let _worker = LocalWorkerHarness::spawn(&ipc_sock.path, &ingest_sock.path);

    let mut stream = connect_with_retry(&ingest_sock.path, Duration::from_secs(30)).await;

    // 1. ping
    send_request(&mut stream, 1, "ping", rmpv::Value::Map(vec![])).await;
    let resp = read_response(&mut stream).await;
    assert_eq!(map_get(&resp, "id").and_then(rmpv::Value::as_u64), Some(1));
    assert_eq!(
        map_get(&resp, "ok").and_then(rmpv::Value::as_bool),
        Some(true)
    );

    // 2. publish_work: 3 encode items -> 3 ordered successful WorkResults.
    let items: Vec<WorkItem> = (0..3)
        .map(|i| work_item("li-req-1", i, 3, "encode", "default"))
        .collect();
    send_request(
        &mut stream,
        2,
        "publish_work",
        publish_work_body(&items, "default", 60_000),
    )
    .await;
    let resp = timeout(Duration::from_secs(60), read_response(&mut stream))
        .await
        .expect("publish_work answered within budget");
    assert_eq!(map_get(&resp, "id").and_then(rmpv::Value::as_u64), Some(2));
    let results = decode_results(&resp);
    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.success,
            "item {i} failed: {:?} {:?}",
            r.error_code, r.error
        );
        assert_eq!(r.item_index, i as u32, "results must be in input order");
        assert_eq!(r.work_item_id, format!("li-req-1.{i}"));
        assert_eq!(r.worker_id.as_deref(), Some("loopback-worker"));
        assert!(!r.result_msgpack.is_empty());
        // The harness cans a msgpack map with smoke=ok.
        let payload: rmpv::Value =
            rmp_serde::from_slice(&r.result_msgpack).expect("payload decodes");
        assert_eq!(
            map_get(&payload, "smoke").and_then(|v| v.as_str().map(str::to_string)),
            Some("ok".to_string())
        );
    }

    // 3. Admission re-check: foreign pool -> typed errors, not a hang.
    let foreign: Vec<WorkItem> = (0..2)
        .map(|i| work_item("li-req-2", i, 2, "encode", "other-pool"))
        .collect();
    send_request(
        &mut stream,
        3,
        "publish_work",
        publish_work_body(&foreign, "other-pool", 30_000),
    )
    .await;
    let resp = timeout(Duration::from_secs(30), read_response(&mut stream))
        .await
        .expect("admission rejection answered");
    let results = decode_results(&resp);
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(!r.success);
        assert_eq!(r.error_code.as_deref(), Some("pool_admission_rejected"));
    }

    // 4. generate op -> typed bad_operation error (streaming lands later).
    let gen = vec![work_item("li-req-3", 0, 1, "generate", "default")];
    send_request(
        &mut stream,
        4,
        "publish_work",
        publish_work_body(&gen, "default", 30_000),
    )
    .await;
    let resp = timeout(Duration::from_secs(30), read_response(&mut stream))
        .await
        .expect("generate rejection answered");
    let results = decode_results(&resp);
    assert_eq!(results.len(), 1);
    assert!(!results[0].success);
    assert_eq!(results[0].error_code.as_deref(), Some("bad_operation"));

    // 5. unknown op -> ok=false envelope, connection stays usable.
    send_request(&mut stream, 5, "warp", rmpv::Value::Map(vec![])).await;
    let resp = read_response(&mut stream).await;
    assert_eq!(
        map_get(&resp, "ok").and_then(rmpv::Value::as_bool),
        Some(false)
    );
    assert!(map_get(&resp, "error")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .contains("unknown op"));
    send_request(&mut stream, 6, "ping", rmpv::Value::Map(vec![])).await;
    let resp = read_response(&mut stream).await;
    assert_eq!(
        map_get(&resp, "ok").and_then(rmpv::Value::as_bool),
        Some(true)
    );

    // 6. Concurrent publish_work calls on separate connections coalesce in
    //    the same scheduler without cross-talk.
    let path_a = ingest_sock.path.clone();
    let path_b = ingest_sock.path.clone();
    let (ra, rb) = tokio::join!(
        async move {
            let mut s = connect_with_retry(&path_a, Duration::from_secs(5)).await;
            let items: Vec<WorkItem> = (0..4)
                .map(|i| work_item("li-conc-a", i, 4, "encode", "default"))
                .collect();
            send_request(
                &mut s,
                10,
                "publish_work",
                publish_work_body(&items, "default", 60_000),
            )
            .await;
            decode_results(&read_response(&mut s).await)
        },
        async move {
            let mut s = connect_with_retry(&path_b, Duration::from_secs(5)).await;
            let items: Vec<WorkItem> = (0..4)
                .map(|i| work_item("li-conc-b", i, 4, "encode", "default"))
                .collect();
            send_request(
                &mut s,
                11,
                "publish_work",
                publish_work_body(&items, "default", 60_000),
            )
            .await;
            decode_results(&read_response(&mut s).await)
        },
    );
    assert_eq!(ra.len(), 4);
    assert_eq!(rb.len(), 4);
    assert!(ra.iter().all(|r| r.success && r.request_id == "li-conc-a"));
    assert!(rb.iter().all(|r| r.success && r.request_id == "li-conc-b"));
}
