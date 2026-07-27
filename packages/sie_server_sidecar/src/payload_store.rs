//! Payload store — read-only side.
//!
//! Workers consume payload_refs written by the gateway
//! (`packages/sie_gateway/src/queue/payload_store.rs`). Only `get` is needed
//! on this side; the gateway writes/deletes.
//!
//! The ref format is:
//!   - local: filesystem path under the worker's configured `base_dir`
//!     (absolute paths are only accepted when they are inside `base_dir`;
//!     relative paths are resolved against it). When no `base_dir` is
//!     configured the reader is disabled for safety.
//!   - cloud: `s3://…` / `gs://…` / `abfs://…` / `abfss://…` URL
//!     (handled when the `cloud-storage` feature is enabled)
//!
//! Security: on Linux, the configured directory is pinned as a file
//! descriptor and refs are opened beneath it with `openat2`, without
//! following symlinks. Only nonblocking regular-file reads are accepted.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[cfg(target_os = "linux")]
use std::path::Path;

use async_trait::async_trait;
#[cfg(target_os = "linux")]
use rustix::fd::OwnedFd;
#[cfg(target_os = "linux")]
use rustix::fs::{FileType, Mode, OFlags, ResolveFlags};
use thiserror::Error;
#[cfg(target_os = "linux")]
use tokio::io::AsyncReadExt;
#[cfg(any(target_os = "linux", feature = "cloud-storage"))]
use tracing::debug;

use crate::observability::metrics::SidecarTelemetry;

#[cfg(feature = "cloud-storage")]
use futures_util::StreamExt;
#[cfg(feature = "cloud-storage")]
use object_store::ObjectStoreExt;

const OBJECT_STORE_SCHEMES: &[&str] = &["s3://", "gs://", "abfs://", "abfss://"];
#[cfg(any(target_os = "linux", feature = "cloud-storage"))]
const MAX_PAYLOAD_BYTES: u64 = crate::prep::media::MAX_OFFLOADED_PAYLOAD_BYTES as u64;

fn object_store_scheme(url: &str) -> Option<&'static str> {
    OBJECT_STORE_SCHEMES
        .iter()
        .copied()
        .find(|scheme| url.starts_with(scheme))
}

#[derive(Debug, Error)]
pub enum PayloadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid payload ref: {0}")]
    InvalidRef(String),
    #[error("payload is too large ({actual} bytes); maximum is {max} bytes")]
    TooLarge { actual: u64, max: u64 },
    #[error("unsupported payload scheme: {0}")]
    Unsupported(String),
    #[error("object store: {0}")]
    ObjectStore(String),
}

#[async_trait]
pub trait PayloadStore: Send + Sync {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError>;
}

/// Decorate the active payload store with one semantic OTel observation per
/// fetch. Concrete stores remain telemetry-free and the facade retains the
/// closed error vocabulary.
pub fn with_telemetry(
    inner: Arc<dyn PayloadStore>,
    telemetry: SidecarTelemetry,
) -> Arc<dyn PayloadStore> {
    Arc::new(TelemetryPayloadStore { inner, telemetry })
}

struct TelemetryPayloadStore {
    inner: Arc<dyn PayloadStore>,
    telemetry: SidecarTelemetry,
}

#[async_trait]
impl PayloadStore for TelemetryPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        if !self.telemetry.is_enabled() {
            return self.inner.get(payload_ref).await;
        }
        let started = Instant::now();
        let result = self.inner.get(payload_ref).await;
        match &result {
            Ok(bytes) => self.telemetry.payload_fetch_completed(
                "success",
                "none",
                started.elapsed(),
                Some(bytes.len()),
            ),
            Err(error) => self.telemetry.payload_fetch_completed(
                "error",
                payload_error_reason(error),
                started.elapsed(),
                None,
            ),
        }
        result
    }
}

fn payload_error_reason(error: &PayloadError) -> &'static str {
    match error {
        PayloadError::Io(error) => match error.kind() {
            std::io::ErrorKind::NotFound => "not_found",
            std::io::ErrorKind::PermissionDenied => "permission_denied",
            _ => "io",
        },
        PayloadError::InvalidRef(_) => "invalid_ref",
        PayloadError::TooLarge { .. } => "too_large",
        PayloadError::Unsupported(_) => "unsupported",
        PayloadError::ObjectStore(_) => "object_store",
    }
}

/// Local filesystem reader.
///
/// * When `base_dir` is `Some`, `payload_ref` is opened relative to a pinned
///   directory capability. Symlink traversal and non-regular files are rejected.
/// * When `base_dir` is `None`, the store rejects all reads — this matches the
///   gateway's refusal to publish payload refs without `SIE_PAYLOAD_STORE_URL`.
pub struct LocalPayloadStore {
    #[cfg(target_os = "linux")]
    base_dir: Option<PathBuf>,
    #[cfg(target_os = "linux")]
    base_dir_fd: Option<OwnedFd>,
    #[cfg(target_os = "linux")]
    base_open_error: Option<String>,
}

impl LocalPayloadStore {
    pub fn new(base_dir: Option<impl Into<PathBuf>>) -> Self {
        let base_dir = base_dir.map(Into::into);
        #[cfg(target_os = "linux")]
        {
            let Some(configured_base_dir) = base_dir else {
                return Self {
                    base_dir: None,
                    base_dir_fd: None,
                    base_open_error: None,
                };
            };
            let configured_base_dir = if configured_base_dir.is_absolute() {
                configured_base_dir
            } else {
                match std::env::current_dir() {
                    Ok(current_dir) => current_dir.join(configured_base_dir),
                    Err(error) => {
                        return Self {
                            base_dir: None,
                            base_dir_fd: None,
                            base_open_error: Some(format!(
                                "failed to resolve local payload base directory: {error}"
                            )),
                        };
                    }
                }
            };
            match rustix::fs::open(
                &configured_base_dir,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            ) {
                Ok(base_dir_fd) => Self {
                    base_dir: Some(configured_base_dir),
                    base_dir_fd: Some(base_dir_fd),
                    base_open_error: None,
                },
                Err(error) => Self {
                    base_dir: Some(configured_base_dir),
                    base_dir_fd: None,
                    base_open_error: Some(format!(
                        "failed to open local payload base directory: {error}"
                    )),
                },
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = base_dir;
            Self {}
        }
    }

    #[cfg(target_os = "linux")]
    fn relative_ref(&self, payload_ref: &str) -> Result<PathBuf, PayloadError> {
        if payload_ref.is_empty() {
            return Err(PayloadError::InvalidRef(payload_ref.to_string()));
        }
        let Some(base_dir) = &self.base_dir else {
            return Err(PayloadError::InvalidRef(format!(
                "payload store not configured; cannot resolve {payload_ref}"
            )));
        };
        let raw = Path::new(payload_ref);
        let relative = if raw.is_absolute() {
            raw.strip_prefix(base_dir).map_err(|_| {
                PayloadError::InvalidRef(format!(
                    "payload ref {payload_ref} escapes base_dir {}",
                    base_dir.display()
                ))
            })?
        } else {
            raw
        };
        let mut normalized = PathBuf::new();
        for component in relative.components() {
            match component {
                std::path::Component::Normal(value) => normalized.push(value),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_) => {
                    return Err(PayloadError::InvalidRef(format!(
                        "payload ref {payload_ref} escapes base_dir {}",
                        base_dir.display()
                    )));
                }
            }
        }
        if normalized.as_os_str().is_empty() {
            return Err(PayloadError::InvalidRef(format!(
                "payload ref {payload_ref} does not identify a file"
            )));
        }
        Ok(normalized)
    }
}

#[async_trait]
impl PayloadStore for LocalPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        #[cfg(target_os = "linux")]
        {
            if let Some(error) = &self.base_open_error {
                return Err(PayloadError::InvalidRef(error.clone()));
            }
            let relative = self.relative_ref(payload_ref)?;
            let base_dir_fd = self.base_dir_fd.as_ref().ok_or_else(|| {
                PayloadError::InvalidRef(
                    "payload store not configured; cannot resolve payload ref".to_owned(),
                )
            })?;
            let file_fd = rustix::fs::openat2(
                base_dir_fd,
                &relative,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
                ResolveFlags::BENEATH | ResolveFlags::NO_MAGICLINKS | ResolveFlags::NO_SYMLINKS,
            )
            .map_err(|error| match error {
                rustix::io::Errno::NOSYS => PayloadError::Unsupported(
                    "local payload store requires Linux openat2 support".to_owned(),
                ),
                rustix::io::Errno::NOENT => PayloadError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "payload ref does not exist",
                )),
                rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR | rustix::io::Errno::XDEV => {
                    PayloadError::InvalidRef("payload ref traverses a disallowed path".to_owned())
                }
                _ => PayloadError::Io(std::io::Error::from(error)),
            })?;
            let stat = rustix::fs::fstat(&file_fd)
                .map_err(|error| PayloadError::Io(std::io::Error::from(error)))?;
            if !FileType::from_raw_mode(stat.st_mode).is_file() {
                return Err(PayloadError::InvalidRef(
                    "payload ref does not identify a regular file".to_owned(),
                ));
            }
            let size = u64::try_from(stat.st_size).map_err(|_| {
                PayloadError::InvalidRef("payload file has an invalid size".to_owned())
            })?;
            if size > MAX_PAYLOAD_BYTES {
                return Err(PayloadError::TooLarge {
                    actual: size,
                    max: MAX_PAYLOAD_BYTES,
                });
            }

            let file = tokio::fs::File::from_std(std::fs::File::from(file_fd));
            let mut bytes = Vec::with_capacity(size as usize);
            file.take(MAX_PAYLOAD_BYTES + 1)
                .read_to_end(&mut bytes)
                .await?;
            if bytes.len() as u64 > MAX_PAYLOAD_BYTES {
                return Err(PayloadError::TooLarge {
                    actual: bytes.len() as u64,
                    max: MAX_PAYLOAD_BYTES,
                });
            }
            debug!(
                path = %self.base_dir.as_ref().expect("base checked").join(relative).display(),
                bytes = bytes.len(),
                "read local payload"
            );
            Ok(bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = payload_ref;
            Err(PayloadError::Unsupported(
                "local payload store requires Linux openat2".to_owned(),
            ))
        }
    }
}

#[cfg(feature = "cloud-storage")]
pub struct ObjectPayloadStore {
    store: Box<dyn object_store::ObjectStore>,
    prefix: object_store::path::Path,
    url_prefix: String,
    scheme: &'static str,
}

#[cfg(feature = "cloud-storage")]
impl ObjectPayloadStore {
    pub fn new_s3(path: &str) -> Result<Self, PayloadError> {
        use object_store::aws::AmazonS3Builder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path)?;
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                PayloadError::Unsupported(format!("failed to build S3 payload store: {e}"))
            })?;
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix: format!("s3://{}", path.trim_end_matches('/')),
            scheme: "s3://",
        })
    }

    pub fn new_gcs(path: &str) -> Result<Self, PayloadError> {
        use object_store::gcp::GoogleCloudStorageBuilder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path)?;
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                PayloadError::Unsupported(format!("failed to build GCS payload store: {e}"))
            })?;
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix: format!("gs://{}", path.trim_end_matches('/')),
            scheme: "gs://",
        })
    }

    pub fn new_azure(path: &str) -> Result<Self, PayloadError> {
        Self::new_azure_with_scheme(path, "abfs://")
    }

    fn new_azure_with_scheme(path: &str, scheme: &'static str) -> Result<Self, PayloadError> {
        use object_store::azure::MicrosoftAzureBuilder;

        let (container, account, prefix) = Self::parse_azure_path(path)?;
        let store = MicrosoftAzureBuilder::from_env()
            .with_account(&account)
            .with_container_name(&container)
            .build()
            .map_err(|e| {
                PayloadError::Unsupported(format!("failed to build Azure payload store: {e}"))
            })?;
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix: format!("{}{}", scheme, path.trim_end_matches('/')),
            scheme,
        })
    }

    fn parse_bucket_prefix(path: &str) -> Result<(String, String), PayloadError> {
        let path = path.trim_matches('/');
        let Some((bucket, prefix)) = path.split_once('/') else {
            if path.is_empty() {
                return Err(PayloadError::InvalidRef(
                    "payload store bucket is empty".into(),
                ));
            }
            return Ok((path.to_string(), String::new()));
        };
        if bucket.is_empty() {
            return Err(PayloadError::InvalidRef(
                "payload store bucket is empty".into(),
            ));
        }
        Ok((bucket.to_string(), prefix.trim_matches('/').to_string()))
    }

    fn parse_azure_path(path: &str) -> Result<(String, String, String), PayloadError> {
        let path = path.trim_matches('/');
        let Some((authority, prefix)) = path.split_once('/') else {
            return Self::parse_azure_authority(path)
                .map(|(container, account)| (container, account, String::new()));
        };
        let (container, account) = Self::parse_azure_authority(authority)?;
        Ok((container, account, prefix.trim_matches('/').to_string()))
    }

    fn parse_azure_authority(authority: &str) -> Result<(String, String), PayloadError> {
        let Some((container, host)) = authority.split_once('@') else {
            return Err(PayloadError::InvalidRef(
                "Azure payload store URL must use <container>@<account>.dfs.core.windows.net"
                    .into(),
            ));
        };
        if container.is_empty() || host.is_empty() {
            return Err(PayloadError::InvalidRef(
                "Azure payload store container and account must be non-empty".into(),
            ));
        }
        let account = host
            .strip_suffix(".dfs.core.windows.net")
            .or_else(|| host.strip_suffix(".blob.core.windows.net"))
            .unwrap_or(host);
        if account.is_empty() {
            return Err(PayloadError::InvalidRef(
                "Azure payload store account must be non-empty".into(),
            ));
        }
        Ok((container.to_string(), account.to_string()))
    }

    fn relative_key<'a>(&self, payload_ref: &'a str) -> Result<&'a str, PayloadError> {
        if payload_ref.is_empty() {
            return Err(PayloadError::InvalidRef(payload_ref.to_string()));
        }

        let key = if payload_ref.starts_with(self.scheme) {
            let prefix = self.url_prefix.trim_end_matches('/');
            let Some(rest) = payload_ref.strip_prefix(prefix) else {
                return Err(PayloadError::InvalidRef(format!(
                    "payload ref {payload_ref} is outside configured store {prefix}"
                )));
            };
            let Some(rest) = rest.strip_prefix('/') else {
                return Err(PayloadError::InvalidRef(format!(
                    "payload ref {payload_ref} is outside configured store {prefix}"
                )));
            };
            rest
        } else if object_store_scheme(payload_ref).is_some() {
            return Err(PayloadError::InvalidRef(format!(
                "payload ref {payload_ref} uses a different object-store scheme"
            )));
        } else {
            payload_ref
        };

        Self::validate_relative_key(key)?;
        Ok(key)
    }

    fn validate_relative_key(key: &str) -> Result<(), PayloadError> {
        if key.is_empty()
            || key.starts_with('/')
            || key.contains('\\')
            || key
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(PayloadError::InvalidRef(key.to_string()));
        }
        Ok(())
    }

    fn object_key(&self, key: &str) -> object_store::path::Path {
        if self.prefix.as_ref().is_empty() {
            object_store::path::Path::from(key)
        } else {
            object_store::path::Path::from(format!("{}/{}", self.prefix, key))
        }
    }
}

#[cfg(feature = "cloud-storage")]
#[async_trait]
impl PayloadStore for ObjectPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        let key = self.relative_key(payload_ref)?;
        let object_key = self.object_key(key);
        let result = self.store.get(&object_key).await.map_err(|e| match e {
            object_store::Error::NotFound { .. } => {
                PayloadError::InvalidRef(format!("payload ref not found: {payload_ref}"))
            }
            other => PayloadError::ObjectStore(format!("read payload {object_key}: {other}")),
        })?;
        if result.meta.size > MAX_PAYLOAD_BYTES {
            return Err(PayloadError::TooLarge {
                actual: result.meta.size,
                max: MAX_PAYLOAD_BYTES,
            });
        }
        let expected_size = result.meta.size;
        let mut stream = result.into_stream();
        let mut bytes = Vec::with_capacity(expected_size as usize);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                PayloadError::ObjectStore(format!("read payload {object_key}: {e}"))
            })?;
            let actual = bytes.len() as u64 + chunk.len() as u64;
            if actual > MAX_PAYLOAD_BYTES {
                return Err(PayloadError::TooLarge {
                    actual,
                    max: MAX_PAYLOAD_BYTES,
                });
            }
            bytes.extend_from_slice(&chunk);
        }
        debug!(key = %object_key, bytes = bytes.len(), "read object-store payload");
        Ok(bytes)
    }
}

/// Factory — mirrors the gateway's `create_payload_store`. Returns an
/// opaque trait object so call sites don't care which backend is in use.
///
/// URL handling:
/// * `None` or an empty string → local store with no base_dir (disabled,
///   rejects all reads). Pods without offload configured stay safe.
/// * `s3://…` / `gs://…` / `abfs://…` / `abfss://…` with the
///   `cloud-storage` feature → object storage reader.
/// * Those object-store URLs without the feature → we can't read from
///   object storage, and we MUST NOT silently treat the URL string as a
///   filesystem path (`LocalPayloadStore::new(Some("s3://…"))` canonicalizes
///   to a broken, confusing error on every request). Return an explicit
///   `Unsupported` so the worker fails fast at startup instead of NAKing
///   every item at runtime.
/// * `file://path` → strip scheme, treat as local filesystem.
/// * Anything else → treated as a local filesystem path.
pub async fn create_payload_store(
    url: Option<&str>,
) -> Result<std::sync::Arc<dyn PayloadStore>, PayloadError> {
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return Ok(std::sync::Arc::new(LocalPayloadStore::new(None::<PathBuf>)));
    };

    if let Some(scheme) = object_store_scheme(url) {
        #[cfg(feature = "cloud-storage")]
        {
            if let Some(rest) = url.strip_prefix("s3://") {
                return Ok(std::sync::Arc::new(ObjectPayloadStore::new_s3(rest)?));
            }
            if let Some(rest) = url.strip_prefix("gs://") {
                return Ok(std::sync::Arc::new(ObjectPayloadStore::new_gcs(rest)?));
            }
            if let Some(rest) = url.strip_prefix(scheme) {
                return Ok(std::sync::Arc::new(
                    ObjectPayloadStore::new_azure_with_scheme(rest, scheme)?,
                ));
            }
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            return Err(PayloadError::Unsupported(format!(
                "payload store URL {url} with scheme {scheme} requires the 'cloud-storage' feature; \
                 rebuild the worker with --features cloud-storage or point \
                 SIE_PAYLOAD_STORE_URL at a local path"
            )));
        }
    }

    let path = url.strip_prefix("file://").unwrap_or(url);
    Ok(std::sync::Arc::new(LocalPayloadStore::new(Some(path))))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cloud-storage")]
    use object_store::ObjectStoreExt;
    #[cfg(feature = "cloud-storage")]
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "cloud-storage")]
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    #[cfg(feature = "cloud-storage")]
    #[derive(Debug)]
    struct CountingObjectStore {
        inner: object_store::memory::InMemory,
        get_calls: std::sync::Arc<AtomicUsize>,
    }

    #[cfg(feature = "cloud-storage")]
    impl std::fmt::Display for CountingObjectStore {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("counting object store")
        }
    }

    #[cfg(feature = "cloud-storage")]
    #[async_trait]
    impl object_store::ObjectStore for CountingObjectStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            options: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, options).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, options).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.get_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: futures_util::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures_util::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures_util::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[cfg(feature = "cloud-storage")]
    const AZURE_ENV_KEYS: &[&str] = &[
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_ACCESS_KEY",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_CLIENT_SECRET",
        "AZURE_STORAGE_TENANT_ID",
        "AZURE_TENANT_ID",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_STORAGE_AUTHORITY_HOST",
        "AZURE_AUTHORITY_HOST",
        "AZURE_STORAGE_TOKEN",
        "AZURE_USE_AZURE_CLI",
    ];

    #[cfg(feature = "cloud-storage")]
    fn azure_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "cloud-storage")]
    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    #[cfg(feature = "cloud-storage")]
    impl EnvGuard {
        fn set(vars: &[(&'static str, String)]) -> Self {
            let saved = AZURE_ENV_KEYS
                .iter()
                .map(|key| (*key, std::env::var(key).ok()))
                .collect();
            for key in AZURE_ENV_KEYS {
                std::env::remove_var(key);
            }
            for (key, value) in vars {
                std::env::set_var(key, value);
            }
            Self { saved }
        }
    }

    #[cfg(feature = "cloud-storage")]
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_absolute_path_inside_base_dir() {
        // Gateway writes absolute paths (e.g. /var/cache/sie/uuid.bin);
        // those are accepted as long as they sit under the configured
        // base_dir.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("req-1_0.bin");
        tokio::fs::write(&path, b"hello").await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let got = store.get(path.to_str().unwrap()).await.unwrap();
        assert_eq!(got, b"hello");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_relative_with_base() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("req-2_0.bin");
        tokio::fs::write(&path, b"world").await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let got = store.get("req-2_0.bin").await.unwrap();
        assert_eq!(got, b"world");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_accepts_empty_and_exact_cap_payloads() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("empty.bin"), [])
            .await
            .unwrap();
        let exact_path = dir.path().join("exact.bin");
        let exact_file = tokio::fs::File::create(&exact_path).await.unwrap();
        exact_file.set_len(MAX_PAYLOAD_BYTES).await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        assert!(store.get("empty.bin").await.unwrap().is_empty());
        assert_eq!(
            store.get("exact.bin").await.unwrap().len() as u64,
            MAX_PAYLOAD_BYTES
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_rejects_final_and_intermediate_symlinks() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().unwrap();
        let base = root.path().join("base");
        let outside = root.path().join("outside");
        tokio::fs::create_dir_all(&base).await.unwrap();
        tokio::fs::create_dir_all(&outside).await.unwrap();
        tokio::fs::write(outside.join("secret.bin"), b"secret")
            .await
            .unwrap();
        symlink(outside.join("secret.bin"), base.join("final-link.bin")).unwrap();
        symlink(&outside, base.join("directory-link")).unwrap();

        let store = LocalPayloadStore::new(Some(base));
        for payload_ref in ["final-link.bin", "directory-link/secret.bin"] {
            let error = store.get(payload_ref).await.unwrap_err();
            assert!(
                matches!(error, PayloadError::InvalidRef(_)),
                "expected symlink ref {payload_ref} to be rejected, got {error:?}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_rejects_fifo_without_blocking() {
        let dir = TempDir::new().unwrap();
        let dir_fd = rustix::fs::open(
            dir.path(),
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .unwrap();
        rustix::fs::mkfifoat(&dir_fd, "payload.pipe", Mode::RUSR | Mode::WUSR).unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(1), store.get("payload.pipe"))
                .await
                .expect("FIFO payload lookup must not block");
        assert!(matches!(result, Err(PayloadError::InvalidRef(_))));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_uses_directory_pinned_at_initialization() {
        let root = TempDir::new().unwrap();
        let configured = root.path().join("payloads");
        let pinned = root.path().join("pinned");
        tokio::fs::create_dir(&configured).await.unwrap();
        tokio::fs::write(configured.join("payload.bin"), b"original")
            .await
            .unwrap();
        let store = LocalPayloadStore::new(Some(configured.clone()));

        tokio::fs::rename(&configured, &pinned).await.unwrap();
        tokio::fs::create_dir(&configured).await.unwrap();
        tokio::fs::write(configured.join("payload.bin"), b"replacement")
            .await
            .unwrap();

        assert_eq!(store.get("payload.bin").await.unwrap(), b"original");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_rejects_oversized_payload_before_reading() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("oversized.bin");
        let file = tokio::fs::File::create(&path).await.unwrap();
        file.set_len(MAX_PAYLOAD_BYTES + 1).await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let err = store.get("oversized.bin").await.unwrap_err();
        assert!(matches!(
            err,
            PayloadError::TooLarge {
                actual,
                max: MAX_PAYLOAD_BYTES
            } if actual == MAX_PAYLOAD_BYTES + 1
        ));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn missing_ref_surfaces_not_found_io() {
        let dir = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let err = store.get("missing.bin").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound),
            "expected NotFound IO for missing payload, got {err:?}"
        );
    }

    #[test]
    fn telemetry_error_reasons_are_closed_and_semantic() {
        assert_eq!(
            payload_error_reason(&PayloadError::Io(std::io::Error::from(
                std::io::ErrorKind::NotFound
            ))),
            "not_found"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::Io(std::io::Error::from(
                std::io::ErrorKind::PermissionDenied
            ))),
            "permission_denied"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::Io(std::io::Error::other("disk"))),
            "io"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::InvalidRef("bad".into())),
            "invalid_ref"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::TooLarge { actual: 2, max: 1 }),
            "too_large"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::Unsupported("scheme".into())),
            "unsupported"
        );
        assert_eq!(
            payload_error_reason(&PayloadError::ObjectStore("remote".into())),
            "object_store"
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn local_get_without_base_dir_is_disabled() {
        // Safety: without a configured base_dir, the reader is disabled
        // so a compromised gateway can't feed us arbitrary paths.
        let store = LocalPayloadStore::new(None::<PathBuf>);
        let err = store.get("/etc/passwd").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn absolute_ref_outside_base_dir_is_rejected() {
        // Classic path traversal: base_dir is a tempdir, and the attacker sends
        // an absolute path outside it. The directory capability must refuse it.
        let base = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(base.path().to_path_buf()));
        let err = store.get("/etc/hosts").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::InvalidRef(_)),
            "expected InvalidRef for out-of-base absolute, got {err:?}"
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn relative_dotdot_ref_cannot_escape_base_dir() {
        // Same thing as a relative ref that tries to climb out.
        let parent = TempDir::new().unwrap();
        let base = parent.path().join("sub");
        tokio::fs::create_dir(&base).await.unwrap();
        // Touch a file in the parent that an attacker might want to read.
        tokio::fs::write(parent.path().join("secret.bin"), b"secret")
            .await
            .unwrap();

        let store = LocalPayloadStore::new(Some(base.clone()));
        let err = store.get("../secret.bin").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::InvalidRef(_)),
            "expected InvalidRef for dotdot escape, got {err:?}"
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn empty_ref_rejected() {
        let dir = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let err = store.get("").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn factory_none_yields_disabled_store() {
        let store = create_payload_store(None).await.unwrap();
        let err = store.get("/etc/passwd").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn factory_empty_url_yields_disabled_store() {
        // Treat `SIE_PAYLOAD_STORE_URL=` the same as unset, not as "the
        // current directory". The previous behaviour tried to use "" as a
        // filesystem path which canonicalized to CWD.
        let store = create_payload_store(Some("")).await.unwrap();
        let err = store.get("foo.bin").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn factory_local_path_roundtrips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pay.bin");
        tokio::fs::write(&path, b"ok").await.unwrap();

        let store = create_payload_store(Some(dir.path().to_str().unwrap()))
            .await
            .unwrap();
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"ok");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn factory_file_scheme_is_stripped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pay.bin");
        tokio::fs::write(&path, b"ok").await.unwrap();

        let url = format!("file://{}", dir.path().display());
        let store = create_payload_store(Some(&url)).await.unwrap();
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"ok");
    }

    #[cfg(not(target_os = "linux"))]
    #[tokio::test]
    async fn local_store_fails_closed_without_openat2() {
        let dir = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        assert!(matches!(
            store.get("payload.bin").await,
            Err(PayloadError::Unsupported(_))
        ));
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_full_ref_inside_prefix() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/pay.bin"),
                PutPayload::from(b"cloud".to_vec()),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let got = store.get("s3://bucket/payloads/pay.bin").await.unwrap();
        assert_eq!(got, b"cloud");
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_empty_payload() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/empty.bin"),
                PutPayload::from(Vec::new()),
            )
            .await
            .unwrap();
        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };

        assert!(store
            .get("s3://bucket/payloads/empty.bin")
            .await
            .unwrap()
            .is_empty());
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_uses_one_request_at_exact_cap() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/exact.bin"),
                PutPayload::from(vec![0; MAX_PAYLOAD_BYTES as usize]),
            )
            .await
            .unwrap();
        let get_calls = std::sync::Arc::new(AtomicUsize::new(0));
        let store = ObjectPayloadStore {
            store: Box::new(CountingObjectStore {
                inner: memory,
                get_calls: get_calls.clone(),
            }),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };

        assert_eq!(
            store
                .get("s3://bucket/payloads/exact.bin")
                .await
                .unwrap()
                .len() as u64,
            MAX_PAYLOAD_BYTES
        );
        assert_eq!(get_calls.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_rejects_oversized_payload_before_reading() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/oversized.bin"),
                PutPayload::from(vec![0; (MAX_PAYLOAD_BYTES + 1) as usize]),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let err = store
            .get("s3://bucket/payloads/oversized.bin")
            .await
            .unwrap_err();
        assert!(matches!(err, PayloadError::TooLarge { .. }));
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_rejects_ref_outside_prefix() {
        use object_store::path::Path;

        let store = ObjectPayloadStore {
            store: Box::new(object_store::memory::InMemory::new()),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let err = store.get("s3://bucket/other/pay.bin").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_azure_full_ref_inside_prefix() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/pay.bin"),
                PutPayload::from(b"cloud".to_vec()),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "abfss://container@account.dfs.core.windows.net/payloads".into(),
            scheme: "abfss://",
        };
        let got = store
            .get("abfss://container@account.dfs.core.windows.net/payloads/pay.bin")
            .await
            .unwrap();
        assert_eq!(got, b"cloud");
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_rejects_different_azure_scheme() {
        use object_store::path::Path;

        let store = ObjectPayloadStore {
            store: Box::new(object_store::memory::InMemory::new()),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let err = store
            .get("abfs://container@account.dfs.core.windows.net/payloads/pay.bin")
            .await
            .unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(feature = "cloud-storage")]
    #[test]
    fn new_azure_parses_container_account_and_prefix() {
        let (container, account, prefix) =
            ObjectPayloadStore::parse_azure_path("payloads@sieacct.dfs.core.windows.net/prefix")
                .unwrap();
        assert_eq!(container, "payloads");
        assert_eq!(account, "sieacct");
        assert_eq!(prefix, "prefix");
    }

    #[cfg(feature = "cloud-storage")]
    #[test]
    fn new_azure_accepts_blob_endpoint_suffix() {
        let (container, account, prefix) =
            ObjectPayloadStore::parse_azure_path("payloads@sieacct.blob.core.windows.net/prefix")
                .unwrap();
        assert_eq!(container, "payloads");
        assert_eq!(account, "sieacct");
        assert_eq!(prefix, "prefix");
    }

    #[cfg(feature = "cloud-storage")]
    #[test]
    fn new_azure_rejects_missing_container_account_separator() {
        let err = ObjectPayloadStore::parse_azure_path("payloads/prefix").unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(feature = "cloud-storage")]
    #[test]
    fn new_azure_builds_from_workload_identity_env() {
        let _lock = azure_env_lock().lock().unwrap();
        let dir = TempDir::new().unwrap();
        let token_file = dir.path().join("token");
        std::fs::write(&token_file, "token").unwrap();
        let _env = EnvGuard::set(&[
            ("AZURE_STORAGE_ACCOUNT_NAME", "env-account".to_string()),
            ("AZURE_CLIENT_ID", "client-id".to_string()),
            ("AZURE_TENANT_ID", "tenant-id".to_string()),
            (
                "AZURE_FEDERATED_TOKEN_FILE",
                token_file.to_string_lossy().to_string(),
            ),
            (
                "AZURE_AUTHORITY_HOST",
                "https://login.microsoftonline.com/".to_string(),
            ),
        ]);

        let store =
            ObjectPayloadStore::new_azure("payloads@sieacct.dfs.core.windows.net/path/to/payloads")
                .unwrap();

        assert_eq!(store.scheme, "abfs://");
        assert_eq!(
            store.url_prefix,
            "abfs://payloads@sieacct.dfs.core.windows.net/path/to/payloads"
        );
        assert_eq!(store.prefix.as_ref(), "path/to/payloads");
    }

    #[cfg(feature = "cloud-storage")]
    #[test]
    fn new_azure_with_abfss_preserves_scheme() {
        let _lock = azure_env_lock().lock().unwrap();
        let dir = TempDir::new().unwrap();
        let token_file = dir.path().join("token");
        std::fs::write(&token_file, "token").unwrap();
        let _env = EnvGuard::set(&[
            ("AZURE_CLIENT_ID", "client-id".to_string()),
            ("AZURE_TENANT_ID", "tenant-id".to_string()),
            (
                "AZURE_FEDERATED_TOKEN_FILE",
                token_file.to_string_lossy().to_string(),
            ),
        ]);

        let store = ObjectPayloadStore::new_azure_with_scheme(
            "payloads@sieacct.dfs.core.windows.net/path/to/payloads",
            "abfss://",
        )
        .unwrap();

        assert_eq!(store.scheme, "abfss://");
        assert_eq!(
            store.url_prefix,
            "abfss://payloads@sieacct.dfs.core.windows.net/path/to/payloads"
        );
        assert_eq!(store.prefix.as_ref(), "path/to/payloads");
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_plain_key_under_configured_prefix() {
        use object_store::path::Path;
        use object_store::PutPayload;

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/pay.bin"),
                PutPayload::from(b"cloud".to_vec()),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"cloud");
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn factory_s3_without_feature_errors_loudly() {
        // Previously this would silently fall back to a LocalPayloadStore
        // whose base_dir was the literal string "s3://…", guaranteeing
        // every read produced a confusing `canonicalize` error. We now
        // refuse at construction time.
        match create_payload_store(Some("s3://bucket/prefix")).await {
            Ok(_) => panic!("expected Unsupported for s3:// without cloud-storage feature"),
            Err(e) => assert!(
                matches!(e, PayloadError::Unsupported(_)),
                "expected Unsupported for s3://, got {e:?}"
            ),
        }
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn factory_abfs_without_feature_errors_loudly() {
        match create_payload_store(Some("abfs://container@account.dfs.core.windows.net/prefix"))
            .await
        {
            Ok(_) => panic!("expected Unsupported for abfs:// without cloud-storage feature"),
            Err(e) => assert!(
                matches!(e, PayloadError::Unsupported(_)),
                "expected Unsupported for abfs://, got {e:?}"
            ),
        }
    }
}
