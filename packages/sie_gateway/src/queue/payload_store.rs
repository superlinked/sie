use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

#[cfg(feature = "cloud-storage")]
use object_store::ObjectStoreExt;

const OBJECT_STORE_SCHEMES: &[&str] = &["s3://", "gs://", "abfs://", "abfss://", "oss://"];

fn object_store_scheme(url: &str) -> Option<&'static str> {
    OBJECT_STORE_SCHEMES
        .iter()
        .copied()
        .find(|scheme| url.starts_with(scheme))
}

/// Validate that a key is a plain filename (no path separators or traversal).
fn validate_key(key: &str) -> Result<(), String> {
    if key.contains('/') || key.contains('\\') || key.contains("..") || key.is_empty() {
        return Err(format!("invalid payload key: {}", key));
    }
    Ok(())
}

/// Trait for storing and retrieving offloaded payloads.
/// Implementations must be Send + Sync for use in async contexts.
#[async_trait]
pub trait PayloadStore: Send + Sync {
    /// Store payload bytes, returning the reference key (used as `payload_ref` in WorkItem).
    async fn put(&self, key: &str, data: &[u8]) -> Result<String, String>;

    /// Delete a previously stored payload by key.
    async fn delete(&self, key: &str) -> Result<(), String>;
}

/// Disabled payload store. Used when `SIE_PAYLOAD_STORE_URL` is unset so
/// queue mode does not silently offload to gateway-local disk that worker
/// pods cannot read.
pub struct DisabledPayloadStore;

#[async_trait]
impl PayloadStore for DisabledPayloadStore {
    async fn put(&self, _key: &str, _data: &[u8]) -> Result<String, String> {
        Err("payload store not configured".to_string())
    }

    async fn delete(&self, _key: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Local filesystem payload store (default).
pub struct LocalPayloadStore {
    base_dir: PathBuf,
}

impl LocalPayloadStore {
    pub async fn new(base_dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = base_dir.as_ref().to_path_buf();
        match tokio::fs::metadata(&dir).await {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("payload store path {} is not a directory", dir.display()),
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                tokio::fs::create_dir_all(&dir).await.map_err(|e| {
                    io::Error::new(
                        e.kind(),
                        format!("create payload store directory {}: {e}", dir.display()),
                    )
                })?;
            }
            Err(e) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("stat payload store directory {}: {e}", dir.display()),
                ));
            }
        }
        Ok(Self { base_dir: dir })
    }
}

#[async_trait]
impl PayloadStore for LocalPayloadStore {
    async fn put(&self, key: &str, data: &[u8]) -> Result<String, String> {
        validate_key(key)?;
        let path = self.base_dir.join(key);
        tokio::fs::write(&path, data)
            .await
            .map_err(|e| format!("write payload {}: {}", path.display(), e))?;
        debug!(path = %path.display(), size = data.len(), "stored payload locally");
        Ok(path.to_string_lossy().to_string())
    }

    async fn delete(&self, key: &str) -> Result<(), String> {
        validate_key(key)?;
        let path = self.base_dir.join(key);
        match tokio::fs::remove_file(&path).await {
            Ok(_) => {
                debug!(path = %path.display(), "deleted local payload");
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(format!("delete payload {}: {}", path.display(), e));
            }
        }
        Ok(())
    }
}

/// Cloud-backed payload store using the `object_store` crate.
#[cfg(feature = "cloud-storage")]
pub struct ObjectStorePayloadStore {
    store: Box<dyn object_store::ObjectStore>,
    prefix: object_store::path::Path,
    url_prefix: String,
}

#[cfg(feature = "cloud-storage")]
impl ObjectStorePayloadStore {
    pub fn new_s3(path: &str) -> io::Result<Self> {
        use object_store::aws::AmazonS3Builder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path, "S3")?;
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("failed to build S3 payload store: {e}"),
                )
            })?;

        let url_prefix = format!("s3://{}", path.trim_end_matches('/'));
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix,
        })
    }

    pub fn new_gcs(path: &str) -> io::Result<Self> {
        use object_store::gcp::GoogleCloudStorageBuilder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path, "GCS")?;
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("failed to build GCS payload store: {e}"),
                )
            })?;

        let url_prefix = format!("gs://{}", path.trim_end_matches('/'));
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix,
        })
    }

    pub fn new_azure(path: &str) -> io::Result<Self> {
        Self::new_azure_with_scheme(path, "abfs://")
    }

    fn new_azure_with_scheme(path: &str, scheme: &'static str) -> io::Result<Self> {
        use object_store::azure::MicrosoftAzureBuilder;

        let (container, account, prefix) = Self::parse_azure_path(path)?;
        let store = MicrosoftAzureBuilder::from_env()
            .with_account(&account)
            .with_container_name(&container)
            .build()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("failed to build Azure payload store: {e}"),
                )
            })?;

        let url_prefix = format!("{}{}", scheme, path.trim_end_matches('/'));
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix,
        })
    }

    fn parse_bucket_prefix(path: &str, provider: &str) -> io::Result<(String, String)> {
        let path = path.trim_matches('/');
        let (bucket, prefix) = match path.split_once('/') {
            Some((bucket, prefix)) => (bucket.to_string(), prefix.trim_matches('/').to_string()),
            None => (path.to_string(), String::new()),
        };
        if bucket.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{provider} payload store bucket must be non-empty"),
            ));
        }
        Ok((bucket, prefix))
    }

    fn parse_azure_path(path: &str) -> io::Result<(String, String, String)> {
        let path = path.trim_matches('/');
        let (authority, prefix) = match path.split_once('/') {
            Some((authority, prefix)) => (authority, prefix.trim_matches('/').to_string()),
            None => (path, String::new()),
        };
        let Some((container, host)) = authority.split_once('@') else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Azure payload store URL must use <container>@<account>.dfs.core.windows.net",
            ));
        };
        if container.is_empty() || host.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Azure payload store container and account must be non-empty",
            ));
        }
        let account = host
            .strip_suffix(".dfs.core.windows.net")
            .or_else(|| host.strip_suffix(".blob.core.windows.net"))
            .unwrap_or(host);
        if account.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Azure payload store account must be non-empty",
            ));
        }
        Ok((container.to_string(), account.to_string(), prefix))
    }

    fn object_key(&self, key: &str) -> object_store::path::Path {
        if self.prefix.as_ref().is_empty() {
            object_store::path::Path::from(key.to_string())
        } else {
            object_store::path::Path::from(format!("{}/{}", self.prefix, key))
        }
    }
}

#[cfg(feature = "cloud-storage")]
#[async_trait]
impl PayloadStore for ObjectStorePayloadStore {
    async fn put(&self, key: &str, data: &[u8]) -> Result<String, String> {
        let obj_key = self.object_key(key);
        self.store
            .put(&obj_key, object_store::PutPayload::from(data.to_vec()))
            .await
            .map_err(|e| format!("put payload {}: {}", obj_key, e))?;

        let payload_ref = format!("{}/{}", self.url_prefix.trim_end_matches('/'), key);
        debug!(key = %obj_key, size = data.len(), "stored payload in cloud store");
        Ok(payload_ref)
    }

    async fn delete(&self, key: &str) -> Result<(), String> {
        let obj_key = self.object_key(key);
        match self.store.delete(&obj_key).await {
            Ok(_) => {
                debug!(key = %obj_key, "deleted payload from cloud store");
                Ok(())
            }
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(format!("delete payload {}: {}", obj_key, e)),
        }
    }
}

/// Factory function to create the appropriate payload store based on URL prefix.
pub async fn create_payload_store(url: &str) -> io::Result<Arc<dyn PayloadStore>> {
    let url = url.trim();
    if url.is_empty() {
        return Ok(Arc::new(DisabledPayloadStore));
    }

    #[cfg(feature = "cloud-storage")]
    {
        if let Some(scheme) = object_store_scheme(url) {
            if let Some(rest) = url.strip_prefix("oss://") {
                return Ok(Arc::new(
                    super::oss_payload_store::OssPayloadStore::from_url(&format!("oss://{rest}"))?,
                ));
            }
            if let Some(rest) = url.strip_prefix("s3://") {
                return Ok(Arc::new(ObjectStorePayloadStore::new_s3(rest)?));
            }
            if let Some(rest) = url.strip_prefix("gs://") {
                return Ok(Arc::new(ObjectStorePayloadStore::new_gcs(rest)?));
            }
            if let Some(rest) = url.strip_prefix("abfs://") {
                return Ok(Arc::new(ObjectStorePayloadStore::new_azure(rest)?));
            }
            if let Some(rest) = url.strip_prefix(scheme) {
                return Ok(Arc::new(ObjectStorePayloadStore::new_azure_with_scheme(
                    rest, scheme,
                )?));
            }
        }
    }

    #[cfg(not(feature = "cloud-storage"))]
    {
        if let Some(scheme) = object_store_scheme(url) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "payload store URL {url} with scheme {scheme} requires the 'cloud-storage' feature; \
                     rebuild the gateway with --features cloud-storage or point SIE_PAYLOAD_STORE_URL at a local path"
                ),
            ));
        }
    }

    Ok(Arc::new(LocalPayloadStore::new(url).await?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cloud-storage")]
    use std::sync::{Mutex, OnceLock};

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

    #[tokio::test]
    async fn test_local_payload_store_put_delete() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = LocalPayloadStore::new(dir.path()).await.unwrap();

        let key = "test_req_0.bin";
        let data = b"hello world";

        let payload_ref = store.put(key, data).await.unwrap();
        assert!(payload_ref.contains(key));

        let path = dir.path().join(key);
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), data);

        store.delete(key).await.unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn test_local_payload_store_delete_nonexistent() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = LocalPayloadStore::new(dir.path()).await.unwrap();
        store.delete("nonexistent.bin").await.unwrap();
    }

    #[tokio::test]
    async fn test_local_payload_store_creates_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let sub = dir.path().join("sub").join("dir");
        let store = LocalPayloadStore::new(&sub).await.unwrap();

        let payload_ref = store.put("test.bin", b"data").await.unwrap();
        assert!(payload_ref.contains("test.bin"));
        assert!(sub.join("test.bin").exists());
    }

    #[tokio::test]
    async fn test_validate_key_rejects_path_traversal() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = LocalPayloadStore::new(dir.path()).await.unwrap();

        assert!(store.put("../escape.bin", b"data").await.is_err());
        assert!(store.put("sub/dir.bin", b"data").await.is_err());
        assert!(store.put("sub\\dir.bin", b"data").await.is_err());
        assert!(store.put("", b"data").await.is_err());
        assert!(store.delete("../escape.bin").await.is_err());
    }

    #[tokio::test]
    async fn test_create_payload_store_local() {
        let dir = tempfile::TempDir::new().unwrap();
        let _store = create_payload_store(dir.path().to_str().unwrap())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_create_payload_store_empty_disables_offload() {
        let store = create_payload_store("").await.unwrap();
        assert!(store.put("test.bin", b"data").await.is_err());
        assert!(store.delete("test.bin").await.is_ok());
    }

    #[tokio::test]
    async fn test_create_payload_store_rejects_file_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("payloads");
        tokio::fs::write(&file, b"not a dir").await.unwrap();

        let err = match create_payload_store(file.to_str().unwrap()).await {
            Ok(_) => panic!("expected file path to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("not a directory"));
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn test_create_payload_store_s3_without_feature() {
        let err = match create_payload_store("s3://my-bucket/prefix").await {
            Ok(_) => panic!("expected s3:// to require cloud-storage"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("cloud-storage"));
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn test_create_payload_store_gs_without_feature() {
        let err = match create_payload_store("gs://my-bucket/prefix").await {
            Ok(_) => panic!("expected gs:// to require cloud-storage"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("cloud-storage"));
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn test_create_payload_store_abfs_without_feature() {
        let err = match create_payload_store("abfs://payloads@sieacct.dfs.core.windows.net/prefix")
            .await
        {
            Ok(_) => panic!("expected abfs:// to require cloud-storage"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("cloud-storage"));
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn test_create_payload_store_oss_without_feature() {
        let err = match create_payload_store("oss://my-bucket/payloads").await {
            Ok(_) => panic!("expected oss:// to require cloud-storage"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("cloud-storage"));
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn test_create_payload_store_trims_before_scheme_classification() {
        let err = match create_payload_store("  oss://my-bucket/payloads  ").await {
            Ok(_) => panic!("expected trimmed oss:// to require cloud-storage"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("cloud-storage"));
    }

    #[cfg(feature = "cloud-storage")]
    mod cloud_tests {
        use super::super::*;
        use super::{azure_env_lock, EnvGuard};

        #[test]
        fn test_parse_bucket_prefix() {
            let (bucket, prefix) =
                ObjectStorePayloadStore::parse_bucket_prefix("my-bucket/path/to/payloads", "S3")
                    .unwrap();
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "path/to/payloads");
        }

        #[test]
        fn test_parse_bucket_only() {
            let (bucket, prefix) =
                ObjectStorePayloadStore::parse_bucket_prefix("my-bucket", "S3").unwrap();
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "");
        }

        #[test]
        fn test_parse_bucket_prefix_rejects_empty_bucket() {
            let err = ObjectStorePayloadStore::parse_bucket_prefix("", "S3").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        }

        #[test]
        fn test_parse_bucket_prefix_trims_slashes() {
            // Parity with the sidecar's parse_bucket_prefix.
            let (bucket, prefix) =
                ObjectStorePayloadStore::parse_bucket_prefix("/my-bucket/path/to/payloads/", "S3")
                    .unwrap();
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "path/to/payloads");
        }

        #[test]
        fn test_parse_azure_path() {
            let (container, account, prefix) = ObjectStorePayloadStore::parse_azure_path(
                "payloads@sieacct.dfs.core.windows.net/path/to/payloads",
            )
            .unwrap();
            assert_eq!(container, "payloads");
            assert_eq!(account, "sieacct");
            assert_eq!(prefix, "path/to/payloads");
        }

        #[test]
        fn test_parse_azure_blob_endpoint_path() {
            let (container, account, prefix) = ObjectStorePayloadStore::parse_azure_path(
                "payloads@sieacct.blob.core.windows.net/path/to/payloads",
            )
            .unwrap();
            assert_eq!(container, "payloads");
            assert_eq!(account, "sieacct");
            assert_eq!(prefix, "path/to/payloads");
        }

        #[test]
        fn test_parse_azure_path_rejects_missing_account_separator() {
            let err = ObjectStorePayloadStore::parse_azure_path("payloads/path").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        }

        #[test]
        fn test_new_azure_builds_from_workload_identity_env() {
            let _lock = azure_env_lock().lock().unwrap();
            let dir = tempfile::TempDir::new().unwrap();
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

            let store = ObjectStorePayloadStore::new_azure(
                "payloads@sieacct.dfs.core.windows.net/path/to/payloads",
            )
            .unwrap();

            assert_eq!(
                store.url_prefix,
                "abfs://payloads@sieacct.dfs.core.windows.net/path/to/payloads"
            );
            assert_eq!(store.prefix.as_ref(), "path/to/payloads");
        }

        #[test]
        fn test_create_payload_store_abfss_with_cloud_feature() {
            let _lock = azure_env_lock().lock().unwrap();
            let dir = tempfile::TempDir::new().unwrap();
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

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _store = runtime.block_on(create_payload_store(
                "abfss://payloads@sieacct.dfs.core.windows.net/path/to/payloads",
            ));
            assert!(_store.is_ok());
        }
    }
}
