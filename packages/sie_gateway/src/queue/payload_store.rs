use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

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

/// Local filesystem payload store (default).
pub struct LocalPayloadStore {
    base_dir: PathBuf,
}

impl LocalPayloadStore {
    pub async fn new(base_dir: impl AsRef<Path>) -> Self {
        let dir = base_dir.as_ref().to_path_buf();
        if !dir.exists() {
            if let Err(e) = tokio::fs::create_dir_all(&dir).await {
                warn!(dir = %dir.display(), error = %e, "failed to create payload store directory");
            }
        }
        Self { base_dir: dir }
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
    pub fn new_s3(path: &str) -> Self {
        use object_store::aws::AmazonS3Builder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path);
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .expect("failed to build S3 payload store");

        let url_prefix = format!("s3://{}", path);
        Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix,
        }
    }

    pub fn new_gcs(path: &str) -> Self {
        use object_store::gcp::GoogleCloudStorageBuilder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path);
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .expect("failed to build GCS payload store");

        let url_prefix = format!("gs://{}", path);
        Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix,
        }
    }

    fn parse_bucket_prefix(path: &str) -> (String, String) {
        match path.split_once('/') {
            Some((bucket, prefix)) => (bucket.to_string(), prefix.to_string()),
            None => (path.to_string(), String::new()),
        }
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
pub async fn create_payload_store(url: &str) -> Arc<dyn PayloadStore> {
    #[cfg(feature = "cloud-storage")]
    {
        if let Some(rest) = url.strip_prefix("s3://") {
            return Arc::new(ObjectStorePayloadStore::new_s3(rest));
        }
        if let Some(rest) = url.strip_prefix("gs://") {
            return Arc::new(ObjectStorePayloadStore::new_gcs(rest));
        }
    }

    #[cfg(not(feature = "cloud-storage"))]
    {
        if url.starts_with("s3://") || url.starts_with("gs://") {
            warn!(
                url = %url,
                "cloud storage payload store requested but 'cloud-storage' feature not enabled — falling back to local"
            );
        }
    }

    Arc::new(LocalPayloadStore::new(url).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_payload_store_put_delete() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = LocalPayloadStore::new(dir.path()).await;

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
        let store = LocalPayloadStore::new(dir.path()).await;
        store.delete("nonexistent.bin").await.unwrap();
    }

    #[tokio::test]
    async fn test_local_payload_store_creates_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let sub = dir.path().join("sub").join("dir");
        let store = LocalPayloadStore::new(&sub).await;

        let payload_ref = store.put("test.bin", b"data").await.unwrap();
        assert!(payload_ref.contains("test.bin"));
        assert!(sub.join("test.bin").exists());
    }

    #[tokio::test]
    async fn test_validate_key_rejects_path_traversal() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = LocalPayloadStore::new(dir.path()).await;

        assert!(store.put("../escape.bin", b"data").await.is_err());
        assert!(store.put("sub/dir.bin", b"data").await.is_err());
        assert!(store.put("sub\\dir.bin", b"data").await.is_err());
        assert!(store.put("", b"data").await.is_err());
        assert!(store.delete("../escape.bin").await.is_err());
    }

    #[tokio::test]
    async fn test_create_payload_store_local() {
        let dir = tempfile::TempDir::new().unwrap();
        let _store = create_payload_store(dir.path().to_str().unwrap()).await;
    }

    #[tokio::test]
    async fn test_create_payload_store_s3_without_feature() {
        let _store = create_payload_store("s3://my-bucket/prefix").await;
    }

    #[tokio::test]
    async fn test_create_payload_store_gs_without_feature() {
        let _store = create_payload_store("gs://my-bucket/prefix").await;
    }

    #[cfg(feature = "cloud-storage")]
    mod cloud_tests {
        use super::super::*;

        #[test]
        fn test_parse_bucket_prefix() {
            let (bucket, prefix) =
                ObjectStorePayloadStore::parse_bucket_prefix("my-bucket/path/to/payloads");
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "path/to/payloads");
        }

        #[test]
        fn test_parse_bucket_only() {
            let (bucket, prefix) = ObjectStorePayloadStore::parse_bucket_prefix("my-bucket");
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "");
        }
    }
}
