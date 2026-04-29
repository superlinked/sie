use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{Event, PollWatcher, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::model_registry::ModelRegistry;

/// Checks whether a directory appears to be a K8s ConfigMap mount.
/// K8s ConfigMap mounts use atomic symlink swaps with a `..data` symlink
/// pointing to the current data directory.
fn is_k8s_configmap_mount(dir: &Path) -> bool {
    let dot_data = dir.join("..data");
    dot_data
        .symlink_metadata()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

pub struct ConfigWatcher {
    _watcher: Box<dyn Watcher + Send>,
    #[allow(dead_code)]
    cancel: tokio::sync::watch::Sender<()>,
}

impl ConfigWatcher {
    pub fn start(
        registry: Arc<ModelRegistry>,
        debounce: Duration,
        force_polling: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bundles_dir = registry.bundles_dir().to_path_buf();
        let models_dir = registry.models_dir().to_path_buf();

        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(());
        let (event_tx, event_rx) = mpsc::channel::<PathBuf>(64);

        // Create filesystem watcher — use PollWatcher for K8s ConfigMap mounts
        let use_polling = force_polling
            || is_k8s_configmap_mount(&bundles_dir)
            || is_k8s_configmap_mount(&models_dir);

        let tx = event_tx.clone();
        let event_handler = move |res: Result<Event, notify::Error>| match res {
            Ok(event) => {
                for path in &event.paths {
                    if let Some(ext) = path.extension() {
                        if ext == "yaml" || ext == "yml" {
                            let _ = tx.try_send(path.clone());
                        }
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "filesystem watch error");
            }
        };

        let mut watcher: Box<dyn Watcher + Send> = if use_polling {
            let poll_config = notify::Config::default()
                .with_poll_interval(Duration::from_secs(5))
                .with_compare_contents(true);
            Box::new(PollWatcher::new(event_handler, poll_config)?)
        } else {
            Box::new(RecommendedWatcher::new(
                event_handler,
                notify::Config::default(),
            )?)
        };

        if use_polling {
            info!("using poll watcher (5s interval, compare_contents=true)");
        } else {
            info!("using native filesystem watcher");
        }

        // Watch directories
        if bundles_dir.exists() {
            watcher.watch(&bundles_dir, RecursiveMode::NonRecursive)?;
            info!(dir = %bundles_dir.display(), "watching bundles directory");
        } else {
            warn!(dir = %bundles_dir.display(), "bundles directory not found, skipping watch");
        }

        if models_dir.exists() {
            watcher.watch(&models_dir, RecursiveMode::Recursive)?;
            info!(dir = %models_dir.display(), "watching models directory");
        } else {
            warn!(dir = %models_dir.display(), "models directory not found, skipping watch");
        }

        // Spawn debounce + reload task
        tokio::spawn(Self::debounce_loop(registry, event_rx, cancel_rx, debounce));

        Ok(Self {
            _watcher: watcher,
            cancel: cancel_tx,
        })
    }

    async fn debounce_loop(
        registry: Arc<ModelRegistry>,
        mut event_rx: mpsc::Receiver<PathBuf>,
        mut cancel_rx: tokio::sync::watch::Receiver<()>,
        debounce: Duration,
    ) {
        let mut pending = false;
        let mut timer = tokio::time::interval(debounce);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Consume first immediate tick
        timer.tick().await;

        loop {
            tokio::select! {
                path = event_rx.recv() => {
                    match path {
                        Some(p) => {
                            debug!(path = %p.display(), "config change detected");
                            pending = true;
                        }
                        None => break, // Channel closed
                    }
                }
                _ = timer.tick(), if pending => {
                    pending = false;
                    info!("reloading model registry (debounced)");
                    registry.reload();
                    info!(
                        bundles = registry.list_bundles().len(),
                        models = registry.list_models().len(),
                        "model registry reloaded"
                    );
                }
                _ = cancel_rx.changed() => {
                    info!("config watcher stopping");
                    break;
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn stop(self) {
        // Dropping cancel_tx signals the debounce loop to stop
        drop(self.cancel);
        // _watcher is dropped automatically
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_is_k8s_configmap_mount_false_for_regular_dir() {
        let dir = TempDir::new().unwrap();
        assert!(!is_k8s_configmap_mount(dir.path()));
    }

    #[cfg(unix)]
    #[test]
    fn test_is_k8s_configmap_mount_true_with_dot_data_symlink() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("..2024_01_01_00_00_00.000000");
        fs::create_dir(&target).unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("..data")).unwrap();
        assert!(is_k8s_configmap_mount(dir.path()));
    }

    #[test]
    fn test_is_k8s_configmap_mount_false_for_regular_file_named_dot_data() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("..data"), "not a symlink").unwrap();
        assert!(!is_k8s_configmap_mount(dir.path()));
    }

    #[test]
    fn test_is_k8s_configmap_mount_false_for_nonexistent_dir() {
        let path = PathBuf::from("/nonexistent/path/that/does/not/exist");
        assert!(!is_k8s_configmap_mount(&path));
    }

    #[tokio::test]
    async fn test_start_with_native_watcher() {
        let dir = TempDir::new().unwrap();
        let bundles_dir = dir.path().join("bundles");
        let models_dir = dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));

        let watcher = ConfigWatcher::start(registry, Duration::from_secs(1), false);
        assert!(watcher.is_ok());
    }

    #[tokio::test]
    async fn test_start_with_forced_polling() {
        let dir = TempDir::new().unwrap();
        let bundles_dir = dir.path().join("bundles");
        let models_dir = dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));

        let watcher = ConfigWatcher::start(registry, Duration::from_secs(1), true);
        assert!(watcher.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_start_auto_detects_k8s_mount() {
        let dir = TempDir::new().unwrap();
        let bundles_dir = dir.path().join("bundles");
        let models_dir = dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();

        // Simulate K8s ConfigMap mount in bundles_dir
        let target = bundles_dir.join("..2024_01_01_00_00_00.000000");
        fs::create_dir(&target).unwrap();
        std::os::unix::fs::symlink(&target, bundles_dir.join("..data")).unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));

        let watcher = ConfigWatcher::start(registry, Duration::from_secs(1), false);
        assert!(watcher.is_ok());
    }
}
