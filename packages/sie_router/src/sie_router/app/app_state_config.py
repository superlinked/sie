from __future__ import annotations

import os
from dataclasses import dataclass, field

# Environment variable names for configuration
ENV_WORKERS = "SIE_ROUTER_WORKERS"
ENV_KUBERNETES = "SIE_ROUTER_KUBERNETES"
ENV_K8S_NAMESPACE = "SIE_ROUTER_K8S_NAMESPACE"
ENV_K8S_SERVICE = "SIE_ROUTER_K8S_SERVICE"
ENV_K8S_PORT = "SIE_ROUTER_K8S_PORT"
ENV_ENABLE_POOLS = "SIE_ROUTER_ENABLE_POOLS"
ENV_HOT_RELOAD = "SIE_ROUTER_HOT_RELOAD"


@dataclass
class AppStateConfig:
    """Configuration for FastAPI app state, passed to lifespan.

    This dataclass holds all startup configuration needed by the lifespan
    context manager to initialize the worker registry and related components.
    """

    worker_urls: list[str] = field(default_factory=list)
    """List of worker URLs for static discovery."""

    use_kubernetes: bool = False
    """Use Kubernetes discovery instead of static."""

    k8s_namespace: str = "default"
    """Kubernetes namespace for discovery."""

    k8s_service: str = "sie-worker"
    """Kubernetes service name for discovery."""

    k8s_port: int = 8080
    """Worker port for Kubernetes discovery."""

    enable_pools: bool = False
    """Enable resource pool management."""

    enable_hot_reload: bool = False
    """Enable hot reload of model configs via file watching."""

    def save_to_env_vars(self) -> None:
        """Serialize configuration to environment variables for uvicorn reload mode."""
        if self.worker_urls:
            os.environ[ENV_WORKERS] = ",".join(self.worker_urls)
        elif ENV_WORKERS in os.environ:
            del os.environ[ENV_WORKERS]

        os.environ[ENV_KUBERNETES] = "true" if self.use_kubernetes else "false"
        os.environ[ENV_K8S_NAMESPACE] = self.k8s_namespace
        os.environ[ENV_K8S_SERVICE] = self.k8s_service
        os.environ[ENV_K8S_PORT] = str(self.k8s_port)
        os.environ[ENV_ENABLE_POOLS] = "true" if self.enable_pools else "false"
        os.environ[ENV_HOT_RELOAD] = "true" if self.enable_hot_reload else "false"

    @classmethod
    def from_env_vars(cls) -> AppStateConfig:
        """Deserialize configuration from environment variables."""
        worker_urls_str = os.environ.get(ENV_WORKERS, "")
        worker_urls = [url.strip() for url in worker_urls_str.split(",") if url.strip()]

        use_kubernetes = os.environ.get(ENV_KUBERNETES, "").lower() == "true"
        k8s_namespace = os.environ.get(ENV_K8S_NAMESPACE, "default")
        k8s_service = os.environ.get(ENV_K8S_SERVICE, "sie-worker")
        k8s_port = int(os.environ.get(ENV_K8S_PORT, "8080"))
        enable_pools = os.environ.get(ENV_ENABLE_POOLS, "").lower() == "true"
        enable_hot_reload = os.environ.get(ENV_HOT_RELOAD, "").lower() == "true"

        return cls(
            worker_urls=worker_urls,
            use_kubernetes=use_kubernetes,
            k8s_namespace=k8s_namespace,
            k8s_service=k8s_service,
            k8s_port=k8s_port,
            enable_pools=enable_pools,
            enable_hot_reload=enable_hot_reload,
        )
