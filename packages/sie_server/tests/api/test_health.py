"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core import gpu_health
from sie_server.core.readiness import mark_not_ready, mark_ready


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


@pytest.fixture(autouse=True)
def _gpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the probe cache between tests and simulate a single GPU.

    The module-level probe cache would otherwise leak a result across tests on a
    GPU runner (issue #1025 review), so a wedged-GPU test could flake an
    unrelated readiness test. Resetting it keeps tests isolated; pinning
    ``device_count`` to 1 makes the probe loop deterministic when a test
    simulates an available GPU.
    """
    gpu_health.reset_gpu_health_cache()
    monkeypatch.setattr(gpu_health.torch.cuda, "device_count", lambda: 1)


class TestHealthEndpoints:
    """Tests for /healthz and /readyz endpoints."""

    def test_healthz_returns_ok(self, client: TestClient) -> None:
        """Liveness probe returns 200 OK."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_readyz_returns_ok_when_ready(self, client: TestClient) -> None:
        """Readiness probe returns 200 OK when ready."""
        # Ensure ready state (TestClient invokes lifespan which calls mark_ready(),
        # but other tests may have modified the global state)
        mark_ready()
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.text == "ok"
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_readyz_returns_503_when_not_ready(self, client: TestClient) -> None:
        """Readiness probe returns 503 when not ready."""
        # Force not-ready state
        mark_not_ready()
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.text == "not ready"
        # Restore ready state for other tests
        mark_ready()

    def test_readyz_returns_503_when_gpu_wedged(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Readiness probe returns 503 when the CUDA context is wedged (issue #1025).

        Reproduces the bug where a wedged GPU (every inference returns 500) still
        passed the readiness probe, so the gateway kept routing to a dead worker.
        """
        mark_ready()
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def boom(device: str) -> None:
            raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", boom)

        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.text == "gpu unhealthy"

    def test_readyz_returns_ok_when_gpu_healthy(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Readiness probe returns 200 when ready and the GPU probe succeeds."""
        mark_ready()
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(gpu_health, "_run_gpu_probe", lambda device: None)

        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.text == "ok"

    def test_readyz_ok_when_gpu_out_of_memory(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient OutOfMemoryError is memory pressure, not a wedge → stays ready.

        OutOfMemoryError subclasses RuntimeError; treating it as a wedge would
        depool/restart a worker that is merely full and will recover (issue #1025).
        """
        mark_ready()
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def oom(device: str) -> None:
            raise gpu_health.torch.cuda.OutOfMemoryError("CUDA out of memory")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", oom)

        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.text == "ok"

    def test_healthz_stays_cheap_when_gpu_wedged(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Liveness probe stays green on a wedged GPU; it must not touch CUDA.

        /healthz is process-alive only so transient back-pressure does not restart
        pods. The GPU probe lives behind /readyz and /livez.
        """
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def fail(device: str) -> None:
            raise AssertionError("/healthz must not run the GPU probe")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", fail)

        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"


class TestLivenessProbe:
    """Tests for the GPU-aware /livez liveness probe (issue #1025)."""

    def test_livez_returns_503_when_gpu_wedged(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wedged CUDA context fails /livez and latches the wedge.

        PyTorch cannot recover a wedged context, so a serving worker exits on
        the latch and its container restarts rather than serving 500s.
        """
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def boom(device: str) -> None:
            raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", boom)

        response = client.get("/livez")
        assert response.status_code == 503
        assert response.text == "gpu unhealthy"

    def test_livez_ignores_ready_state(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Liveness must not depend on the ready flag, so draining never restarts a pod.

        During graceful shutdown the worker marks itself not-ready (to drain), but
        the process and GPU are fine — /livez must stay green.
        """
        mark_not_ready()
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(gpu_health, "_run_gpu_probe", lambda device: None)

        response = client.get("/livez")
        assert response.status_code == 200
        assert response.text == "ok"
        mark_ready()

    def test_livez_returns_ok_on_cpu_worker(self, client: TestClient) -> None:
        """On a CPU-only worker there is no GPU to wedge, so /livez is green."""
        response = client.get("/livez")
        assert response.status_code == 200
        assert response.text == "ok"
