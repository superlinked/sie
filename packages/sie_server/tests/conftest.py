"""Shared pytest fixtures for sie_server tests.

Provides server lifecycle management for integration tests.
All server management is inline to avoid cross-package dependencies.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, TextIO

import httpx
import pytest
from sie_sdk import SIEClient
from sie_sdk.client.async_ import SIEAsyncClient

logger = logging.getLogger(__name__)

# Project root (for finding models directory, Dockerfiles, etc.)
_project_root = Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session")
def device() -> str:
    """Auto-detected device for integration tests (cuda:0, mps, or cpu)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        # torch not installed — caller will fall back to CPU.
        pass
    return "cpu"


def _find_free_port(start: int = 8090, end: int = 8200) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    msg = f"No free port found in range {start}-{end}"
    raise RuntimeError(msg)


def _wait_for_health(
    url: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Wait for server health endpoint to respond 200."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            response = httpx.get(f"{url}/healthz", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(poll_interval_s)
    return False


class _SIEServerProcess:
    """Test-local SIE server process with model dependency resolution.

    This deliberately lives with the server tests to keep subprocess
    management self-contained.
    """

    def __init__(
        self,
        *,
        port: int,
        models_dir: str | Path,
        instrumentation: bool = False,
    ) -> None:
        self._port = port
        models_path = Path(models_dir)
        self._models_dir = models_path if models_path.is_absolute() else _project_root / models_path
        self._instrumentation = instrumentation
        self._process: subprocess.Popen[str] | None = None
        self._requirements_path: Path | None = None
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None

    def _resolve_dependencies(self, model: str, device: str) -> str:
        command = [
            sys.executable,
            "-m",
            "sie_server.cli",
            "resolve-deps",
            "--models-dir",
            str(self._models_dir),
            "--models",
            model,
        ]
        if not device.lower().startswith("cuda"):
            command.append("--cpu")
        result = subprocess.run(  # noqa: S603 — intentional subprocess call
            command,
            cwd=_project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        if result.returncode != 0:
            logger.warning("Failed to resolve test server dependencies: %s", result.stderr)
            return ""
        return result.stdout

    def start(self, model: str, device: str) -> None:
        if self._process is not None:
            raise RuntimeError("SIE test server is already running")

        requirements = self._resolve_dependencies(model, device)
        if requirements.strip():
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                delete=False,
                prefix="sie-server-test-requirements-",
            ) as requirements_file:
                requirements_file.write(requirements)
                self._requirements_path = Path(requirements_file.name)

        # Preserve the already-materialized workspace while layering
        # model-specific requirements. Project mode would reconcile the
        # virtual root and remove the workspace packages installed by CI.
        command = ["uv", "run", "--no-project", "--python", sys.executable]
        if self._requirements_path is not None:
            command.extend(["--with-requirements", str(self._requirements_path)])
        command.extend(
            [
                "python",
                "-m",
                "sie_server.cli",
                "serve",
                "--port",
                str(self._port),
                "--host",
                "127.0.0.1",
                "--device",
                device,
                "--models-dir",
                str(self._models_dir),
                "--models",
                model,
            ]
        )
        if self._instrumentation:
            command.append("--tracing")

        self._log_file = tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            suffix=".log",
            delete=False,
            prefix="sie-server-test-",
        )
        self._log_path = Path(self._log_file.name)
        logger.info("Starting test server: %s", " ".join(command))
        self._process = subprocess.Popen(  # noqa: S603 — intentional subprocess call
            command,
            cwd=_project_root,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def wait_ready(self, timeout_s: float) -> None:
        if self._process is None:
            raise RuntimeError("SIE test server has not been started")
        if _wait_for_health(self.get_url(), timeout_s=timeout_s, proc=self._process):
            return

        log_file = self._log_file
        log_path = self._log_path
        if log_file is None or log_path is None:
            raise RuntimeError("SIE test server log is not initialized")
        log_file.flush()
        output = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(output.rstrip().splitlines()[-80:])
        pytest.fail(f"SIE test server did not become ready within {timeout_s:.0f}s.\n{tail}")

    def get_url(self) -> str:
        if self._process is None:
            raise RuntimeError("SIE test server is not running")
        return f"http://127.0.0.1:{self._port}"

    def stop(self) -> None:
        process = self._process
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    # The child may exit between inspecting it and signaling its process group.
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        # The child may exit after the timeout but before the fallback signal.
                        pass
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=5)
        finally:
            self._process = None

            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            if self._log_path is not None:
                self._log_path.unlink(missing_ok=True)
                self._log_path = None
            if self._requirements_path is not None:
                self._requirements_path.unlink(missing_ok=True)
                self._requirements_path = None


@pytest.fixture(scope="session")
def sie_server_process_factory() -> type[_SIEServerProcess]:
    """Return the test-local configurable server process."""
    return _SIEServerProcess


# =============================================================================
# Subprocess-based SIE server (for regular integration tests)
# =============================================================================


@pytest.fixture(scope="session")
def sie_server(device: str) -> Generator[str]:
    """Start a SIE server via subprocess for integration tests.

    Yields the server URL. Server is stopped after all tests in the module.

    Usage:
        @pytest.mark.integration
        def test_something(sie_server: str):
            client = SIEClient(sie_server)
            # ... test code ...
    """
    models_dir = _project_root / "packages" / "sie_server" / "models"
    port = _find_free_port(8090, 8200)

    # Start server with default-bundle models for integration testing:
    # - bge-m3 (embedding with dense/sparse/multivector)
    # - gliner-bert-tiny (extraction) — only when gliner is installed
    models = "BAAI/bge-m3:bge_m3_flag,NeuML/gliner-bert-tiny"

    cmd = [
        sys.executable,
        "-m",
        "sie_server.cli",
        "serve",
        "-p",
        str(port),
        "-d",
        device,
        "--models-dir",
        str(models_dir),
        "-m",
        models,
    ]

    logger.info("Starting SIE server: %s", " ".join(cmd))

    proc = subprocess.Popen(  # noqa: S603 — intentional subprocess call
        cmd,
        cwd=_project_root,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    url = f"http://localhost:{port}"

    try:
        health_timeout = float(os.environ.get("SIE_TEST_SERVER_TIMEOUT", "120"))
        if not _wait_for_health(url, timeout_s=health_timeout, proc=proc):
            returncode = proc.poll()
            if returncode is None:
                proc.terminate()
                proc.wait(timeout=10)
                pytest.fail(f"Server failed to start within {health_timeout:.0f}s — check server output above")
            pytest.fail(f"Server process exited before health check passed (exit code {returncode})")

        logger.info("Integration test server ready at %s", url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logger.info("Integration test server stopped")


@pytest.fixture
def sie_client(sie_server: str) -> SIEClient:
    """Create an SIEClient connected to the test server.

    Usage:
        @pytest.mark.integration
        def test_something(sie_client: SIEClient):
            result = sie_client.encode("model", [Item(text="hello")])
    """
    return SIEClient(sie_server, timeout_s=180.0)


@pytest.fixture
def async_client(sie_server: str) -> SIEAsyncClient:
    return SIEAsyncClient(sie_server, timeout_s=180.0)


# =============================================================================
# Docker-based SIE server (for Docker image integration tests)
# =============================================================================


def _get_docker_client() -> Any:
    """Get Docker client, or skip test if unavailable."""
    try:
        import docker

        return docker.from_env(timeout=600)
    except ImportError:
        pytest.skip("docker package not installed")
    except Exception as e:  # noqa: BLE001 — Docker API errors are varied
        pytest.skip(f"Docker not available: {e}")
    # Unreachable: both except branches call pytest.skip (NoReturn).
    raise RuntimeError("unreachable")


def _build_docker_image(
    dockerfile: str = "Dockerfile.cpu",
    tag: str = "sie-server:test",
) -> None:
    """Build SIE Docker image using docker buildx (supports BuildKit features).

    In CI, the image should be pre-built by the workflow (set SIE_DOCKER_IMAGE).
    This function is used for local development only.
    """
    dockerfile_path = _project_root / "packages" / "sie_server" / dockerfile

    if not dockerfile_path.exists():
        pytest.fail(f"Dockerfile not found: {dockerfile_path}")

    logger.info("Building SIE Docker image from %s", dockerfile_path)

    # Use docker buildx for BuildKit support (required for --mount=type=cache)
    # Build for linux/amd64 to avoid ARM64 compatibility issues with some packages
    # Use --progress=plain to get streamable output (default auto uses TTY features)
    cmd = [
        "docker",
        "buildx",
        "build",
        "--progress=plain",
        "--platform",
        "linux/amd64",
        "-f",
        f"packages/sie_server/{dockerfile}",
        "-t",
        tag,
        "--build-arg",
        "BUNDLE=default",
        "--load",  # Load into local docker images
        str(_project_root),
    ]

    logger.info("Docker build command: %s", " ".join(cmd))

    proc: subprocess.Popen[str] | None = None
    try:
        # Stream build output in real-time
        proc = subprocess.Popen(  # noqa: S603 — intentional subprocess call
            cmd,
            cwd=_project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines: list[str] = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                logger.info("[docker build] %s", line)

        returncode = proc.wait(timeout=600)

        if returncode != 0:
            output = "\n".join(output_lines[-50:])
            pytest.fail(f"Docker build failed with exit code {returncode}.\nOutput:\n{output}")

        logger.info("SIE Docker image built: %s", tag)

    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        pytest.fail("Docker build timed out after 10 minutes")
    except Exception as e:  # noqa: BLE001 — Docker build errors are varied
        pytest.fail(f"Failed to build Docker image: {e}")


@pytest.fixture(scope="session")
def sie_docker_server() -> Generator[str]:
    """Build and start SIE Docker container for tests.

    Yields the server URL. Container is stopped after all tests in the module.

    This fixture tests the actual Docker image, catching issues like:
    - Missing directories (e.g., HF cache)
    - Permission problems
    - Dependency issues

    Set SIE_DOCKER_IMAGE env var to use a pre-built image (skips build).

    Regression test for: https://github.com/superlinked/sie-internal/issues/10
    """
    docker_client = _get_docker_client()

    # Use pre-built image if SIE_DOCKER_IMAGE is set, otherwise build
    image_tag = os.environ.get("SIE_DOCKER_IMAGE", "")
    if image_tag:
        logger.info("Using pre-built Docker image: %s", image_tag)
    else:
        image_tag = "sie-server:test"
        _build_docker_image(dockerfile="Dockerfile.cpu", tag=image_tag)

    # Find free port
    port = _find_free_port(8090, 8200)

    # Use host's HF cache to speed up model downloads
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hf_cache = hf_home / "hub"
    hf_cache.mkdir(parents=True, exist_ok=True)

    # Use a small model for faster testing
    model = "sentence-transformers/all-MiniLM-L6-v2"

    container_config = {
        "image": image_tag,
        "detach": True,
        "ports": {"8080/tcp": port},
        "command": [
            "serve",
            "--host",
            "0.0.0.0",  # noqa: S104 — intentional bind to all interfaces in container
            "--port",
            "8080",
            "--models-dir",
            "/app/models",
            "--device",
            "cpu",
            "-m",
            model,
        ],
        "remove": True,
        "volumes": {
            str(hf_cache): {"bind": "/app/.cache/huggingface/hub", "mode": "rw"},
        },
        "environment": {
            "HF_HOME": "/app/.cache/huggingface",
            # Propagate the deployment-env tag from the host (set to "ci" by
            # the GH workflow, "development" by `mise run serve`) into the
            # container so heartbeats from the dockerised sie-server don't
            # land in the "unknown" telemetry bucket.
            "SIE_DEPLOYMENT_ENV": os.environ.get("SIE_DEPLOYMENT_ENV", "development"),
        },
    }

    logger.info("Starting SIE Docker container on port %d", port)

    container = docker_client.containers.run(**container_config)
    container_id = container.id
    url = f"http://localhost:{port}"

    try:
        # Wait for container to be running
        start = time.monotonic()
        while time.monotonic() - start < 30:
            container.reload()
            if container.status == "running":
                break
            time.sleep(1.0)
        else:
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Container did not start within 30s. Logs:\n{logs}")

        # Wait for health check (longer timeout for model download)
        if not _wait_for_health(url, timeout_s=600.0, poll_interval_s=2.0):
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Container health check failed. Logs:\n{logs}")

        logger.info("Docker test server ready at %s", url)
        yield url

    finally:
        try:
            container = docker_client.containers.get(container_id)
            container.stop(timeout=10)
        except Exception as e:  # noqa: BLE001 — Docker cleanup must not raise
            logger.warning("Error stopping container: %s", e)
        logger.info("Docker test server stopped")


@pytest.fixture(scope="session")
def sie_docker_client(sie_docker_server: str) -> SIEClient:
    """Create an SIEClient connected to the Docker test server.

    Named ``sie_docker_client`` (not ``docker_client``) to avoid clashing with the
    local ``docker_client`` variable inside fixtures that hold the actual Docker
    SDK client.
    """
    return SIEClient(sie_docker_server, timeout_s=180.0)


# =============================================================================
# Docker-based SIE Gateway (for gateway image tests)
# =============================================================================


def _build_config_image(tag: str = "sie-config:test") -> None:
    """Build SIE Config Service Docker image using docker buildx.

    In CI, the image should be pre-built by the workflow (set SIE_CONFIG_IMAGE).
    This function is used for local development only.
    """
    dockerfile_path = _project_root / "packages" / "sie_config" / "Dockerfile"

    if not dockerfile_path.exists():
        pytest.fail(f"Gateway Dockerfile not found: {dockerfile_path}")

    logger.info("Building SIE Config Service Docker image from %s", dockerfile_path)

    cmd = [
        "docker",
        "buildx",
        "build",
        "--progress=plain",
        "--platform",
        "linux/amd64",
        "-f",
        "packages/sie_config/Dockerfile",
        "-t",
        tag,
        "--load",
        str(_project_root),
    ]

    logger.info("Docker build command: %s", " ".join(cmd))

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=_project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines: list[str] = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                logger.info("[docker build] %s", line)

        returncode = proc.wait(timeout=600)

        if returncode != 0:
            output = "\n".join(output_lines[-50:])
            pytest.fail(f"Config service Docker build failed with exit code {returncode}.\nOutput:\n{output}")

        logger.info("SIE Config Service Docker image built: %s", tag)

    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        pytest.fail("Config service Docker build timed out after 10 minutes")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to build Config Service Docker image: {e}")


@pytest.fixture(scope="session")
def sie_docker_config() -> Generator[str]:
    """Build and start SIE Config Service Docker container for tests.

    Yields the server URL. Container is stopped after all tests in the session.

    Starts the config service -- just validates the image
    can start and respond to health checks.

    Set SIE_CONFIG_IMAGE env var to use a pre-built image (skips build).
    """
    docker_client = _get_docker_client()

    image_tag = os.environ.get("SIE_CONFIG_IMAGE", "")
    if image_tag:
        logger.info("Using pre-built Config Service Docker image: %s", image_tag)
    else:
        image_tag = "sie-config:test"
        _build_config_image(tag=image_tag)

    port = _find_free_port(8090, 8200)

    container_config = {
        "image": image_tag,
        "detach": True,
        "ports": {"8080/tcp": port},
        "command": [
            "--port",
            "8080",
            "--host",
            "0.0.0.0",  # noqa: S104
        ],
        "remove": True,
    }

    logger.info("Starting SIE Config Service Docker container on port %d", port)

    container = docker_client.containers.run(**container_config)
    container_id = container.id
    url = f"http://localhost:{port}"

    try:
        start = time.monotonic()
        while time.monotonic() - start < 30:
            container.reload()
            if container.status == "running":
                break
            time.sleep(1.0)
        else:
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Gateway container did not start within 30s. Logs:\n{logs}")

        if not _wait_for_health(url, timeout_s=60.0, poll_interval_s=1.0):
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Gateway container health check failed. Logs:\n{logs}")

        logger.info("Gateway Docker test server ready at %s", url)
        yield url

    finally:
        try:
            container = docker_client.containers.get(container_id)
            container.stop(timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error stopping gateway container: %s", e)
        logger.info("Gateway Docker test server stopped")


# =============================================================================
# Docker-based SIE Gateway (for gateway image tests)
# =============================================================================


def _build_gateway_image(tag: str = "sie-gateway:test") -> None:
    """Build SIE Gateway Docker image using docker buildx.

    In CI, the image should be pre-built by the workflow (set SIE_GATEWAY_IMAGE).
    This function is used for local development only.
    """
    dockerfile_path = _project_root / "packages" / "sie_gateway" / "Dockerfile"

    if not dockerfile_path.exists():
        pytest.fail(f"Gateway Dockerfile not found: {dockerfile_path}")

    logger.info("Building SIE Gateway Docker image from %s", dockerfile_path)

    cmd = [
        "docker",
        "buildx",
        "build",
        "--progress=plain",
        "--platform",
        "linux/amd64",
        "-f",
        "packages/sie_gateway/Dockerfile",
        "-t",
        tag,
        "--load",
        str(_project_root),
    ]

    logger.info("Docker build command: %s", " ".join(cmd))

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=_project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines: list[str] = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                logger.info("[docker build] %s", line)

        returncode = proc.wait(timeout=600)

        if returncode != 0:
            output = "\n".join(output_lines[-50:])
            pytest.fail(f"Gateway Docker build failed with exit code {returncode}.\nOutput:\n{output}")

        logger.info("SIE Gateway Docker image built: %s", tag)

    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        pytest.fail("Gateway Docker build timed out after 10 minutes")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to build Gateway Docker image: {e}")


@pytest.fixture(scope="session")
def sie_docker_gateway() -> Generator[str]:
    """Build and start SIE Gateway Docker container for tests.

    Yields the server URL. Container is stopped after all tests in the session.

    Starts the gateway without any worker URLs and without Kubernetes discovery --
    just validates the image can start and respond to health/readiness probes.

    Set SIE_GATEWAY_IMAGE env var to use a pre-built image (skips build).
    """
    docker_client = _get_docker_client()

    image_tag = os.environ.get("SIE_GATEWAY_IMAGE", "")
    if image_tag:
        logger.info("Using pre-built Gateway Docker image: %s", image_tag)
    else:
        image_tag = "sie-gateway:test"
        _build_gateway_image(tag=image_tag)

    port = _find_free_port(8090, 8200)

    container_config = {
        "image": image_tag,
        "detach": True,
        "ports": {"8080/tcp": port},
        # Override the Dockerfile CMD to skip --kubernetes (no cluster to
        # discover workers from in the test environment).
        "command": [
            "--port",
            "8080",
            "--host",
            "0.0.0.0",  # noqa: S104
        ],
        "remove": True,
    }

    logger.info("Starting SIE Gateway Docker container on port %d", port)

    container = docker_client.containers.run(**container_config)
    container_id = container.id
    url = f"http://localhost:{port}"

    try:
        start = time.monotonic()
        while time.monotonic() - start < 30:
            container.reload()
            if container.status == "running":
                break
            time.sleep(1.0)
        else:
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Gateway container did not start within 30s. Logs:\n{logs}")

        if not _wait_for_health(url, timeout_s=60.0, poll_interval_s=1.0):
            logs = container.logs().decode("utf-8", errors="replace")
            pytest.fail(f"Gateway container health check failed. Logs:\n{logs}")

        logger.info("Gateway Docker test server ready at %s", url)
        yield url

    finally:
        try:
            container = docker_client.containers.get(container_id)
            container.stop(timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error stopping gateway container: %s", e)
        logger.info("Gateway Docker test server stopped")
