"""Shared pytest fixtures for sie_server tests.

Provides server lifecycle management for integration tests.
All server management is inline to avoid cross-package dependencies.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

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
        pass
    return "cpu"


def _find_free_port(start: int = 8090, end: int = 8200) -> int:
    """Find an available port in the given range.

    Binds to 0.0.0.0 (not 127.0.0.1) so the check matches Docker's bind
    address and avoids false positives where the port appears free on
    localhost but is already taken by a Docker container on 0.0.0.0.
    """
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))  # noqa: S104
                return port
        except OSError:
            continue
    msg = f"No free port found in range {start}-{end}"
    raise RuntimeError(msg)


def _wait_for_health(url: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> bool:
    """Wait for server health endpoint to respond 200."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            response = httpx.get(f"{url}/healthz", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(poll_interval_s)
    return False


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
    mise_path = shutil.which("mise")
    if mise_path is None:
        pytest.skip("mise not found in PATH - required for integration tests")

    models_dir = _project_root / "packages" / "sie_server" / "models"
    port = _find_free_port(8090, 8200)

    # Start server with default-bundle models for integration testing:
    # - bge-m3 (embedding with dense/sparse/multivector)
    # - gliner-bert-tiny (extraction) — only when gliner is installed
    models = "BAAI/bge-m3:bge_m3_flag,NeuML/gliner-bert-tiny"

    cmd = [
        mise_path,
        "run",
        "serve",
        "--",
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
        if not _wait_for_health(url, timeout_s=health_timeout):
            proc.terminate()
            proc.wait(timeout=10)
            pytest.fail(f"Server failed to start within {health_timeout:.0f}s — check server output above")

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
def docker_client(sie_docker_server: str) -> SIEClient:
    """Create an SIEClient connected to the Docker test server."""
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
