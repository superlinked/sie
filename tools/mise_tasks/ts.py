#!/usr/bin/env python3
# fmt: off
#MISE description="TypeScript SDK tasks"
#USAGE arg "<cmd>" help="Command: build, test, test-integration, test-node22, test-node24, test-browser, test-matrix, lint, lint-fix, typecheck"
# fmt: on

"""TypeScript SDK tasks.

This task handles:
- Building the SDK and integrations
- Running unit tests, integration tests, browser tests
- Multi-Node version testing (Node 22, 24)
- Linting and type checking
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Add common to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.colors import (
    log,
    log_error,
    log_success,
)
from common.env import apply_mise_env, get_usage_flag

apply_mise_env()

SDK_DIR = Path("packages/sie_ts_sdk")
INTEGRATIONS_DIR = Path("integrations")
HTTP_OK = 200


def is_port_in_use(port: int) -> bool:
    """Return whether any localhost TCP address cannot bind the port."""
    try:
        addresses = socket.getaddrinfo(
            "localhost",
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return True

    seen: set[tuple[int, tuple[object, ...]]] = set()
    for family, socktype, proto, _canonname, sockaddr in addresses:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        key = (family, sockaddr)
        if key in seen:
            continue
        seen.add(key)
        try:
            with socket.socket(family, socktype, proto) as probe:
                probe.bind(sockaddr)
        except OSError:
            return True

    return not seen


def run_pnpm(args: list[str], cwd: Path | None = None) -> int:
    """Run pnpm command."""
    result = subprocess.run(["mise", "exec", "--", "pnpm", *args], cwd=cwd, check=False)  # noqa: S607, S603 — intentional partial path
    return result.returncode


def cmd_build() -> int:
    """Build TypeScript SDK and integrations."""
    log("## Building TypeScript SDK and integrations")
    return run_pnpm(["run", "-r", "build"])


def cmd_test() -> int:
    """Run TypeScript SDK and integration unit tests."""
    log("## Running TypeScript SDK and integration unit tests")
    return run_pnpm(["run", "-r", "test"])


def wait_for_server(url: str, timeout: int = 180) -> bool:
    """Wait for server to be ready.

    Args:
        url: Health check URL.
        timeout: Maximum seconds to wait.

    Returns:
        True if server is ready, False if timeout.
    """
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310 — intentional HTTP health check
                if response.status == HTTP_OK:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    return False


def cmd_test_integration() -> int:
    """Run TypeScript integration tests (starts SIE server)."""
    log("## Running TypeScript integration tests (SDK + frameworks)")
    log("   This starts a SIE server and runs tests against it.")
    log("")

    test_port = int(os.environ.get("SIE_TEST_PORT", "8081"))
    server_process: subprocess.Popen | None = None

    def cleanup() -> None:
        """Clean up server process."""
        nonlocal server_process
        log("")
        log("Stopping SIE server...")

        if server_process is not None:
            try:
                # Send SIGTERM to process group
                pgid = os.getpgid(server_process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                log("   Server process group already stopped.")
                log("   Server cleanup complete.")
                return

            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            else:
                try:
                    server_process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass

        log("   Server cleanup complete.")

    atexit.register(cleanup)

    # Start server if not already running
    health_url = f"http://localhost:{test_port}/readyz"
    if not is_port_in_use(test_port):
        log(f"Starting SIE server on port {test_port}...")

        # Start server in new process group so we can kill it cleanly
        server_process = subprocess.Popen(  # noqa: S603 — intentional subprocess call
            ["mise", "run", "serve", "--", "-p", str(test_port)],  # noqa: S607 — intentional partial path
            start_new_session=True,
        )

        log("   Waiting for server to start...")

        if not wait_for_server(health_url, timeout=180):
            # Check if process died
            if server_process.poll() is not None:
                log_error("Server process died")
                return 1
            log_error("Server failed to start within 180s")
            return 1

        log_success(f"Server ready at http://localhost:{test_port}")
    else:
        if not wait_for_server(health_url, timeout=10):
            log_error(f"Port {test_port} is in use, but {health_url} is not ready")
            return 1
        log(f"Using existing ready server on port {test_port}")

    log("")
    os.environ["SIE_SERVER_URL"] = f"http://localhost:{test_port}"

    # Run SDK integration tests
    log("### SDK integration tests")
    if run_pnpm(["run", "test:integration"], cwd=SDK_DIR) != 0:
        return 1

    # Run framework integration tests
    for integration in sorted(INTEGRATIONS_DIR.glob("sie_ts_*")):
        config_file = integration / "vitest.integration.config.ts"
        if integration.is_dir() and config_file.exists():
            test_files = list((integration / "tests" / "integration").glob("**/*.integration.test.ts"))
            if not test_files:
                log("")
                log(f"### {integration.name} integration tests")
                log("   Skipping: no integration test files")
                continue

            log("")
            log(f"### {integration.name} integration tests")
            if run_pnpm(["run", "test:integration"], cwd=integration) != 0:
                return 1

    log("")
    log_success("All integration tests passed!")
    return 0


def cmd_test_node22() -> int:
    """Run TypeScript SDK unit tests with Node.js 22."""
    log("## Running TypeScript SDK unit tests with Node.js 22")
    result = subprocess.run(
        ["mise", "exec", "node@22", "--", "pnpm", "run", "test"],  # noqa: S607 — intentional partial path
        cwd=SDK_DIR,
        check=False,
    )
    if result.returncode == 0:
        log_success("Node.js 22 tests passed!")
    return result.returncode


def cmd_test_node24() -> int:
    """Run TypeScript SDK unit tests with Node.js 24."""
    log("## Running TypeScript SDK unit tests with Node.js 24")
    result = subprocess.run(
        ["mise", "exec", "node@24", "--", "pnpm", "run", "test"],  # noqa: S607 — intentional partial path
        cwd=SDK_DIR,
        check=False,
    )
    if result.returncode == 0:
        log_success("Node.js 24 tests passed!")
    return result.returncode


def cmd_test_browser() -> int:
    """Run TypeScript SDK browser compatibility tests."""
    log("## Running TypeScript SDK browser compatibility tests")
    log("   Using Playwright to test in real browser environment")
    log("")

    # Build SDK first (browser tests load from dist/)
    log("### Building SDK...")
    if run_pnpm(["run", "build"], cwd=SDK_DIR) != 0:
        return 1

    # Check if Playwright is installed
    result = subprocess.run(
        ["mise", "exec", "--", "pnpm", "exec", "playwright", "--version"],  # noqa: S607 — intentional partial path
        cwd=SDK_DIR,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        log("### Installing Playwright...")
        if run_pnpm(["exec", "playwright", "install", "chromium"], cwd=SDK_DIR) != 0:
            return 1

    log("")
    log("### Running browser tests...")
    if run_pnpm(["run", "test:browser"], cwd=SDK_DIR) != 0:
        return 1

    log("")
    log_success("Browser tests passed!")
    return 0


def cmd_test_matrix() -> int:
    """Run full test matrix (Node 22, Node 24, Browser)."""
    log("## Running full test matrix (Node 22, Node 24, Browser)")
    log("")

    log("=== Node.js 22 ===")
    if cmd_test_node22() != 0:
        return 1
    log("")

    log("=== Node.js 24 ===")
    if cmd_test_node24() != 0:
        return 1
    log("")

    log("=== Browser (Chromium) ===")
    if cmd_test_browser() != 0:
        return 1
    log("")

    log_success("All environment tests passed!")
    return 0


def cmd_lint() -> int:
    """Lint TypeScript SDK and integrations."""
    log("## Linting TypeScript SDK and integrations")
    return run_pnpm(["run", "-r", "lint"])


def cmd_lint_fix() -> int:
    """Fix TypeScript SDK and integration lint issues."""
    log("## Fixing TypeScript SDK and integration lint issues")
    return run_pnpm(["run", "-r", "lint-fix"])


def cmd_typecheck() -> int:
    """Type check TypeScript SDK and integrations."""
    log("## Type checking TypeScript SDK and integrations")
    return run_pnpm(["run", "-r", "typecheck"])


def show_usage() -> None:
    """Show usage information."""
    log("Usage: mise run ts -- <command>")
    log("")
    log("Commands:")
    log("  build            Build SDK and integrations (ESM + CJS + types)")
    log("  test             Run unit tests for SDK and integrations (mocked, fast)")
    log("  test-integration Run integration tests (starts real SIE server)")
    log("  test-node22      Run SDK unit tests with Node.js 22")
    log("  test-node24      Run SDK unit tests with Node.js 24")
    log("  test-browser     Run SDK browser compatibility tests (Playwright)")
    log("  test-matrix      Run full test matrix (Node 22 + Node 24 + Browser)")
    log("  lint             Check linting for SDK and integrations")
    log("  lint-fix         Fix lint issues")
    log("  typecheck        Run TypeScript type checker")


COMMANDS = {
    "build": cmd_build,
    "test": cmd_test,
    "test-integration": cmd_test_integration,
    "test-node22": cmd_test_node22,
    "test-node24": cmd_test_node24,
    "test-browser": cmd_test_browser,
    "test-matrix": cmd_test_matrix,
    "lint": cmd_lint,
    "lint-fix": cmd_lint_fix,
    "typecheck": cmd_typecheck,
}


def main() -> int:
    """Main entry point for the ts task."""
    cmd = get_usage_flag("cmd")

    if not cmd:
        show_usage()
        return 1

    handler = COMMANDS.get(cmd)
    if not handler:
        log_error(f"Unknown command: {cmd}")
        show_usage()
        return 1

    return handler()


if __name__ == "__main__":
    sys.exit(main())
