#!/usr/bin/env python3
# fmt: off
#MISE description="Helm chart operations for SIE cluster deployment"
#USAGE arg "[command]" help="Command: dependencies, lint, template, install, upgrade, uninstall, status"
#USAGE arg "[args]..." help="Additional arguments to pass to helm"
# fmt: on

"""Helm chart operations for SIE cluster deployment.

This task handles:
- Linting the Helm chart
- Rendering templates locally
- Installing/upgrading/uninstalling releases
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.colors import (
    log,
    log_error,
    log_success,
)
from common.env import apply_mise_env, get_usage_flag, resolve_project_root

apply_mise_env()

# Configuration
CHART_DIR = Path("deploy/helm/sie-cluster")
RELEASE_NAME = os.environ.get("SIE_HELM_RELEASE", "sie")
NAMESPACE = os.environ.get("SIE_HELM_NAMESPACE", "sie")
DEFAULT_VALIDATION_ARGS = ["--set", "payloadStore.enabled=false"]


@contextmanager
def _helm_config_sync_lock() -> Iterator[None]:
    """Serialize chart file sync because Helm reads those dirs during render."""
    root = resolve_project_root()
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    lock_path = Path(tempfile.gettempdir()) / f"sie-helm-config-sync-{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise RuntimeError(f"Helm config lock is not a regular file: {lock_path}")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _sync_configs_to_helm() -> None:
    """Copy bundle and model configs into the Helm chart's files/ directory.

    The embeddedConfigs feature uses .Files.Glob to generate ConfigMaps from
    files/bundles/*.yaml and files/models/*.yaml. These files are gitignored
    and must be copied from packages/sie_server/ before any helm operation
    that renders templates (template, install, upgrade, lint).
    """
    root = resolve_project_root()
    src_bundles = root / "packages" / "sie_server" / "bundles"
    src_models = root / "packages" / "sie_server" / "models"
    helm_files = root / "deploy" / "helm" / "sie-cluster" / "files"
    dst_bundles = helm_files / "bundles"
    dst_models = helm_files / "models"

    # Clean and re-copy bundles
    if dst_bundles.exists():
        shutil.rmtree(dst_bundles)
    dst_bundles.mkdir(parents=True, exist_ok=True)
    for f in src_bundles.glob("*.yaml"):
        shutil.copy2(f, dst_bundles / f.name)

    # Clean and re-copy models
    if dst_models.exists():
        shutil.rmtree(dst_models)
    dst_models.mkdir(parents=True, exist_ok=True)
    for f in src_models.glob("*.yaml"):
        shutil.copy2(f, dst_models / f.name)


def _cleanup_helm_configs() -> None:
    """Remove copied configs from Helm chart files/ directory."""
    root = resolve_project_root()
    helm_files = root / "deploy" / "helm" / "sie-cluster" / "files"
    for d in (helm_files / "bundles", helm_files / "models"):
        if d.exists():
            shutil.rmtree(d)


def check_helm() -> bool:
    """Check if helm is available."""
    result = subprocess.run(["mise", "exec", "--", "helm", "version"], capture_output=True, check=False)  # noqa: S607 — intentional partial path
    if result.returncode != 0:
        log_error("helm not found. Run: mise install")
        return False
    return True


def run_helm(args: list[str]) -> int:
    """Run helm command."""
    result = subprocess.run(["mise", "exec", "--", "helm", *args], check=False)  # noqa: S607, S603 — intentional partial path
    return result.returncode


def dependency_command() -> list[str]:
    """Return the locked chart dependency preparation command."""
    return ["dependency", "build", str(CHART_DIR)]


def validation_args(extra_args: list[str]) -> list[str]:
    """Provide deterministic, non-secret values for local chart validation."""
    return [*DEFAULT_VALIDATION_ARGS, *extra_args]


def show_help() -> None:
    """Show usage information."""
    log("SIE Helm Chart Operations")
    log("")
    log("Usage: mise run helm -- <command> [options]")
    log("")
    log("Commands:")
    log("  dependencies      Build dependencies from Chart.yaml and Chart.lock")
    log("  lint              Lint the Helm chart")
    log("  template          Render templates locally")
    log("  install           Install to cluster (dry-run by default)")
    log("  install --apply   Actually install to cluster")
    log("  upgrade           Upgrade existing installation")
    log("  uninstall         Remove from cluster")
    log("  status            Show release status")
    log("")
    log("Environment variables:")
    log(f"  SIE_HELM_RELEASE    Release name (default: {RELEASE_NAME})")
    log(f"  SIE_HELM_NAMESPACE  Namespace (default: {NAMESPACE})")
    log("")
    log("Examples:")
    log("  mise run helm -- dependencies")
    log("  mise run helm -- lint")
    log("  mise run helm -- template --set gateway.replicas=3")
    log("  mise run helm -- install --apply --set workers.pools.l4.enabled=true")


def cmd_dependencies(extra_args: list[str]) -> int:
    """Prepare locked chart dependencies."""
    if extra_args:
        log_error("dependencies does not accept additional arguments")
        return 1
    log(f"[helm] Preparing locked dependencies for: {CHART_DIR}")
    return run_helm(dependency_command())


def cmd_lint(extra_args: list[str]) -> int:
    """Lint the Helm chart."""
    log(f"[helm] Linting chart: {CHART_DIR}")
    if run_helm(["lint", str(CHART_DIR), *validation_args(extra_args)]) != 0:
        return 1
    log_success("Lint passed!")
    return 0


def cmd_template(extra_args: list[str]) -> int:
    """Render templates locally."""
    log(f"[helm] Rendering templates for: {CHART_DIR}")
    return run_helm(
        [
            "template",
            RELEASE_NAME,
            str(CHART_DIR),
            "--namespace",
            NAMESPACE,
            *validation_args(extra_args),
        ]
    )


def cmd_install(extra_args: list[str]) -> int:
    """Install to cluster."""
    apply = "--apply" in extra_args
    filtered_args = [a for a in extra_args if a != "--apply"]

    if not apply:
        log("[helm] Dry-run install (add --apply to actually install)")
        return run_helm(
            [
                "install",
                RELEASE_NAME,
                str(CHART_DIR),
                "--namespace",
                NAMESPACE,
                "--create-namespace",
                "--dry-run",
                *filtered_args,
            ]
        )

    log(f"[helm] Installing {RELEASE_NAME} to namespace {NAMESPACE}")
    if (
        run_helm(
            [
                "install",
                RELEASE_NAME,
                str(CHART_DIR),
                "--namespace",
                NAMESPACE,
                "--create-namespace",
                "--wait",
                "--timeout=15m",
                *filtered_args,
            ]
        )
        != 0
    ):
        return 1
    log_success("Installation complete!")
    return 0


def cmd_upgrade(extra_args: list[str]) -> int:
    """Upgrade existing installation."""
    log(f"[helm] Upgrading {RELEASE_NAME} in namespace {NAMESPACE}")
    if (
        run_helm(
            [
                "upgrade",
                RELEASE_NAME,
                str(CHART_DIR),
                "--namespace",
                NAMESPACE,
                "--wait",
                "--timeout=15m",
                *extra_args,
            ]
        )
        != 0
    ):
        return 1
    log_success("Upgrade complete!")
    return 0


def cmd_uninstall(extra_args: list[str]) -> int:
    """Uninstall from cluster."""
    log(f"[helm] Uninstalling {RELEASE_NAME} from namespace {NAMESPACE}")
    if (
        run_helm(
            [
                "uninstall",
                RELEASE_NAME,
                "--namespace",
                NAMESPACE,
                *extra_args,
            ]
        )
        != 0
    ):
        return 1
    log_success("Uninstall complete!")
    return 0


def cmd_status(extra_args: list[str]) -> int:
    """Show release status."""
    log(f"[helm] Status of {RELEASE_NAME}")
    return run_helm(
        [
            "status",
            RELEASE_NAME,
            "--namespace",
            NAMESPACE,
            *extra_args,
        ]
    )


def main() -> int:
    """Main entry point for the helm task."""
    if not check_helm():
        return 1

    command = get_usage_flag("command")

    # Get extra args from sys.argv (after the command)
    extra_args = sys.argv[2:]

    if not command or command in ("-h", "--help", "help"):
        show_help()
        return 0

    commands = {
        "dependencies": cmd_dependencies,
        "lint": cmd_lint,
        "template": cmd_template,
        "install": cmd_install,
        "upgrade": cmd_upgrade,
        "uninstall": cmd_uninstall,
        "status": cmd_status,
    }

    handler = commands.get(command)
    if not handler:
        log_error(f"Unknown command: {command}")
        show_help()
        return 1

    # Commands that render templates need bundle/model configs in files/
    needs_configs = command in ("lint", "template", "install", "upgrade")
    if needs_configs:
        if cmd_dependencies([]) != 0:
            return 1
        with _helm_config_sync_lock():
            _sync_configs_to_helm()
            try:
                return handler(extra_args)
            finally:
                _cleanup_helm_configs()
    return handler(extra_args)


if __name__ == "__main__":
    sys.exit(main())
