#!/usr/bin/env python3
# fmt: off
#MISE description="Helm chart operations for SIE cluster deployment"
#USAGE arg "[command]" help="Command: dependencies, lint, template, package, install, upgrade, uninstall, status"
#USAGE arg "[args]..." help="Additional arguments to pass to helm"
# fmt: on

"""Helm chart operations for SIE cluster deployment.

This task handles:
- Preparing the locked chart dependencies, reusing the archives in charts/
  when they already match a Chart.lock that is current for Chart.yaml
- Linting the Helm chart
- Rendering templates locally
- Installing/upgrading/uninstalling releases
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib
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


def dependency_repository_commands() -> list[list[str]]:
    """Return idempotent Helm repo setup commands derived from Chart.yaml."""
    root = resolve_project_root()
    chart = root / CHART_DIR / "Chart.yaml"
    repositories: list[str] = []
    for line in chart.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("repository:"):
            continue
        repository = stripped.split(":", maxsplit=1)[1].strip().strip("\"'")
        if repository.startswith(("https://", "http://")) and repository not in repositories:
            repositories.append(repository)
    return [
        [
            "repo",
            "add",
            f"sie-dependency-{hashlib.sha256(repository.encode()).hexdigest()[:12]}",
            repository,
            "--force-update",
        ]
        for repository in repositories
    ]


# A dependency's fields, in the order Helm writes them for the lock digest.
_DEPENDENCY_FIELDS = ("name", "version", "repository", "condition", "alias")


def _dependency_entries(path: Path) -> list[dict[str, str]] | None:
    """Return the ``dependencies`` entries of Chart.yaml or Chart.lock.

    Only plain ``key: value`` fields are read. Anything else under the list,
    such as tags, gives None, and the caller then treats the lock as unknown.
    """
    entries: list[dict[str, str]] = []
    inside = False
    for raw in path.read_text().splitlines():
        line = raw.split(" #", maxsplit=1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "-")):
            inside = line == "dependencies:"
            continue
        if not inside:
            continue
        item = line.strip()
        if item.startswith("- "):
            entries.append({})
            item = item[2:].strip()
        key, separator, value = item.partition(":")
        if not separator or not entries or key not in _DEPENDENCY_FIELDS:
            return None
        entries[-1][key] = value.strip().strip("\"'")
    return entries


def _as_helm_writes(entry: dict[str, str]) -> dict[str, str]:
    """Return one dependency the way Helm serializes it: ordered, empty optional fields left out."""
    return {key: entry.get(key, "") for key in _DEPENDENCY_FIELDS if key in ("name", "repository") or entry.get(key)}


def lock_matches_chart() -> bool:
    """Return whether Chart.lock was generated from the current Chart.yaml.

    Helm stores a digest of both dependency lists in the lock and refuses to
    build once Chart.yaml has changed without a new lock. The same digest is
    computed here the way Helm does it: SHA-256 over the JSON of the two lists.
    A digest that cannot be computed counts as no match.
    """
    chart = resolve_project_root() / CHART_DIR
    lock = chart / "Chart.lock"
    if not lock.is_file() or not (chart / "Chart.yaml").is_file():
        return False
    requested = _dependency_entries(chart / "Chart.yaml")
    locked = _dependency_entries(lock)
    if requested is None or locked is None:
        return False
    payload = json.dumps(
        [[_as_helm_writes(entry) for entry in requested], [_as_helm_writes(entry) for entry in locked]],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    for char in "&<>":  # Go's JSON encoder escapes these three.
        payload = payload.replace(char, f"\\u{ord(char):04x}")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"digest: sha256:{digest}" in lock.read_text().splitlines()


def locked_dependency_archives() -> dict[str, tuple[str, str]] | None:
    """Return each archive Chart.lock pins with its chart name and version, or None when unknown."""
    lock = resolve_project_root() / CHART_DIR / "Chart.lock"
    locked = _dependency_entries(lock) if lock.is_file() else None
    if not locked or not all(entry.get("name") and entry.get("version") for entry in locked):
        return None
    return {f"{entry['name']}-{entry['version']}.tgz": (entry["name"], entry["version"]) for entry in locked}


def _archive_identity(archive: Path) -> tuple[str, str] | None:
    """Return the chart name and version an archive declares, or None when it cannot be read."""
    try:
        with tarfile.open(archive, "r:gz") as chart:
            for member in chart:
                if member.isfile() and member.name.count("/") == 1 and member.name.endswith("/Chart.yaml"):
                    manifest = chart.extractfile(member).read().decode()
                    break
            else:
                return None
    except (OSError, EOFError, tarfile.TarError, zlib.error, UnicodeDecodeError):
        return None
    declared: dict[str, str] = {}
    for line in manifest.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in ("name", "version") and key not in declared:
            declared[key] = value.split(" #", maxsplit=1)[0].strip().strip("\"'")
    return declared.get("name", ""), declared.get("version", "")


def locked_dependencies_present() -> bool:
    """Return whether charts/ holds exactly what the current Chart.yaml locks.

    Helm downloads every dependency again on each build, even when the
    archive is already there. No build is needed when the lock is still the
    one generated from Chart.yaml and charts/ matches it exactly: nothing
    missing, nothing left over from another version, and every archive
    declaring the chart name and version the lock pins.
    """
    expected = locked_dependency_archives()
    charts = resolve_project_root() / CHART_DIR / "charts"
    if expected is None or not charts.is_dir() or not lock_matches_chart():
        return False
    return {entry.name for entry in charts.iterdir()} == set(expected) and all(
        _archive_identity(charts / name) == identity for name, identity in expected.items()
    )


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
    log("  package           Package and validate a chart with embedded configs (--destination DIR)")
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
    for command in dependency_repository_commands():
        if run_helm(command) != 0:
            return 1
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


def validate_packaged_configs(archive: Path) -> None:
    """Require the retained chart to contain the exact checked-in catalogs."""
    root = resolve_project_root()
    chart_name = CHART_DIR.name
    expected = {
        f"{chart_name}/files/{kind}/{path.name}": path.read_bytes()
        for kind in ("bundles", "models")
        for path in (root / "packages/sie_server" / kind).glob("*.yaml")
    }
    actual = {}
    with tarfile.open(archive, "r:gz") as chart:
        for member in chart.getmembers():
            if not any(member.name.startswith(f"{chart_name}/files/{kind}/") for kind in ("bundles", "models")):
                continue
            if not member.isfile() or member.name in actual:
                raise ValueError("packaged chart catalog must contain unique regular files")
            contents = chart.extractfile(member)
            if contents is None:
                raise ValueError("packaged chart catalog member has no contents")
            actual[member.name] = contents.read()
    if not expected or actual != expected:
        raise ValueError("packaged chart catalogs differ from the checked-in bundle/model configs")


def cmd_package(extra_args: list[str]) -> int:
    """Retain only an archive that passes checks after config staging is removed."""
    parser = argparse.ArgumentParser(prog="mise run helm -- package")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(extra_args)
    with tempfile.TemporaryDirectory(prefix="sie-helm-package-") as temporary:
        if run_helm(["package", str(CHART_DIR), "--destination", temporary]) != 0:
            return 1
        archives = list(Path(temporary).glob("*.tgz"))
        if len(archives) != 1:
            raise ValueError("Helm package must produce exactly one chart archive")
        archive = archives[0]
        validate_packaged_configs(archive)
        # Validation must depend on the archive, never on loose staged files.
        _cleanup_helm_configs()
        if run_helm(["lint", str(archive), *validation_args([])]) != 0:
            return 1
        for options in ([], ["--set", "config.enabled=false", "--set", "gateway.embeddedConfigs.enabled=true"]):
            result = subprocess.run(  # noqa: S603
                [  # noqa: S607 - use the repository's mise-managed Helm executable
                    "mise",
                    "exec",
                    "--",
                    "helm",
                    "template",
                    RELEASE_NAME,
                    str(archive),
                    "--namespace",
                    NAMESPACE,
                    *validation_args(options),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                log_error(result.stderr)
                return result.returncode
        args.destination.mkdir(parents=True, exist_ok=True)
        destination = args.destination / archive.name
        if destination.exists() and destination.read_bytes() != archive.read_bytes():
            raise ValueError("refusing to replace a different retained chart archive")
        shutil.copyfile(archive, destination)
        log_success(f"Validated chart retained at {destination}")
    return 0


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
        "package": cmd_package,
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
    needs_configs = command in ("lint", "template", "package", "install", "upgrade")
    if needs_configs:
        # An explicit `dependencies` always builds. Before the other commands
        # the build runs unless charts/ already holds what Chart.yaml locks.
        if locked_dependencies_present():
            log(f"[helm] Locked dependencies already present for: {CHART_DIR}")
        elif cmd_dependencies([]) != 0:
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
