"""Check every saved image overlay with the Docker-pinned PEP751 reader.

These dry runs check installer compatibility for all release targets and the
documented CPU arm64 targets without installing model packages. Actual image
builds additionally verify the installed inventory before copying the receipt.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from sie_server.bundle_lock import UV_VERSION, Target, check_lock

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPILER_SPEC = importlib.util.spec_from_file_location("bundle_lock_compiler", _REPO_ROOT / "tools/ci/bundle_lock.py")
assert _COMPILER_SPEC is not None
assert _COMPILER_SPEC.loader is not None
compiler = importlib.util.module_from_spec(_COMPILER_SPEC)
_COMPILER_SPEC.loader.exec_module(compiler)
_TARGETS = compiler.targets(_REPO_ROOT)


def _uv_executable() -> str:
    # uv run prepends the project environment, which contains the separately
    # root-pinned uv. CI resolves the Docker installer before entering it.
    found = os.environ.get("SIE_BUNDLE_LOCK_UV") or shutil.which("uv")
    if not found:
        pytest.fail("SIE_RUN_IMAGE_RESOLUTION=1 but uv is not installed")
    version = subprocess.run(  # noqa: S603 - version-check the explicitly selected uv executable
        [found, "--version"], check=True, text=True, capture_output=True
    ).stdout.split()
    if version[:2] != ["uv", UV_VERSION]:
        pytest.fail(
            f"bundle image validation requires the Docker-pinned uv {UV_VERSION}; found {version[:2]} at {found}"
        )
    return found


def test_missing_uv_fails_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_BUNDLE_LOCK_UV", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(pytest.fail.Exception, match="uv is not installed"):
        _uv_executable()


def test_explicit_installer_wins_over_the_project_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BUNDLE_LOCK_UV", "/docker-pinned/uv")
    monkeypatch.setattr(shutil, "which", lambda _name: pytest.fail("must not select the project uv"))

    def version(command, **_kwargs):
        assert command == ["/docker-pinned/uv", "--version"]
        return SimpleNamespace(stdout=f"uv {UV_VERSION}\n")

    monkeypatch.setattr(subprocess, "run", version)
    assert _uv_executable() == "/docker-pinned/uv"


def test_explicit_wrong_installer_is_not_a_version_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BUNDLE_LOCK_UV", "/wrong/uv")
    monkeypatch.setattr(shutil, "which", lambda _name: pytest.fail("must not silently fall back"))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="uv 0.5.31\n"))
    with pytest.raises(pytest.fail.Exception, match="requires the Docker-pinned uv"):
        _uv_executable()


@pytest.mark.skipif(
    os.environ.get("SIE_RUN_IMAGE_RESOLUTION") != "1",
    reason="opt-in installer compatibility check",
)
@pytest.mark.parametrize("target", _TARGETS, ids=[target.name for target in _TARGETS])
def test_saved_bundle_is_consumed_without_fresh_resolution(target: Target) -> None:
    check_lock(_REPO_ROOT / compiler.LOCKS, _REPO_ROOT / compiler.BUNDLES, _REPO_ROOT / "uv.lock", target)
    with tempfile.TemporaryDirectory() as temp:
        command = [
            _uv_executable(),
            "pip",
            "sync",
            "--dry-run",
            "--no-config",
            "--python",
            sys.executable,
            "--python-platform",
            target.python_platform,
            "--python-version",
            "3.12",
            "--target",
            str(Path(temp) / "bundle-libs"),
            str(_REPO_ROOT / compiler.LOCKS / target.name / "pylock.toml"),
        ]
        completed = subprocess.run(  # noqa: S603 - fixed installer arguments and checked repository lock
            command,
            check=False,
            capture_output=True,
            text=True,
            env=compiler.resolver_environment(),
            cwd=temp,
            timeout=900,
        )
    assert completed.returncode == 0, (completed.stderr or completed.stdout or "")[-4000:]
