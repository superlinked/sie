"""Resolve each release-matrix bundle the way its image install does.

The image build runs on linux/amd64. ``uv pip install --dry-run`` applies that
marker environment on any host, with the same ``--target``, constraint, and
requirements arguments as the Dockerfiles. pip on macOS evaluates markers for
Darwin and misses the linux nvidia conflict those constraints used to create.
``--no-config`` keeps the repo's uv torch index from standing in for PyPI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from sie_server.bundle_requirements import lock_constraint_lines, parse_uv_lock, resolve_bundle_requirements

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUNDLES = _REPO_ROOT / "packages" / "sie_server" / "bundles"
# CUDA 12 and 13 images are Ubuntu 22.04 (glibc 2.35). manylinux_2_28 rejects
# wheels such as sglang 0.5.20, which publish only manylinux_2_34 tags.
_LINUX_PLATFORM = "x86_64-manylinux_2_35"
_PYTHON_VERSION = "3.12"
# Dockerfile.cuda12's non-default index branch. Embedding is not in the
# release matrix today; keep the same case split as the image.
_SGLANG_CUDA12_BUNDLES = frozenset({"sglang", "sglang-embedding", "sglang-vision-extract"})
_SCRUBBED_ENV = (
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_STRATEGY",
    "UV_TORCH_BACKEND",
    "UV_CONSTRAINT",
    "UV_OVERRIDE",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_CONSTRAINT",
)


def _release_targets() -> list[tuple[str, str]]:
    data = json.loads((_REPO_ROOT / ".github" / "release-matrix.json").read_text())
    targets = [(str(platform), str(bundle)) for platform in data["platforms"] for bundle in data["bundles"]]
    targets.extend((str(item["platform"]), str(item["bundle"])) for item in data.get("include", []))
    return targets


_TARGETS = _release_targets()


def _uv_executable() -> str:
    sibling = Path(sys.executable).resolve().with_name("uv")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = shutil.which("uv")
    if found:
        return found
    pytest.fail(
        "SIE_RUN_IMAGE_RESOLUTION=1 but uv is not installed; the resolution job must not pass without resolving"
    )


def _image_requirements(platform: str, bundle: str) -> list[str]:
    bundle_path = _BUNDLES / f"{bundle}.yaml"
    deps = yaml.safe_load(bundle_path.read_text())["deps"]
    # Dockerfile.cpu passes --cpu, which drops flash-attn, xformers, and fla-core.
    return resolve_bundle_requirements(deps, exclude_cuda=platform == "cpu")


def _resolve_command(platform: str, bundle: str, target: Path, requirements: Path, constraints: Path) -> list[str]:
    """Arguments that affect resolution, matched to the platform Dockerfile."""
    command = [
        _uv_executable(),
        "pip",
        "install",
        "--dry-run",
        "--no-config",
        "--python-platform",
        _LINUX_PLATFORM,
        "--python-version",
        _PYTHON_VERSION,
        "--target",
        str(target),
    ]
    if platform == "cuda13":
        # packages/sie_server/Dockerfile.cuda13
        command.extend(
            [
                "--index-strategy",
                "unsafe-best-match",
                "--prerelease=allow",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu130",
                "--extra-index-url",
                "https://docs.sglang.ai/whl/cu130/",
                "--constraint",
                str(constraints),
            ]
        )
    elif platform == "cpu":
        # packages/sie_server/Dockerfile.cpu. pip searches every configured index;
        # uv's default first-index strategy would not.
        command.extend(
            [
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "--extra-index-url",
                "https://pypi.org/simple",
                "--index-strategy",
                "unsafe-best-match",
                "-c",
                str(constraints),
            ]
        )
    else:
        # packages/sie_server/Dockerfile.cuda12
        if bundle in _SGLANG_CUDA12_BUNDLES:
            command.extend(
                [
                    "--extra-index-url",
                    "https://docs.sglang.ai/whl/cu129/",
                    "--index-strategy",
                    "unsafe-best-match",
                ]
            )
        command.extend(["-c", str(constraints)])
    command.extend(["-r", str(requirements)])
    return command


def _resolver_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _SCRUBBED_ENV:
        env.pop(key, None)
    return env


def test_missing_uv_fails_the_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(pytest.fail.Exception, match="uv is not installed"):
        _uv_executable()


@pytest.mark.skipif(
    os.environ.get("SIE_RUN_IMAGE_RESOLUTION") != "1",
    reason="needs network access to package indexes",
)
@pytest.mark.parametrize(
    ("platform", "bundle"),
    _TARGETS,
    ids=[f"{platform}-{bundle}" for platform, bundle in _TARGETS],
)
def test_release_bundle_constraints_resolve(platform: str, bundle: str) -> None:
    requirements = _image_requirements(platform, bundle)
    if not requirements:
        return
    constraints = lock_constraint_lines(requirements, parse_uv_lock(_REPO_ROOT / "uv.lock"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        requirements_path = root / "requirements.txt"
        constraints_path = root / "constraints.txt"
        target = root / "bundle-libs"
        target.mkdir()
        requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        constraints_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        command = _resolve_command(platform, bundle, target, requirements_path, constraints_path)
        assert str(target) in command
        assert str(requirements_path) in command
        assert str(constraints_path) in command
        completed = subprocess.run(  # noqa: S603 - fixed uv resolution command built above
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_resolver_env(),
            cwd=tmp,
            timeout=900,
        )
    details = (completed.stderr or completed.stdout or "")[-4000:]
    assert completed.returncode == 0, details
