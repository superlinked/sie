"""Portable native audio-wheel build contract."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from packages.sie_audio_prep import build_wheel as audio_prep_wheel


def _write_valid_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("sie_audio_prep/sie_audio_prep.abi3.so", b"extension")
        wheel.writestr(
            f"sie_audio_prep-{audio_prep_wheel.AUDIO_PREP_VERSION}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: sie-audio-prep\nVersion: {audio_prep_wheel.AUDIO_PREP_VERSION}\n",
        )


def _write_minimal_source_tree(project_root: Path) -> Path:
    crate = project_root / "packages" / "sie_audio_prep"
    (crate / "src").mkdir(parents=True)
    (project_root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
    (crate / "Cargo.toml").write_text("[package]\nname = 'test'\n", encoding="utf-8")
    (crate / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (crate / "src" / "lib.rs").write_text("pub fn test() {}\n", encoding="utf-8")
    return crate


def test_portable_worker_wheel_tag_is_validated(tmp_path: Path) -> None:
    assert audio_prep_wheel.AUDIO_WHEEL_COMPATIBILITY == "manylinux_2_28"
    wheel_path = tmp_path / audio_prep_wheel.AUDIO_WHEEL_FILENAME
    _write_valid_wheel(wheel_path)

    audio_prep_wheel._validate_wheel(wheel_path)

    host_wheel = tmp_path / audio_prep_wheel.AUDIO_WHEEL_FILENAME.replace("manylinux_2_28", "manylinux_2_34")
    _write_valid_wheel(host_wheel)
    with pytest.raises(RuntimeError, match="unexpected sie-audio-prep wheel tag"):
        audio_prep_wheel._validate_wheel(host_wheel)


def test_wheel_cache_digest_includes_build_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = _write_minimal_source_tree(tmp_path)

    portable_digest = audio_prep_wheel._source_digest(crate)
    monkeypatch.setattr(audio_prep_wheel, "_BUILD_FINGERPRINT", "different-toolchain")

    assert audio_prep_wheel._source_digest(crate) != portable_digest


@pytest.mark.parametrize("kind", ["symlink", "shared"])
def test_wheel_cache_rejects_unsafe_preexisting_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _write_minimal_source_tree(tmp_path)
    monkeypatch.setattr(audio_prep_wheel.sys, "platform", "linux")
    monkeypatch.setattr(audio_prep_wheel.platform, "machine", lambda: "x86_64")
    cache_parent = tmp_path / "cache"
    cache_parent.mkdir()
    cache_root = cache_parent / f"sie-audio-prep-wheels-{os.getuid()}"
    if kind == "symlink":
        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        cache_root.symlink_to(attacker_dir, target_is_directory=True)
    else:
        cache_root.mkdir(mode=0o755)
        cache_root.chmod(0o755)
    monkeypatch.setattr(audio_prep_wheel.tempfile, "gettempdir", lambda: str(cache_parent))

    with pytest.raises(RuntimeError, match="unsafe audio prep wheel cache directory"):
        audio_prep_wheel.build_audio_prep_wheel(tmp_path)


def test_wheel_validation_rejects_symlinks(tmp_path: Path) -> None:
    wheel_path = tmp_path / audio_prep_wheel.AUDIO_WHEEL_FILENAME
    _write_valid_wheel(wheel_path)
    symlink = tmp_path / f"linked-{audio_prep_wheel.AUDIO_WHEEL_FILENAME}"
    symlink.symlink_to(wheel_path)

    with pytest.raises(RuntimeError, match="did not produce a wheel"):
        audio_prep_wheel._validate_wheel(symlink)


def test_wheel_cache_reuses_valid_artifact_from_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    crate = _write_minimal_source_tree(project_root)
    monkeypatch.setattr(audio_prep_wheel.sys, "platform", "linux")
    monkeypatch.setattr(audio_prep_wheel.platform, "machine", lambda: "x86_64")
    cache_parent = tmp_path / "cache"
    cache_parent.mkdir()
    monkeypatch.setattr(audio_prep_wheel.tempfile, "gettempdir", lambda: str(cache_parent))

    cache_root = cache_parent / f"sie-audio-prep-wheels-{os.getuid()}"
    cache_root.mkdir(mode=0o700)
    output_dir = cache_root / audio_prep_wheel._source_digest(crate)
    output_dir.mkdir(mode=0o700)
    wheel_path = output_dir / audio_prep_wheel.AUDIO_WHEEL_FILENAME
    _write_valid_wheel(wheel_path)

    assert audio_prep_wheel.build_audio_prep_wheel(project_root) == wheel_path
    assert (cache_root.stat().st_mode & 0o777) == 0o700


def test_optional_build_skips_when_toolchain_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # required=False (the Modal sandbox path) must degrade to a wheel-less
    # image on hosts without uvx/zig; the default stays a hard failure for
    # deploy paths.
    monkeypatch.setattr(audio_prep_wheel.sys, "platform", "linux")
    monkeypatch.setattr(audio_prep_wheel.platform, "machine", lambda: "x86_64")
    cache_parent = tmp_path / "cache"
    cache_parent.mkdir()
    monkeypatch.setattr(audio_prep_wheel.tempfile, "gettempdir", lambda: str(cache_parent))
    monkeypatch.setattr(audio_prep_wheel.shutil, "which", lambda name: None)
    project_root = tmp_path / "project"
    (project_root / "packages" / "sie_audio_prep" / "src").mkdir(parents=True)
    # Workspace-root lock (#2339): the digest fails loudly without it.
    (project_root / "Cargo.lock").write_text("version = 4\n")

    assert audio_prep_wheel.build_audio_prep_wheel(project_root, required=False) is None
    assert "skipped: uvx and zig not on PATH" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="required to build portable"):
        audio_prep_wheel.build_audio_prep_wheel(project_root)


def test_missing_workspace_lock_fails_loudly(tmp_path: Path) -> None:
    crate = tmp_path / "packages" / "sie_audio_prep"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\nname = 'test'\n")
    with pytest.raises(RuntimeError, match=r"workspace Cargo\.lock not found"):
        audio_prep_wheel._source_digest(crate)


def test_optional_build_skips_on_non_linux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(audio_prep_wheel.sys, "platform", "darwin")

    assert audio_prep_wheel.build_audio_prep_wheel(tmp_path, required=False) is None
    assert "Linux x86_64 only" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="deployed from Linux x86_64"):
        audio_prep_wheel.build_audio_prep_wheel(tmp_path)
