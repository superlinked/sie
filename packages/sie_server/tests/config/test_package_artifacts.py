from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from sie_server.config.package_artifacts import (
    PackageArtifactDeclaration,
    PackageArtifactMode,
    verify_staged_package_artifacts,
)


def _write_manifest(root: Path, *, artifact_path: str = "layout/model.bin") -> PackageArtifactDeclaration:
    artifact = root / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"immutable model bytes")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "artifacts": [
                    {
                        "path": artifact_path,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return PackageArtifactDeclaration(
        mode=PackageArtifactMode.STAGED,
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def test_verifies_manifest_and_every_staged_artifact(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)

    verified = verify_staged_package_artifacts(declaration)

    assert verified.root == tmp_path
    assert verified.manifest_sha256 == declaration.manifest_sha256
    assert verified.artifact_count == 1


def test_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    mismatched = PackageArtifactDeclaration(
        mode=PackageArtifactMode.STAGED,
        manifest_path=declaration.manifest_path,
        manifest_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="manifest sha256"):
        verify_staged_package_artifacts(mismatched)


def test_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    artifact = tmp_path / "layout" / "model.bin"
    original = artifact.read_bytes()
    artifact.write_bytes(b"X" + original[1:])

    with pytest.raises(ValueError, match="artifact sha256"):
        verify_staged_package_artifacts(declaration)


def test_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path, artifact_path="../model.bin")

    with pytest.raises(ValueError, match="manifest is invalid"):
        verify_staged_package_artifacts(declaration)


def test_rejects_symlinked_artifact(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    declaration = _write_manifest(tmp_path)
    artifact = tmp_path / "layout" / "model.bin"
    target = tmp_path_factory.mktemp("outside-root") / "target.bin"
    artifact.rename(target)
    artifact.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        verify_staged_package_artifacts(declaration)


def test_rejects_unlisted_file(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    (tmp_path / "undeclared.bin").write_bytes(b"effective but unbound")

    with pytest.raises(ValueError, match="unlisted file"):
        verify_staged_package_artifacts(declaration)


def test_rejects_unlisted_symlink(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    (tmp_path / "undeclared-link.bin").symlink_to(tmp_path / "layout" / "model.bin")

    with pytest.raises(ValueError, match="symlink"):
        verify_staged_package_artifacts(declaration)


def test_rejects_unlisted_directory(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    (tmp_path / "undeclared-directory").mkdir()

    with pytest.raises(ValueError, match="unlisted directory"):
        verify_staged_package_artifacts(declaration)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO nodes are unavailable on this platform")
def test_rejects_special_node(tmp_path: Path) -> None:
    declaration = _write_manifest(tmp_path)
    os.mkfifo(tmp_path / "undeclared-fifo")

    with pytest.raises(ValueError, match="special node"):
        verify_staged_package_artifacts(declaration)
