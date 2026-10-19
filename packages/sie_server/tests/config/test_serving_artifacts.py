from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError
from sie_server.config import serving_artifacts
from sie_server.config.serving_artifacts import (
    CTRANSLATE2_TORCH_VERSION,
    CTRANSLATE2_TRANSFORMERS_VERSION,
    CTRANSLATE2_VERSION,
    SERVING_ARTIFACT_SCHEMA,
    ServingArtifactDeclaration,
    ServingArtifactManifest,
    atomic_rename_directory_noreplace,
    canonical_manifest_bytes,
    verify_and_materialize_serving_artifact,
)

SOURCE_ID = "google/source-model"
SOURCE_REVISION = "a" * 40
DERIVED_ID = "superlinked/derived-model"
DERIVED_REVISION = "b" * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage_artifact(
    root: Path,
    *,
    source_id: str = SOURCE_ID,
    source_revision: str = SOURCE_REVISION,
) -> tuple[ServingArtifactDeclaration, Path]:
    repo_root = root / "models--superlinked--derived-model"
    snapshot = repo_root / "snapshots" / DERIVED_REVISION
    snapshot.mkdir(parents=True)
    artifacts = {
        "config.json": (b'{"model_type":"Transformer"}\n', "model.config"),
        "model.bin": (b"converted model bytes", "model.weights"),
        "tokenizer_config.json": (b'{"tokenizer_class":"T5Tokenizer"}\n', "tokenizer.config"),
        "spiece.model": (b"sentencepiece bytes", "tokenizer.sentencepiece"),
    }
    entries = []
    for relative, (content, role) in artifacts.items():
        path = snapshot / relative
        path.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": len(content),
                "sha256": _sha256(content),
            }
        )
    tokenizer_config = artifacts["tokenizer_config.json"][0]
    manifest = ServingArtifactManifest.model_validate(
        {
            "schema": SERVING_ARTIFACT_SCHEMA,
            "source": {"hf_id": source_id, "hf_revision": source_revision},
            "converter": {
                "name": "ct2-transformers-converter",
                "version": CTRANSLATE2_VERSION,
                "torch_version": CTRANSLATE2_TORCH_VERSION,
                "transformers_version": CTRANSLATE2_TRANSFORMERS_VERSION,
                "compute_type": "bfloat16",
                "recipe_sha256": "c" * 64,
            },
            "runtime": {"minimum_version": CTRANSLATE2_VERSION, "maximum_version": CTRANSLATE2_VERSION},
            "tokenizer": {
                "class_name": "T5Tokenizer",
                "config_sha256": _sha256(tokenizer_config),
                "files": [
                    {"path": "spiece.model", "role": "tokenizer.sentencepiece"},
                    {"path": "tokenizer_config.json", "role": "tokenizer.config"},
                ],
            },
            "artifacts": entries,
        }
    )
    manifest_bytes = canonical_manifest_bytes(manifest)
    manifest_path = snapshot / "sie-serving-artifact.json"
    manifest_path.write_bytes(manifest_bytes)
    declaration = ServingArtifactDeclaration(
        format="ctranslate2",
        repo_id=DERIVED_ID,
        revision=DERIVED_REVISION,
        manifest_path="sie-serving-artifact.json",
        manifest_sha256=_sha256(manifest_bytes),
        compute_type="bfloat16",
    )
    return declaration, repo_root


def _link_through_shared_blob_store(repo_root: Path, relative: str, *, store: Path | None = None) -> Path:
    """Lay one snapshot file out the way huggingface_hub stores a Xet-backed blob."""
    snapshot_entry = repo_root / "snapshots" / DERIVED_REVISION / relative
    content = snapshot_entry.read_bytes()
    xet_hash = _sha256(b"xet:" + content)
    store = store if store is not None else repo_root.parent / "blobs"
    store_entry = store / xet_hash[:2] / xet_hash
    store_entry.parent.mkdir(parents=True, exist_ok=True)
    store_entry.write_bytes(content)
    store_entry.chmod(0o444)
    repo_blob = repo_root / "blobs" / _sha256(content)[:40]
    repo_blob.parent.mkdir(exist_ok=True)
    repo_blob.symlink_to(os.path.relpath(store_entry, repo_blob.parent))
    snapshot_entry.unlink()
    snapshot_entry.symlink_to(os.path.relpath(repo_blob, snapshot_entry.parent))
    return store_entry


def _materialize(
    declaration: ServingArtifactDeclaration,
    repo_root: Path,
    cache_root: Path,
):
    return verify_and_materialize_serving_artifact(
        declaration,
        source_hf_id=SOURCE_ID,
        source_hf_revision=SOURCE_REVISION,
        cached_repo_root=repo_root,
        materialized_cache_root=cache_root,
    )


@pytest.mark.parametrize("path", ["/manifest.json", "../manifest.json", "a//manifest.json", "a\\manifest.json"])
def test_declaration_rejects_unsafe_manifest_path(path: str) -> None:
    with pytest.raises(ValidationError, match=r"relative POSIX|must not contain"):
        ServingArtifactDeclaration(
            format="ctranslate2",
            repo_id=DERIVED_ID,
            revision=DERIVED_REVISION,
            manifest_path=path,
            manifest_sha256="d" * 64,
            compute_type="bfloat16",
        )


def test_declaration_rejects_mutable_revision_and_uppercase_digest() -> None:
    with pytest.raises(ValidationError):
        ServingArtifactDeclaration(
            format="ctranslate2",
            repo_id=DERIVED_ID,
            revision="main",
            manifest_path="manifest.json",
            manifest_sha256="D" * 64,
            compute_type="bfloat16",
        )


def test_atomic_publish_refuses_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        atomic_rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_materialization_falls_back_when_noreplace_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")

    def unsupported_rename(_source: Path, destination: Path) -> None:
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), destination)

    monkeypatch.setattr(serving_artifacts, "atomic_rename_directory_noreplace", unsupported_rename)

    first = _materialize(declaration, repo_root, tmp_path / "materialized")
    second = _materialize(declaration, repo_root, tmp_path / "materialized")

    assert first.root == second.root
    assert (first.root / "model.bin").read_bytes() == b"converted model bytes"


def test_portable_publish_requires_nonempty_source_and_absent_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    with pytest.raises(ValueError, match="nonempty"):
        serving_artifacts._rename_verified_nonempty_directory(source, destination)

    (source / "model.bin").write_bytes(b"model")
    destination.mkdir()
    with pytest.raises(FileExistsError):
        serving_artifacts._rename_verified_nonempty_directory(source, destination)


def test_concurrent_portable_publishers_cannot_replace_nonempty_winner(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    sources = [tmp_path / "source-a", tmp_path / "source-b"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "model.bin").write_bytes(str(index).encode())
    barrier = threading.Barrier(2)

    def publish(source: Path) -> int | None:
        barrier.wait()
        try:
            serving_artifacts._rename_verified_nonempty_directory(source, destination)
        except OSError as exc:
            return exc.errno
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, sources))

    assert results.count(None) == 1
    assert next(result for result in results if result is not None) in {errno.EEXIST, errno.ENOTEMPTY}
    assert (destination / "model.bin").read_bytes() in {b"0", b"1"}


def test_materializes_canonical_read_only_tree_and_reuses_verified_bytes(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    cache_root = tmp_path / "materialized"

    first = _materialize(declaration, repo_root, cache_root)
    second = _materialize(declaration, repo_root, cache_root)

    assert first.root == second.root == cache_root / declaration.manifest_sha256
    assert first.manifest_sha256 == declaration.manifest_sha256
    assert first.artifact_count == 4
    assert sorted(path.relative_to(first.root).as_posix() for path in first.root.rglob("*") if path.is_file()) == [
        "config.json",
        "model.bin",
        "spiece.model",
        "tokenizer_config.json",
    ]
    assert all(path.stat().st_mode & 0o222 == 0 for path in first.root.rglob("*") if path.is_file())


def test_materialized_bytes_are_independent_from_later_source_mutation(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    source = repo_root / "snapshots" / DERIVED_REVISION / "model.bin"
    source.chmod(0o444)

    verified = _materialize(declaration, repo_root, tmp_path / "materialized")
    served = verified.root / "model.bin"

    assert source.stat().st_ino != served.stat().st_ino
    source.chmod(0o644)
    source.write_bytes(b"mutated after admission")
    assert served.read_bytes() == b"converted model bytes"
    assert served.stat().st_mode & 0o222 == 0


def test_rejects_source_identity_mismatch(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub", source_id="other/source")

    with pytest.raises(ValueError, match="source identity"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_effective_profile_compute_type_mismatch(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    mismatched = declaration.model_copy(update={"compute_type": "int8_bfloat16"})

    with pytest.raises(ValueError, match="compute_type does not match the effective profile"):
        _materialize(mismatched, repo_root, tmp_path / "materialized")


def test_rejects_manifest_digest_and_noncanonical_json(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    manifest_path = repo_root / "snapshots" / DERIVED_REVISION / declaration.manifest_path
    parsed = json.loads(manifest_path.read_bytes())
    noncanonical = json.dumps(parsed, indent=2).encode()
    manifest_path.write_bytes(noncanonical)
    matching = declaration.model_copy(update={"manifest_sha256": _sha256(noncanonical)})

    with pytest.raises(ValueError, match="not canonical"):
        _materialize(matching, repo_root, tmp_path / "materialized")


def test_rejects_file_drift_before_materialization(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    (repo_root / "snapshots" / DERIVED_REVISION / "model.bin").write_bytes(b"drifted model bytes")

    with pytest.raises(ValueError, match=r"size|sha256"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_tokenizer_class_mismatch_before_publishing_materialized_root(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    manifest_path = repo_root / "snapshots" / DERIVED_REVISION / declaration.manifest_path
    payload = json.loads(manifest_path.read_bytes())
    payload["tokenizer"]["class_name"] = "OtherTokenizer"
    manifest = ServingArtifactManifest.model_validate(payload)
    manifest_bytes = canonical_manifest_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    declaration = declaration.model_copy(update={"manifest_sha256": _sha256(manifest_bytes)})
    cache_root = tmp_path / "materialized"

    with pytest.raises(ValueError, match="tokenizer class"):
        _materialize(declaration, repo_root, cache_root)

    assert not (cache_root / declaration.manifest_sha256).exists()


def test_rejects_source_symlink_escaping_repository_cache(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    model = repo_root / "snapshots" / DERIVED_REVISION / "model.bin"
    outside = tmp_path / "outside.bin"
    model.rename(outside)
    model.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_accepts_files_hugging_face_keeps_in_its_cache_wide_shared_blob_store(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    for relative in ("model.bin", "spiece.model", declaration.manifest_path):
        _link_through_shared_blob_store(repo_root, relative)
    model = repo_root / "snapshots" / DERIVED_REVISION / "model.bin"
    assert model.resolve().is_relative_to(tmp_path / "hub" / "blobs")

    verified = _materialize(declaration, repo_root, tmp_path / "materialized")

    assert verified.artifact_count == 4
    assert (verified.root / "model.bin").read_bytes() == b"converted model bytes"
    assert (verified.root / "model.bin").stat().st_ino != model.resolve().stat().st_ino


@pytest.mark.parametrize(
    "relocate",
    [
        pytest.param(
            lambda hub, path: hub.parent / "elsewhere" / "blobs" / path.parent.name / path.name, id="other-cache"
        ),
        pytest.param(lambda hub, path: hub / "blobs" / path.name, id="no-prefix-directory"),
        pytest.param(lambda hub, path: hub / "blobs" / "zz" / path.name, id="wrong-prefix"),
        pytest.param(lambda hub, path: hub / "blobs" / path.parent.name / "model.bin", id="not-a-content-hash"),
        pytest.param(lambda hub, path: hub / "other" / path.parent.name / path.name, id="other-directory"),
    ],
)
def test_rejects_source_outside_the_shared_blob_store_layout(
    tmp_path: Path, relocate: Callable[[Path, Path], Path]
) -> None:
    hub = tmp_path / "hub"
    declaration, repo_root = _stage_artifact(hub)
    store_entry = _link_through_shared_blob_store(repo_root, "model.bin")
    moved = relocate(hub, store_entry)
    moved.parent.mkdir(parents=True, exist_ok=True)
    store_entry.rename(moved)
    repo_blob = next((repo_root / "blobs").iterdir())
    repo_blob.unlink()
    repo_blob.symlink_to(os.path.relpath(moved, repo_blob.parent))

    with pytest.raises(ValueError, match=r"escapes its repository cache: 'model\.bin'"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_a_shared_blob_store_linked_from_outside_the_cache(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    declaration, repo_root = _stage_artifact(hub)
    outside = tmp_path / "outside-store"
    outside.mkdir()
    (hub / "blobs").symlink_to(outside, target_is_directory=True)
    _link_through_shared_blob_store(repo_root, "model.bin")

    with pytest.raises(ValueError, match=r"escapes its repository cache: 'model\.bin'"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_manifest_outside_the_repository_cache_and_shared_blob_store(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    declaration, repo_root = _stage_artifact(hub)
    _link_through_shared_blob_store(repo_root, declaration.manifest_path, store=tmp_path / "elsewhere" / "blobs")

    with pytest.raises(ValueError, match="manifest is not a safe regular file"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_snapshot_revision_alias(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    snapshot = repo_root / "snapshots" / DERIVED_REVISION
    actual_snapshot = repo_root / "snapshots" / ("e" * 40)
    snapshot.rename(actual_snapshot)
    snapshot.symlink_to(actual_snapshot, target_is_directory=True)

    with pytest.raises(ValueError, match="snapshot escapes"):
        _materialize(declaration, repo_root, tmp_path / "materialized")


def test_rejects_partial_or_extra_reused_tree(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    cache_root = tmp_path / "materialized"
    destination = cache_root / declaration.manifest_sha256
    destination.mkdir(parents=True)
    (destination / "extra.bin").write_bytes(b"unbound")
    (destination / "extra.bin").chmod(0o444)
    destination.chmod(0o555)

    with pytest.raises(ValueError, match="unlisted file"):
        _materialize(declaration, repo_root, cache_root)


def test_rejects_writable_reused_tree(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    cache_root = tmp_path / "materialized"
    verified = _materialize(declaration, repo_root, cache_root)
    verified.root.chmod(0o755)

    with pytest.raises(ValueError, match="root is writable"):
        _materialize(declaration, repo_root, cache_root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO nodes are unavailable on this platform")
def test_rejects_special_node_in_reused_tree(tmp_path: Path) -> None:
    declaration, repo_root = _stage_artifact(tmp_path / "hub")
    cache_root = tmp_path / "materialized"
    destination = cache_root / declaration.manifest_sha256
    destination.mkdir(parents=True)
    os.mkfifo(destination / "model.bin")
    destination.chmod(0o555)

    try:
        with pytest.raises(ValueError, match="special node"):
            _materialize(declaration, repo_root, cache_root)
    finally:
        destination.chmod(0o700)
        (destination / "model.bin").unlink()
        destination.rmdir()
