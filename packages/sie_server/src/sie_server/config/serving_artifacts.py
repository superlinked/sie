"""Immutable derived serving-artifact declarations and materialization."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SERVING_ARTIFACT_KEY = "serving_artifact"
SERVING_ARTIFACT_PATH_KEY = "artifact_path"
SERVING_ARTIFACT_COMPUTE_TYPE_KEY = "ct2_compute_type"

SERVING_ARTIFACT_SCHEMA = "sie-ctranslate2-serving-artifact-v1"
CTRANSLATE2_VERSION = "4.8.1"
CTRANSLATE2_TORCH_VERSION = "2.9.1"
CTRANSLATE2_TRANSFORMERS_VERSION = "4.57.6"
CTRANSLATE2_COMPUTE_TYPES = (
    "bfloat16",
    "float16",
    "float32",
    "int8",
    "int8_bfloat16",
    "int8_float16",
    "int8_float32",
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"
_REPO_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"
_ROLE_PATTERN = r"^[a-z][a-z0-9_.-]{0,63}$"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_TOKENIZER_CONFIG_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


def _validate_relative_posix_path(value: str, *, label: str) -> str:
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"{label} must be a relative POSIX path")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{label} must not contain empty, '.' or '..' segments")
    if PurePosixPath(value).is_absolute():
        raise ValueError(f"{label} must be relative")
    return value


class ServingArtifactDeclaration(BaseModel):
    """Catalog identity of one immutable derived serving repository snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["ctranslate2"]
    repo_id: str = Field(pattern=_REPO_ID_PATTERN)
    revision: str = Field(pattern=_REVISION_PATTERN)
    manifest_path: str = Field(min_length=1, max_length=1024)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    compute_type: str = Field(pattern=_ROLE_PATTERN)

    @field_validator("manifest_path")
    @classmethod
    def validate_manifest_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value, label="serving artifact manifest_path")

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, value: str) -> str:
        if value not in CTRANSLATE2_COMPUTE_TYPES:
            raise ValueError("serving artifact compute_type is unsupported")
        return value


class ServingArtifactSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_id: str = Field(pattern=_REPO_ID_PATTERN)
    hf_revision: str = Field(pattern=_REVISION_PATTERN)


class ServingArtifactConverter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["ct2-transformers-converter"]
    version: Literal["4.8.1"]
    torch_version: Literal["2.9.1"]
    transformers_version: Literal["4.57.6"]
    compute_type: str = Field(pattern=_ROLE_PATTERN)
    recipe_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, value: str) -> str:
        if value not in CTRANSLATE2_COMPUTE_TYPES:
            raise ValueError("serving artifact compute_type is unsupported")
        return value


class ServingArtifactRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_version: Literal["4.8.1"]
    maximum_version: Literal["4.8.1"]


class ServingArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=1024)
    role: str = Field(pattern=_ROLE_PATTERN)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value, label="serving artifact path")

    @field_validator("size_bytes", mode="before")
    @classmethod
    def reject_boolean_size(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("serving artifact size_bytes must be an integer")
        return value


class TokenizerArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=1024)
    role: str = Field(pattern=r"^tokenizer\.[a-z][a-z0-9_.-]{0,53}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value, label="tokenizer artifact path")


class ServingArtifactTokenizer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    class_name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    files: tuple[TokenizerArtifactReference, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_unique_files(self) -> ServingArtifactTokenizer:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("tokenizer artifact inventory contains duplicate paths")
        return self


class ServingArtifactManifest(BaseModel):
    """Closed canonical manifest for a CTranslate2 serving tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["sie-ctranslate2-serving-artifact-v1"] = Field(alias="schema")
    source: ServingArtifactSource
    converter: ServingArtifactConverter
    runtime: ServingArtifactRuntime
    tokenizer: ServingArtifactTokenizer
    artifacts: tuple[ServingArtifactEntry, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> ServingArtifactManifest:
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("serving artifact manifest contains duplicate paths")
        sorted_paths = sorted(paths)
        for parent, candidate in pairwise(sorted_paths):
            if candidate.startswith(f"{parent}/"):
                raise ValueError("serving artifact manifest contains a file/directory path collision")

        artifacts_by_path = {artifact.path: artifact for artifact in self.artifacts}
        tokenizer_paths = {entry.path for entry in self.tokenizer.files}
        declared_tokenizer_paths = {
            artifact.path for artifact in self.artifacts if artifact.role.startswith("tokenizer.")
        }
        if tokenizer_paths != declared_tokenizer_paths:
            raise ValueError("tokenizer file inventory must exactly match tokenizer-role artifacts")
        for entry in self.tokenizer.files:
            artifact = artifacts_by_path.get(entry.path)
            if artifact is None or artifact.role != entry.role:
                raise ValueError("tokenizer file role must match its artifact entry")
        tokenizer_configs = [
            artifacts_by_path[entry.path] for entry in self.tokenizer.files if entry.role == "tokenizer.config"
        ]
        if len(tokenizer_configs) != 1 or tokenizer_configs[0].sha256 != self.tokenizer.config_sha256:
            raise ValueError("tokenizer config digest must match exactly one tokenizer.config artifact")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedServingArtifact:
    root: Path
    manifest_sha256: str
    repo_id: str
    revision: str
    compute_type: str
    artifact_count: int
    manifest: ServingArtifactManifest


def parse_serving_artifact_declaration(loadtime: dict[str, object]) -> ServingArtifactDeclaration | None:
    """Parse one effective profile's nested serving-artifact declaration."""
    for key in (
        SERVING_ARTIFACT_PATH_KEY,
        SERVING_ARTIFACT_COMPUTE_TYPE_KEY,
    ):
        if key in loadtime:
            raise ValueError(f"'{key}' is loader-owned and must not be declared")

    if SERVING_ARTIFACT_KEY not in loadtime:
        return None
    value = loadtime[SERVING_ARTIFACT_KEY]
    try:
        return ServingArtifactDeclaration.model_validate(value)
    except ValueError as exc:
        raise ValueError("loadtime.serving_artifact is invalid") from exc


def canonical_manifest_bytes(manifest: ServingArtifactManifest) -> bytes:
    """Return the one accepted byte representation of a serving manifest."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def normalized_recipe_sha256(recipe: dict[str, object]) -> str:
    """Hash a normalized conversion recipe without filesystem or timestamp data."""
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(
    declaration: ServingArtifactDeclaration,
    *,
    cached_repo_root: Path,
) -> tuple[Path, ServingArtifactManifest]:
    repo_root = cached_repo_root.resolve(strict=True)
    snapshot_root = repo_root / "snapshots" / declaration.revision
    try:
        resolved_snapshot = snapshot_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("derived serving artifact snapshot is not staged at the declared revision") from exc
    if (
        resolved_snapshot != snapshot_root
        or not resolved_snapshot.is_dir()
        or not resolved_snapshot.is_relative_to(repo_root)
    ):
        raise ValueError("derived serving artifact snapshot escapes its repository cache")

    manifest_path = snapshot_root.joinpath(*PurePosixPath(declaration.manifest_path).parts)
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("derived serving artifact manifest does not exist") from exc
    if not resolved_manifest.is_file() or not resolved_manifest.is_relative_to(repo_root):
        raise ValueError("derived serving artifact manifest is not a safe regular file")
    if resolved_manifest.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("derived serving artifact manifest exceeds the size limit")

    manifest_bytes = resolved_manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != declaration.manifest_sha256:
        raise ValueError("derived serving artifact manifest sha256 does not match the catalog declaration")
    try:
        manifest = ServingArtifactManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise ValueError("derived serving artifact manifest is invalid") from exc
    if manifest_bytes != canonical_manifest_bytes(manifest):
        raise ValueError("derived serving artifact manifest is not canonical JSON")
    return resolved_snapshot, manifest


def _verify_manifest_compatibility(
    declaration: ServingArtifactDeclaration,
    manifest: ServingArtifactManifest,
    *,
    source_hf_id: str,
    source_hf_revision: str,
    runtime_version: str,
) -> None:
    if manifest.source.hf_id != source_hf_id or manifest.source.hf_revision != source_hf_revision:
        raise ValueError("derived serving artifact source identity does not match the model catalog")
    if manifest.converter.version != CTRANSLATE2_VERSION:
        raise ValueError("derived serving artifact converter version is unsupported")
    if manifest.converter.torch_version != CTRANSLATE2_TORCH_VERSION:
        raise ValueError("derived serving artifact torch converter version is unsupported")
    if manifest.converter.transformers_version != CTRANSLATE2_TRANSFORMERS_VERSION:
        raise ValueError("derived serving artifact transformers converter version is unsupported")
    if manifest.converter.compute_type != declaration.compute_type:
        raise ValueError("derived serving artifact compute_type does not match the effective profile")
    if runtime_version not in {manifest.runtime.minimum_version, manifest.runtime.maximum_version}:
        raise ValueError("derived serving artifact runtime compatibility does not include this worker")
    if declaration.format != "ctranslate2":
        raise ValueError("unsupported derived serving artifact format")
    if declaration.manifest_path in {artifact.path for artifact in manifest.artifacts}:
        raise ValueError("derived serving artifact manifest must not be part of the served file inventory")


def _expected_directories(manifest: ServingArtifactManifest) -> set[str]:
    return {
        parent.as_posix()
        for artifact in manifest.artifacts
        for parent in PurePosixPath(artifact.path).parents
        if parent != PurePosixPath(".")
    }


def _verify_materialized_tree(root: Path, manifest: ServingArtifactManifest) -> None:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("materialized serving artifact tree does not exist") from exc
    if resolved_root != root or not root.is_dir():
        raise ValueError("materialized serving artifact root must be a non-symlink directory")
    if root.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("materialized serving artifact root is writable")

    expected_files = {artifact.path for artifact in manifest.artifacts}
    expected_directories = _expected_directories(manifest)
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    if entry.is_symlink():
                        raise ValueError(f"materialized serving artifact tree contains a symlink: {relative!r}")
                    if entry.is_dir(follow_symlinks=False):
                        if relative not in expected_directories:
                            raise ValueError(
                                f"materialized serving artifact tree contains an unlisted directory: {relative!r}"
                            )
                        observed_directories.add(relative)
                        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                            raise ValueError(f"materialized serving artifact directory is writable: {relative!r}")
                        scan(path)
                    elif entry.is_file(follow_symlinks=False):
                        if relative not in expected_files:
                            raise ValueError(
                                f"materialized serving artifact tree contains an unlisted file: {relative!r}"
                            )
                        observed_files.add(relative)
                    else:
                        raise ValueError(f"materialized serving artifact tree contains a special node: {relative!r}")
        except OSError as exc:
            raise ValueError("materialized serving artifact tree cannot be inventoried") from exc

    scan(root)
    missing_files = sorted(expected_files - observed_files)
    missing_directories = sorted(expected_directories - observed_directories)
    if missing_files or missing_directories:
        raise ValueError(
            "materialized serving artifact tree does not match the manifest: "
            f"missing_files={missing_files!r}, missing_directories={missing_directories!r}"
        )

    for artifact in manifest.artifacts:
        path = root.joinpath(*PurePosixPath(artifact.path).parts)
        resolved = path.resolve(strict=True)
        if resolved != path or not path.is_file() or not resolved.is_relative_to(root):
            raise ValueError(f"materialized serving artifact is not a safe regular file: {artifact.path!r}")
        file_stat = path.stat()
        if file_stat.st_size != artifact.size_bytes:
            raise ValueError(f"materialized serving artifact size does not match the manifest: {artifact.path!r}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"materialized serving artifact sha256 does not match the manifest: {artifact.path!r}")
        if file_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"materialized serving artifact is writable: {artifact.path!r}")


def _safe_source_file(snapshot_root: Path, repo_root: Path, artifact: ServingArtifactEntry) -> Path:
    source_path = snapshot_root.joinpath(*PurePosixPath(artifact.path).parts)
    try:
        resolved = source_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"derived serving artifact is missing: {artifact.path!r}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(repo_root):
        raise ValueError(f"derived serving artifact escapes its repository cache: {artifact.path!r}")
    file_stat = resolved.stat()
    if file_stat.st_size != artifact.size_bytes:
        raise ValueError(f"derived serving artifact size does not match the manifest: {artifact.path!r}")
    if _sha256_file(resolved) != artifact.sha256:
        raise ValueError(f"derived serving artifact sha256 does not match the manifest: {artifact.path!r}")
    return resolved


def _make_read_only(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for file_name in file_names:
            (directory_path / file_name).chmod(0o444)
        for directory_name in directory_names:
            (directory_path / directory_name).chmod(0o555)
        directory_path.chmod(0o555)


def _remove_staging_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        directory_path.chmod(0o700)
        for directory_name in directory_names:
            (directory_path / directory_name).chmod(0o700)
        # Unlinking read-only files only requires a writable parent directory.
    shutil.rmtree(root)


def atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        result = renameat2(
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(source)),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(1),
        )
    else:
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unavailable")
        result = renamex_np(
            ctypes.c_char_p(os.fsencode(source)),
            ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(4),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), destination)


def _rename_verified_nonempty_directory(source: Path, destination: Path) -> None:
    """Atomically publish a verified tree on filesystems without no-replace rename."""
    if source.is_symlink() or not source.is_dir() or next(source.iterdir(), None) is None:
        raise ValueError("portable serving-artifact publication requires a nonempty directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
    # This private cache has only cooperating materializers. Every valid tree
    # is nonempty, so a concurrent winner makes this rename fail instead of
    # being replaced. Keep the strict primitive for general create-only paths.
    source.rename(destination)


def _publish_materialized_tree(source: Path, destination: Path) -> None:
    try:
        atomic_rename_directory_noreplace(source, destination)
    except OSError as exc:
        unsupported_errors = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
        if exc.errno not in unsupported_errors:
            raise
        # Some FUSE filesystems expose renameat2 but reject RENAME_NOREPLACE.
        _rename_verified_nonempty_directory(source, destination)


def _verify_tokenizer_config(root: Path, manifest: ServingArtifactManifest) -> None:
    config_ref = next(entry for entry in manifest.tokenizer.files if entry.role == "tokenizer.config")
    config_path = root.joinpath(*PurePosixPath(config_ref.path).parts)
    if config_path.stat().st_size > _MAX_TOKENIZER_CONFIG_BYTES:
        raise ValueError("materialized tokenizer config exceeds the size limit")
    try:
        value = json.loads(config_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("materialized tokenizer config is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("tokenizer_class") != manifest.tokenizer.class_name:
        raise ValueError("materialized tokenizer class does not match the serving artifact manifest")


def _materialize(
    *,
    snapshot_root: Path,
    cached_repo_root: Path,
    cache_root: Path,
    manifest_sha256: str,
    manifest: ServingArtifactManifest,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root.resolve(strict=True)
    destination = cache_root / manifest_sha256
    if destination.exists() or destination.is_symlink():
        _verify_materialized_tree(destination, manifest)
        _verify_tokenizer_config(destination, manifest)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=f".{manifest_sha256}.", dir=cache_root))
    staging.chmod(0o700)
    repo_root = cached_repo_root.resolve(strict=True)
    try:
        for artifact in manifest.artifacts:
            source = _safe_source_file(snapshot_root, repo_root, artifact)
            target = staging.joinpath(*PurePosixPath(artifact.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A hardlink cannot be made immutable for the worker: the cache
            # owner can chmod and rewrite the shared inode after admission.
            # Always create an independently owned inode, then verify the copy.
            shutil.copyfile(source, target)

        _make_read_only(staging)
        _verify_materialized_tree(staging, manifest)
        _verify_tokenizer_config(staging, manifest)
        try:
            _publish_materialized_tree(staging, destination)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _remove_staging_tree(staging)
            _verify_materialized_tree(destination, manifest)
            _verify_tokenizer_config(destination, manifest)
        return destination
    except Exception:
        _remove_staging_tree(staging)
        raise


def verify_and_materialize_serving_artifact(
    declaration: ServingArtifactDeclaration,
    *,
    source_hf_id: str,
    source_hf_revision: str,
    cached_repo_root: Path,
    materialized_cache_root: Path,
    runtime_version: str = CTRANSLATE2_VERSION,
) -> VerifiedServingArtifact:
    """Verify an exact derived snapshot and atomically materialize its served files."""
    snapshot_root, manifest = _load_manifest(declaration, cached_repo_root=cached_repo_root)
    _verify_manifest_compatibility(
        declaration,
        manifest,
        source_hf_id=source_hf_id,
        source_hf_revision=source_hf_revision,
        runtime_version=runtime_version,
    )
    root = _materialize(
        snapshot_root=snapshot_root,
        cached_repo_root=cached_repo_root,
        cache_root=materialized_cache_root,
        manifest_sha256=declaration.manifest_sha256,
        manifest=manifest,
    )
    return VerifiedServingArtifact(
        root=root,
        manifest_sha256=declaration.manifest_sha256,
        repo_id=declaration.repo_id,
        revision=declaration.revision,
        compute_type=declaration.compute_type,
        artifact_count=len(manifest.artifacts),
        manifest=manifest,
    )
