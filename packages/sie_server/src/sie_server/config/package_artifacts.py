"""Immutable staged-artifact contract for package-backed adapters."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PACKAGE_ARTIFACT_MODE_KEY = "package_artifact_mode"
PACKAGE_ARTIFACT_MANIFEST_PATH_KEY = "package_artifact_manifest_path"
PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY = "package_artifact_manifest_sha256"
PACKAGE_ARTIFACT_ROOT_KEY = "package_artifact_root"

_DECLARATION_KEYS = frozenset(
    {
        PACKAGE_ARTIFACT_MODE_KEY,
        PACKAGE_ARTIFACT_MANIFEST_PATH_KEY,
        PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY,
    }
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA256_HEX_LENGTH = 64
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


class PackageArtifactMode(StrEnum):
    """How a package-backed adapter obtains non-code model artifacts."""

    BUNDLED = "bundled"
    LIVE = "live"
    STAGED = "staged"


@dataclass(frozen=True, slots=True)
class PackageArtifactDeclaration:
    mode: PackageArtifactMode
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedPackageArtifacts:
    root: Path
    manifest_sha256: str
    artifact_count: int


class PackageArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value or value.startswith("/"):
            raise ValueError("artifact path must be a relative POSIX path")
        segments = value.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise ValueError("artifact path must not contain empty, '.' or '..' segments")
        if PurePosixPath(value).is_absolute():
            raise ValueError("artifact path must be relative")
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def reject_boolean_size(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("artifact size_bytes must be an integer")
        return value


class PackageArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(alias="schema")
    artifacts: tuple[PackageArtifactEntry, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> PackageArtifactManifest:
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("package artifact manifest contains duplicate paths")
        return self


def parse_package_artifact_declaration(loadtime: dict[str, Any]) -> PackageArtifactDeclaration:
    """Parse the reserved declaration fields from one effective load-time profile."""
    if PACKAGE_ARTIFACT_ROOT_KEY in loadtime:
        raise ValueError(f"'{PACKAGE_ARTIFACT_ROOT_KEY}' is loader-owned and must not be declared")

    mode_value = loadtime.get(PACKAGE_ARTIFACT_MODE_KEY)
    has_manifest_fields = any(key in loadtime for key in _DECLARATION_KEYS - {PACKAGE_ARTIFACT_MODE_KEY})
    if mode_value is None:
        if has_manifest_fields:
            raise ValueError(f"'{PACKAGE_ARTIFACT_MODE_KEY}' is required when declaring a package artifact manifest")
        return PackageArtifactDeclaration(mode=PackageArtifactMode.BUNDLED)
    if not isinstance(mode_value, str):
        raise ValueError(f"'{PACKAGE_ARTIFACT_MODE_KEY}' must be one of bundled, live, or staged")
    try:
        mode = PackageArtifactMode(mode_value)
    except ValueError as exc:
        raise ValueError(f"'{PACKAGE_ARTIFACT_MODE_KEY}' must be one of bundled, live, or staged") from exc

    manifest_path_value = loadtime.get(PACKAGE_ARTIFACT_MANIFEST_PATH_KEY)
    manifest_sha256 = loadtime.get(PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY)
    if mode != PackageArtifactMode.STAGED:
        if manifest_path_value is not None or manifest_sha256 is not None:
            raise ValueError(f"package artifact mode '{mode.value}' must not declare a staged manifest")
        return PackageArtifactDeclaration(mode=mode)

    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        raise ValueError("staged package artifacts require a non-empty manifest path")
    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_absolute():
        raise ValueError("staged package artifact manifest path must be absolute")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != _SHA256_HEX_LENGTH:
        raise ValueError("staged package artifacts require a lowercase 64-hex manifest sha256")
    if any(character not in "0123456789abcdef" for character in manifest_sha256):
        raise ValueError("staged package artifacts require a lowercase 64-hex manifest sha256")
    return PackageArtifactDeclaration(
        mode=mode,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def has_package_artifact_declaration(loadtime: dict[str, Any]) -> bool:
    return bool((_DECLARATION_KEYS | {PACKAGE_ARTIFACT_ROOT_KEY}) & loadtime.keys())


def _verify_exact_artifact_inventory(
    *,
    root: Path,
    manifest_path: Path,
    manifest: PackageArtifactManifest,
) -> None:
    """Require the returned root to contain only the manifest's regular files."""
    expected_files = {artifact.path for artifact in manifest.artifacts}
    expected_directories = {
        parent.as_posix()
        for artifact_path in expected_files
        for parent in PurePosixPath(artifact_path).parents
        if parent != PurePosixPath(".")
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    try:
                        if entry.is_symlink():
                            raise ValueError(f"staged package artifact tree must not contain symlinks: {relative!r}")
                        if entry.is_dir(follow_symlinks=False):
                            if relative not in expected_directories:
                                raise ValueError(
                                    f"staged package artifact tree contains an unlisted directory: {relative!r}"
                                )
                            observed_directories.add(relative)
                            scan(path)
                            continue
                        if entry.is_file(follow_symlinks=False):
                            if path == manifest_path:
                                continue
                            if relative not in expected_files:
                                raise ValueError(
                                    f"staged package artifact tree contains an unlisted file: {relative!r}"
                                )
                            observed_files.add(relative)
                            continue
                    except OSError as exc:
                        raise ValueError(
                            f"staged package artifact tree entry cannot be inspected: {relative!r}"
                        ) from exc
                    raise ValueError(f"staged package artifact tree contains a special node: {relative!r}")
        except OSError as exc:
            raise ValueError("staged package artifact tree cannot be inventoried") from exc

    scan(root)
    missing_files = sorted(expected_files - observed_files)
    missing_directories = sorted(expected_directories - observed_directories)
    if missing_files or missing_directories:
        raise ValueError(
            "staged package artifact tree does not match the manifest: "
            f"missing_files={missing_files!r}, missing_directories={missing_directories!r}"
        )


def verify_staged_package_artifacts(declaration: PackageArtifactDeclaration) -> VerifiedPackageArtifacts:
    """Verify one staged manifest and every file it names without network access.

    The deployment contract keeps the containing artifact root immutable and
    read-only for the worker lifetime; this admission check therefore does not
    attempt to synchronize against concurrent filesystem mutation.
    """
    if (
        declaration.mode != PackageArtifactMode.STAGED
        or declaration.manifest_path is None
        or declaration.manifest_sha256 is None
    ):
        raise ValueError("package artifact verification requires a complete staged declaration")

    manifest_path = declaration.manifest_path
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("staged package artifact manifest does not exist") from exc
    if resolved_manifest != manifest_path or not manifest_path.is_file():
        raise ValueError("staged package artifact manifest must be a regular non-symlink file")
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("staged package artifact manifest exceeds the size limit")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != declaration.manifest_sha256:
        raise ValueError("staged package artifact manifest sha256 does not match the catalog declaration")
    try:
        manifest = PackageArtifactManifest.model_validate(json.loads(manifest_bytes))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("staged package artifact manifest is invalid") from exc

    root = manifest_path.parent.resolve(strict=True)
    _verify_exact_artifact_inventory(root=root, manifest_path=manifest_path, manifest=manifest)
    for artifact in manifest.artifacts:
        artifact_path = root.joinpath(*PurePosixPath(artifact.path).parts)
        try:
            resolved_artifact = artifact_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"staged package artifact is missing: {artifact.path!r}") from exc
        if resolved_artifact != artifact_path or not artifact_path.is_file():
            raise ValueError(f"staged package artifact must be a regular non-symlink file: {artifact.path!r}")
        if not resolved_artifact.is_relative_to(root):
            raise ValueError(f"staged package artifact escapes the manifest root: {artifact.path!r}")
        stat = artifact_path.stat()
        if stat.st_size != artifact.size_bytes:
            raise ValueError(f"staged package artifact size does not match the manifest: {artifact.path!r}")
        digest = hashlib.sha256()
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(f"staged package artifact sha256 does not match the manifest: {artifact.path!r}")

    return VerifiedPackageArtifacts(
        root=root,
        manifest_sha256=declaration.manifest_sha256,
        artifact_count=len(manifest.artifacts),
    )
