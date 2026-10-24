"""Resolve admitted LoRA sources through the configured model-cache policy.

This module neither declares nor authorizes adapters. Callers must resolve their
served alias and admit its source/revision before calling this load-time helper.
It has no PEFT, GLiNER or model-runtime dependency.
"""

from pathlib import Path

from huggingface_hub.utils import validate_repo_id
from sie_sdk.cache import ensure_model_cached, get_cache_config

from sie_server.config.model import is_immutable_revision

_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _check_files(directory: Path, *, blobs: Path | None = None) -> Path:
    for filename in _FILES:
        path = directory / filename
        resolved = path.resolve()
        if not resolved.is_relative_to(directory) and (blobs is None or not resolved.is_relative_to(blobs)):
            raise ValueError("LoRA artifact escapes its declared directory or repository blobs")
        if not path.is_file():
            raise ValueError(f"LoRA source ships no {filename}")
    return directory


def _check_repository(repository: Path) -> None:
    # The configured cache root may intentionally be a symlink; it is already
    # canonicalized. Derived repositories and layout directories may not alias
    # another repository or snapshot.
    for directory in (repository, *(repository / name for name in ("snapshots", "refs", "blobs"))):
        if directory.resolve() != directory:
            raise ValueError("LoRA cache repository directories cannot be aliases")


def resolve_lora_source(source: str, revision: str | None = None) -> Path:
    """Return the admitted two-file adapter from local, cluster or allowed Hub storage.

    Explicit local directories (including a selected root symlink) are supported
    without a Hub revision. Hub pins must be immutable commits. An unpinned Hub
    source uses only its cached ``refs/main`` commit, never an arbitrary snapshot.
    Standard snapshot-file links to this repository's blobs remain supported.
    Missing files, redirected paths or a missing exact revision fail closed.
    """
    directory = Path(source)
    if directory.is_dir():
        if revision is not None:
            raise ValueError("A local LoRA directory cannot honor a Hugging Face revision pin")
        directory = directory.resolve()
        repository = directory.parent.parent
        if (
            directory.parent.name == "snapshots"
            and is_immutable_revision(directory.name)
            and repository.name.startswith("models--")
        ):
            _check_repository(repository)
            return _check_files(directory, blobs=repository / "blobs")
        return _check_files(directory)

    validate_repo_id(source)
    if revision is not None and not is_immutable_revision(revision):
        raise ValueError("LoRA revision must be an immutable 40-hex commit SHA")
    config = get_cache_config()
    repository = config.local_cache.resolve() / f"models--{source.replace('/', '--')}"
    _check_repository(repository)
    if revision is None:
        reference = repository / "refs" / "main"
        if reference.is_symlink():
            raise ValueError("LoRA default revision cannot be a file alias")
        if reference.is_file() and not is_immutable_revision(reference.read_text().strip()):
            raise ValueError("LoRA cached default revision must be an immutable commit SHA")
    cached = ensure_model_cached(
        source,
        config,
        revision=revision or "main",
        allow_patterns=list(_FILES),
        required_files=list(_FILES),
    )
    if cached.resolve() != repository:
        raise ValueError("LoRA cache returned a different repository")
    _check_repository(repository)

    if revision is None:
        reference = repository / "refs" / "main"
        if reference.is_symlink():
            raise ValueError("LoRA default revision cannot be a file alias")
        revision = reference.read_text().strip()
        if not is_immutable_revision(revision):
            raise ValueError("LoRA cached default revision must be an immutable commit SHA")
    directory = repository / "snapshots" / revision
    if directory.resolve() != directory:
        raise ValueError("LoRA snapshot cannot alias another revision or repository")
    return _check_files(directory, blobs=repository / "blobs")
