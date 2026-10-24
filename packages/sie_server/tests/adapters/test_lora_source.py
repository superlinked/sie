"""LoRA cache resolution with every network connection denied."""

import socket
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest
from huggingface_hub.utils import HFValidationError
from sie_server.adapters._lora_source import resolve_lora_source

SHA = "a" * 40
OTHER_SHA = "b" * 40
FILES = ("adapter_config.json", "adapter_model.safetensors")


def snapshot(root: Path, source: str = "tenant-a/adapter", revision: str = SHA) -> Path:
    repository = root / f"models--{source.replace('/', '--')}"
    directory = repository / "snapshots" / revision
    directory.mkdir(parents=True)
    for filename in FILES:
        (directory / filename).write_bytes(f"{source}:{revision}:{filename}".encode())
    (repository / "refs").mkdir(exist_ok=True)
    (repository / "refs" / "main").write_text(revision)
    return directory


@pytest.fixture
def cache_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(local))
    monkeypatch.setenv("SIE_CLUSTER_CACHE", str(cluster))
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", Mock(side_effect=AssertionError("Hub forbidden")))
    monkeypatch.setattr(huggingface_hub, "file_exists", Mock(side_effect=AssertionError("Hub metadata forbidden")))
    return local, cluster


@pytest.mark.parametrize("revision", [None, SHA])
@pytest.mark.parametrize("tier", ["local", "cluster"])
@pytest.mark.parametrize("fallback", ["false", "true"])
def test_exact_adapter_from_cache_without_hub(
    cache_policy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tier: str, revision: str | None, fallback: str
) -> None:
    local, cluster = cache_policy
    monkeypatch.setenv("SIE_HF_FALLBACK", fallback)
    original = snapshot(local if tier == "local" else cluster)

    directory = resolve_lora_source("tenant-a/adapter", revision)

    assert directory == local / "models--tenant-a--adapter" / "snapshots" / SHA
    assert [path.read_bytes() for path in (directory / name for name in FILES)] == [
        (original / name).read_bytes() for name in FILES
    ]


def test_other_revision_cannot_satisfy_pin(cache_policy: tuple[Path, Path]) -> None:
    local, cluster = cache_policy
    snapshot(local, revision=OTHER_SHA)
    snapshot(cluster, revision=OTHER_SHA)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        resolve_lora_source("tenant-a/adapter", SHA)


@pytest.mark.parametrize("filename", FILES)
def test_incomplete_snapshot_fails_closed(cache_policy: tuple[Path, Path], filename: str) -> None:
    local, _ = cache_policy
    directory = snapshot(local)
    (directory / filename).unlink()
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        resolve_lora_source("tenant-a/adapter", SHA)


def test_default_branch_never_chooses_an_arbitrary_snapshot(cache_policy: tuple[Path, Path]) -> None:
    local, _ = cache_policy
    directory = snapshot(local)
    (directory.parent.parent / "refs" / "main").unlink()
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        resolve_lora_source("tenant-a/adapter")


def test_default_reference_cannot_alias_another_tenant(cache_policy: tuple[Path, Path]) -> None:
    local, _ = cache_policy
    directory = snapshot(local)
    other = snapshot(local, source="tenant-b/adapter", revision=OTHER_SHA)
    reference = directory.parent.parent / "refs" / "main"
    reference.unlink()
    reference.symlink_to(other.parent.parent / "refs" / "main")
    with pytest.raises(ValueError, match="file alias"):
        resolve_lora_source("tenant-a/adapter")


def test_configured_root_symlink_preserves_exact_repository(
    cache_policy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local, _ = cache_policy
    directory = snapshot(local)
    selected = tmp_path / "selected"
    selected.symlink_to(local, target_is_directory=True)
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(selected))
    assert resolve_lora_source("tenant-a/adapter", SHA) == directory


@pytest.mark.parametrize("revision", ["main", "../other", "a" * 39, "A" * 40])
def test_mutable_or_malformed_pin_is_refused(cache_policy: tuple[Path, Path], revision: str) -> None:
    with pytest.raises(ValueError, match="immutable"):
        resolve_lora_source("tenant-a/adapter", revision)


@pytest.mark.parametrize("source", ["../tenant-b/adapter", "a/b/c", "s3://bucket/adapter", "tenant\\adapter"])
def test_invalid_repository_is_refused(cache_policy: tuple[Path, Path], source: str) -> None:
    with pytest.raises(HFValidationError):
        resolve_lora_source(source)


def test_local_root_symlink_is_supported_but_pin_is_not(cache_policy: tuple[Path, Path], tmp_path: Path) -> None:
    actual = snapshot(tmp_path / "fixtures")
    selected = tmp_path / "selected"
    selected.symlink_to(actual, target_is_directory=True)
    assert resolve_lora_source(str(selected)) == actual
    with pytest.raises(ValueError, match="local LoRA directory cannot honor"):
        resolve_lora_source(str(selected), SHA)


def test_local_file_escape_is_refused(cache_policy: tuple[Path, Path], tmp_path: Path) -> None:
    actual = snapshot(tmp_path / "fixtures")
    outside = tmp_path / "outside"
    outside.write_bytes(b"not this adapter")
    (actual / FILES[1]).unlink()
    (actual / FILES[1]).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        resolve_lora_source(str(actual))


def test_repository_cannot_alias_another_tenant(cache_policy: tuple[Path, Path]) -> None:
    local, _ = cache_policy
    other = snapshot(local, source="tenant-b/adapter")
    (local / "models--tenant-a--adapter").symlink_to(other.parent.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        resolve_lora_source("tenant-a/adapter", SHA)


@pytest.mark.parametrize("part", ["snapshots", "refs", "blobs"])
def test_repository_layout_cannot_alias_another_tenant(cache_policy: tuple[Path, Path], part: str) -> None:
    local, _ = cache_policy
    other = snapshot(local, source="tenant-b/adapter").parent.parent
    (other / "blobs").mkdir()
    repository = local / "models--tenant-a--adapter"
    repository.mkdir()
    (repository / part).symlink_to(other / part, target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        resolve_lora_source("tenant-a/adapter", SHA)


def test_snapshot_cannot_alias_another_revision(cache_policy: tuple[Path, Path]) -> None:
    local, _ = cache_policy
    other = snapshot(local, revision=OTHER_SHA)
    (other.parent / SHA).symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="another revision"):
        resolve_lora_source("tenant-a/adapter", SHA)


@pytest.mark.parametrize("other_tenant", [False, True])
def test_hf_file_links_are_limited_to_this_repository_blobs(
    cache_policy: tuple[Path, Path], other_tenant: bool
) -> None:
    local, _ = cache_policy
    directory = snapshot(local)
    repository = snapshot(local, source="tenant-b/adapter").parent.parent if other_tenant else directory.parent.parent
    blob = repository / "blobs" / "digest"
    blob.parent.mkdir()
    blob.write_bytes(b"blob")
    (directory / FILES[1]).unlink()
    (directory / FILES[1]).symlink_to(blob)
    if other_tenant:
        with pytest.raises(ValueError, match="escapes"):
            resolve_lora_source("tenant-a/adapter", SHA)
    else:
        assert resolve_lora_source("tenant-a/adapter", SHA) == directory


def test_hub_fallback_uses_the_configured_cache_and_exact_pin(
    cache_policy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    local, _ = cache_policy
    monkeypatch.setenv("SIE_HF_FALLBACK", "true")
    download = Mock(side_effect=lambda *_args, **_kwargs: str(snapshot(local)))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "file_exists", Mock(return_value=False))

    assert resolve_lora_source("tenant-a/adapter", SHA) == local / "models--tenant-a--adapter" / "snapshots" / SHA
    download.assert_called_once()
    assert download.call_args.args == ("tenant-a/adapter",)
    assert download.call_args.kwargs["revision"] == SHA
    assert download.call_args.kwargs["cache_dir"] == str(local)
    assert download.call_args.kwargs["allow_patterns"] == list(FILES)


@pytest.mark.parametrize("selected_alias", [False, True])
def test_explicit_local_hf_snapshot_supports_owned_blob_links(
    cache_policy: tuple[Path, Path], tmp_path: Path, selected_alias: bool
) -> None:
    directory = snapshot(tmp_path / "explicit-cache")
    repository = directory.parent.parent
    blobs = repository / "blobs"
    blobs.mkdir()
    for filename in FILES:
        original = (directory / filename).read_bytes()
        (blobs / filename).write_bytes(original)
        (directory / filename).unlink()
        (directory / filename).symlink_to(Path("../../blobs") / filename)
    selected = directory
    if selected_alias:
        selected = tmp_path / "selected"
        selected.symlink_to(directory, target_is_directory=True)

    assert resolve_lora_source(str(selected)) == directory
    with pytest.raises(ValueError, match="local LoRA directory cannot honor"):
        resolve_lora_source(str(selected), SHA)


@pytest.mark.parametrize("escape", ["other-repository", "blobs-directory-alias", "non-hf-layout"])
def test_explicit_local_snapshot_does_not_allow_unowned_blob_links(
    cache_policy: tuple[Path, Path], tmp_path: Path, escape: str
) -> None:
    directory = snapshot(tmp_path / "explicit-cache")
    repository = directory.parent.parent
    outside = snapshot(tmp_path / "explicit-cache", source="tenant-b/adapter").parent.parent / "blobs"
    outside.mkdir()
    (outside / "weights").write_bytes(b"other repository")
    if escape == "blobs-directory-alias":
        (repository / "blobs").symlink_to(outside, target_is_directory=True)
        target = repository / "blobs" / "weights"
    elif escape == "non-hf-layout":
        plain = tmp_path / "plain"
        directory.rename(plain)
        directory = plain
        target = outside / "weights"
    else:
        target = outside / "weights"
    (directory / FILES[1]).unlink()
    (directory / FILES[1]).symlink_to(target)
    with pytest.raises(ValueError, match=r"escapes|aliases"):
        resolve_lora_source(str(directory))


def complete_hub_snapshot(local: Path) -> Mock:
    def download(*_args: object, **_kwargs: object) -> str:
        repository = local / "models--tenant-a--adapter"
        directory = repository / "snapshots" / SHA
        directory.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            (directory / filename).write_bytes(f"complete:{filename}".encode())
        (repository / "refs").mkdir(exist_ok=True)
        (repository / "refs" / "main").write_text(SHA)
        return str(directory)

    return Mock(side_effect=download)


@pytest.mark.parametrize("missing", FILES)
@pytest.mark.parametrize("revision", [None, SHA])
@pytest.mark.parametrize("tier", ["cluster", "hub"])
def test_partial_local_snapshot_continues_to_correct_tier(
    cache_policy: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    revision: str | None,
    tier: str,
) -> None:
    local, cluster = cache_policy
    partial = snapshot(local)
    (partial / missing).unlink()
    download = None
    if tier == "cluster":
        expected = snapshot(cluster)
    else:
        monkeypatch.setenv("SIE_HF_FALLBACK", "true")
        download = complete_hub_snapshot(local)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        monkeypatch.setattr(huggingface_hub, "file_exists", Mock(return_value=False))
        expected = partial

    resolved = resolve_lora_source("tenant-a/adapter", revision)

    assert resolved == partial
    assert all((resolved / name).is_file() for name in FILES)
    if download is not None:
        download.assert_called_once()
        assert download.call_args.kwargs["revision"] == (revision or "main")
        assert download.call_args.kwargs["allow_patterns"] == list(FILES)
    else:
        assert [(resolved / name).read_bytes() for name in FILES] == [(expected / name).read_bytes() for name in FILES]


@pytest.mark.parametrize("cached_revision", [SHA, OTHER_SHA])
@pytest.mark.parametrize("tier", ["cluster", "hub"])
def test_default_ref_miss_continues_to_correct_tier(
    cache_policy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, cached_revision: str, tier: str
) -> None:
    local, cluster = cache_policy
    old = snapshot(local, revision=cached_revision)
    old_bytes = [(old / name).read_bytes() for name in FILES]
    (old.parent.parent / "refs" / "main").unlink()
    download = None
    if tier == "cluster":
        snapshot(cluster)
    else:
        monkeypatch.setenv("SIE_HF_FALLBACK", "true")
        download = complete_hub_snapshot(local)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        monkeypatch.setattr(huggingface_hub, "file_exists", Mock(return_value=False))

    resolved = resolve_lora_source("tenant-a/adapter")

    assert resolved == old.parent / SHA
    assert all((resolved / name).is_file() for name in FILES)
    if cached_revision != SHA:
        assert [(old / name).read_bytes() for name in FILES] == old_bytes
    if download is not None:
        download.assert_called_once()
        assert download.call_args.kwargs["revision"] == "main"


@pytest.mark.parametrize("fallback", [False, True])
def test_partial_cluster_hit_requires_complete_files_before_return(
    cache_policy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    local, cluster = cache_policy
    partial = snapshot(cluster)
    (partial / FILES[1]).unlink()
    if not fallback:
        with pytest.raises(RuntimeError, match="fallback is disabled"):
            resolve_lora_source("tenant-a/adapter", SHA)
        assert (local / partial.relative_to(cluster) / FILES[0]).is_file()
        assert not (local / partial.relative_to(cluster) / FILES[1]).exists()
    else:
        monkeypatch.setenv("SIE_HF_FALLBACK", "true")
        download = complete_hub_snapshot(local)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        monkeypatch.setattr(huggingface_hub, "file_exists", Mock(return_value=False))
        resolved = resolve_lora_source("tenant-a/adapter", SHA)
        assert all((resolved / name).is_file() for name in FILES)
        download.assert_called_once()


@pytest.mark.parametrize("filename", FILES)
def test_explicit_incomplete_directory_still_fails_locally(
    cache_policy: tuple[Path, Path], tmp_path: Path, filename: str
) -> None:
    directory = snapshot(tmp_path / "explicit")
    (directory / filename).unlink()
    with pytest.raises(ValueError, match=f"ships no {filename}"):
        resolve_lora_source(str(directory))
