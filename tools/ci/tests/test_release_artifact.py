from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ci import build_audio_prep_release_asset as audio
from tools.ci import build_sidecar_release_asset as sidecar
from tools.ci import publish_helm_archive as helm
from tools.ci import release_artifact as artifact
from tools.ci import restore_release_artifact as restore

IDENTITY = {"version": "0.7.4", "tag_name": "v0.7.4", "source_revision": "a" * 40, "run_id": "1234"}


def test_provenance_roundtrip_and_identical_restamp(tmp_path):
    (tmp_path / "tested.whl").write_bytes(b"tested wheel")
    first = artifact.create_manifest(tmp_path, kind="python", **IDENTITY)
    assert artifact.create_manifest(tmp_path, kind="python", **IDENTITY) == first
    assert artifact.validate_manifest(tmp_path, kind="python", **IDENTITY) == first


@pytest.mark.parametrize("field", ["version", "tag_name", "source_revision", "run_id", "repository"])
def test_wrong_provenance_identity_is_rejected(tmp_path, field):
    (tmp_path / "tested.whl").write_bytes(b"tested wheel")
    data = artifact.create_manifest(tmp_path, kind="python", **IDENTITY)
    data[field] = "wrong"
    (tmp_path / artifact.MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance mismatch"):
        artifact.validate_manifest(tmp_path, **IDENTITY)


@pytest.mark.parametrize("change", ["bytes", "missing", "extra", "symlink"])
def test_provenance_requires_exact_payload_file_set(tmp_path, change):
    payload = tmp_path / "image.tar"
    payload.write_bytes(b"tested")
    artifact.create_manifest(tmp_path, kind="docker", **IDENTITY)
    if change == "bytes":
        payload.write_bytes(b"unseen")
    elif change == "missing":
        payload.unlink()
    elif change == "extra":
        (tmp_path / "extra").write_text("unexpected")
    else:
        (tmp_path / "extra").symlink_to(payload)
    with pytest.raises(ValueError, match="artifact"):
        artifact.validate_manifest(tmp_path, **IDENTITY)


def test_provenance_cannot_replace_different_existing_output(tmp_path):
    payload = tmp_path / "image.tar"
    payload.write_bytes(b"tested")
    artifact.create_manifest(tmp_path, kind="docker", **IDENTITY)
    payload.write_bytes(b"unseen")
    with pytest.raises(ValueError, match="replace different"):
        artifact.create_manifest(tmp_path, kind="docker", **IDENTITY)


def test_audio_builder_supports_same_source_and_output_path(tmp_path, monkeypatch):
    wheel = tmp_path / "audio.whl"
    wheel.write_bytes(b"native wheel")
    validated = []
    module = SimpleNamespace(
        build_audio_prep_wheel=lambda *_args, **_kwargs: wheel,
        _validate_wheel=validated.append,
    )
    monkeypatch.setattr(audio, "load_build_wheel", lambda *_: module)
    assert audio.main(["--out", str(tmp_path)]) == 0
    assert validated == [wheel]
    assert wheel.read_bytes() == b"native wheel"


@pytest.mark.parametrize("glibc", ["2.17", "2.36", "2.38"])
def test_sidecar_binary_checks_architecture_and_glibc_floor(tmp_path, monkeypatch, glibc):
    binary = tmp_path / "sidecar"
    binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + b"\x3e\0")
    monkeypatch.setattr(
        sidecar,
        "capture",
        lambda command: f"GLIBC_{glibc}" if "--version-info" in command else "(NEEDED) Shared library: [libc.so.6]",
    )
    if glibc == "2.38":
        with pytest.raises(ValueError, match="glibc requirement"):
            sidecar.inspect_abi(binary)
    else:
        assert sidecar.inspect_abi(binary)["glibc_minimum"] == glibc
    binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + b"\xb7\0")
    with pytest.raises(ValueError, match="x86_64"):
        sidecar.inspect_abi(binary)


@pytest.mark.parametrize("remote", [b"tested chart", b"different chart", None])
def test_chart_publisher_only_uploads_missing_exact_archive(tmp_path, monkeypatch, remote):
    chart = tmp_path / "sie-cluster-0.7.4.tgz"
    chart.write_bytes(b"tested chart")
    artifact.create_manifest(tmp_path, kind="helm", **IDENTITY)
    calls = []
    uploaded = False

    def execute(command, **_kwargs):
        nonlocal uploaded
        calls.append(command)
        if command[1] == "push":
            uploaded = True
        if command[1] == "pull":
            if remote is None and not uploaded:
                return subprocess.CompletedProcess(command, 1, "", "manifest unknown")
            destination = Path(command[-1]) / chart.name
            destination.write_bytes(remote if remote is not None else chart.read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(helm.subprocess, "run", execute)
    kwargs = {key: value for key, value in IDENTITY.items() if key != "tag_name"}
    if remote == b"different chart":
        with pytest.raises(ValueError, match="remote chart differs"):
            helm.publish(tmp_path, **kwargs)
    else:
        helm.publish(tmp_path, **kwargs)
    assert uploaded == (remote is None)
    assert all(command[1] in {"pull", "push"} for command in calls)


@pytest.mark.parametrize("mode", ["missing", "valid", "wrong_sha", "expired"])
def test_retry_restores_only_original_retained_bytes(tmp_path, monkeypatch, mode):
    (tmp_path / "image.tar").write_bytes(b"retained tested bytes")
    artifact.create_manifest(tmp_path, kind="docker", **IDENTITY)
    record = {
        "name": "docker-service-sie-config-0.7.4",
        "expired": mode == "expired",
        "workflow_run": {"id": 1234, "head_sha": "b" * 40 if mode == "wrong_sha" else IDENTITY["source_revision"]},
    }
    pages = [{"artifacts": [] if mode == "missing" else [record]}]
    monkeypatch.setattr(restore.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps(pages))
    commands = []
    monkeypatch.setattr(restore.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    if mode in {"wrong_sha", "expired"}:
        with pytest.raises(ValueError, match="original release artifact"):
            restore.restore(tmp_path, name=record["name"], kind="docker", **IDENTITY)
        assert commands == []
    else:
        assert restore.restore(tmp_path, name=record["name"], kind="docker", **IDENTITY) == (mode == "valid")
        assert all(command[:3] == ["gh", "run", "download"] for command in commands)


def test_retry_evidence_must_match_retained_image_archive(tmp_path, monkeypatch):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "image.tar").write_bytes(b"retained tested bytes")
    artifact.create_manifest(archive_dir, kind="docker", **IDENTITY)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    record = {
        "name": "docker-evidence-service-sie-config-0.7.4",
        "expired": False,
        "workflow_run": {"id": 1234, "head_sha": IDENTITY["source_revision"]},
    }
    monkeypatch.setattr(
        restore.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps([{"artifacts": [record]}])
    )
    monkeypatch.setattr(restore.subprocess, "run", lambda *_args, **_kwargs: None)
    provenance = evidence / "provenance.json"
    provenance.write_bytes((archive_dir / "provenance.json").read_bytes())
    kwargs = {"name": record["name"], "kind": "docker", "evidence_of": archive_dir, **IDENTITY}
    assert restore.restore(evidence, **kwargs)
    provenance.write_text("different image evidence")
    with pytest.raises(ValueError, match="differs from its retained archive"):
        restore.restore(evidence, **kwargs)
