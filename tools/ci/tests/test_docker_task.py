from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.ci.release_artifact import create_manifest
from tools.mise_tasks import docker_task

FULL_SHA = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64
VERSION = "0.7.4"
IMAGE = f"ghcr.io/superlinked/sie-config:v{VERSION}"
MATRIX = docker_task.DEFAULT_MATRIX


@pytest.fixture
def complete_source(tmp_path: Path, monkeypatch):
    bundles = tmp_path / "packages/sie_server/bundles"
    bundles.mkdir(parents=True)
    for name in ("default", "ctranslate2", "sglang", "transformers5", "sglang-cu130", "tensorrt-llm"):
        platform = "cuda13" if name in {"sglang-cu130", "tensorrt-llm"} else "cuda12"
        (bundles / f"{name}.yaml").write_text(f"name: {name}\nplatform: {platform}\n")
    monkeypatch.setattr(docker_task, "ROOT", tmp_path)
    return docker_task.load_release_matrix(MATRIX)


def metadata(image: str = IMAGE):
    return {"image": image, "image_id": IMAGE_ID, "os": "linux", "architecture": "amd64"}


def archive(tmp_path: Path, image: str = IMAGE):
    (tmp_path / "image.tar").write_bytes(b"same tested image bytes")
    return create_manifest(
        tmp_path,
        kind="docker",
        version=VERSION,
        tag_name=f"v{VERSION}",
        source_revision=FULL_SHA,
        run_id="1234",
        metadata=metadata(image),
    )


def test_release_matrix_resolves_exact_ten_pairs(complete_source):
    assert {(target.platform, target.bundle) for target in complete_source} == {
        (platform, bundle)
        for platform in ("cuda12", "cpu")
        for bundle in ("default", "ctranslate2", "sglang", "transformers5")
    } | {("cuda13", "sglang-cu130"), ("cuda13", "tensorrt-llm")}
    assert len(complete_source) == 10


def test_release_matrix_fails_closed_for_absent_bundle(complete_source, tmp_path):
    (tmp_path / "packages/sie_server/bundles/ctranslate2.yaml").unlink()
    with pytest.raises(ValueError, match="release bundle does not exist: ctranslate2"):
        docker_task.load_release_matrix(MATRIX)


def test_bundle_requires_its_adapter_source(complete_source, tmp_path):
    (tmp_path / "packages/sie_server/bundles/ctranslate2.yaml").write_text(
        "adapters: [sie_server.adapters.ctranslate2]\n"
    )
    with pytest.raises(ValueError, match="adapter source is missing"):
        docker_task.load_release_matrix(MATRIX)


def test_release_matrix_rejects_duplicates(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"platforms": ["cpu", "cpu"], "bundles": ["default"]}))
    with pytest.raises(ValueError, match="duplicate"):
        docker_task.load_release_matrix(path)


def test_release_matrix_rejects_bundle_platform_disagreement(complete_source):
    with pytest.raises(ValueError, match="disagrees"):
        docker_task.validate_target(docker_task.ServerTarget("cuda13", "default"))


def test_build_commands_only_load_source_bound_images():
    for command in (
        docker_task.build_server_command(
            registry="ghcr.io/superlinked",
            version=VERSION,
            target=docker_task.ServerTarget("cpu", "default"),
            source_revision=FULL_SHA,
        ),
        docker_task.build_service_command(
            registry="ghcr.io/superlinked",
            version=VERSION,
            service="sie-config",
            source_revision=FULL_SHA,
        ),
    ):
        assert "--load" in command
        assert "--push" not in command
        assert f"org.opencontainers.image.revision={FULL_SHA}" in command
        assert "org.opencontainers.image.source=https://github.com/superlinked/sie" in command
    with pytest.raises(ValueError, match="full 40-character"):
        docker_task.validate_source_revision("abc123")
    with pytest.raises(SystemExit):
        docker_task.parser().parse_args(
            [
                "build-service",
                "--registry",
                "local",
                "--version",
                "0.0.0",
                "--service",
                "sie-config",
                "--source-revision",
                FULL_SHA,
                "--push",
            ]
        )


def test_complete_set_verified_before_alias_commands(complete_source, monkeypatch, tmp_path):
    commands = []

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError("incomplete")

    monkeypatch.setattr(docker_task, "verify_release", fail_verification)
    monkeypatch.setattr(docker_task, "run", commands.append)
    with pytest.raises(RuntimeError, match="incomplete"):
        docker_task.move_aliases(
            "ghcr.io/superlinked",
            VERSION,
            complete_source,
            evidence_dir=tmp_path,
            source_revision=FULL_SHA,
            run_id="1234",
        )
    assert commands == []


def test_expected_release_set_has_fifteen_tags_and_six_names(complete_source):
    images = docker_task.expected_versioned_images("ghcr.io/superlinked", VERSION, complete_source)
    assert len(images) == len(set(images)) == 15
    assert len({image.split(":")[0] for image in images}) == 6
    assert f"ghcr.io/superlinked/sie-server-rust:v{VERSION}-cuda12-sm89" in images


def test_export_saves_inspected_image_and_records_same_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_task, "inspect_loaded", lambda *_: metadata())
    commands = []

    def save(command):
        commands.append(command)
        Path(command[4]).write_bytes(b"tested bytes")

    monkeypatch.setattr(docker_task, "run", save)
    manifest = docker_task.export_image(IMAGE, tmp_path, version=VERSION, source_revision=FULL_SHA, run_id="1234")
    assert commands == [["docker", "image", "save", "--output", str(tmp_path / "image.tar"), IMAGE]]
    assert manifest["metadata"] == metadata()
    assert manifest["files"][0]["size"] == len(b"tested bytes")


@pytest.mark.parametrize("change", ["revision", "archive", "image", "digest"])
def test_archive_rejects_mismatched_binding_before_push(tmp_path, monkeypatch, change):
    data = archive(tmp_path)
    if change == "revision":
        data["source_revision"] = "c" * 40
    if change == "image":
        data["metadata"]["image"] = IMAGE + "-other"
    if change == "digest":
        data["metadata"]["image_id"] = "sha256:" + "c" * 64
    (tmp_path / "provenance.json").write_text(json.dumps(data))
    if change == "archive":
        (tmp_path / "image.tar").write_bytes(b"different")
    commands = []
    monkeypatch.setattr(docker_task, "run", commands.append)
    monkeypatch.setattr(docker_task, "inspect_loaded", lambda *_: metadata())
    with pytest.raises(ValueError, match="mismatch"):
        docker_task.publish_archive(IMAGE, tmp_path, version=VERSION, source_revision=FULL_SHA, run_id="1234")
    assert not any(command[1] == "push" for command in commands)


@pytest.mark.parametrize("existing", [None, IMAGE_ID, "sha256:" + "c" * 64])
def test_publication_loads_same_archive_and_never_rebuilds_or_clobbers(tmp_path, monkeypatch, existing):
    archive(tmp_path)
    commands = []
    monkeypatch.setattr(docker_task, "run", commands.append)
    monkeypatch.setattr(docker_task, "inspect_loaded", lambda *_: metadata())
    monkeypatch.setattr(
        docker_task, "remote_image_id", lambda _image, **kwargs: existing if kwargs.get("allow_missing") else IMAGE_ID
    )
    if existing is not None and existing != IMAGE_ID:
        with pytest.raises(ValueError, match="overwrite"):
            docker_task.publish_archive(IMAGE, tmp_path, version=VERSION, source_revision=FULL_SHA, run_id="1234")
    else:
        docker_task.publish_archive(IMAGE, tmp_path, version=VERSION, source_revision=FULL_SHA, run_id="1234")
    assert commands[0] == ["docker", "image", "load", "--input", str(tmp_path / "image.tar")]
    assert all("build" not in command and "buildx" not in command for command in commands)
    assert (["docker", "push", IMAGE] in commands) == (existing is None)


def test_remote_inspection_does_not_treat_authorization_failure_as_absent(monkeypatch):
    monkeypatch.setattr(
        docker_task.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "unauthorized"),
    )
    with pytest.raises(RuntimeError, match="cannot inspect"):
        docker_task.remote_image_id(IMAGE, allow_missing=True)


def test_full_set_verifier_rejects_source_or_remote_digest_mismatch(complete_source, tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for offset, image in enumerate(
        docker_task.expected_versioned_images("ghcr.io/superlinked", VERSION, complete_source)
    ):
        data = {
            "schema": 1,
            "repository": "superlinked/sie",
            "kind": "docker",
            "version": VERSION,
            "tag_name": f"v{VERSION}",
            "source_revision": FULL_SHA,
            "run_id": "1234",
            "metadata": metadata(image),
        }
        (evidence / f"{offset}.json").write_text(json.dumps(data))
    monkeypatch.setattr(docker_task, "remote_image_id", lambda *_: IMAGE_ID)
    kwargs = {"evidence_dir": evidence, "source_revision": FULL_SHA, "run_id": "1234"}
    docker_task.verify_release("ghcr.io/superlinked", VERSION, complete_source, **kwargs)
    monkeypatch.setattr(docker_task, "remote_image_id", lambda *_: "sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="differs"):
        docker_task.verify_release("ghcr.io/superlinked", VERSION, complete_source, **kwargs)
    (evidence / "0.json").unlink()
    with pytest.raises(ValueError, match="exact complete"):
        docker_task.verify_release("ghcr.io/superlinked", VERSION, complete_source, **kwargs)
