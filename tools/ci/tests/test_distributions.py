from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.ci import distributions as packages


def python_archives(tmp_path, name="sie-sdk", version="0.7.2"):
    wheel = tmp_path / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    metadata = f"Name: {name}\nVersion: {version}\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
    sdist = tmp_path / f"{name}-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo(f"{name}-{version}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata.encode()))
    return wheel, sdist


def test_pr_checks_actual_versions_and_release_checks_coordinated_versions():
    actual = packages.manifests("python")
    assert len(actual) == 13
    assert "sie-config" in actual
    assert "sie-mcp" in actual
    assert len(packages.manifests("npm")) == 5
    with pytest.raises(ValueError, match="actual"):
        packages.manifests("python", "999.0.0")


def test_archives_require_both_formats_and_exact_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(packages, "manifests", lambda *args: {"sie-sdk": ("0.7.2", tmp_path)})
    wheel, sdist = python_archives(tmp_path)
    assert set(packages.verify("python", tmp_path)) == {wheel, sdist}
    sdist.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        packages.verify("python", tmp_path)


def test_wrong_distribution_version_and_unexpected_files_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(packages, "manifests", lambda *args: {"sie-sdk": ("0.7.4", tmp_path)})
    python_archives(tmp_path)
    with pytest.raises(ValueError, match="version"):
        packages.verify("python", tmp_path)


def test_clean_python_consumes_wheel_and_sdist_outside_source(tmp_path, monkeypatch):
    archives = python_archives(tmp_path)
    calls = []
    monkeypatch.setattr(packages, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    packages.clean_python(list(archives))
    assert len(calls) == 2
    for args, kwargs in calls:
        assert "--isolated" in args
        assert "--no-project" in args
        assert "-I" in args
        assert str(kwargs["cwd"]) != str(packages.ROOT)
        assert "sys.prefix" in args[-1]
        assert "--no-deps" not in args


def test_npm_metadata_requires_shipped_entrypoints_and_resolved_workspace(tmp_path):
    package = tmp_path / "package.tgz"
    metadata = json.dumps({"name": "@superlinked/sie-sdk", "version": "0.7.2", "main": "dist/index.cjs"}).encode()
    with tarfile.open(package, "w:gz") as archive:
        member = tarfile.TarInfo("package/package.json")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
    with pytest.raises(ValueError, match="missing packed entrypoint"):
        packages.archive_metadata(package)


def test_python_build_retains_only_package_archives_not_uv_gitignore(tmp_path, monkeypatch):
    monkeypatch.setattr(packages, "manifests", lambda *args: {"sie-sdk": ("0.7.2", tmp_path)})

    def run(*args, **kwargs):
        if "build" in args:
            directory = Path(args[-1])
            python_archives(directory)
            (directory / ".gitignore").write_text("*\n")

    monkeypatch.setattr(packages, "run", run)
    monkeypatch.setattr(packages, "clean_python", lambda archives: None)
    output = tmp_path / "artifacts"
    packages.build("python", output, "")
    assert len(list(output.iterdir())) == 2
    assert not (output / ".gitignore").exists()


@pytest.mark.parametrize("conflicting", [False, True])
def test_pypi_retry_accepts_only_exact_existing_bytes(tmp_path, monkeypatch, conflicting):
    source = tmp_path / "artifacts"
    source.mkdir()
    wheel, sdist = python_archives(source)
    monkeypatch.setattr(packages, "manifests", lambda *args: {"sie-sdk": ("0.7.2", tmp_path)})
    records = [
        {
            "filename": wheel.name,
            "digests": {"sha256": "0" * 64 if conflicting else hashlib.sha256(wheel.read_bytes()).hexdigest()},
        }
    ]
    monkeypatch.setattr(
        packages.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps({"urls": records}).encode())
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    if conflicting:
        with pytest.raises(ValueError, match="different bytes"):
            packages.prepare_pypi(source, tmp_path / "pending", "0.7.2")
    else:
        packages.prepare_pypi(source, tmp_path / "pending", "0.7.2")
        assert (tmp_path / "pending" / sdist.name).read_bytes() == sdist.read_bytes()
        assert not (tmp_path / "pending" / wheel.name).exists()


def test_npm_retry_checks_all_versions_before_any_upload(tmp_path, monkeypatch):
    archive = tmp_path / "package.tgz"
    archive.write_bytes(b"tested archive")
    monkeypatch.setattr(packages, "verify", lambda *args: [archive])
    monkeypatch.setattr(packages, "archive_metadata", lambda path: ("@superlinked/sie-sdk", "0.7.4"))
    uploads = []
    monkeypatch.setattr(packages, "run", lambda *args: uploads.append(args))
    monkeypatch.setattr(
        packages.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, '"sha512-different"', "")
    )
    with pytest.raises(ValueError, match="different bytes"):
        packages.publish_npm(tmp_path, "0.7.4")
    assert uploads == []


def npm_publish_fixture(tmp_path, monkeypatch, replies, *, count=1):
    archives = [tmp_path / f"package-{index}.tgz" for index in range(count)]
    for archive in archives:
        archive.write_bytes(b"tested archive")
    monkeypatch.setattr(packages, "verify", lambda *args: archives)
    monkeypatch.setattr(packages, "archive_metadata", lambda path: (f"@superlinked/{path.stem}", "0.7.4"))
    calls, uploads = [], []
    results = iter(replies)

    def query(args, **kwargs):
        calls.append(args)
        return next(results)

    monkeypatch.setattr(packages.subprocess, "run", query)
    monkeypatch.setattr(packages, "run", lambda *args: uploads.append(args))
    return calls, uploads


def npm_missing():
    return subprocess.CompletedProcess([], 1, '{"error":{"code":"E404"}}', "npm error code E404")


@pytest.mark.parametrize(
    ("latest", "tag"), [("0.7.5", "release-v0.7.4"), ("0.7.3", "latest"), ("0.7.10", "release-v0.7.4")]
)
def test_npm_publish_preserves_newer_latest_on_historical_repair(tmp_path, monkeypatch, latest, tag):
    calls, uploads = npm_publish_fixture(
        tmp_path, monkeypatch, [npm_missing(), subprocess.CompletedProcess([], 0, json.dumps(latest), "")]
    )
    packages.publish_npm(tmp_path, "0.7.4")
    assert calls[0][2:4] == ["@superlinked/package-0@0.7.4", "dist.integrity"]
    assert calls[1][2:4] == ["@superlinked/package-0", "dist-tags.latest"]
    assert len(uploads) == 1
    assert uploads[0][3:5] == ("--tag", tag)
    assert uploads[0][:2] == ("npm", "publish")


def test_first_npm_publication_uses_latest(tmp_path, monkeypatch):
    _, uploads = npm_publish_fixture(tmp_path, monkeypatch, [npm_missing(), npm_missing()])
    packages.publish_npm(tmp_path, "0.7.4")
    assert uploads[0][3:5] == ("--tag", "latest")


@pytest.mark.parametrize("reply", ["not-json", "null", "{}", "[]", '"0.7.5-rc.1"', '"bogus"', '"01.2.3"', '""'])
def test_malformed_latest_blocks_all_pending_uploads(tmp_path, monkeypatch, reply):
    _, uploads = npm_publish_fixture(
        tmp_path,
        monkeypatch,
        [
            npm_missing(),
            subprocess.CompletedProcess([], 0, '"0.7.3"', ""),
            npm_missing(),
            subprocess.CompletedProcess([], 0, reply, ""),
        ],
        count=2,
    )
    with pytest.raises(ValueError, match="malformed npm"):
        packages.publish_npm(tmp_path, "0.7.4")
    assert uploads == []


@pytest.mark.parametrize("code", ["E500", "E401", "ETIMEDOUT"])
def test_latest_query_failure_does_not_publish(tmp_path, monkeypatch, code):
    _, uploads = npm_publish_fixture(
        tmp_path,
        monkeypatch,
        [
            npm_missing(),
            subprocess.CompletedProcess([], 1, json.dumps({"error": {"code": code}}), code),
        ],
    )
    with pytest.raises(ValueError, match="cannot check npm"):
        packages.publish_npm(tmp_path, "0.7.4")
    assert uploads == []


def test_identical_existing_npm_version_never_touches_latest(tmp_path, monkeypatch):
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(b"tested archive").digest()).decode()
    calls, uploads = npm_publish_fixture(
        tmp_path, monkeypatch, [subprocess.CompletedProcess([], 0, json.dumps(integrity), "")]
    )
    packages.publish_npm(tmp_path, "0.7.4")
    assert len(calls) == 1
    assert uploads == []
