from __future__ import annotations

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
