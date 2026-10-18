"""Adversarial build-input and installed-metadata checks; no model imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from sie_server.bundle_lock import (
    Target,
    _check_requirement_source,
    artifact_sources,
    canonical_digest,
    check_lock,
    digest,
    policy_digest,
    read_packages,
    required_exceptions,
    root_constraints,
    validate_exceptions,
    verify_inventory,
)

TARGET = Target("cpu", "default", "amd64")
HASH = "a" * 64
WHEEL_URL = "https://files.pythonhosted.org/packages/demo-1.0-py3-none-any.whl"


def write_distribution(installed: Path, name="demo", version="1.0", *, requires=(), direct=None):
    info = installed / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    text = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
    text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
    (info / "METADATA").write_text(text)
    if direct is not None:
        (info / "direct_url.json").write_text(json.dumps(direct))
    return info


@pytest.fixture
def locked(tmp_path):
    root_lock = tmp_path / "uv.lock"
    root_lock.write_text(
        'version = 1\n[[package]]\nname = "demo"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{url = "{WHEEL_URL}", hash = "sha256:{HASH}"}}]\n'
    )
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    (bundles / "default.yaml").write_text('name: default\ndeps:\n  demo: ">=1,<2"\n')
    locks = tmp_path / "locks"
    folder = locks / TARGET.name
    folder.mkdir(parents=True)
    lock = folder / "pylock.toml"
    lock.write_text(
        'lock-version = "1.0"\n[[packages]]\nname = "demo"\nversion = "1.0"\n'
        f'wheels = [{{url = "{WHEEL_URL}", hashes = {{ sha256 = "{HASH}" }} }}]\n'
    )
    manifest = {
        "schema_version": 1,
        "uv_version": "0.9.28",
        "policy_sha256": policy_digest(),
        "target": TARGET.as_dict(),
        "inputs": {"uv_lock_sha256": digest(root_lock), "requirements_sha256": canonical_digest(["demo>=1,<2"])},
        "lock_sha256": digest(lock),
        "exceptions_sha256": canonical_digest({}),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (locks / "exceptions.json").write_text(json.dumps({"schema_version": 1, "targets": {}}))
    return locks, bundles, root_lock


def test_clean_metadata_inventory_is_checked_without_importing_package(locked, tmp_path):
    checked = check_lock(*locked, TARGET)
    installed = tmp_path / "installed"
    write_distribution(installed)
    (installed / "demo.py").write_text('raise AssertionError("must not import installed code")')
    receipt = verify_inventory(checked, installed, TARGET)
    assert receipt["packages"] == [{"name": "demo", "version": "1.0", "direct_url": None}]
    assert receipt["lock_sha256"] == checked["manifest"]["lock_sha256"]


@pytest.mark.parametrize("mutation", ["root", "requirements", "target", "lock", "exception", "policy"])
def test_stale_inputs_and_wrong_target_fail_before_install(locked, mutation):
    locks, bundles, root_lock = locked
    manifest_path = locks / TARGET.name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "root":
        root_lock.write_text(root_lock.read_text() + "\n# different root lock\n")
    elif mutation == "requirements":
        (bundles / "default.yaml").write_text('deps:\n  demo: ">=1.1,<2"\n')
    elif mutation == "target":
        manifest["target"]["architecture"] = "arm64"
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "lock":
        lock = locks / TARGET.name / "pylock.toml"
        lock.write_text(lock.read_text() + "\n# altered\n")
    elif mutation == "policy":
        manifest["policy_sha256"] = "c" * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        (locks / "exceptions.json").write_text(
            json.dumps({"schema_version": 1, "targets": {TARGET.name: {"demo": {"version": "2.0"}}}})
        )
    with pytest.raises(ValueError, match=r"bundle .* (changed|differs)"):
        check_lock(*locked, TARGET)


@pytest.mark.parametrize("mutation", ["missing", "extra", "version", "duplicate", "source", "dependency"])
def test_actual_inventory_drift_is_rejected(locked, tmp_path, mutation):
    checked = check_lock(*locked, TARGET)
    installed = tmp_path / "installed"
    if mutation != "missing":
        write_distribution(
            installed,
            version="1.1" if mutation == "version" else "1.0",
            requires=("absent>=1",) if mutation == "dependency" else (),
            direct={"url": "https://unapproved.example/demo.whl", "archive_info": {"hashes": {"sha256": HASH}}}
            if mutation == "source"
            else None,
        )
    if mutation == "extra":
        write_distribution(installed, "unlocked")
    if mutation == "duplicate":
        write_distribution(installed, version="0.9")
    with pytest.raises(ValueError, match=r"(installed|distribution)"):
        verify_inventory(checked, installed, TARGET)


def test_requested_extra_dependency_is_not_silently_omitted(locked, tmp_path):
    checked = check_lock(*locked, TARGET)
    checked["requirements"] = ["demo[feature]>=1"]
    installed = tmp_path / "installed"
    write_distribution(installed, requires=('absent>=1; extra == "feature"',))
    with pytest.raises(ValueError, match="requires absent"):
        verify_inventory(checked, installed, TARGET)


def test_inactive_dependency_marker_does_not_charge_another_platform(locked, tmp_path):
    checked = check_lock(*locked, TARGET)
    installed = tmp_path / "installed"
    write_distribution(installed, requires=('absent>=1; sys_platform == "win32"',))
    verify_inventory(checked, installed, TARGET)


def test_actual_python_patch_dependency_is_checked(locked, tmp_path, monkeypatch):
    checked = check_lock(*locked, TARGET)
    installed = tmp_path / "installed"
    write_distribution(installed, requires=('absent>=1; python_full_version >= "3.12.1"',))
    monkeypatch.setattr("sie_server.bundle_lock.host_platform.python_version", lambda: "3.12.12")
    with pytest.raises(ValueError, match="requires absent"):
        verify_inventory(checked, installed, TARGET)


def test_unsupported_image_combination_is_explicit(locked):
    with pytest.raises(ValueError, match="unsupported bundle lock target"):
        check_lock(*locked, Target("cuda13", "default", "arm64"))


def test_exact_exception_rejects_wrong_version_source_and_stale_record(locked):
    packages = read_packages(locked[0] / TARGET.name / "pylock.toml", TARGET)
    package = packages["demo"]
    package["version"] = "1.1"
    required = required_exceptions(packages, locked[2])
    approved = {"demo": required["demo"] | {"reason": "Declared engine requires demo>=1.1; root1.0 is incompatible."}}
    validate_exceptions(packages, locked[2], approved)
    assert root_constraints(locked[2], approved) == ["demo==1.1"]
    with pytest.raises(ValueError, match="missing"):
        validate_exceptions(packages, locked[2], {})
    package["wheels"][0]["url"] = "https://changed.example/demo-1.0-py3-none-any.whl"
    with pytest.raises(ValueError, match="not exact"):
        validate_exceptions(packages, locked[2], approved)
    with pytest.raises(ValueError, match="stale"):
        validate_exceptions({}, locked[2], approved)


def test_root_pins_do_not_use_a_post_release_wildcard(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname="torch"\nversion="2.9.1+cu129"\nsource={registry="https://download.pytorch.org/whl/cu129"}\n'
    )
    assert root_constraints(lock, {}) == ["torch==2.9.1"]


@pytest.mark.parametrize(
    "source",
    [
        {"archive": {"url": "https://files.pythonhosted.org/demo.whl", "hashes": {}}},
        {"archive": {"url": "https://user:secret@example.test/demo.whl", "hashes": {"sha256": HASH}}},
        {"vcs": {"type": "git", "url": "https://github.com/example/demo", "commit-id": "main"}},
    ],
)
def test_unbound_or_credential_bearing_sources_are_rejected(source):
    with pytest.raises(ValueError, match=r"(requires|must use)"):
        artifact_sources({"name": "demo", "version": "1", **source})


@pytest.mark.parametrize("actual_commit", ["c" * 40, "b" * 40])
def test_vcs_commit_is_verified_from_installed_metadata(locked, tmp_path, actual_commit):
    checked = check_lock(*locked, TARGET)
    checked["packages"]["demo"] = {
        "name": "demo",
        "version": "1.0",
        "vcs": {"type": "git", "url": "https://github.com/example/demo", "commit-id": "b" * 40},
    }
    installed = tmp_path / "installed"
    write_distribution(
        installed,
        direct={"url": "https://github.com/example/demo", "vcs_info": {"vcs": "git", "commit_id": actual_commit}},
    )
    if actual_commit == "b" * 40:
        verify_inventory(checked, installed, TARGET)
    else:
        with pytest.raises(ValueError, match="VCS source"):
            verify_inventory(checked, installed, TARGET)


def test_arm64_lock_cannot_silently_reuse_x86_wheel(locked):
    path = locked[0] / TARGET.name / "pylock.toml"
    path.write_text(path.read_text().replace("py3-none-any.whl", "cp312-cp312-manylinux_2_28_x86_64.whl"))
    with pytest.raises(ValueError, match="architecture"):
        read_packages(path, Target("cpu", "default", "arm64"))


@pytest.mark.parametrize("source", ["archive", "vcs"])
def test_declared_source_cannot_be_substituted_with_same_version(source):
    if source == "archive":
        requirement = Requirement(f"demo @ {WHEEL_URL}")
        package = {"archive": {"url": WHEEL_URL}}
        _check_requirement_source(requirement, package)
        package["archive"]["url"] = "https://example.test/different.whl"
    else:
        requirement = Requirement("demo @ git+https://github.com/example/demo@" + "b" * 40)
        package = {"vcs": {"url": "https://github.com/example/demo", "commit-id": "b" * 40}}
        _check_requirement_source(requirement, package)
        package["vcs"]["commit-id"] = "c" * 40
    with pytest.raises(ValueError, match=r"declared .* source"):
        _check_requirement_source(requirement, package)


def test_direct_root_artifact_is_not_silently_omitted(locked):
    locks, _bundles, root_lock = locked
    root_lock.write_text(root_lock.read_text().replace('registry = "https://pypi.org/simple"', f'url = "{WHEEL_URL}"'))
    packages = read_packages(locks / TARGET.name / "pylock.toml", TARGET)
    assert root_constraints(root_lock, {}) == ["demo==1.0"]
    assert required_exceptions(packages, root_lock) == {}
    packages["demo"]["wheels"][0]["hashes"]["sha256"] = "b" * 64
    assert "demo" in required_exceptions(packages, root_lock)
