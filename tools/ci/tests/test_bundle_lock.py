"""The saved bundle policy is public, complete and rejects drift."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sie_server.bundle_lock import Target

from tools.ci import bundle_lock


def test_target_matrix_preserves_cpu_arm64_in_addition_to_every_release():
    selected = bundle_lock.targets(bundle_lock.ROOT)
    assert len(selected) == 16
    assert len({target.name for target in selected}) == 16
    assert {target.bundle for target in selected if target.architecture == "arm64"} == {
        "default",
        "ctranslate2",
        "sglang",
        "transformers5",
    }
    assert all(target.platform == "cpu" for target in selected if target.architecture == "arm64")


def test_ambient_private_index_or_hash_bypass_is_not_a_compilation_input(monkeypatch):
    monkeypatch.setenv("UV_NO_VERIFY_HASHES", "1")
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.test/private")
    monkeypatch.setenv("UV_OVERRIDE", "unreviewed.txt")
    monkeypatch.setenv("SIE_BUNDLE_CONTROL", "retained")
    env = bundle_lock.resolver_environment()
    assert not any(key.startswith(("UV_", "PIP_")) for key in env)
    assert env["SIE_BUNDLE_CONTROL"] == "retained"


class Response(io.BytesIO):
    def __init__(self, data=b"public wheel bytes", url="https://example.test/demo-1-py3-none-any.whl"):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url


def missing_hash(tmp_path, url="https://example.test/demo-1-py3-none-any.whl"):
    path = tmp_path / "pylock.toml"
    path.write_text(
        'lock-version = "1.0"\n[[packages]]\nname = "demo"\nversion = "1"\n'
        f"wheels = [{{url = {json.dumps(url)}, hashes = {{}} }}]\n"
    )
    return path


def test_missing_index_hash_is_bound_to_actual_downloaded_bytes(tmp_path, monkeypatch):
    path = missing_hash(tmp_path)
    monkeypatch.setattr(bundle_lock, "urlopen", lambda *_args, **_kwargs: Response())
    completed = bundle_lock.complete_artifact_hashes(path)
    expected = hashlib.sha256(b"public wheel bytes").hexdigest()
    assert completed[0]["sha256"] == expected
    assert completed[0]["bytes"] == len(b"public wheel bytes")
    assert expected in path.read_text()
    monkeypatch.setattr(bundle_lock, "urlopen", lambda *_args, **_kwargs: pytest.fail("must not redownload"))
    assert bundle_lock.complete_artifact_hashes(path) == []


@pytest.mark.parametrize("url", ["http://example.test/wheel", "https://user:password@example.test/wheel"])
def test_untrusted_missing_hash_source_fails_before_network(tmp_path, monkeypatch, url):
    path = missing_hash(tmp_path, url)
    prior = path.read_bytes()
    monkeypatch.setattr(bundle_lock, "urlopen", lambda *_args, **_kwargs: pytest.fail("must not fetch"))
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        bundle_lock.complete_artifact_hashes(path)
    assert path.read_bytes() == prior


def test_insecure_redirect_and_oversized_hash_completion_do_not_save(tmp_path, monkeypatch):
    path = missing_hash(tmp_path)
    prior = path.read_bytes()
    monkeypatch.setattr(bundle_lock, "urlopen", lambda *_args, **_kwargs: Response(url="http://example.test/redirect"))
    with pytest.raises(ValueError, match="redirect"):
        bundle_lock.complete_artifact_hashes(path)
    assert path.read_bytes() == prior
    monkeypatch.setattr(Response, "read", lambda *_args: b"a" * (1024 * 1024))
    monkeypatch.setattr(bundle_lock, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(ValueError, match="64MiB"):
        bundle_lock.complete_artifact_hashes(path)
    assert path.read_bytes() == prior


def test_saved_target_and_index_policy_cannot_drift(tmp_path, monkeypatch):
    target = Target("cpu", "default", "amd64")
    folder = tmp_path / bundle_lock.LOCKS / target.name
    folder.mkdir(parents=True)
    monkeypatch.setattr(bundle_lock, "targets", lambda _root: [target])
    monkeypatch.setattr(bundle_lock, "check_lock", lambda *_args, **_kwargs: {"manifest": {"indexes": ["private"]}})
    with pytest.raises(ValueError, match="index policy"):
        bundle_lock.check_all(tmp_path)
    (folder.parent / "unexpected").mkdir()
    with pytest.raises(ValueError, match="target set"):
        bundle_lock.check_all(tmp_path)


def test_repository_locks_bind_current_public_inputs():
    bundle_lock.check_all(bundle_lock.ROOT)


def test_compile_another_checkout_binds_its_compiler_policy(tmp_path, monkeypatch):
    target = Target("cpu", "default", "amd64")
    compiler = tmp_path / "tools/ci/bundle_lock.py"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("# Another checkout's independently reviewed compiler.\n")
    locks = tmp_path / bundle_lock.LOCKS
    locks.mkdir(parents=True)
    (locks / "exceptions.json").write_text(json.dumps({"schema_version": 1, "targets": {}}))
    bundles = tmp_path / bundle_lock.BUNDLES
    bundles.mkdir(parents=True)
    (bundles / "default.yaml").write_text('deps:\n  demo: ">=1,<2"\n')
    url = "https://files.pythonhosted.org/packages/demo-1.0-py3-none-any.whl"
    sha = "a" * 64
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "demo"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{url = "{url}", hash = "sha256:{sha}"}}]\n'
    )
    calls = []

    def controlled_resolver(command, **kwargs):
        calls.append(command)
        if command == ["uv", "--version"]:
            return SimpleNamespace(stdout="uv 0.9.28\n")
        assert command[:3] == ["uv", "pip", "compile"]
        directory = Path(kwargs["cwd"])
        assert (directory / "requirements.in").read_text() == "demo>=1,<2\n"
        assert (directory / "constraints.txt").read_text() == "demo==1.0\n"
        (directory / "pylock.toml").write_text(
            'lock-version = "1.0"\n[[packages]]\nname = "demo"\nversion = "1.0"\n'
            f'wheels = [{{url = "{url}", hashes = {{sha256 = "{sha}"}}}}]\n'
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bundle_lock.subprocess, "run", controlled_resolver)
    bundle_lock.compile_target(tmp_path, target, "uv", "2026-10-18T00:00:00Z", None)
    assert len(calls) == 2
    checked = bundle_lock.check_lock(locks, bundles, tmp_path / "uv.lock", target, compiler=compiler)
    assert checked["manifest"]["policy_sha256"] == bundle_lock.policy_digest(compiler)
    assert checked["manifest"]["policy_sha256"] != bundle_lock.policy_digest()
    assert checked["packages"]["demo"]["version"] == "1.0"
    with pytest.raises(ValueError, match="policy changed"):
        bundle_lock.check_lock(locks, bundles, tmp_path / "uv.lock", target)
    compiler.write_text("# Unreviewed change in the other checkout.\n")
    with pytest.raises(ValueError, match="policy changed"):
        bundle_lock.check_lock(locks, bundles, tmp_path / "uv.lock", target, compiler=compiler)


def test_direct_platform_pins_keep_the_global_index_cutoff(tmp_path, monkeypatch):
    target = Target("cuda13", "sglang-cu130", "amd64")
    locks = tmp_path / bundle_lock.LOCKS
    locks.mkdir(parents=True)
    url = "https://example.test/demo-1.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    pin = {"version": "1.0+cu130", "url": url, "sha256": "a" * 64, "reason": "Undated native platform wheel."}
    (locks / "exceptions.json").write_text(json.dumps({"targets": {}, "source_pins": {target.name: {"demo": pin}}}))
    monkeypatch.setattr(bundle_lock, "bundle_requirements", lambda *_args: ["demo==1.0"])
    monkeypatch.setattr(bundle_lock, "root_constraints", lambda *_args: ["packaging==24.2"])
    calls = []

    def compile_probe(command, **kwargs):
        if command == ["uv", "--version"]:
            return SimpleNamespace(stdout="uv 0.9.28\n")
        calls.append(command)
        constraints = (Path(kwargs["cwd"]) / "constraints.txt").read_text().splitlines()
        assert constraints == ["packaging==24.2", f"demo @ {url}#sha256={'a' * 64}"]
        assert command[command.index("--exclude-newer") + 1] == "2026-10-18T00:00:00Z"
        assert "--exclude-newer-package" not in command
        raise RuntimeError("stop before resolver")

    monkeypatch.setattr(bundle_lock.subprocess, "run", compile_probe)
    with pytest.raises(RuntimeError, match="stop before resolver"):
        bundle_lock.compile_target(tmp_path, target, "uv", "2026-10-18T00:00:00Z", None)
    assert len(calls) == 1
