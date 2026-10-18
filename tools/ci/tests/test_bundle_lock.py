"""The saved bundle policy is public, complete and rejects drift."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from sie_server.bundle_lock import Target

from tools.ci import bundle_lock


def test_target_matrix_preserves_cpu_arm64_in_addition_to_every_release():
    selected = bundle_lock.targets(bundle_lock.ROOT)
    assert len(selected) == 15
    assert len({target.name for target in selected}) == 15
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
