from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from tools.ci import release_guard as guard
from tools.ci import release_recovery as recovery

SHA = "a" * 40
NOW = datetime(2026, 9, 3, tzinfo=UTC)


@pytest.mark.parametrize("version", ["0.7.2", "0.7.3", "0.0.1", "0.7.4-rc.1", "v0.7.4", "01.2.3", "bad"])
def test_new_publication_rejects_seed_and_nonstable_versions(version):
    with pytest.raises(ValueError, match=r"release|publication|original|archive|successful"):
        guard.stable_version(version)


def context():
    return {
        "GITHUB_REPOSITORY": "superlinked/sie",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_PROTECTED": "true",
        "PUBLIC_RELEASE_PUBLISHING_ENABLED": "true",
        "GITHUB_SHA": SHA,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("GITHUB_REPOSITORY", "someone/fork"),
        ("GITHUB_EVENT_NAME", "workflow_dispatch"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_REF_PROTECTED", "false"),
        ("PUBLIC_RELEASE_PUBLISHING_ENABLED", "false"),
        ("GITHUB_SHA", "b" * 40),
    ],
)
def test_actual_writer_guard_rejects_untrusted_context(key, value):
    guard.trusted_context(context(), SHA)
    with pytest.raises(ValueError, match=r"release|publication|original|archive|successful"):
        guard.trusted_context({**context(), key: value}, SHA)


def test_seed_checks_fixed_real_release_and_commit(monkeypatch):
    calls = []

    def api(path):
        calls.append(path)
        return (
            {"tag_name": "v0.7.3", "draft": False, "prerelease": False}
            if path.startswith("releases/")
            else {"object": {"sha": SHA, "type": "commit"}}
        )

    monkeypatch.setattr(guard, "api", api)
    monkeypatch.setattr(guard, "command", lambda *args: SHA if args[1] == "rev-parse" else "")
    assert guard.stable_release(guard.SEED_VERSION) == SHA
    assert calls == ["releases/tags/v0.7.3", "git/ref/tags/v0.7.3"]


def test_seed_manifest_is_not_a_new_publication_candidate():
    assert guard.seed_manifest({".": "0.7.3"}) is True
    assert guard.seed_manifest({".": "0.7.4"}) is False
    with pytest.raises(ValueError, match="below"):
        guard.seed_manifest({".": "0.7.2"})
    with pytest.raises(ValueError, match="one coordinated"):
        guard.seed_manifest({".": "0.7.3", "other": "0.7.4"})


def test_seed_requires_actual_tag_ancestry_in_a_controlled_repository(tmp_path, monkeypatch):
    def command(*args):
        return subprocess.check_output(args, cwd=tmp_path, text=True, stderr=subprocess.DEVNULL).strip()  # noqa: S603

    command("git", "init", "-b", "main")
    command("git", "config", "user.name", "Release test")
    command("git", "config", "user.email", "release-test@example.invalid")
    command("git", "config", "commit.gpgsign", "false")
    command("git", "config", "core.hooksPath", "/dev/null")
    command("git", "commit", "--allow-empty", "-m", "test seed")
    command("git", "-c", "tag.gpgsign=false", "tag", "v0.7.3")
    source = command("git", "rev-parse", "HEAD")
    command("git", "remote", "add", "origin", str(tmp_path))
    command("git", "commit", "--allow-empty", "-m", "after seed")
    monkeypatch.setattr(guard, "command", command)
    monkeypatch.setattr(
        guard,
        "api",
        lambda path: (
            {"tag_name": "v0.7.3", "draft": False, "prerelease": False}
            if path.startswith("releases/")
            else {"object": {"sha": source, "type": "commit"}}
        ),
    )
    assert guard.stable_release("0.7.3") == source
    with pytest.raises(ValueError, match="original release SHA"):
        guard.stable_release("0.7.3", "b" * 40)
    command("git", "checkout", "--orphan", "unrelated")
    command("git", "commit", "--allow-empty", "-m", "unrelated")
    with pytest.raises(subprocess.CalledProcessError):
        guard.stable_release("0.7.3")


@pytest.mark.parametrize(
    "release",
    [
        {},
        {"tag_name": "v0.7.3", "draft": True, "prerelease": False},
        {"tag_name": "v0.7.3", "draft": False, "prerelease": True},
        {"tag_name": "v0.7.2", "draft": False, "prerelease": False},
    ],
)
def test_missing_or_unstable_seed_cannot_bootstrap(monkeypatch, release):
    monkeypatch.setattr(guard, "api", lambda path: release)
    with pytest.raises(ValueError, match="genuine stable GitHub Release"):
        guard.stable_release(guard.SEED_VERSION)


def run_record():
    return {
        "id": 123,
        "event": "push",
        "head_branch": "main",
        "head_sha": SHA,
        "path": ".github/workflows/release.yml",
        "status": "completed",
        "conclusion": "failure",
        "repository": {"full_name": "superlinked/sie"},
        "head_repository": {"full_name": "superlinked/sie"},
        "created_at": (NOW - timedelta(days=1)).isoformat(),
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", 999),
        ("event", "workflow_dispatch"),
        ("head_branch", "feature"),
        ("head_sha", "b" * 40),
        ("path", ".github/workflows/other.yml"),
        ("status", "in_progress"),
        ("conclusion", "success"),
        ("repository", {"full_name": "someone/fork"}),
        ("head_repository", {"full_name": "someone/fork"}),
        ("created_at", (NOW - timedelta(days=30)).isoformat()),
        ("created_at", (NOW + timedelta(days=1)).isoformat()),
    ],
)
def test_recovery_cannot_change_original_provenance(key, value):
    recovery.validate_run(run_record(), original_run=123, source_sha=SHA, now=NOW)
    with pytest.raises(ValueError, match=r"release|publication|original|archive|successful"):
        recovery.validate_run({**run_record(), key: value}, original_run=123, source_sha=SHA, now=NOW)


def artifact():
    return {
        "name": "python-distributions",
        "workflow_run": {"id": 123, "head_sha": SHA, "head_branch": "main"},
        "expired": False,
        "expires_at": (NOW + timedelta(days=1)).isoformat(),
        "digest": "sha256:" + "b" * 64,
        "size_in_bytes": 500,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "wrong"),
        ("expired", True),
        ("expires_at", NOW.isoformat()),
        ("digest", ""),
        ("size_in_bytes", 0),
        ("workflow_run", {"id": 999, "head_sha": SHA, "head_branch": "main"}),
        ("workflow_run", {"id": 123, "head_sha": "b" * 40, "head_branch": "main"}),
    ],
)
def test_recovery_rejects_missing_expired_or_unbound_archives(key, value):
    kwargs = {"original_run": 123, "source_sha": SHA, "now": NOW}
    recovery.validate_artifacts([artifact()], {"python-distributions"}, **kwargs)
    with pytest.raises(ValueError, match=r"release|publication|original|archive|successful"):
        recovery.validate_artifacts([{**artifact(), key: value}], {"python-distributions"}, **kwargs)


def test_retry_selector_only_reruns_failed_original_jobs():
    jobs = [
        {"id": 0, "name": "release-please", "conclusion": "success"},
        {"id": 1, "name": "python-publish", "conclusion": "failure"},
        {"id": 2, "name": "npm-publish", "conclusion": "success"},
        {"id": 3, "name": "docker / push-server", "conclusion": "failure"},
    ]
    assert recovery.selected_jobs(jobs, "python") == [1]
    assert recovery.selected_jobs(jobs, "docker") == [3]
    assert recovery.selected_jobs(jobs, "all") == [1, 3]
    with pytest.raises(ValueError, match=r"release|publication|original|archive|successful"):
        recovery.selected_jobs(jobs, "npm")


def test_archive_recovery_scope_contains_all_families():
    assert len(recovery.artifact_names("docker", "0.7.4")) == 15
    assert recovery.artifact_names("native", "0.7.4") == {"native-sidecar-0.7.4"}


def test_retry_never_replays_release_please_or_only_completion():
    jobs = [
        {"id": 0, "name": "release-please", "conclusion": "success"},
        {"id": 1, "name": "complete", "conclusion": "failure"},
        {"id": 2, "name": "python-publish", "conclusion": "skipped"},
    ]
    with pytest.raises(ValueError, match="skipped-only"):
        recovery.selected_jobs(jobs, "all")
    jobs[0]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="must not be rerun"):
        recovery.selected_jobs(jobs, "all")


def test_retry_only_makes_one_api_call_for_matrix_failures():
    jobs = [
        {"id": 0, "name": "release-please", "conclusion": "success"},
        {"id": 1, "name": "docker / push-server (cpu)", "conclusion": "failure"},
        {"id": 2, "name": "docker / push-service (gateway)", "conclusion": "failure"},
        {"id": 3, "name": "docker / complete", "conclusion": "failure"},
    ]
    assert recovery.retry_endpoint(jobs, "docker", 123) == "actions/runs/123/rerun-failed-jobs"
    jobs.append({"id": 4, "name": "npm-publish", "conclusion": "failure"})
    with pytest.raises(ValueError, match="other failed jobs"):
        recovery.retry_endpoint(jobs, "docker", 123)
    assert recovery.retry_endpoint(jobs, "npm", 123) == "actions/jobs/4/rerun"
    assert recovery.retry_endpoint(jobs, "all", 123) == "actions/runs/123/rerun-failed-jobs"


def test_failed_builders_and_skipped_publisher_completion_are_not_recovery():
    jobs = [
        {"id": 0, "name": "release-please", "conclusion": "success"},
        {"id": 1, "name": "docker / complete", "conclusion": "failure"},
        {"id": 2, "name": "docker / publish", "conclusion": "skipped"},
    ]
    with pytest.raises(ValueError, match="skipped-only"):
        recovery.retry_endpoint(jobs, "all", 123)
    jobs += [
        {"id": 3, "name": "python / build", "conclusion": "failure"},
        {"id": 4, "name": "npm-publish", "conclusion": "failure"},
    ]
    with pytest.raises(ValueError, match="failed builders"):
        recovery.retry_endpoint(jobs, "all", 123)
