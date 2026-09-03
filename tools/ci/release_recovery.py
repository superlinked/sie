#!/usr/bin/env python3
"""Request a retry on the original release run; never publish from this dispatch."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime, timedelta

from tools.ci.release_guard import REPOSITORY, api, command, stable_release, stable_version, trusted_context

FAMILIES = ("python", "npm", "docker", "helm", "audio", "native")
FAILED = {"failure", "timed_out", "cancelled", "action_required"}
PUBLISH_JOB = re.compile(
    r"^(python-publish|npm-publish)$|^(docker|helm|audio|native) / "
    r"(push-server|push-service|publish|verify|alias)(?:\s|\(|$)"
)


def artifact_names(family: str, version: str) -> set[str]:
    if family in {"python", "npm"}:
        return {f"{family}-distributions"}
    if family == "docker":
        pairs = [
            (platform, bundle)
            for platform in ("cpu", "cuda12")
            for bundle in ("default", "ctranslate2", "sglang", "transformers5")
        ]
        pairs += [("cuda13", "sglang-cu130"), ("cuda13", "tensorrt-llm")]
        return {f"docker-server-{platform}-{bundle}-{version}" for platform, bundle in pairs} | {
            f"docker-service-{service}-{version}"
            for service in ("sie-gateway", "sie-config", "sie-mcp", "sie-server-sidecar", "sie-server-rust")
        }
    return {f"{ {'helm': 'helm-sie-cluster', 'audio': 'audio-prep', 'native': 'native-sidecar'}[family] }-{version}"}


def validate_run(run: dict, *, original_run: int, source_sha: str, now: datetime) -> None:
    expected = {
        "id": original_run,
        "event": "push",
        "head_branch": "main",
        "head_sha": source_sha,
        "path": ".github/workflows/release.yml",
        "status": "completed",
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("original run must be the completed release.yml push for the exact release SHA")
    if (
        run.get("repository", {}).get("full_name") != REPOSITORY
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
    ):
        raise ValueError("original run must belong to the public repository, not a fork")
    age = now - datetime.fromisoformat(run["created_at"])
    if not timedelta(0) <= age < timedelta(days=30):
        raise ValueError("original run is outside GitHub's 30-day rerun window")
    if run.get("conclusion") == "success":
        raise ValueError("successful releases do not need recovery")


def validate_artifacts(
    artifacts: list[dict], wanted: set[str], *, original_run: int, source_sha: str, now: datetime
) -> None:
    for name in wanted:
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous original archive: {name}")
        artifact = matches[0]
        linkage = artifact.get("workflow_run", {})
        if (
            linkage.get("id") != original_run
            or linkage.get("head_sha") != source_sha
            or linkage.get("head_branch") != "main"
        ):
            raise ValueError(f"archive is not bound to the original release run: {name}")
        if artifact.get("expired") is not False or datetime.fromisoformat(artifact["expires_at"]) <= now:
            raise ValueError(f"original archive has expired: {name}")
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", ""))
            or artifact.get("size_in_bytes", 0) <= 0
        ):
            raise ValueError(f"original archive has no immutable digest: {name}")


def selected_jobs(jobs: list[dict], family: str) -> list[int]:
    release_jobs = [job for job in jobs if job.get("name") == "release-please"]
    if len(release_jobs) != 1 or release_jobs[0].get("conclusion") != "success":
        raise ValueError("original release-please must remain successful; it must not be rerun")
    prefixes = ("python-publish",) if family == "python" else ("npm-publish",) if family == "npm" else (f"{family} /",)
    if family == "all":
        prefixes = ("python-publish", "npm-publish", "docker /", "helm /", "audio /", "native /")
    selected = [
        job["id"]
        for job in jobs
        if job.get("conclusion") in FAILED
        and PUBLISH_JOB.match(job.get("name", ""))
        and job.get("name", "").startswith(prefixes)
    ]
    if not selected:
        raise ValueError(
            "no failed original family jobs to rerun; skipped-only publication requires operator diagnosis"
        )
    return selected


def retry_endpoint(jobs: list[dict], family: str, original_run: int) -> str:
    selected = selected_jobs(jobs, family)
    if family != "all" and len(selected) == 1:
        return f"actions/jobs/{selected[0]}/rerun"
    other_failures = [
        job
        for job in jobs
        if job.get("conclusion") in FAILED
        and job["id"] not in selected
        and job.get("name", "").split(" / ")[-1] != "complete"
    ]
    if other_failures:
        raise ValueError("retry would also rerun other failed jobs; use all only after resolving failed builders")
    return f"actions/runs/{original_run}/rerun-failed-jobs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--original-run", required=True, type=int)
    parser.add_argument("--family", choices=["all", *FAMILIES], default="all")
    args = parser.parse_args()
    stable_version(args.version)
    if args.original_run <= 0 or str(args.original_run) == os.environ.get("GITHUB_RUN_ID"):
        raise ValueError("recovery requires a different original run")
    trusted_context(dict(os.environ), os.environ["GITHUB_SHA"], recovery=True)
    if command("git", "rev-parse", "HEAD") != os.environ["GITHUB_SHA"]:
        raise ValueError("recovery must execute the reviewed dispatch commit")
    source_sha = stable_release(args.version)
    now = datetime.now(UTC)
    run = api(f"actions/runs/{args.original_run}")
    validate_run(run, original_run=args.original_run, source_sha=source_sha, now=now)
    pages = json.loads(
        command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPOSITORY}/actions/runs/{args.original_run}/artifacts?per_page=100",
        )
    )
    artifacts = [artifact for page in pages for artifact in page["artifacts"]]
    families = FAMILIES if args.family == "all" else (args.family,)
    wanted = set().union(*(artifact_names(family, args.version) for family in families))
    validate_artifacts(artifacts, wanted, original_run=args.original_run, source_sha=source_sha, now=now)
    pages = json.loads(
        command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPOSITORY}/actions/runs/{args.original_run}/jobs?filter=latest&per_page=100",
        )
    )
    endpoint = retry_endpoint([job for page in pages for job in page["jobs"]], args.family, args.original_run)
    command("gh", "api", "--method", "POST", f"repos/{REPOSITORY}/{endpoint}")
    print(
        f"Requested {args.family} recovery on original run {args.original_run}; "
        "no artifacts published by this dispatch."
    )


if __name__ == "__main__":
    main()
