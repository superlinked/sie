#!/usr/bin/env python3
"""Verify the real public release boundary before invoking any writer."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPOSITORY = "superlinked/sie"
SEED_VERSION = "0.7.3"
SHA = re.compile(r"[0-9a-f]{40}")


def stable_version(version: str, *, new: bool = True) -> tuple[int, ...]:
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("release version must be stable X.Y.Z")
    parts = tuple(map(int, version.split(".")))
    if new and parts <= tuple(map(int, SEED_VERSION.split("."))):
        raise ValueError("new publication must be newer than the 0.7.3 seed")
    return parts


def seed_manifest(manifest: dict) -> bool:
    if set(manifest) != {"."}:
        raise ValueError("release manifest must contain one coordinated root version")
    version = stable_version(manifest["."], new=False)
    seed = stable_version(SEED_VERSION, new=False)
    if version < seed:
        raise ValueError("release manifest must not move below the 0.7.3 seed")
    return version == seed


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()  # noqa: S603


def api(path: str) -> dict:
    return json.loads(command("gh", "api", f"repos/{REPOSITORY}/{path}"))


def stable_release(version: str, source_sha: str | None = None) -> str:
    stable_version(version, new=False)
    tag = f"v{version}"
    release = api(f"releases/tags/{tag}")
    if release.get("tag_name") != tag or release.get("draft") is not False or release.get("prerelease") is not False:
        raise ValueError(f"{tag} must have a genuine stable GitHub Release")
    obj = api(f"git/ref/tags/{tag}")["object"]
    for _ in range(5):
        if obj["type"] == "commit":
            break
        if obj["type"] != "tag" or not SHA.fullmatch(obj["sha"]):
            raise ValueError("release tag does not resolve to a commit")
        obj = api(f"git/tags/{obj['sha']}")["object"]
    sha = obj["sha"]
    if obj["type"] != "commit" or not SHA.fullmatch(sha) or (source_sha is not None and sha != source_sha):
        raise ValueError("release tag commit differs from the original release SHA")
    command("git", "fetch", "--no-tags", "origin", f"refs/tags/{tag}")
    if command("git", "rev-parse", "FETCH_HEAD^{commit}") != sha:
        raise ValueError("fetched release tag differs from GitHub tag identity")
    command("git", "merge-base", "--is-ancestor", sha, "HEAD")
    return sha


def published_event(environment: dict[str, str], event: dict, source_sha: str) -> str:
    release = event.get("release", {})
    tag = release.get("tag_name", "")
    version = tag.removeprefix("v")
    stable_version(version)
    if (
        environment.get("GITHUB_REPOSITORY") != REPOSITORY
        or event.get("repository", {}).get("full_name") != REPOSITORY
        or environment.get("GITHUB_EVENT_NAME") != "release"
        or environment.get("GITHUB_REF_PROTECTED") != "true"
        or event.get("action") != "published"
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or tag != f"v{version}"
        or environment.get("GITHUB_REF") != f"refs/tags/{tag}"
        or not SHA.fullmatch(source_sha)
        or environment.get("GITHUB_SHA") != source_sha
    ):
        raise ValueError("publication requires the exact stable published release event and tag SHA")
    return version


def protected_main_ancestor(source_sha: str) -> None:
    if api("branches/main").get("protected") is not True:
        raise ValueError("publication requires independently verified protected main")
    command("git", "fetch", "--no-tags", "origin", "refs/heads/main")
    command("git", "merge-base", "--is-ancestor", source_sha, "FETCH_HEAD")


def event_payload(environment: dict[str, str]) -> dict:
    return json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text())


def trusted_context(
    environment: dict[str, str], source_sha: str, *, recovery: bool = False, event: dict | None = None
) -> None:
    expected = {"GITHUB_REPOSITORY": REPOSITORY, "PUBLIC_RELEASE_PUBLISHING_ENABLED": "true"}
    if recovery:
        expected.update(
            GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_REF="refs/heads/main", GITHUB_REF_PROTECTED="true"
        )
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError("publication requires activated protected public main and the correct event")
    if not SHA.fullmatch(source_sha) or environment.get("GITHUB_SHA") != source_sha:
        raise ValueError("publication SHA must be the original workflow SHA")
    if not recovery:
        published_event(environment, event if event is not None else event_payload(environment), source_sha)


def prepare_release(environment: dict[str, str], event: dict) -> dict[str, str]:
    source_sha = environment["GITHUB_SHA"]
    version = published_event(environment, event, source_sha)
    if command("git", "rev-parse", "HEAD") != source_sha:
        raise ValueError("release event checkout differs from the tagged commit")
    stable_release(SEED_VERSION)
    stable_release(version, source_sha)
    protected_main_ancestor(source_sha)
    return {"sha": source_sha, "version": version, "tag_name": f"v{version}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["seed", "prepare", "build", "publish"])
    parser.add_argument("--version", default="")
    parser.add_argument("--tag-name", default="")
    parser.add_argument("--source-ref", default="")
    args = parser.parse_args()
    if args.mode == "seed":
        at_seed = seed_manifest(json.loads(Path(".release-please-manifest.json").read_text()))
        stable_release(SEED_VERSION)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"at_seed={str(at_seed).lower()}\n")
        return
    environment = dict(os.environ)
    if args.mode == "prepare":
        identity = prepare_release(environment, event_payload(environment))
        with Path(environment["GITHUB_OUTPUT"]).open("a") as output:
            output.writelines(f"{key}={value}\n" for key, value in identity.items())
        return
    if not SHA.fullmatch(args.source_ref) or command("git", "rev-parse", "HEAD") != args.source_ref:
        raise ValueError("checkout is not the exact requested source SHA")
    if args.version:
        stable_version(args.version)
        if args.tag_name != f"v{args.version}":
            raise ValueError("release tag/version mismatch")
        stable_release(args.version, args.source_ref)
    elif args.tag_name or args.mode == "publish":
        raise ValueError("publication requires an exact stable release version")
    if args.mode == "publish":
        event = event_payload(environment)
        trusted_context(environment, args.source_ref, event=event)
        if published_event(environment, event, args.source_ref) != args.version:
            raise ValueError("publisher version differs from the original release event")
        protected_main_ancestor(args.source_ref)


if __name__ == "__main__":
    main()
