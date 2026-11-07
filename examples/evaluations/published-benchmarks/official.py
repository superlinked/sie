"""Look up a model's official number in the public MTEB results repository, with the standard library only."""

import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

RESULTS_REPOSITORY = "embeddings-benchmark/results"
COMMIT_URL = f"https://api.github.com/repos/{RESULTS_REPOSITORY}/commits/main"
CONTENTS_URL = f"https://api.github.com/repos/{RESULTS_REPOSITORY}/contents/results"
RAW_URL = f"https://raw.githubusercontent.com/{RESULTS_REPOSITORY}"
BLOB_URL = f"https://github.com/{RESULTS_REPOSITORY}/blob"
SHA_MEDIA_TYPE = "application/vnd.github.sha"
JSON_MEDIA_TYPE = "application/vnd.github+json"
USER_AGENT = "published-benchmarks-example"
TIMEOUT_S = 30.0
MAX_BYTES = 5 * 1024 * 1024
MAX_TEXT = 100
RATE_LIMIT_CODES = frozenset({403, 429})
MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
DIRECTORY_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")
VERSION_PREFIX = re.compile(r"\d+(?:\.\d+)*")
FOUND = "found"
NOT_FOUND = "not_found"
FAILED = "failed"
SKIPPED = "skipped"

Fetch = Callable[[str, str | None], bytes]


class NotFound(Exception):
    pass


class LookupFailed(Exception):
    pass


class Miss(Exception):
    pass


@dataclass(frozen=True)
class Lookup:
    status: str
    detail: str
    baseline: dict[str, Any] | None = None


def checked_baseline_value(value: float) -> float:
    if not math.isfinite(value) or value <= 0 or value > 1:
        raise ValueError("a baseline value must be finite, above 0 and at most 1")
    return value


def plain_text(value: Any) -> str | None:
    if isinstance(value, str) and len(value) <= MAX_TEXT and value.isascii() and value.isprintable():
        return value
    return None


def http_fetch(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if accept is not None:
        headers["Accept"] = accept
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as response:
            body = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NotFound(url) from None
        hint = ", probably GitHub's rate limit" if exc.code in RATE_LIMIT_CODES else ""
        raise LookupFailed(f"HTTP {exc.code} from {urlsplit(url).hostname}{hint}") from None
    except OSError as exc:
        raise LookupFailed(f"could not reach {urlsplit(url).hostname} ({type(exc).__name__})") from None
    if len(body) > MAX_BYTES:
        raise LookupFailed(f"a response from {urlsplit(url).hostname} is larger than {MAX_BYTES} bytes")
    return body


def describe(baseline: Mapping[str, Any]) -> str:
    return f"{baseline['metric']} {baseline['value']} from {baseline['source']}"


def _json(body: bytes, label: str) -> Any:
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        raise LookupFailed(f"{label} is not JSON") from None


def _main_commit(fetch: Fetch) -> str:
    try:
        body = fetch(COMMIT_URL, SHA_MEDIA_TYPE)
    except NotFound:
        raise Miss(f"{RESULTS_REPOSITORY} has no main branch") from None
    text = body.decode("ascii", "replace").strip()
    if COMMIT_ID.fullmatch(text):
        return text
    payload = _json(body, "the main commit")
    sha = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(sha, str) or not COMMIT_ID.fullmatch(sha):
        raise LookupFailed("the main commit has no commit id")
    return sha


def _revision_directories(fetch: Fetch, model: str, folder: str, sha: str) -> list[str]:
    try:
        body = fetch(f"{CONTENTS_URL}/{folder}?ref={sha}", JSON_MEDIA_TYPE)
    except NotFound:
        raise Miss(f"{model} is not in {RESULTS_REPOSITORY}") from None
    entries = _json(body, "the model's directory listing")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise LookupFailed("the model's directory listing is not a list of entries")
    names = [entry.get("name") for entry in entries if entry.get("type") == "dir"]
    return sorted(
        name for name in names if isinstance(name, str) and DIRECTORY_NAME.fullmatch(name) and name not in {".", ".."}
    )


def _main_value(payload: Any, label: str, split: str, metric: str) -> float:
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if isinstance(scores, dict) and split not in scores:
        raise Miss(f"{label} has no {split} scores")
    rows = scores.get(split) if isinstance(scores, dict) else None
    if not isinstance(rows, list):
        raise LookupFailed(f"{label} has no list of {split} scores")
    defaults = [row for row in rows if isinstance(row, dict) and row.get("hf_subset") == "default"]
    if len(defaults) != 1:
        raise LookupFailed(f"{label} has {len(defaults)} {split} entries for the default subset, not one")
    if metric not in defaults[0]:
        raise Miss(f"{label} has no {metric}")
    value = defaults[0][metric]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LookupFailed(f"{label} has no number for {metric}")
    try:
        return checked_baseline_value(float(value))
    except (ValueError, OverflowError):
        raise LookupFailed(f"{label} has a {metric} that is not above 0 and at most 1") from None


def _version_key(version: str | None) -> tuple[int, ...]:
    match = VERSION_PREFIX.match(version or "")
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def _preferred(found: list[dict[str, Any]], served_revision: str | None) -> dict[str, Any]:
    for result in found:
        if result["model_revision"] == served_revision:
            return result
    return max(found, key=lambda result: _version_key(result["mteb_version"]))


def _find_result(
    model: str, task_name: str, split: str, metric: str, served_revision: str | None, fetch: Fetch
) -> dict:
    folder = model.replace("/", "__")
    file_name = f"{task_name.removeprefix('mteb/')}.json"
    sha = _main_commit(fetch)
    found: list[dict[str, Any]] = []
    problems: list[str] = []
    misses: list[str] = []
    for directory in _revision_directories(fetch, model, folder, sha):
        try:
            body = fetch(f"{RAW_URL}/{sha}/results/{folder}/{directory}/{file_name}", None)
        except NotFound:
            continue
        label = f"{directory}/{file_name}"
        try:
            payload = _json(body, label)
            value = _main_value(payload, label, split, metric)
        except LookupFailed as exc:
            problems.append(str(exc))
            continue
        except Miss as exc:
            misses.append(str(exc))
            continue
        found.append(
            {
                "kind": "official",
                "metric": metric,
                "value": value,
                "source": f"{BLOB_URL}/{sha}/results/{folder}/{directory}/{file_name}",
                "model_revision": directory,
                "dataset_revision": plain_text(payload.get("dataset_revision")),
                "mteb_version": plain_text(payload.get("mteb_version")),
            }
        )
    if found:
        return _preferred(found, served_revision)
    if problems:
        raise LookupFailed(problems[0])
    raise Miss(misses[0] if misses else f"{model} has no {file_name} in {RESULTS_REPOSITORY}")


def lookup_official(
    model: str,
    task_name: str,
    split: str,
    metric: str,
    served_revision: str | None = None,
    fetch: Fetch = http_fetch,
) -> Lookup:
    """Find the number; every failure is an answer, never an exception, because a lookup must not block a run."""
    if not MODEL_ID.fullmatch(model):
        return Lookup(NOT_FOUND, f"{model!r} is not an <org>/<name> model id")
    try:
        baseline = _find_result(model, task_name, split, metric, served_revision, fetch)
    except Miss as exc:
        return Lookup(NOT_FOUND, str(exc))
    except LookupFailed as exc:
        return Lookup(FAILED, str(exc))
    except Exception as exc:  # noqa: BLE001
        return Lookup(FAILED, f"the lookup failed ({type(exc).__name__})")
    return Lookup(FOUND, describe(baseline), baseline)
