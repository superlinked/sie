#!/usr/bin/env python3
"""Score the recorded searches offline. No network, no API key, no install.

    python3 score.py

Reads everything from evidence/, which `python3 fetch.py` downloads: the pinned
searches and the recorded encode responses. Computes MaxSim between each
question and each of its four candidate passages, then reports whether the
passage that answers the question beat the near-duplicates from the same page.

Every check is fail-closed. A response that does not match its digest, a
request the pinned inputs do not rebuild, a missing call, a passage with no
recorded vector, or a vector whose token count does not match the response's
own `num_tokens` all exit non-zero. Nothing is scored around a missing input.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import maxsim
from maxsim import InputError


def expected_url(manifest: dict[str, Any]) -> str:
    """The URL every recorded call must carry.

    The model and the path come from maxsim.py, not from the manifest, so a
    recording cannot tell the scorer which model it was allowed to use. The host
    does come from the manifest, because a run against a regional endpoint or a
    self-hosted cluster is legitimate.
    """
    if manifest.get("model") != maxsim.MODEL:
        raise InputError(f"manifest records model {manifest.get('model')!r}; this scorer scores {maxsim.MODEL!r}")
    if manifest.get("path") != maxsim.ENCODE_PATH:
        raise InputError(
            f"manifest records path {manifest.get('path')!r}; the pinned model needs {maxsim.ENCODE_PATH!r}"
        )
    return manifest["endpoint"].rstrip("/") + maxsim.ENCODE_PATH


def token_vectors(entry: dict[str, Any], dims: set[int]) -> dict[str, list[list[float]]]:
    """The token vectors one encode response returned, keyed by item id.

    MaxSim zips a query token against a passage token, and `zip` stops at the
    shorter one, so a short row would silently score on a prefix. Every row is
    therefore checked to be the declared width, `dims` accumulates across the
    run so a query and its passages cannot be compared at different widths, and
    the ids a response echoes have to be exactly the ids its request sent.
    """
    sent = [item["id"] for item in entry["request"]["body"]["items"]]
    returned = [item["id"] for item in entry["response"]["items"]]
    if returned != sent:
        raise InputError(f"{entry['slug']}: the response echoes {returned}, not the {sent} its request sent")
    out: dict[str, list[list[float]]] = {}
    for item in entry["response"]["items"]:
        block = item["multivector"]
        values = block["values"]
        if len(values) != block["num_tokens"]:
            raise InputError(f"{entry['slug']}: {item['id']} has {len(values)} rows for {block['num_tokens']} tokens")
        if not values:
            raise InputError(f"{entry['slug']}: {item['id']} carries no token vectors")
        width = block.get("token_dims")
        if not isinstance(width, int) or width <= 0:
            raise InputError(f"{entry['slug']}: {item['id']} declares token_dims {width!r}")
        for row in values:
            if len(row) != width:
                raise InputError(f"{entry['slug']}: {item['id']} has a token vector of {len(row)}, not {width}")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise InputError(f"{entry['slug']}: {item['id']} carries a value that is not a finite number")
        if item["id"] in out:
            raise InputError(f"{entry['slug']}: {item['id']} appears twice in one response")
        dims.add(width)
        out[item["id"]] = values
    return out


def score() -> dict[str, Any]:
    cases_doc = maxsim.load_cases()
    manifest = maxsim.read_json(maxsim.MANIFEST_PATH)
    url = expected_url(manifest)

    allowed = maxsim.allowed_revisions(manifest)
    observed_revisions: set[str] = set()
    recorded: dict[tuple[str, str], Any] = {}
    for entry in maxsim.read_json(maxsim.CALLS_PATH)["calls"]:
        key = (entry["case"], entry["kind"])
        if key in recorded:
            raise InputError(f"Duplicate recorded call for {key[0]} {key[1]}")
        if entry["kind"] not in maxsim.KINDS:
            raise InputError(f"{entry['slug']}: unknown call kind {entry['kind']!r}")
        if entry["status"] != 200:
            raise InputError(f"{entry['slug']}: recorded HTTP {entry['status']}")
        if entry["request"]["url"] != url:
            raise InputError(f"{entry['slug']}: recorded URL is {entry['request']['url']}, not {url}")
        if maxsim.sha256_bytes(maxsim.compact_json(entry["response"])) != entry["response_sha256"]:
            raise InputError(f"{entry['slug']}: recorded response does not match its response_sha256")
        observed_revisions.add(maxsim.check_revision(entry["slug"], entry, allowed))
        recorded[key] = entry
    if observed_revisions != allowed:
        unused = sorted(allowed - observed_revisions)
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")

    expected = {(case["slug"], kind) for case in cases_doc["cases"] for kind in maxsim.KINDS}
    maxsim.check_call_set(set(recorded), expected, manifest, "call")

    dims: set[int] = set()
    results = []
    for case in cases_doc["cases"]:
        slug = case["slug"]
        calls = {}
        for kind in maxsim.KINDS:
            entry = recorded.get((slug, kind))
            if entry is None:
                raise InputError(f"{slug}: no recorded {kind} call in calls.json")
            calls[kind] = entry

        # The pinned question and passages must rebuild both requests. One side
        # is inputs/, the other is calls.json, and neither comes from the other.
        if calls["query"]["request"]["body"] != maxsim.query_body(case):
            raise InputError(f"{slug}: pinned question does not rebuild the recorded query request")
        if calls["passages"]["request"]["body"] != maxsim.passages_body(case):
            raise InputError(f"{slug}: pinned passages do not rebuild the recorded passages request")

        query_vectors = token_vectors(calls["query"], dims)
        if list(query_vectors) != [f"{slug}-query"]:
            raise InputError(f"{slug}: the query call returned {list(query_vectors)}")
        passage_vectors = token_vectors(calls["passages"], dims)
        for passage in case["passages"]:
            if passage["id"] not in passage_vectors:
                raise InputError(f"{slug}: no recorded vector for {passage['id']}")

        results.append(maxsim.rank_passages(case, query_vectors[f"{slug}-query"], passage_vectors))

    if len(dims) != 1:
        raise InputError(f"the recorded token vectors are not all the same width: {sorted(dims)}")

    wins = sum(1 for result in results if result["rank"] == 1)
    return {"searches": len(results), "wins": wins, "token_dims": next(iter(dims)), "results": results}


def main() -> int:
    try:
        summary = score()
    except InputError as error:
        print(f"FAILED: {error}")
        return 1

    for result in summary["results"]:
        verdict = "answer first" if result["rank"] == 1 else f"answer at {result['rank']}"
        print(
            f"{result['slug']:<32} {verdict:<16}"
            f" {result['answer_maxsim']:>7.3f} against {result['closest_other_maxsim']:>7.3f}"
            f"  ({result['closest_other_id']})"
        )

    print()
    print(f"{summary['searches']} searches, {summary['token_dims']} dimensions per token")
    print(
        f"the answer passage beat the closest same-page passage in {summary['wins']} of {summary['searches']} searches"
    )
    lost = [result["slug"] for result in summary["results"] if result["rank"] != 1]
    if lost:
        print(f"it lost in: {', '.join(lost)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
