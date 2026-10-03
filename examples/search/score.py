#!/usr/bin/env python3
"""Rank the recorded run offline. No network, no API key, no install.

    python3 score.py

Reads everything from evidence/, which `python3 fetch.py` downloads: the pinned
corpus, the pinned questions and the recorded encode responses. Scores every
question against every passage and reports where each answer lands.

Every check is fail-closed. A passage whose text no longer matches its digest,
a response that does not match its digest, a request the pinned inputs do not
rebuild, a passage with no recorded vector, or a duplicate vector all exit
non-zero. Nothing is ranked around a missing input.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import retrieval
from retrieval import InputError

# The published claim, and the reason this exits non-zero if it stops holding.
MAX_ANSWER_RANK = 3


def expected_url(manifest: dict[str, Any]) -> str:
    """The URL every recorded call must carry.

    The model and the path come from retrieval.py, not from the manifest, so a
    recording cannot tell the scorer which model it was allowed to use. The host
    does come from the manifest, because a run against a regional endpoint or a
    self-hosted cluster is legitimate.
    """
    if manifest.get("model") != retrieval.MODEL:
        raise InputError(f"manifest records model {manifest.get('model')!r}; this scorer scores {retrieval.MODEL!r}")
    if manifest.get("path") != retrieval.ENCODE_PATH:
        raise InputError(
            f"manifest records path {manifest.get('path')!r}; the pinned model needs {retrieval.ENCODE_PATH!r}"
        )
    return manifest["endpoint"].rstrip("/") + retrieval.ENCODE_PATH


def vectors_from(entry: dict[str, Any], into: dict[str, list[float]], dims: set[int]) -> None:
    """Take the vectors one response returned, refusing anything unusable.

    A vector whose declared `dims` disagrees with its own row length, or that
    carries a value which is not a finite number, would still rank; it would
    just rank something meaningless. `dims` accumulates across the run, so a
    batch encoded at a different width is caught rather than averaged in.
    """
    sent = [item["id"] for item in entry["request"]["body"]["items"]]
    returned = [item["id"] for item in entry["response"]["items"]]
    if returned != sent:
        # Batches merge into one map, so a response carrying another batch's ids
        # would quietly move vectors between passages.
        raise InputError(f"{entry['slug']}: the response echoes {len(returned)} ids that are not the {len(sent)} sent")
    for item in entry["response"]["items"]:
        if item["id"] in into:
            raise InputError(f"{entry['slug']}: {item['id']} was already encoded by another call")
        block = item["dense"]
        values = block["values"]
        if block.get("dims") != len(values):
            raise InputError(
                f"{entry['slug']}: {item['id']} declares {block.get('dims')} dims for {len(values)} values"
            )
        if not values:
            raise InputError(f"{entry['slug']}: {item['id']} carries an empty vector")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InputError(f"{entry['slug']}: {item['id']} carries a value that is not a finite number")
        dims.add(len(values))
        into[item["id"]] = values


def score() -> dict[str, Any]:
    passages = retrieval.load_corpus()
    queries = retrieval.load_queries(passages)
    manifest = retrieval.read_json(retrieval.MANIFEST_PATH)
    url = expected_url(manifest)

    allowed = retrieval.allowed_revisions(manifest)
    observed_revisions: set[str] = set()
    recorded: dict[str, Any] = {}
    for entry in retrieval.read_json(retrieval.CALLS_PATH)["calls"]:
        if entry["slug"] in recorded:
            raise InputError(f"Duplicate recorded call {entry['slug']}")
        if entry["status"] != 200:
            raise InputError(f"{entry['slug']}: recorded HTTP {entry['status']}")
        if entry["request"]["url"] != url:
            raise InputError(f"{entry['slug']}: recorded URL is {entry['request']['url']}, not {url}")
        if retrieval.sha256_bytes(retrieval.compact_json(entry["response"])) != entry["response_sha256"]:
            raise InputError(f"{entry['slug']}: recorded response does not match its response_sha256")
        observed_revisions.add(retrieval.check_revision(entry["slug"], entry, allowed))
        recorded[entry["slug"]] = entry
    if observed_revisions != allowed:
        unused = sorted(allowed - observed_revisions)
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")

    corpus_calls = retrieval.corpus_bodies(passages)
    retrieval.check_call_set(set(recorded), {name for name, _ in corpus_calls} | {"queries"}, manifest, "call")

    # The pinned inputs must rebuild every request that was sent. One side is
    # inputs/, the other is calls.json, and neither is derived from the other.
    dims: set[int] = set()
    passage_vectors: dict[str, list[float]] = {}
    for name, body in corpus_calls:
        entry = recorded.get(name)
        if entry is None:
            raise InputError(f"{name}: no recorded call in calls.json")
        if entry["request"]["body"] != body:
            raise InputError(f"{name}: pinned corpus does not rebuild the recorded request body")
        vectors_from(entry, passage_vectors, dims)

    query_entry = recorded.get("queries")
    if query_entry is None:
        raise InputError("queries: no recorded call in calls.json")
    if query_entry["request"]["body"] != retrieval.query_body(queries):
        raise InputError("queries: pinned questions do not rebuild the recorded request body")
    query_vectors: dict[str, list[float]] = {}
    vectors_from(query_entry, query_vectors, dims)
    if len(dims) != 1:
        raise InputError(f"the recorded vectors are not all the same width: {sorted(dims)}")

    for passage in passages:
        if passage["id"] not in passage_vectors:
            raise InputError(f"{passage['id']}: no recorded vector")
    for query in queries:
        if query["id"] not in query_vectors:
            raise InputError(f"{query['id']}: no recorded vector")

    results = retrieval.rank_answers(queries, passages, query_vectors, passage_vectors)
    ranks = [result["rank"] for result in results]
    return {
        "dims": next(iter(dims)),
        "passages": len(passages),
        "queries": len(results),
        "worst_rank": max(ranks),
        "rank_one": sum(1 for rank in ranks if rank == 1),
        "results": results,
    }


def main() -> int:
    try:
        summary = score()
    except InputError as error:
        print(f"FAILED: {error}")
        return 1

    for result in summary["results"]:
        top = result["top"][0]
        print(f"{result['id']:<22} answer at rank {result['rank']}   top hit {top['id']} ({top['score']:.4f})")

    print()
    print(f"{summary['passages']} passages, {summary['queries']} questions, {summary['dims']}-dimensional vectors")
    print(f"the answer is in the top {summary['worst_rank']} for every question")
    print(f"it ranks first for {summary['rank_one']} of the {summary['queries']}")

    if summary["worst_rank"] > MAX_ANSWER_RANK:
        print(f"FAILED: an answer fell outside the top {MAX_ANSWER_RANK}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
