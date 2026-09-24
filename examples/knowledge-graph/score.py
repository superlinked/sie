#!/usr/bin/env python3
"""Count the recorded graph offline. No network, no API key, no install.

    python3 score.py

Reads everything from evidence/, which `python3 fetch.py` downloads: the
pinned paragraphs, the recorded calls and the hand review. Prints the five
figures the /knowledge-graph task page publishes.

Every check is fail-closed. A paragraph whose text no longer matches its
digest, a candidate with a missing call, a response that does not match its
digest, a request the pinned input does not rebuild, a relations call whose
metadata is not the entities call's own output, or a flagged edge the model
never returned all exit non-zero. Nothing is counted around a missing input.
"""

from __future__ import annotations

import sys
from typing import Any

import graph
from graph import InputError


def expected_url(manifest: dict[str, Any], model: str) -> str:
    """The URL every recorded call must carry.

    The path is derived here from the pinned model, not read from the manifest,
    so a recording cannot tell the scorer which endpoint it was allowed to use.
    The host does come from the manifest, because a run against a regional
    endpoint or a self-hosted cluster is legitimate.
    """
    path = graph.extract_path(model)
    if manifest.get("path") != path:
        raise InputError(f"manifest records path {manifest.get('path')!r}; the pinned model needs {path!r}")
    if manifest.get("model") != model:
        raise InputError(f"manifest model is {manifest.get('model')!r}, the pinned model is {model!r}")
    return manifest["endpoint"].rstrip("/") + path


def load_recorded(manifest: dict[str, Any], model: str) -> dict[tuple[str, str], Any]:
    """Recorded calls keyed by (candidate, kind), rejecting duplicates."""
    doc = graph.read_json(graph.CALLS_PATH)
    recorded: dict[tuple[str, str], Any] = {}
    url = expected_url(manifest, model)
    allowed = graph.allowed_revisions(manifest)
    observed_revisions: set[str] = set()
    for entry in doc["calls"]:
        key = (entry["candidate"], entry["kind"])
        if key in recorded:
            raise InputError(f"Duplicate recorded call for {key[0]} {key[1]}")
        if entry["kind"] not in graph.KINDS:
            raise InputError(f"{entry['slug']}: unknown call kind {entry['kind']!r}")
        if entry["status"] != 200:
            raise InputError(f"{entry['slug']}: recorded HTTP {entry['status']}")
        if entry["request"]["url"] != url:
            raise InputError(f"{entry['slug']}: recorded URL is {entry['request']['url']}, not {url}")
        if graph.sha256_bytes(graph.compact_json(entry["response"])) != entry["response_sha256"]:
            raise InputError(f"{entry['slug']}: recorded response does not match its response_sha256")
        if set(entry["response"]["item"]) != set(graph.ITEM_KEYS):
            raise InputError(f"{entry['slug']}: recorded item does not carry the extract response fields")
        observed_revisions.add(graph.check_revision(entry["slug"], entry, allowed))
        recorded[key] = entry
    if observed_revisions != allowed:
        unused = sorted(allowed - observed_revisions)
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")
    return recorded


def resolve(candidates_doc: dict[str, Any], recorded: dict[tuple[str, str], Any]) -> dict[str, dict[str, Any]]:
    """Per candidate, the entities and relations the model returned.

    The pinned paragraph must rebuild both recorded request bodies, and the
    relations call must carry the entities call's own output as its metadata.
    Neither side of those comparisons is derived from the other.
    """
    model = candidates_doc["model"]
    resolved: dict[str, dict[str, Any]] = {}
    for candidate in candidates_doc["candidates"]:
        cid = candidate["id"]
        calls = {}
        for kind in graph.KINDS:
            entry = recorded.get((cid, kind))
            if entry is None:
                raise InputError(f"{cid}: no recorded {kind} call in calls.json")
            if entry["request"]["model"] != model:
                raise InputError(f"{cid}: recorded {kind} call used {entry['request']['model']}, not {model}")
            calls[kind] = entry

        entities = calls["entities"]["response"]["item"]["entities"]
        if graph.entities_body(candidate) != calls["entities"]["request"]["body"]:
            raise InputError(f"{cid}: pinned text does not rebuild the recorded entities request")
        if graph.relations_body(candidate, entities) != calls["relations"]["request"]["body"]:
            raise InputError(f"{cid}: the relations request is not this paragraph plus its own returned entities")

        relations = calls["relations"]["response"]["item"]["relations"]
        for relation in relations:
            if relation["relation"] not in candidate["relation_labels"]:
                raise InputError(f"{cid}: returned relation {relation['relation']!r} is not one of the sent labels")
        for entity in entities:
            if candidate["text"][entity["start"] : entity["end"]] != entity["text"]:
                raise InputError(f"{cid}: entity offsets do not land on the span the model returned")
        resolved[cid] = {"candidate": candidate, "entities": entities, "relations": relations}
    return resolved


# Every edge a person read against its source paragraph, as the run first
# recorded them (dataset revision fc13484f). `inputs/review.json` records the
# readings that found something; this is the coverage the first sentence of
# that file claims, written out so the scorer can hold each displayed edge to
# it. Sixteen edges: eleven across the proof paragraphs and five in the hero.
#
# The page's claim is that a person read every edge it shows. Checking only the
# flagged triples left that claim unenforced, so a displayed edge nobody had
# read would have scored clean. Narrowing a schema removes edges from a run and
# never adds one, which is why this set is a superset of what any later run can
# display, and why an edge outside it means a person has not read it.
REVIEWED_EDGES = frozenset(
    {
        ("flex-credit-facility", "Citibank, N.A.", "administrative agent of", "Flex Ltd."),
        ("flex-credit-facility", "Flex Ltd.", "borrower under", "Credit Agreement"),
        ("flex-credit-facility", "credit facility", "commitment amount", "$1.45 billion"),
        ("ford-jdi-display", "Ford Escape", "equipped with", "8” display"),
        ("ford-jdi-display", "Lincoln Corsair", "equipped with", "8” display"),
        ("fresenius-morphine", "Fresenius Kabi", "headquartered in", "LAKE ZURICH"),
        ("fresenius-morphine", "Fresenius Kabi", "operating company of", "Fresenius Group"),
        ("fresenius-morphine", "Fresenius Kabi", "recalls", "Simplist® 2 mg/1 mL"),
        ("tarsus-alkeus", "Alkeus Pharmaceuticals, Inc.", "incorporated in", "Delaware"),
        ("tarsus-alkeus", "Apex 2026 Merger Sub, Inc.", "subsidiary of", "Tarsus Pharmaceuticals, Inc."),
        ("tarsus-alkeus", "Tarsus Pharmaceuticals, Inc.", "acquired", "Alkeus Pharmaceuticals, Inc."),
        ("veracyte-convergent", "Convergent", "develops", "UroAmp"),
        ("veracyte-convergent", "Convergent", "develops", "urine tumor DNA technology"),
        ("veracyte-convergent", "Convergent", "focused on", "bladder cancer"),
        ("veracyte-convergent", "Convergent", "subsidiary of", "Veracyte"),
        ("veracyte-convergent", "Veracyte", "acquired", "Convergent"),
    }
)


def match_reviews(review_doc: dict[str, Any], resolved: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Confirm every flagged edge is one this run supports, or one it no longer asks for.

    The review is a superset of the current run. A schema that stops asking for
    a relation stops producing the edges built on it, and a reading taken
    before that change still stands as a record of what a person read. What is
    not allowed is a flagged edge missing while its relation is still being
    sent: that would be a review of something this run does not support, which
    is the case this check exists to catch.
    """
    matched = []
    for flagged in review_doc["flagged"]:
        cid = flagged["candidate"]
        if cid not in resolved:
            raise InputError(f"review flags {cid}, which is not a pinned candidate")
        role = resolved[cid]["candidate"]["page_role"]
        if not role.startswith("proof"):
            raise InputError(f"review flags {cid}, whose page_role is {role!r}, not a proof paragraph")
        triple = (flagged["head"], flagged["relation"], flagged["tail"])
        hits = [
            relation
            for relation in resolved[cid]["relations"]
            if (relation["head"], relation["relation"], relation["tail"]) == triple
        ]
        if not hits and flagged["relation"] not in resolved[cid]["candidate"]["relation_labels"]:
            continue
        if len(hits) != 1:
            raise InputError(f"review flags {triple} on {cid}, which the model returned {len(hits)} times")
        if flagged["verdict"] not in review_doc["verdicts"]:
            raise InputError(f"{cid}: unknown verdict {flagged['verdict']!r}")
        matched.append({**flagged, "score": hits[0]["score"]})
    return matched


def score() -> dict[str, Any]:
    candidates_doc = graph.load_candidates()
    review_doc = graph.read_json(graph.REVIEW_PATH)
    manifest = graph.read_json(graph.MANIFEST_PATH)
    recorded = load_recorded(manifest, candidates_doc["model"])
    expected = {(candidate["id"], kind) for candidate in candidates_doc["candidates"] for kind in graph.KINDS}
    graph.check_call_set(set(recorded), expected, manifest, "call")
    resolved = resolve(candidates_doc, recorded)
    displayed = graph.shown(candidates_doc)
    hero = displayed[0]
    proof = displayed[1:]

    hero_edges = len(resolved[hero["id"]]["relations"])
    proof_edges = sum(len(resolved[c["id"]]["relations"]) for c in proof)
    reviews = match_reviews(review_doc, resolved)

    unreviewed = [
        (c["id"], edge["head"], edge["relation"], edge["tail"])
        for c in displayed
        for edge in resolved[c["id"]]["relations"]
        if (c["id"], edge["head"], edge["relation"], edge["tail"]) not in REVIEWED_EDGES
    ]
    if unreviewed:
        listing = "; ".join(f"{cid}: {h} -[{r}]-> {t}" for cid, h, r, t in unreviewed)
        raise InputError(f"{len(unreviewed)} displayed edge(s) carry no recorded reading: {listing}")

    return {
        "candidates_recorded": len(candidates_doc["candidates"]),
        "candidates_shown": len(displayed),
        "hero_edges": hero_edges,
        "proof_edges": proof_edges,
        "edges_drawn": hero_edges + proof_edges,
        "flagged_edges": len(reviews),
        "displayed": [
            {
                "id": c["id"],
                "page_role": c["page_role"],
                "entities": len(resolved[c["id"]]["entities"]),
                "edges": len(resolved[c["id"]]["relations"]),
                "flagged": sum(1 for review in reviews if review["candidate"] == c["id"]),
            }
            for c in displayed
        ],
        "not_shown": [
            {"id": c["id"], "edges": len(resolved[c["id"]]["relations"]), "reason": c["selection_note"]}
            for c in candidates_doc["candidates"]
            if c["page_role"] == "not shown"
        ],
        "reviews": reviews,
    }


def main() -> int:
    try:
        summary = score()
    except InputError as error:
        print(f"FAILED: {error}")
        return 1

    print(f"{'candidate':<24} {'role':<22} {'entities':>8} {'edges':>6}")
    for row in summary["displayed"]:
        print(f"{row['id']:<24} {row['page_role']:<22} {row['entities']:>8} {row['edges']:>6}")
    print()
    for row in summary["not_shown"]:
        print(f"{row['id']:<24} {'not shown':<22} {'':>8} {row['edges']:>6}   {row['reason']}")
    print()
    print(f"{summary['candidates_recorded']} paragraphs recorded, {summary['candidates_shown']} shown on the page")
    print(
        f"{summary['proof_edges']} edges across the {len(summary['displayed']) - 1} proof paragraphs "
        f"and {summary['hero_edges']} in the hero graph, {summary['edges_drawn']} drawn in total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
