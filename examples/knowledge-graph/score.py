#!/usr/bin/env python3
"""Count the recorded graph offline. No network, no API key, no install.

    python3 score.py

Reads the pinned paragraphs from data/, the recorded calls from calls.json,
and the hand review from data/review.json, then prints the five figures the
/knowledge-graph task page publishes.

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


def load_recorded() -> dict[tuple[str, str], Any]:
    """Recorded calls keyed by (candidate, kind), rejecting duplicates."""
    doc = graph.read_json(graph.CALLS_PATH)
    recorded: dict[tuple[str, str], Any] = {}
    expected_url = f"{doc['endpoint']}{doc['path']}"
    for entry in doc["calls"]:
        key = (entry["candidate"], entry["kind"])
        if key in recorded:
            raise InputError(f"Duplicate recorded call for {key[0]} {key[1]}")
        if entry["kind"] not in graph.KINDS:
            raise InputError(f"{entry['slug']}: unknown call kind {entry['kind']!r}")
        if entry["status"] != 200:
            raise InputError(f"{entry['slug']}: recorded HTTP {entry['status']}")
        if entry["request"]["url"] != expected_url:
            raise InputError(f"{entry['slug']}: recorded URL is {entry['request']['url']}, not {expected_url}")
        if graph.sha256_bytes(graph.compact_json(entry["response"])) != entry["response_sha256"]:
            raise InputError(f"{entry['slug']}: recorded response does not match its response_sha256")
        if set(entry["response"]["item"]) != set(graph.ITEM_KEYS):
            raise InputError(f"{entry['slug']}: recorded item does not carry the extract response fields")
        recorded[key] = entry
    return recorded


def resolve(candidates_doc: dict[str, Any], recorded: dict[tuple[str, str], Any]) -> dict[str, dict[str, Any]]:
    """Per candidate, the entities and relations the model returned.

    The pinned paragraph must rebuild both recorded request bodies, and the
    relations call must carry the entities call's own output as its metadata.
    Neither side of those comparisons is derived from the other.
    """
    model = candidates_doc["model"]
    path = graph.extract_path(model)
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
            if not entry["request"]["url"].endswith(path):
                raise InputError(f"{cid}: recorded {kind} URL does not address {model}")
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


def match_reviews(review_doc: dict[str, Any], resolved: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Confirm every flagged edge is an edge the model actually returned."""
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
        if len(hits) != 1:
            raise InputError(f"review flags {triple} on {cid}, which the model returned {len(hits)} times")
        if flagged["verdict"] not in review_doc["verdicts"]:
            raise InputError(f"{cid}: unknown verdict {flagged['verdict']!r}")
        matched.append({**flagged, "score": hits[0]["score"]})
    return matched


def score() -> dict[str, Any]:
    candidates_doc = graph.load_candidates()
    review_doc = graph.read_json(graph.REVIEW_PATH)
    resolved = resolve(candidates_doc, load_recorded())
    displayed = graph.shown(candidates_doc)
    hero = displayed[0]
    proof = displayed[1:]

    hero_edges = len(resolved[hero["id"]]["relations"])
    proof_edges = sum(len(resolved[c["id"]]["relations"]) for c in proof)
    reviews = match_reviews(review_doc, resolved)
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

    print(f"{'candidate':<24} {'role':<22} {'entities':>8} {'edges':>6} {'flagged':>8}")
    for row in summary["displayed"]:
        print(f"{row['id']:<24} {row['page_role']:<22} {row['entities']:>8} {row['edges']:>6} {row['flagged']:>8}")
    print()
    for row in summary["not_shown"]:
        print(f"{row['id']:<24} {'not shown':<22} {'':>8} {row['edges']:>6}   {row['reason']}")
    print()
    for review in summary["reviews"]:
        triple = f"{review['head']} -[{review['relation']}]-> {review['tail']}"
        print(f"{review['verdict']:<14} {triple}")
        print(f"{'':<14} {review['note']}")
    print()
    print(f"{summary['candidates_recorded']} paragraphs recorded, {summary['candidates_shown']} shown on the page")
    print(
        f"{summary['proof_edges']} edges across the {len(summary['displayed']) - 1} proof paragraphs "
        f"and {summary['hero_edges']} in the hero graph, {summary['edges_drawn']} drawn in total"
    )
    print(f"a hand review flagged {summary['flagged_edges']} of the {summary['proof_edges']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
