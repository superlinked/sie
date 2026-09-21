#!/usr/bin/env python3
"""Re-derive the published ranks from the recorded run, offline.

    python3 fetch.py
    python3 score.py

Standard library only. No API key, no network, no inference spend. Both ranks
come out of evidence/: the text one from the pages' markdown, the visual one
from the recorded ColPali multivectors.

    python3 score.py --baseline    # the BM25 side alone, from inputs/, no recording

This fails rather than skipping. A missing file, a page whose bytes no longer
match their digest, a response that does not match its response_sha256, a
request the pinned inputs do not rebuild, a page with no recorded multivector, a
multivector returned twice, a declared width that disagrees with its own values,
a non-finite score, or a relevant page that falls outside the published rank all
exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import retrieval


def load_calls() -> dict[str, dict[str, Any]]:
    path = retrieval.EVIDENCE / "calls.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    calls = json.loads(path.read_text(encoding="utf-8"))["calls"]
    by_slug: dict[str, dict[str, Any]] = {}
    for call in calls:
        if call["slug"] in by_slug:
            raise SystemExit(f"calls.json records {call['slug']!r} twice")
        by_slug[call["slug"]] = call
    return by_slug


def check_call(call: dict[str, Any], expected_body: dict[str, Any], manifest: dict[str, Any]) -> None:
    if call["status"] != 200:
        raise SystemExit(f"{call['slug']}: recorded status {call['status']}")
    if call["request"]["body"] != expected_body:
        raise SystemExit(
            f"{call['slug']}: the recorded request is not the one the pinned inputs rebuild. "
            "Either inputs/ changed or the recording is of something else."
        )
    expected_url = manifest["endpoint"].rstrip("/") + manifest["path"]
    if call["request"]["url"] != expected_url:
        raise SystemExit(f"{call['slug']}: recorded URL {call['request']['url']} is not {expected_url}")
    digest = retrieval.sha256_bytes(retrieval.compact_json(call["response"]))
    if digest != call["response_sha256"]:
        raise SystemExit(f"{call['slug']}: response digest {digest} is not the recorded {call['response_sha256']}")


def response_items(call: dict[str, Any], expected_ids: list[str]) -> list[dict[str, Any]]:
    """The response items for one call, bound to that call's own request.

    A response digest proves a response has not been edited. It does not tie
    the response to the request beside it, so without this a page batch could
    carry another batch's answers and still satisfy a global coverage check.
    Comparing the id sequence in order closes that.
    """
    items = call["response"]["items"]
    got = [item.get("id") for item in items]
    if got != expected_ids:
        raise SystemExit(
            f"{call['slug']}: the response does not answer its own request.\n"
            f"  requested: {', '.join(expected_ids)}\n"
            f"  returned:  {', '.join(str(value) for value in got)}"
        )
    return items


def multivector(item: dict[str, Any]) -> list[list[float]]:
    rows = retrieval.decode_multivector(item["multivector"]["float16_base64"])
    if len(rows) != item["multivector"]["tokens"]:
        raise SystemExit(f"{item['id']}: declares {item['multivector']['tokens']} tokens and carries {len(rows)}")
    if rows and len(rows[0]) != item["multivector"]["dim"]:
        raise SystemExit(f"{item['id']}: declares width {item['multivector']['dim']} and carries {len(rows[0])}")
    return rows


def score_comparison(
    comparison: dict[str, Any],
    pages: list[dict[str, Any]],
    calls: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    consumed: set[str],
) -> dict[str, Any]:
    for page in pages:
        retrieval.check_page_bytes(page)

    query_slug = f"{comparison['id']}/query"
    if query_slug not in calls:
        raise SystemExit(f"calls.json has no query call for {comparison['id']}")
    check_call(calls[query_slug], retrieval.query_body(comparison), manifest)
    consumed.add(query_slug)
    # Exactly one item, carrying the id the request asked for.
    query_vectors = multivector(response_items(calls[query_slug], [comparison["id"]])[0])

    page_vectors: dict[int, list[list[float]]] = {}
    batch = 0
    covered = 0
    while covered < len(pages):
        slug = f"{comparison['id']}/pages-{batch:03d}"
        if slug not in calls:
            raise SystemExit(f"calls.json stops at {covered} of {len(pages)} pages for {comparison['id']}")
        chunk = pages[covered : covered + len(calls[slug]["request"]["body"]["items"])]
        check_call(calls[slug], retrieval.page_body(chunk), manifest)
        consumed.add(slug)
        # The ids this batch asked for, in order, taken from the request body
        # check_call has just proven equal to the one the pinned inputs rebuild.
        requested = [item["id"] for item in calls[slug]["request"]["body"]["items"]]
        for item in response_items(calls[slug], requested):
            corpus_id = int(item["id"])
            if corpus_id in page_vectors:
                raise SystemExit(f"{comparison['id']}: page {corpus_id} has two recorded multivectors")
            page_vectors[corpus_id] = multivector(item)
        covered += len(chunk)
        batch += 1

    expected = {page["corpus_id"] for page in pages}
    if set(page_vectors) != expected:
        missing = sorted(expected - set(page_vectors))
        extra = sorted(set(page_vectors) - expected)
        raise SystemExit(f"{comparison['id']}: missing {missing[:5]}, unexpected {extra[:5]}")

    visual = retrieval.visual_rank(query_vectors, page_vectors)
    text = retrieval.bm25_rank(comparison["query"], pages)
    relevant = comparison["relevant_corpus_id"]
    return {
        "id": comparison["id"],
        "label": comparison["label"],
        "pages": len(pages),
        "text_rank": [cid for cid, _ in text].index(relevant) + 1,
        "visual_rank": [cid for cid, _ in visual].index(relevant) + 1,
        "visual_score": dict(visual)[relevant],
    }


def recorded_results(manifest: dict[str, Any], comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The manifest's own result rows, one per comparison, keyed by id.

    Rejects duplicates, extras and gaps rather than letting a lookup silently
    pick a row. A slug-keyed list collapsed to a dict keeps the LAST match, so
    two rows sharing an id would let this check and the ranking check read
    different rows.
    """
    results = manifest.get("results")
    if not isinstance(results, list):
        raise SystemExit("manifest.json has no results list, so there is nothing to check the ranks against")
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        identifier = row.get("id")
        if identifier in by_id:
            raise SystemExit(f"manifest.json records {identifier!r} twice")
        by_id[identifier] = row
    wanted = {comparison["id"] for comparison in comparisons}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SystemExit(f"manifest.json has no recorded result for {', '.join(missing)}")
    extra = sorted(set(by_id) - wanted)
    if extra:
        raise SystemExit(f"manifest.json records a result nothing scores: {', '.join(extra)}")
    return by_id


def check_against_recorded(row: dict[str, Any], expected: dict[str, Any]) -> None:
    """The ranks derived from calls.json must be the ranks the run recorded.

    The two sides are different artifacts: `row` is recomputed here from the
    multivectors in calls.json and the markdown in inputs/pages.json, while
    `expected` was written into manifest.json by run.py at run time. That makes
    this a check on the recording rather than a restatement of it. It is the
    only check covering the BM25 side, whose markdown no response digest spans.
    """
    for key in ("text_rank", "visual_rank", "pages"):
        if row[key] != expected.get(key if key != "pages" else "candidate_pages"):
            recorded = expected.get(key if key != "pages" else "candidate_pages")
            raise SystemExit(f"{row['id']}: derived {key} {row[key]} does not match the recorded {recorded}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the published ranks from the recorded run")
    parser.add_argument("--baseline", action="store_true", help="BM25 only, straight from inputs/, no recording read")
    args = parser.parse_args()

    comparisons = retrieval.load_comparisons()
    by_document = retrieval.load_pages()

    if args.baseline:
        print(f"BM25 over the ViDoRe markdown, k1={retrieval.K1}, b={retrieval.B}\n")
        print(f"{'comparison':<36}{'pages':>6}{'text rank':>11}")
        for comparison in comparisons:
            pages = by_document[comparison["doc_id"]]
            text = retrieval.bm25_rank(comparison["query"], pages)
            rank = [cid for cid, _ in text].index(comparison["relevant_corpus_id"]) + 1
            print(f"{comparison['label']:<36}{len(pages):>6}{rank:>11}")
        return 0

    manifest_path = retrieval.EVIDENCE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing. Run: python3 fetch.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != retrieval.MODEL:
        raise SystemExit(f"manifest names model {manifest.get('model')!r}, this example scores {retrieval.MODEL!r}")
    if manifest.get("model_revision") != retrieval.MODEL_REVISION:
        raise SystemExit(
            f"manifest names model revision {manifest.get('model_revision')!r}, "
            f"this example scores {retrieval.MODEL_REVISION!r}. Different weights produce different ranks."
        )
    calls = load_calls()
    recorded = recorded_results(manifest, comparisons)

    print(f"{manifest['model']} on {manifest['endpoint']}, recorded {manifest['run_date']}")
    print(f"BM25 baseline, k1={retrieval.K1}, b={retrieval.B}\n")
    print(f"{'comparison':<36}{'pages':>6}{'text':>6}{'visual':>8}{'maxsim':>10}")

    rows = []
    consumed: set[str] = set()
    for comparison in comparisons:
        row = score_comparison(comparison, by_document[comparison["doc_id"]], calls, manifest, consumed)
        check_against_recorded(row, recorded[row["id"]])
        rows.append(row)
        print(
            f"{row['label']:<36}{row['pages']:>6}{row['text_rank']:>6}{row['visual_rank']:>8}{row['visual_score']:>10.3f}"
        )

    # A call nothing scores is either evidence for a claim this example does not
    # make, or a leftover. Either way it does not travel silently.
    unconsumed = sorted(set(calls) - consumed)
    if unconsumed:
        raise SystemExit(f"calls.json holds {len(unconsumed)} call(s) nothing scores: {', '.join(unconsumed[:5])}")

    scored = sum(row["pages"] for row in rows)
    leading = sum(1 for row in rows if row["visual_rank"] == 1)
    print()
    print(f"{len(rows)} comparisons over {scored} pages")
    print(f"the visual side puts the benchmark page first in {leading} of {len(rows)}")
    print(f"the text side never does, at best rank {min(row['text_rank'] for row in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
