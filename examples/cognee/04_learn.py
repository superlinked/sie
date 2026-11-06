"""Distill a correction, then verify stored lessons in a separate process."""

import argparse
import json
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from sie_cognee.retrieval import SourceChunk, retrieve_chunks

from common import ROOT, Demo, initialize, json_value, launch, seed_memory

MISSING_STATE_RECOVERY = (
    "Stored embedding identity is missing. Restore the teaching run's original memory directory, including "
    "embedding.json and cognee/sie-cognee.json, or run 04_learn.py --seed with a fresh WORKSHOP_NAMESPACE "
    "and verify that new teaching run."
)


class MissingMemoryStateError(ValueError):
    """The local identity markers needed to verify stored memory are absent."""


def distillation_summary(result: Any) -> dict[str, Any]:
    payload = json_value(result)
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise ValueError("Unexpected Cognee distillation response.")
    documents = payload.get("documents")
    if not isinstance(documents, list) or any(
        not isinstance(document, str) or not document.strip() for document in documents
    ):
        raise ValueError("Distillation documents must be actual lesson texts.")
    return {
        **payload,
        "accepted_document_count": len(documents),
        "persisted_lessons": payload["status"] == "completed" and bool(documents),
    }


def exact_lesson_matches(documents: list[str], chunks: list[SourceChunk]) -> list[dict[str, Any]]:
    """Match complete accepted texts, ignoring whitespace; headings alone prove nothing."""
    return [
        {"accepted_document_index": index, "chunk_id": chunk.chunk_id}
        for index, document in enumerate(documents)
        for chunk in chunks
        if " ".join(document.split()) == " ".join(chunk.text.split())
    ]


def verification_summary(demo: Demo, prior_run: Path) -> dict[str, Any]:
    """Check the explicit teaching run and stored vector identity before Cognee imports."""
    configuration = json.loads((prior_run / "configuration.json").read_text())
    summary = distillation_summary(json.loads((prior_run / "actual-distillation.json").read_text()))
    if not isinstance(configuration, dict) or configuration.get("dataset") != demo.dataset:
        raise ValueError("The teaching run uses a different dataset; use its original WORKSHOP_NAMESPACE.")
    models = configuration.get("models")
    if not isinstance(models, dict):
        raise ValueError("The teaching run is missing its recorded model configuration.")
    expected = {key: demo.models[key] for key in ("embedding", "embedding_dimensions")}
    if {key: models.get(key) for key in expected} != expected:
        raise ValueError("Use the teaching run's embedding model and dimensions when verifying its memory.")
    embedding_manifest = demo.storage / "embedding.json"
    cognee_marker = demo.storage / "cognee/sie-cognee.json"
    if not embedding_manifest.is_file() or not cognee_marker.is_file():
        raise MissingMemoryStateError(MISSING_STATE_RECOVERY)
    stored = json.loads(embedding_manifest.read_text())
    owned = json.loads(cognee_marker.read_text())
    if stored != expected or owned != {
        "embedding_model": expected["embedding"],
        "embedding_dimensions": expected["embedding_dimensions"],
    }:
        raise ValueError("Stored embedding identity differs from the teaching run; verify its original memory.")
    return summary


async def verify(demo: Demo, prior_run: Path) -> None:
    summary = verification_summary(demo, prior_run)
    cognee = initialize(demo, learning=False)
    from cognee.api.v1.search import SearchType

    fresh_session = f"{demo.namespace}-verify-{uuid4().hex[:10]}"
    demo.show("verification-input", {"teaching_run": str(prior_run.resolve()), "recorded_distillation": summary})
    demo.show("session-ids", {"fresh": fresh_session})
    new_query = demo.learning["query"]
    fresh_response = await cognee.recall(
        new_query,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[demo.dataset],
        session_id=fresh_session,
        scope="graph",
    )
    demo.show("fresh-session-answer-MODEL-GENERATED", fresh_response)
    demo.show("fresh-session", await cognee.session.get_session(session_id=fresh_session))
    chunks = await retrieve_chunks(new_query, dataset=demo.dataset, top_k=demo.limits["retrieve_top_k"])
    demo.show("fresh-session-retrieved-documents", chunks)
    matches = exact_lesson_matches(summary["documents"], chunks)
    demo.show(
        "learning-evidence",
        {
            "teaching_run": str(prior_run.resolve()),
            "distillation_status": summary["status"],
            "accepted_document_count": summary["accepted_document_count"],
            "exact_text_matches": matches,
            "note": "Matches identify complete model-generated lesson texts, not independently verified policy facts.",
        },
    )
    if summary["persisted_lessons"] and not matches:
        print(
            "No complete accepted lesson matched the retrieved chunks; inspect the documents before claiming transfer."
        )


async def teach(demo: Demo, *, seed: bool) -> None:
    cognee = initialize(demo, learning=True)
    from cognee.api.v1.search import SearchType

    if seed:
        await seed_memory(demo, cognee)
    teaching_session = f"{demo.namespace}-teaching-{uuid4().hex[:10]}"
    correction = (ROOT / "data/correction.txt").read_text()
    demo.show("authored-correction", {"source_id": "CORRECTION-01", "text": correction})
    demo.show("session-ids", {"teaching": teaching_session})
    response = await cognee.recall(
        correction + "\nWhat policy correction should be remembered for future workshops?",
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[demo.dataset],
        session_id=teaching_session,
        scope="graph",
    )
    demo.show("correction-response-MODEL-GENERATED", response)
    demo.show("teaching-session", await cognee.session.get_session(session_id=teaching_session))
    context = await cognee.recall(
        "workshop readiness correction",
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[demo.dataset],
        session_id=teaching_session,
        scope="session_context",
    )
    demo.show("usable-session-context-before-distillation", context)
    print(
        "Distilling session: selecting gated entries, proposing lessons, judging and persisting accepted documents…",
        flush=True,
    )
    result = await cognee.session.distill_session(
        teaching_session, demo.dataset, max_batches=demo.limits["distill_batches"]
    )
    summary = distillation_summary(result)
    demo.show("actual-distillation", summary)
    if not summary["persisted_lessons"]:
        print("No accepted lesson documents were persisted; verification will retain this status.", flush=True)
    command = (
        f"WORKSHOP_NAMESPACE={shlex.quote(demo.namespace)} uv run 04_learn.py "
        f"--verify-run {shlex.quote(str(demo.run.resolve()))}"
    )
    print(f"\nTeaching finished. Run this from the example folder in a new process:\n{command}", flush=True)


async def main(demo: Demo) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--seed", action="store_true", help="Ingest the three initial records before teaching")
    modes.add_argument("--verify-run", type=Path, help="Verify stored lessons from a prior teaching run; no teaching")
    args = parser.parse_args()
    if args.verify_run is not None:
        try:
            await verify(demo, args.verify_run)
        except MissingMemoryStateError:
            parser.error(MISSING_STATE_RECOVERY)
    else:
        await teach(demo, seed=args.seed)


if __name__ == "__main__":
    launch("04-learn", main)
