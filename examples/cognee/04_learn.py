"""Distill a correction, then verify stored lessons in a separate process."""

import argparse
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from sie_cognee.retrieval import SourceChunk, retrieve_chunks

from common import (
    MEMORY_STATE_RECOVERY,
    ROOT,
    Demo,
    MemoryStateError,
    initialize,
    json_value,
    launch,
    read_state_object,
    record_failure,
    require_input,
    seed_memory,
)


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
    configuration = read_state_object(prior_run / "configuration.json")
    try:
        summary = distillation_summary(read_state_object(prior_run / "actual-distillation.json"))
    except ValueError:
        raise MemoryStateError(MEMORY_STATE_RECOVERY) from None
    if configuration.get("dataset") != demo.dataset:
        raise MemoryStateError(MEMORY_STATE_RECOVERY)
    models = configuration.get("models")
    if not isinstance(models, dict):
        raise MemoryStateError(MEMORY_STATE_RECOVERY)
    expected = {key: demo.models[key] for key in ("embedding", "embedding_dimensions")}
    if {key: models.get(key) for key in expected} != expected:
        raise MemoryStateError(MEMORY_STATE_RECOVERY)
    embedding_manifest = demo.storage / "embedding.json"
    cognee_marker = demo.storage / "cognee/sie-cognee.json"
    try:
        if not embedding_manifest.is_file() or not cognee_marker.is_file():
            raise MemoryStateError(MEMORY_STATE_RECOVERY)
    except OSError:
        raise MemoryStateError(MEMORY_STATE_RECOVERY) from None
    stored = read_state_object(embedding_manifest)
    owned = read_state_object(cognee_marker)
    if stored != expected or owned != {
        "embedding_model": expected["embedding"],
        "embedding_dimensions": expected["embedding_dimensions"],
    }:
        raise MemoryStateError(MEMORY_STATE_RECOVERY)
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
    correction = require_input(ROOT / "data/correction.txt").read_text(encoding="utf-8")
    cognee = initialize(demo, learning=True)
    from cognee.api.v1.search import SearchType

    if seed:
        await seed_memory(demo, cognee)
    teaching_session = f"{demo.namespace}-teaching-{uuid4().hex[:10]}"
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
        except MemoryStateError as error:
            saved = record_failure(demo.run, error)
            note = "" if saved else " Could not write failure.json."
            parser.error(MEMORY_STATE_RECOVERY + note)
    else:
        await teach(demo, seed=args.seed)


if __name__ == "__main__":
    launch("04-learn", main)
