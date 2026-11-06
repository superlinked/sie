"""Distill an authored correction, inspect the actual outcome, then ask in a fresh session."""

import argparse
from typing import Any
from uuid import uuid4

from common import ROOT, Demo, initialize, json_value, launch, run_decision, seed_memory


def distillation_summary(result: Any) -> dict[str, Any]:
    payload = json_value(result)
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise ValueError("Unexpected Cognee distillation response.")
    documents = payload.get("documents")
    if not isinstance(documents, list) or any(not isinstance(document, str) for document in documents):
        raise ValueError("Distillation documents must be actual lesson texts.")
    return {
        **payload,
        "accepted_document_count": len(documents),
        "persisted_lessons": payload["status"] == "completed" and bool(documents),
    }


async def main(demo: Demo) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="Ingest the three initial records first")
    args = parser.parse_args()
    cognee = initialize(demo, learning=True)
    from cognee.api.v1.search import SearchType

    if args.seed:
        await seed_memory(demo, cognee)
    baseline = await run_decision(demo, demo.decision["query"], prefix="before")
    teaching_session = f"{demo.namespace}-teaching-{uuid4().hex[:10]}"
    fresh_session = f"{demo.namespace}-new-{uuid4().hex[:10]}"
    correction = (ROOT / "data/correction.txt").read_text()
    demo.show("authored-correction", {"source_id": "CORRECTION-01", "text": correction})
    demo.show("session-ids", {"teaching": teaching_session, "fresh": fresh_session})
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
        print("No accepted lesson documents were persisted. The fresh-session check will still run.", flush=True)
    new_query = demo.decision["new_query"]
    # Fresh session has never seen the correction. This generated answer is shown,
    # but only the separately retrieved document chunks enter the decision model.
    fresh_response = await cognee.recall(
        new_query,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[demo.dataset],
        session_id=fresh_session,
        scope="graph",
    )
    demo.show("fresh-session-answer-MODEL-GENERATED", fresh_response)
    demo.show("fresh-session", await cognee.session.get_session(session_id=fresh_session))
    after = await run_decision(demo, new_query, prefix="after")
    demo.show(
        "comparison",
        {
            "before": baseline,
            "after": after,
            "choice_changed": baseline["choice"] != after["choice"],
            "accepted_document_count": summary["accepted_document_count"],
            "note": "Different input and session; this is an inspectable transfer experiment, not a causal evaluation.",
        },
    )


if __name__ == "__main__":
    launch("04-learn", main)
