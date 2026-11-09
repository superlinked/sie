"""Ingest three authored facts and inspect raw chunks separately from graph context."""

import argparse

from sie_cognee.retrieval import retrieve_chunks

from common import Demo, initialize, launch, seed_memory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--graph", action="store_true", help="Also display model-derived graph context (not decision evidence)"
)


async def main(demo: Demo) -> None:
    args = parser.parse_args()
    cognee = initialize(demo)
    await seed_memory(demo, cognee)
    query = demo.decision["query"]
    demo.show("source-chunks", await retrieve_chunks(query, dataset=demo.dataset, top_k=5))
    if args.graph:
        # Optional dependency: configuration must precede every Cognee import.
        from cognee.api.v1.search import SearchType

        context = await cognee.recall(
            query, query_type=SearchType.HYBRID_COMPLETION, datasets=[demo.dataset], only_context=True, scope="graph"
        )
        demo.show("model-derived-graph-context-NOT-source-evidence", context)


if __name__ == "__main__":
    launch("01-memory", main)
