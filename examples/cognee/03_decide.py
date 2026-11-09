"""Retrieve source chunks, rerank, and answer a focused question with typed probabilities."""

import argparse

from common import Demo, initialize, launch, run_decision, seed_memory


async def main(demo: Demo) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="Ingest the three initial records first")
    parser.add_argument("--query", help="Override the configured workshop question")
    args = parser.parse_args()
    cognee = initialize(demo)
    if args.seed:
        await seed_memory(demo, cognee)
    await run_decision(demo, args.query or demo.decision["query"])


if __name__ == "__main__":
    launch("03-decide", main)
