"""Explicitly record new native vectors through a user-supplied SIE endpoint."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sie_sdk import SIEClient

from evidence import ARM_DIMENSIONS, DEFAULT_ROOT, REVISION, inputs, read_json, verify
from score import rank, vector

NATIVE_MODELS = {
    "sie-siglip384": "google/siglip-so400m-patch14-384",
    "sie-siglip2-base224": "google/siglip2-base-patch16-224",
}


def encode_vectors(
    client: Any, root: Path, images: list[dict], queries: list[dict], arm: str
) -> dict[str, dict[str, list[float]]]:
    model = NATIVE_MODELS[arm]
    width = ARM_DIMENSIONS[arm]
    indexed = {}
    for start in range(0, len(images), 10):
        batch = images[start : start + 10]
        items = [
            {"images": [{"data": (root / "inputs/original-context" / row["file"]).read_bytes(), "format": "jpeg"}]}
            for row in batch
        ]
        replies = client.encode(
            model,
            items,
            is_query=False,
            output_types=["dense"],
            output_dtype="float32",
            wait_for_capacity=False,
            max_oom_retries=0,
        )
        if len(replies) != len(batch):
            raise ValueError("Index call did not return one vector per submitted image")
        for row, reply in zip(batch, replies, strict=True):
            indexed[row["image_id"]] = vector([float(value) for value in reply["dense"]], width)
    replies = client.encode(
        model,
        [{"text": row["query"]} for row in queries],
        is_query=True,
        output_types=["dense"],
        output_dtype="float32",
        wait_for_capacity=False,
        max_oom_retries=0,
    )
    if len(replies) != len(queries):
        raise ValueError("Query call did not return one vector per submitted query")
    encoded_queries = {
        row["query_id"]: vector([float(value) for value in reply["dense"]], width)
        for row, reply in zip(queries, replies, strict=True)
    }
    return {"index": indexed, "query": encoded_queries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arm", choices=list(NATIVE_MODELS), default="sie-siglip384")
    parser.add_argument("--url", default=os.environ.get("SIE_URL"))
    parser.add_argument("--output", type=Path, default=Path("runs/native-vectors.json"))
    parser.add_argument(
        "--live", action="store_true", help="Send three native Encode requests to your configured endpoint"
    )
    args = parser.parse_args()
    verify(args.evidence)
    images, queries, grades = inputs(args.evidence)
    profiles = read_json(args.evidence, "inputs/original-context/arms.json")
    profile = next(row["native_profile"] for row in profiles if row["arm_id"] == args.arm)
    plan = {
        "model": NATIVE_MODELS[args.arm],
        "requested_checkpoint_revision": profile["checkpoint_revision"],
        "native_index_batches": [10, 10],
        "native_query_batch": 8,
        "image_bytes": "The exact hash-verified prepared JPEGs, without re-encoding or metadata",
        "index_is_query": False,
        "query_is_query": True,
    }
    if not args.live:
        print(json.dumps(plan, indent=2))
        return
    if not args.url:
        parser.error("Set SIE_URL to your compatible SIE endpoint, or pass --url")
    if args.output.exists():
        parser.error("The output already exists; choose a new path for this run")
    with SIEClient(base_url=args.url, api_key=os.environ.get("SIE_API_KEY"), timeout_s=120) as client:
        values = encode_vectors(client, args.evidence, images, queries, args.arm)
    order = [row["image_id"] for row in images]
    observations = []
    for query in queries:
        query_id = query["query_id"]
        ranking, scores = rank(values["index"], values["query"][query_id], order)
        observations.append(
            {
                "query_id": query_id,
                "query": query["query"],
                "ranking": ranking,
                "cosines": scores,
                "top1_grade": grades[query_id][ranking[0]],
            }
        )
    result = {
        "evidence_revision": REVISION,
        "arm": args.arm,
        "requested_settings": plan,
        "vectors": values,
        "observations": observations,
        "first_result_grade2": sum(row["top1_grade"] == 2 for row in observations),
        "independent_query_families": len(queries),
        "settings_scope": "The checkpoint is requested, not independently attested by these returned vectors.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "first_result_grade2": result["first_result_grade2"]}))


if __name__ == "__main__":
    main()
