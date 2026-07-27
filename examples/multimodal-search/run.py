from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCES_PATH = ROOT / "data" / "sources.json"
IMAGE_DIR = ROOT / "data" / "images"
MODEL = "google/siglip-so400m-patch14-384"
EXPECTED_DIMENSIONS = 1152
EXPECTED_TOP_MATCH = "red-leather-handbag.png"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return value


def load_and_verify_sources() -> dict[str, Any]:
    sources = read_json(SOURCES_PATH)
    if sources.get("synthetic_or_generated_images") is not False:
        raise ValueError("Image manifest must reject synthetic images")
    if sources.get("query", {}).get("provenance") != "authored retrieval query":
        raise ValueError("Query provenance is missing")
    images = sources.get("images")
    if not isinstance(images, list) or len(images) != 6:
        raise ValueError("Expected exactly six source images")

    names: set[str] = set()
    for image in images:
        relative = Path(image["file"])
        if relative.parts[:1] != ("images",) or len(relative.parts) != 2:
            raise ValueError(f"Invalid image path: {relative}")
        path = ROOT / "data" / relative
        if path.name in names:
            raise ValueError(f"Duplicate image: {path.name}")
        names.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != image["byte_length"]:
            raise ValueError(f"Image byte length changed: {path.name}")
        if sha256_file(path) != image["sha256"]:
            raise ValueError(f"Image checksum changed: {path.name}")
        required_attribution = ("source", "license", "license_url", "creator")
        if not all(image.get(field) for field in required_attribution):
            raise ValueError(f"Missing source or attribution: {path.name}")
    if EXPECTED_TOP_MATCH not in names:
        raise ValueError("Expected target image is missing")
    return sources


def build_query_audit_envelope(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "SIEClient.encode",
        "model": MODEL,
        "input": {"text": sources["query"]["text"]},
        "wait_for_capacity": True,
        "provision_timeout_s": 900,
    }


def build_image_audit_envelope(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "SIEClient.encode",
        "model": MODEL,
        "input": [
            {
                "id": Path(image["file"]).name,
                "images": [
                    {
                        "file": Path(image["file"]).name,
                        "sha256": image["sha256"],
                        "byte_length": image["byte_length"],
                    }
                ],
            }
            for image in sources["images"]
        ],
        "wait_for_capacity": True,
        "provision_timeout_s": 900,
    }


def dense_vector(result: Any) -> list[float]:
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected encode result: {type(result).__name__}")
    dense = result.get("dense")
    if isinstance(dense, dict):
        dense = dense.get("values") or dense.get("vector")
    if hasattr(dense, "tolist"):
        dense = dense.tolist()
    if not isinstance(dense, list):
        raise TypeError("Encode response has no dense vector")
    vector = [float(value) for value in dense]
    if len(vector) != EXPECTED_DIMENSIONS:
        raise ValueError(f"Expected {EXPECTED_DIMENSIONS} dimensions, received {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains a non-finite value")
    return vector


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector length mismatch")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compare a zero vector")
    return dot / (left_norm * right_norm)


def evaluate(
    sources: dict[str, Any],
    query_response: dict[str, Any],
    image_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(image_responses) != len(sources["images"]):
        raise ValueError("SIE did not return one embedding per image")
    query_vector = dense_vector(query_response)
    matches = []
    for image, response in zip(sources["images"], image_responses, strict=True):
        matches.append(
            {
                "rank": 0,
                "file": Path(image["file"]).name,
                "score": cosine(query_vector, dense_vector(response)),
                "sha256": image["sha256"],
            }
        )
    matches.sort(key=lambda item: item["score"], reverse=True)
    for rank, match in enumerate(matches, start=1):
        match["rank"] = rank
    if matches[0]["file"] != EXPECTED_TOP_MATCH:
        raise ValueError(f"Expected {EXPECTED_TOP_MATCH}, received {matches[0]['file']}")
    return {
        "query": sources["query"]["text"],
        "metric": "cosine_similarity",
        "image_count": len(matches),
        "sorted_matches": matches,
    }


def run_search() -> dict[str, Any]:
    from sie_sdk import SIEClient

    sources = load_and_verify_sources()
    base_url = os.environ.get("SIE_BASE_URL", "http://127.0.0.1:8080")
    api_key = os.environ.get("SIE_API_KEY") or None
    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    query_response = to_jsonable(
        client.encode(
            MODEL,
            {"text": sources["query"]["text"]},
            wait_for_capacity=True,
            provision_timeout_s=900,
        )
    )
    image_paths = [ROOT / "data" / image["file"] for image in sources["images"]]
    image_responses = to_jsonable(
        client.encode(
            MODEL,
            [{"id": path.name, "images": [path]} for path in image_paths],
            wait_for_capacity=True,
            provision_timeout_s=900,
        )
    )
    if not isinstance(image_responses, list):
        raise TypeError("Expected a list of image encode responses")
    evaluation = evaluate(sources, query_response, image_responses)
    return {
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint": base_url,
        "model": MODEL,
        "audit_envelopes": {
            "query": build_query_audit_envelope(sources),
            "images": build_image_audit_envelope(sources),
        },
        "raw": {
            "query_response": query_response,
            "image_responses": image_responses,
        },
        "evaluation": evaluation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search six licensed images with text and public SIE")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_search()
    if args.output:
        write_json(args.output, result)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
