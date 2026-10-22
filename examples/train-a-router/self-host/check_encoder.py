"""Check the pinned encoder; --smoke additionally makes one synthetic SDK call."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np
from sie_sdk import SIEClient

MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
DIMENSIONS = 2560
INSTRUCTION = "Represent the knowledge and reasoning required to answer this question correctly."


class SingleSendClient(SIEClient):
    def _record_retry(self):
        raise RuntimeError("No automatic retry: inspect the server before making another request")


def check_metadata(metadata):
    if (
        metadata.get("name") != MODEL
        or metadata.get("revision") != REVISION
        or metadata.get("loaded") is not True
        or metadata.get("dims", {}).get("dense") != DIMENSIONS
    ):
        raise RuntimeError("The exact encoder is not loaded with 2,560 output dimensions")


def check_vector(reply):
    if not isinstance(reply, list) or len(reply) != 1:
        raise RuntimeError("Expected exactly one embedding")
    item = reply[0]
    if item.get("id") != "encoder-smoke" or item.get("model") != MODEL:
        raise RuntimeError("The response does not match the request")
    vector = item.get("dense")
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise RuntimeError("Wrong vector dimensions")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in vector):
        raise RuntimeError("The vector contains nonfinite or nonnumeric values")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    # BF16 normalization precedes the float32 output cast; feature collection
    # subsequently renormalizes in float64 before saving float32 vectors.
    if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-2):
        raise RuntimeError("Expected a normalized, nonzero vector")
    return norm, hashlib.sha256(json.dumps(vector, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    timeout = 300 if args.smoke else 3
    # An explicit empty key prevents inheriting any Cloud credentials.
    with SingleSendClient(
        "http://127.0.0.1:8080", api_key="", timeout_s=timeout, read_timeout_s=timeout
    ) as client:
        metadata = client.get_model(MODEL)
        check_metadata(metadata)
        if not args.smoke:
            return
        started = time.perf_counter()
        reply = client.encode(
            MODEL,
            [{"id": "encoder-smoke", "text": "Which is an even integer? A. 3 B. 4 C. 5"}],
            output_types=["dense"],
            output_dtype="float32",
            instruction=INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=150,
        )
        elapsed = time.perf_counter() - started
        norm, digest = check_vector(reply)
        if client.last_retry_count != 0:
            raise RuntimeError("The SDK unexpectedly retried the request")
        print(json.dumps({
            "status": "complete",
            "model": MODEL,
            "revision": REVISION,
            "dimensions": DIMENSIONS,
            "instruction": INSTRUCTION,
            "is_query": True,
            "vector_l2_norm": norm,
            "vector_sha256": digest,
            "execution_bundle_sha256": client.last_model_revision,
            "elapsed_seconds": elapsed,
            "physical_encoding_calls": 1,
            "answer_model_calls": 0,
            "observed_cloud_usd": None,
            "hosting_cost": "unpriced",
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
