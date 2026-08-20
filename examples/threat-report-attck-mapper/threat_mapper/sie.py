from __future__ import annotations

import json
import time
from typing import Any, Protocol

import numpy as np
from sie_sdk.types import Item


class SIEClientProtocol(Protocol):
    def encode(self, model: str, items: Item | list[Item], **kwargs: Any) -> Any: ...

    def score(self, model: str, query: Item, items: list[Item], **kwargs: Any) -> Any: ...

    def extract(self, model: str, item: Item, **kwargs: Any) -> Any: ...

    def generate(self, model: str, prompt: str, **kwargs: Any) -> Any: ...


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


def dense_vector(result: Any) -> np.ndarray:
    value = jsonable(result)
    dense: Any = value.get("dense") if isinstance(value, dict) else None
    if isinstance(dense, dict):
        dense = dense.get("values") or dense.get("vector")
    if not isinstance(dense, list) or not dense:
        raise TypeError("Embedding response has no dense vector")
    vector = np.asarray(dense, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding response has a zero-length vector")
    return vector / norm


def request_record(
    stage: str,
    requested_model: str,
    response: Any,
    latency_ms: float,
    *,
    function: str,
) -> dict[str, Any]:
    payload = jsonable(response)
    if not isinstance(payload, dict):
        payload = {}
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    usage = request.get("usage") if isinstance(request.get("usage"), dict) else payload.get("usage", {})
    return {
        "stage": stage,
        "function": function,
        "requested_model": requested_model,
        "runtime_model": payload.get("model", requested_model),
        "request_id": request.get("id"),
        "credits_debited": request.get("credits_debited"),
        "rate_book_version": request.get("rate_book_version") or usage.get("rate_book_version"),
        "execution_identity_sha256": request.get("execution_identity_sha256"),
        "latency_ms": round(latency_ms, 1),
    }


def encode_texts(
    client: SIEClientProtocol,
    model: str,
    texts: list[str],
    *,
    instruction: str | None,
    is_query: bool,
    batch_size: int,
    provision_timeout_s: float,
    stage: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    calls: list[dict[str, Any]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        items = [Item(id=f"{stage}-{offset + index}", text=text) for index, text in enumerate(batch)]
        started = time.perf_counter()
        response = client.encode(
            model,
            items,
            output_types=["dense"],
            instruction=instruction,
            is_query=is_query,
            wait_for_capacity=True,
            provision_timeout_s=provision_timeout_s,
        )
        elapsed = (time.perf_counter() - started) * 1000
        responses = response if isinstance(response, list) else [response]
        if len(responses) != len(items):
            raise RuntimeError(f"Embedding batch returned {len(responses)} rows for {len(items)} inputs")
        vectors.extend(dense_vector(row) for row in responses)
        calls.append(
            request_record(
                f"{stage}_{offset // batch_size}",
                model,
                responses[0],
                elapsed,
                function="encode",
            )
        )
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), calls
    return np.stack(vectors), calls


def parse_generated_json(response: Any) -> dict[str, Any]:
    payload = jsonable(response)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise TypeError("SIE generate response has no text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("SIE generate response was not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("SIE generate response must be one JSON object")
    return value
