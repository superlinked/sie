#!/usr/bin/env -S uv run --frozen --project . --package sie-sdk python
# fmt: off
#MISE description="Smoke test a SIE cluster (encode, score, extract, optional generate) and measure cold-start latency"
#USAGE arg "<endpoint>" help="Cluster endpoint URL (e.g. http://localhost:8080 or LB URL)"
#USAGE flag "--gpu <gpu>" help="GPU/machine profile to route to (e.g. g5, l4)" default=""
#USAGE flag "--model <model>" help="Encode model to test" default="sentence-transformers/all-MiniLM-L6-v2"
#USAGE flag "--score-model <model>" help="Score model to test" default="cross-encoder/ms-marco-MiniLM-L-6-v2"
#USAGE flag "--extract-model <model>" help="Extract model to test" default="urchade/gliner_small-v2.1"
#USAGE flag "--generate" help="Also test the generate endpoint" default="false"
#USAGE flag "--only-generate" help="Skip encode/score/extract and only test generate" default="false"
#USAGE flag "--generate-model <model>" help="Generate model to test (required with --generate or --only-generate)" default=""
#USAGE flag "--generate-prompt <prompt>" help="Prompt for the generate test" default="Say hi in one sentence."
#USAGE flag "--generate-tokens <n>" help="Maximum new tokens for the generate test" default="64"
#USAGE flag "--generate-mode <mode>" help="Generate test mode: native, chat, or both" default="native"
#USAGE flag "--gpu-agnostic-chat-probe" help="Send a no-GPU chat-completions provisioning probe before generate" default="false"
#USAGE flag "--only-capacity-probe" help="Only send the gpu-agnostic chat provisioning probe; do not wait for generation" default="false"
#USAGE flag "--check-docs" help="Probe that GET /docs answers 200 before the API tests" default="false"
#USAGE flag "--skip-extract" help="Skip the extract test (for deployments without an extract model)" default="false"
#USAGE flag "--rerank-model <model>" help="Also test the OpenAI/Cohere /v1/rerank surface with this model" default=""
#USAGE flag "--openai-embeddings" help="Also test the OpenAI-compatible /v1/embeddings surface (encode model)" default="false"
#USAGE flag "--embeddings-determinism" help="Assert two identical /v1/embeddings calls return identical vectors" default="false"
#USAGE flag "--embeddings-batch <n>" help="Batch size for an informational /v1/embeddings throughput probe (0 = skip)" default="0"
#USAGE flag "--generate-streaming" help="Also exercise streaming for the selected generate mode(s)" default="false"
#USAGE flag "-n --iterations <n>" help="Number of iterations to run" default="1"
#USAGE flag "-t --timeout <seconds>" help="Request timeout in seconds (increase for cold start)" default="600"
#USAGE flag "-v --verbose" help="Show query details and full result data" default="false"
# fmt: on

"""Smoke test a SIE cluster and measure time to first response.

Runs encode, score, extract, and optional generate requests against a live cluster.
Measures wall-clock time (including node scale-up and model loading)
and reports server-side timing breakdown.

Usage:
  mise run smoketest -- http://localhost:8080
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --gpu g5
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --gpu g5 -n 5
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --gpu g5 -n 3 -v
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --generate --generate-model Qwen/Qwen3.6-27B:rtx-pro-6000
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --only-generate --generate-model Qwen/Qwen3.6-27B:rtx-pro-6000
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --only-generate --gpu rtx6000 --generate-mode both --gpu-agnostic-chat-probe --generate-model Qwen/Qwen3.6-27B
  mise run smoketest -- http://my-cluster.elb.amazonaws.com --only-capacity-probe --generate-model Qwen/Qwen3.6-27B
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from common.colors import log_error
from common.env import apply_mise_env, get_usage_bool, get_usage_flag, get_usage_int

apply_mise_env()

from sie_sdk import SIEClient
from sie_sdk.types import ChatMessage, Item

# ---------------------------------------------------------------------------
# ANSI + formatting
# ---------------------------------------------------------------------------

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def _ft(s: float) -> str:
    """Format seconds for display."""
    return f"{s / 60:.1f}min" if s >= 60 else f"{s:.1f}s"


def _fms(ms: float | None) -> str:
    """Format milliseconds for display."""
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

ENCODE_QUERY = Item(text="A spaceship is traveling from Earth to Mars.")
SCORE_QUERY = Item(text="Where should we land on Mars?")
SCORE_ITEMS = [
    Item(id="relevant", text="Flat regions near the equator with low elevation are ideal."),
    Item(id="irrelevant", text="Saturn has beautiful rings made of ice."),
]
EXTRACT_QUERY = Item(text="Captain Kirk commands the starship Enterprise for Starfleet.")
EXTRACT_LABELS = ["person", "organization", "ship"]
GENERATE_PROMPT = "Say hi in one sentence."
GENERATE_MODES = {"native", "chat", "both"}
OPENAI_CAPACITY_PROBE_CODES = {
    "provisioning": "PROVISIONING",
    "MODEL_LOADING": "MODEL_LOADING",
    "RESOURCE_EXHAUSTED": "RESOURCE_EXHAUSTED",
}
OPENAI_PROVISIONING_RETRY_AFTER = "60"
EMBEDDINGS_INPUTS = ["hello world", "apple silicon"]
RERANK_QUERY = "apple silicon"
RERANK_DOCUMENTS = ["M-series chips", "the weather today", "Metal GPU inference"]
# Raw-HTTP loading markers: lazy loading is non-blocking on most paths — the
# first touch returns 503 with one of these codes plus Retry-After and expects
# the client to retry. The SDK's wait_for_capacity does this for the
# native surfaces; these markers mirror it for the OpenAI surfaces the SDK
# does not wrap (/v1/embeddings, /v1/rerank).
LOADING_RETRY_MARKERS = ("MODEL_LOADING", "PROVISIONING", "LORA_LOADING")
LOADING_RETRY_MAX_DELAY_S = 5.0


@dataclasses.dataclass
class TestResult:
    name: str
    wall_s: float
    ok: bool
    lines: list[str] = dataclasses.field(default_factory=list)
    error: str = ""


def _run(name: str, fn: Callable[[], dict[str, Any]], verbose: bool) -> TestResult:
    """Run a single test, capture wall time, return structured result."""
    t0 = time.monotonic()
    try:
        data = fn()
        wall_s = time.monotonic() - t0
        lines = _format_result(name, data, wall_s, verbose)
        return TestResult(name, wall_s, True, lines)
    except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
        wall_s = time.monotonic() - t0
        return TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e))


def _chat_messages(prompt: str) -> list[ChatMessage]:
    """Build the OpenAI-compatible chat request payload."""
    return [{"role": "user", "content": prompt}]


def _validated_generate(data: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Assert the blocking-generate contract: non-empty text, clean finish."""
    text = data.get("text")
    if not isinstance(text, str) or not text:
        msg = "generate returned no text"
        raise ValueError(msg)
    finish_reason = data.get("finish_reason")
    if finish_reason not in ("stop", "length"):
        msg = f"unexpected finish_reason: {finish_reason!r}"
        raise ValueError(msg)
    return {**data, "_prompt": prompt}


def _validated_chat_completion(data: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Assert the chat-completion contract before the lenient formatting."""
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict) or "message" not in choices[0]:
        msg = "chat completion returned no choices[0].message"
        raise ValueError(msg)
    return _format_chat_completion(data, prompt)


def _format_chat_completion(data: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Normalize an OpenAI-compatible chat completion for smoke output."""
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return {
        "text": content or "",
        "usage": data.get("usage"),
        "_prompt": prompt,
    }


def _gpu_agnostic_route(gpu: str | None) -> str | None:
    """Drop the machine profile while preserving an explicit pool route."""
    if not gpu or "/" not in gpu:
        return None
    pool_name, _machine_profile = gpu.split("/", 1)
    return f"{pool_name}/" if pool_name else None


def _openai_probe_headers(api_key: str | None, gpu_route: str | None) -> dict[str, str]:
    """Build raw HTTP headers for the OpenAI-compatible capacity probe."""
    headers = {"content-type": "application/json", "accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if not gpu_route:
        return headers
    if "/" in gpu_route:
        pool_name, machine_profile = gpu_route.split("/", 1)
        if pool_name:
            headers["X-SIE-Pool"] = pool_name
        if machine_profile:
            headers["X-SIE-MACHINE-PROFILE"] = machine_profile
    else:
        headers["X-SIE-MACHINE-PROFILE"] = gpu_route
    return headers


def _validate_openai_capacity_probe_response(response: httpx.Response) -> tuple[str, str, str]:
    """Validate the non-2xx OpenAI-compatible not-ready wire contract."""
    if response.status_code == 202:
        msg = "Gateway returned deprecated HTTP 202 provisioning; expected HTTP 503"
        raise ValueError(msg)
    if response.status_code != 503:
        msg = f"Expected HTTP 200 or 503 from chat capacity probe, got {response.status_code}"
        raise ValueError(msg)
    try:
        data = response.json()
    except ValueError as e:
        msg = "Expected JSON OpenAI error envelope from chat capacity probe"
        raise ValueError(msg) from e
    if not isinstance(data, dict) or "detail" in data:
        msg = "Expected OpenAI error envelope without native detail field"
        raise ValueError(msg)
    error = data.get("error")
    if not isinstance(error, dict):
        msg = "Expected OpenAI error object in chat capacity probe response"
        raise ValueError(msg)
    code = error.get("code")
    if not isinstance(code, str) or code not in OPENAI_CAPACITY_PROBE_CODES:
        msg = f"Unexpected OpenAI capacity probe error code: {code!r}"
        raise ValueError(msg)
    expected_header_code = OPENAI_CAPACITY_PROBE_CODES[code]
    header_code = response.headers.get("X-SIE-Error-Code")
    if header_code != expected_header_code:
        msg = f"Expected X-SIE-Error-Code={expected_header_code}, got {header_code!r}"
        raise ValueError(msg)
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        msg = "Expected Retry-After on OpenAI-compatible not-ready response"
        raise ValueError(msg)
    if code == "provisioning" and retry_after != OPENAI_PROVISIONING_RETRY_AFTER:
        msg = f"Expected provisioning Retry-After={OPENAI_PROVISIONING_RETRY_AFTER}, got {retry_after!r}"
        raise ValueError(msg)
    return code, header_code, retry_after


def _run_gpu_agnostic_chat_probe(
    endpoint: str,
    api_key: str | None,
    gpu_route: str | None,
    model: str,
    prompt: str,
    generate_tokens: int,
    timeout: float,
    verbose: bool,
) -> TestResult:
    """Exercise the OpenAI-compatible scale-from-zero wire contract without waiting for generation."""
    name = f"generate/chat capacity-probe(any) · {model}"
    t0 = time.monotonic()
    try:
        response = httpx.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": _chat_messages(prompt),
                "max_tokens": generate_tokens,
                "stream": False,
            },
            headers=_openai_probe_headers(api_key, gpu_route),
            timeout=min(timeout, 30.0),
        )
        wall_s = time.monotonic() - t0
        if response.status_code == 200:
            data = response.json()
            lines = _format_result(name, _format_chat_completion(data, prompt), wall_s, verbose)
            lines.insert(1, f"  probe: {GREEN}warm capacity returned HTTP 200{RESET}")
            return TestResult(name, wall_s, True, lines)

        code, header_code, retry_after = _validate_openai_capacity_probe_response(response)
        lines = [
            f"  wall: {BOLD}{_ft(wall_s)}{RESET}",
            f"  probe: {GREEN}not-ready contract observed{RESET} via HTTP 503",
            f"  error: code={code}, x-sie-error-code={header_code}",
            f"  retry-after: {retry_after}s",
        ]
        if verbose:
            lines.append(f"  {DIM}prompt: {prompt!r}{RESET}")
            lines.append(f"  {DIM}body: {response.text}{RESET}")
        lines.append(f"  {GREEN}OK{RESET}")
        return TestResult(name, wall_s, True, lines)
    except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
        wall_s = time.monotonic() - t0
        return TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e))


def _loading_retry_delay(response: httpx.Response) -> float | None:
    """Return the retry delay for a retryable not-ready 503, else None."""
    if response.status_code != 503 or not any(marker in response.text for marker in LOADING_RETRY_MARKERS):
        return None
    retry_after = response.headers.get("Retry-After")
    try:
        delay = float(retry_after) if retry_after else LOADING_RETRY_MAX_DELAY_S
    except ValueError:
        delay = LOADING_RETRY_MAX_DELAY_S
    return max(0.0, min(delay, LOADING_RETRY_MAX_DELAY_S))


def _post_json_ready(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout_s: float,
) -> httpx.Response:
    """POST to an OpenAI-compatible surface, retrying while the model loads.

    Any response other than a retryable loading 503 (including terminal errors
    like MODEL_LOAD_FAILED) is returned immediately for the caller's assertion;
    a genuinely stuck load fails via the timeout bound.
    """
    headers = {"content-type": "application/json", "accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            msg = f"{path} still not ready after {timeout_s:.0f}s"
            raise TimeoutError(msg)
        response = httpx.post(f"{endpoint.rstrip('/')}{path}", json=payload, headers=headers, timeout=remaining)
        delay = _loading_retry_delay(response)
        if delay is None:
            return response
        time.sleep(min(delay, remaining))


def _fetch_openai_embeddings(
    endpoint: str, api_key: str | None, model: str, inputs: list[str], timeout_s: float
) -> list[list[float]]:
    """Fetch /v1/embeddings vectors, asserting the response shape."""
    response = _post_json_ready(endpoint, "/v1/embeddings", {"model": model, "input": inputs}, api_key, timeout_s)
    if response.status_code != 200:
        msg = f"HTTP {response.status_code}: {response.text[:300]}"
        raise ValueError(msg)
    rows = response.json().get("data") or []
    if len(rows) != len(inputs) or any(not row.get("embedding") for row in rows):
        msg = f"expected {len(inputs)} non-empty vectors, got {len(rows)}"
        raise ValueError(msg)
    dims = {len(row["embedding"]) for row in rows}
    if len(dims) != 1:
        msg = f"embedding dimensions disagree within one batch: {sorted(dims)}"
        raise ValueError(msg)
    return [row["embedding"] for row in rows]


def _run_check_docs(endpoint: str, verbose: bool) -> TestResult:
    """Probe that GET /docs answers 200."""

    def fn() -> dict[str, Any]:
        response = httpx.get(f"{endpoint.rstrip('/')}/docs", timeout=30.0)
        if response.status_code != 200:
            msg = f"HTTP {response.status_code}"
            raise ValueError(msg)
        return {}

    return _run("GET /docs", fn, verbose)


def _run_openai_embeddings(
    endpoint: str,
    api_key: str | None,
    model: str,
    timeout_s: float,
    verbose: bool,
    determinism: bool,
    batch: int,
) -> list[TestResult]:
    """Probe the OpenAI /v1/embeddings surface: shape, optional determinism + throughput."""
    results: list[TestResult] = []
    first: list[list[float]] = []

    def fetch() -> dict[str, Any]:
        first.extend(_fetch_openai_embeddings(endpoint, api_key, model, EMBEDDINGS_INPUTS, timeout_s))
        return {"dense": first[0]}

    shape = _run(f"embeddings(openai) · {model}", fetch, verbose)
    results.append(shape)

    if determinism and shape.ok:

        def refetch() -> dict[str, Any]:
            again = _fetch_openai_embeddings(endpoint, api_key, model, EMBEDDINGS_INPUTS, timeout_s)
            if again != first:
                msg = "identical inputs returned different vectors"
                raise ValueError(msg)
            return {"dense": again[0]}

        results.append(_run(f"embeddings(openai)/determinism · {model}", refetch, verbose))

    if batch > 0 and shape.ok:
        name = f"embeddings(openai)/throughput · {model}"
        inputs = [f"doc {i} about search and apple silicon inference" for i in range(batch)]
        t0 = time.monotonic()
        try:
            _fetch_openai_embeddings(endpoint, api_key, model, inputs, timeout_s)
            wall_s = time.monotonic() - t0
            docs_s = batch / max(wall_s, 1e-9)
            lines = [
                f"  wall: {BOLD}{_ft(wall_s)}{RESET}",
                f"  throughput: {batch} docs -> {docs_s:.0f} docs/s",
                f"  {GREEN}OK{RESET}",
            ]
            results.append(TestResult(name, wall_s, True, lines))
        except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
            wall_s = time.monotonic() - t0
            results.append(TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e)))
    return results


def _run_openai_rerank(endpoint: str, api_key: str | None, model: str, timeout_s: float, verbose: bool) -> TestResult:
    """Probe the OpenAI/Cohere /v1/rerank surface: two ranked results, descending."""
    name = f"rerank(openai) · {model}"
    t0 = time.monotonic()
    try:
        response = _post_json_ready(
            endpoint,
            "/v1/rerank",
            {"model": model, "query": RERANK_QUERY, "documents": RERANK_DOCUMENTS, "top_n": 2},
            api_key,
            timeout_s,
        )
        wall_s = time.monotonic() - t0
        if response.status_code != 200:
            msg = f"HTTP {response.status_code}: {response.text[:300]}"
            raise ValueError(msg)
        ranked = response.json().get("results") or []
        if len(ranked) != 2:
            msg = f"expected 2 reranked results, got {len(ranked)}"
            raise ValueError(msg)
        if ranked[0]["relevance_score"] < ranked[1]["relevance_score"]:
            msg = "rerank results are not sorted by relevance_score"
            raise ValueError(msg)
        scores = ", ".join(f"[{r.get('index')}]={r['relevance_score']:.4f}" for r in ranked)
        lines = [f"  wall: {BOLD}{_ft(wall_s)}{RESET}", f"  ranking: {scores}"]
        if verbose:
            lines.append(f"  {DIM}query: {RERANK_QUERY!r}{RESET}")
            for doc in RERANK_DOCUMENTS:
                lines.append(f"  {DIM}  doc: {doc!r}{RESET}")
        lines.append(f"  {GREEN}OK{RESET}")
        return TestResult(name, wall_s, True, lines)
    except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
        wall_s = time.monotonic() - t0
        return TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e))


def _run_stream_generate(
    client: SIEClient, model: str, prompt: str, generate_tokens: int, timeout: float, verbose: bool
) -> TestResult:
    """Consume the SIE-native /v1/generate SSE stream and assert its contract."""
    name = f"generate/native-stream · {model}"
    t0 = time.monotonic()
    try:
        parts: list[str] = []
        terminal: dict[str, Any] = {}
        for chunk in client.stream_generate(
            model, prompt, max_new_tokens=generate_tokens, wait_for_capacity=True, provision_timeout_s=timeout
        ):
            if chunk.get("text_delta"):
                parts.append(chunk["text_delta"])
            if chunk.get("done"):
                terminal = chunk
        wall_s = time.monotonic() - t0
        text = "".join(parts)
        if not text:
            msg = "stream produced no text"
            raise ValueError(msg)
        if not terminal:
            msg = "stream ended without a terminal done chunk"
            raise ValueError(msg)
        finish_reason = terminal.get("finish_reason")
        if finish_reason not in ("stop", "length"):
            msg = f"unexpected finish_reason: {finish_reason!r}"
            raise ValueError(msg)
        data: dict[str, Any] = {"text": text, "usage": terminal.get("usage"), "_prompt": prompt}
        if terminal.get("ttft_ms") is not None:
            data["ttft_ms"] = terminal["ttft_ms"]
        return TestResult(name, wall_s, True, _format_result(name, data, wall_s, verbose))
    except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
        wall_s = time.monotonic() - t0
        return TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e))


def _run_stream_chat(
    client: SIEClient, model: str, prompt: str, generate_tokens: int, timeout: float, verbose: bool
) -> TestResult:
    """Consume the OpenAI chat SSE stream; report tok/s from the usage event."""
    name = f"generate/chat-stream · {model}"
    t0 = time.monotonic()
    try:
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        for chunk in client.stream_chat_completions(
            model,
            _chat_messages(prompt),
            max_tokens=generate_tokens,
            stream_options={"include_usage": True},
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        ):
            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                # Thinking models (e.g. Qwen3.5) stream `reasoning` deltas
                # before any `content`; a short smoke budget can finish inside
                # the reasoning phase, so both count as streamed output.
                if isinstance(delta, dict):
                    for field in ("content", "reasoning"):
                        if delta.get(field):
                            parts.append(delta[field])
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
        wall_s = time.monotonic() - t0
        text = "".join(parts)
        if not text:
            msg = "chat stream produced no content or reasoning deltas"
            raise ValueError(msg)
        lines = _format_result(name, {"text": text, "usage": usage, "_prompt": prompt}, wall_s, verbose)
        completion_tokens = (usage or {}).get("completion_tokens")
        if completion_tokens:
            tok_s = completion_tokens / max(wall_s, 1e-9)
            lines.insert(-1, f"  generation: {completion_tokens} tokens -> {tok_s:.1f} tok/s (incl. network)")
        else:
            lines.insert(-1, f"  {DIM}generation: (no usage event from child){RESET}")
        return TestResult(name, wall_s, True, lines)
    except Exception as e:  # noqa: BLE001 — smoketest must capture all failures
        wall_s = time.monotonic() - t0
        return TestResult(name, wall_s, False, [f"  {RED}FAILED: {e}{RESET}"], str(e))


def _format_result(name: str, data: dict[str, Any], wall_s: float, verbose: bool) -> list[str]:
    """Build display lines from a result dict."""
    lines = [f"  wall: {BOLD}{_ft(wall_s)}{RESET}"]

    # Timing (encode only)
    timing = data.get("timing")
    if timing:
        server_ms = timing.get("total_ms")
        if server_ms is not None:
            parts = [
                f"{k.replace('_ms', '')}={_fms(timing[k])}"
                for k in ("queue_ms", "tokenization_ms", "inference_ms")
                if timing.get(k) is not None
            ]
            lines.append(f"  server: {_fms(server_ms)} ({', '.join(parts)})")
            overhead = wall_s - server_ms / 1000
            if overhead > 1.0:
                lines.append(f"  {YELLOW}overhead: {_ft(overhead)} (scale-up + model load + network){RESET}")

    # Encode output shape
    if "dense" in data:
        parts = [f"dense={len(data['dense'])}d"]
        if "sparse" in data:
            parts.append("sparse")
        if "multivector" in data:
            s = data["multivector"].shape
            parts.append(f"multivector={s[0]}x{s[1]}")
        lines.append(f"  output: {', '.join(parts)}")

    # Score results
    scores = data.get("scores")
    if scores:
        ranked = ", ".join(f"{s['item_id']}={s['score']:.4f}" for s in scores)
        ok = scores[0]["item_id"] == "relevant"
        check = f"{GREEN}correct{RESET}" if ok else f"{YELLOW}unexpected{RESET}"
        lines.append(f"  ranking: {check} ({ranked})")

    # Extract results
    entities = data.get("entities")
    if entities:
        ents = ", ".join(f"{e['text']}[{e['label']}]" for e in entities)
        lines.append(f"  entities: {ents}")

    # Generate results
    text = data.get("text")
    usage = data.get("usage")
    if isinstance(text, str):
        preview = " ".join(text.split())
        if len(preview) > 120:
            preview = f"{preview[:117]}..."
        lines.append(f"  text: {preview!r}")
        if isinstance(usage, dict):
            parts = [
                f"{name}={usage[name]}"
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                if usage.get(name) is not None
            ]
            if parts:
                lines.append(f"  usage: {', '.join(parts)}")
        timing_parts = [f"{name}={_fms(data[name])}" for name in ("ttft_ms", "tpot_ms") if data.get(name) is not None]
        if timing_parts:
            lines.append(f"  generation: {', '.join(timing_parts)}")

    # Verbose: query details
    if verbose:
        if "dense" in data:
            lines.append(f"  {DIM}query: {ENCODE_QUERY['text']!r}{RESET}")
        elif scores:
            lines.append(f"  {DIM}query: {SCORE_QUERY['text']!r}{RESET}")
            for item in SCORE_ITEMS:
                lines.append(f"  {DIM}  item[{item['id']}]: {item['text']!r}{RESET}")
        elif entities:
            lines.append(f"  {DIM}query: {EXTRACT_QUERY['text']!r}{RESET}")
            lines.append(f"  {DIM}labels: {EXTRACT_LABELS}{RESET}")
        elif isinstance(text, str):
            prompt = data.get("_prompt", GENERATE_PROMPT)
            lines.append(f"  {DIM}prompt: {prompt!r}{RESET}")

    lines.append(f"  {GREEN}OK{RESET}")
    return lines


# ---------------------------------------------------------------------------
# Iteration runner
# ---------------------------------------------------------------------------


def run_iteration(
    client: SIEClient,
    endpoint: str,
    api_key: str | None,
    gpu_agnostic_route: str | None,
    models: dict[str, str],
    timeout: float,
    verbose: bool,
    include_generate: bool,
    only_generate: bool,
    generate_prompt: str,
    generate_tokens: int,
    generate_mode: str,
    gpu_agnostic_chat_probe: bool,
    only_capacity_probe: bool,
    check_docs: bool,
    skip_extract: bool,
    rerank_model: str,
    openai_embeddings: bool,
    embeddings_determinism: bool,
    embeddings_batch: int,
    generate_streaming: bool,
) -> list[TestResult]:
    """Run one smoke-test pass, optionally including or limiting to generate."""
    results: list[TestResult] = []
    if check_docs:
        results.append(_run_check_docs(endpoint, verbose))
    if not only_generate:
        # The raw-HTTP OpenAI probes run BEFORE their SDK siblings so they make
        # the cold first touch of each model: /v1/embeddings exercises the
        # blocking lazy load and /v1/rerank the 503 MODEL_LOADING retry loop
        # After an SDK probe the model would be warm and the retry
        # path dead in every live run.
        if openai_embeddings or embeddings_determinism or embeddings_batch > 0:
            results.extend(
                _run_openai_embeddings(
                    endpoint,
                    api_key,
                    models["encode"],
                    timeout,
                    verbose,
                    embeddings_determinism,
                    embeddings_batch,
                )
            )
        results.append(
            _run(
                f"encode · {models['encode']}",
                lambda: client.encode(
                    models["encode"],
                    ENCODE_QUERY,
                    wait_for_capacity=True,
                    provision_timeout_s=timeout,
                ),
                verbose,
            )
        )
        if rerank_model:
            results.append(_run_openai_rerank(endpoint, api_key, rerank_model, timeout, verbose))
        results.append(
            _run(
                f"score · {models['score']}",
                lambda: client.score(
                    models["score"],
                    SCORE_QUERY,
                    SCORE_ITEMS,
                    wait_for_capacity=True,
                    provision_timeout_s=timeout,
                ),
                verbose,
            )
        )
        if not skip_extract:
            results.append(
                _run(
                    f"extract · {models['extract']}",
                    lambda: client.extract(
                        models["extract"],
                        EXTRACT_QUERY,
                        labels=EXTRACT_LABELS,
                        wait_for_capacity=True,
                        provision_timeout_s=timeout,
                    ),
                    verbose,
                )
            )
    if include_generate:
        if gpu_agnostic_chat_probe:
            results.append(
                _run_gpu_agnostic_chat_probe(
                    endpoint,
                    api_key,
                    gpu_agnostic_route,
                    models["generate"],
                    generate_prompt,
                    generate_tokens,
                    timeout,
                    verbose,
                )
            )
        if only_capacity_probe:
            return results
        if generate_mode in {"native", "both"}:
            results.append(
                _run(
                    f"generate/native · {models['generate']}",
                    lambda: _validated_generate(
                        client.generate(
                            models["generate"],
                            generate_prompt,
                            max_new_tokens=generate_tokens,
                            wait_for_capacity=True,
                            provision_timeout_s=timeout,
                        ),
                        generate_prompt,
                    ),
                    verbose,
                )
            )
            if generate_streaming:
                results.append(
                    _run_stream_generate(client, models["generate"], generate_prompt, generate_tokens, timeout, verbose)
                )
        if generate_mode in {"chat", "both"}:
            results.append(
                _run(
                    f"generate/chat · {models['generate']}",
                    lambda: _validated_chat_completion(
                        client.chat_completions(
                            models["generate"],
                            _chat_messages(generate_prompt),
                            max_tokens=generate_tokens,
                            wait_for_capacity=True,
                            provision_timeout_s=timeout,
                        ),
                        generate_prompt,
                    ),
                    verbose,
                )
            )
            if generate_streaming:
                results.append(
                    _run_stream_chat(client, models["generate"], generate_prompt, generate_tokens, timeout, verbose)
                )
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(all_iterations: list[list[TestResult]], total_s: float) -> int:
    """Print summary table across all iterations. Returns exit code."""
    n_iter = len(all_iterations)
    passed = failed = 0

    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}Summary ({n_iter} iteration{'s' if n_iter > 1 else ''}){RESET}")
    print(f"{'=' * 65}")

    if n_iter == 1:
        for r in all_iterations[0]:
            s = f"{GREEN}PASS{RESET}" if r.ok else f"{RED}FAIL{RESET}"
            print(f"  {s}  {r.name:<45} {_ft(r.wall_s):>8}")
            passed += r.ok
            failed += not r.ok
    else:
        # Group by test name across iterations. Ordered union rather than just
        # iteration 0: conditional probes (determinism/throughput run only when
        # the shape probe passed) may be absent from any one iteration.
        test_names = list(dict.fromkeys(r.name for it in all_iterations for r in it))
        for name in test_names:
            times = [r.wall_s for it in all_iterations for r in it if r.name == name and r.ok]
            fails = sum(1 for it in all_iterations for r in it if r.name == name and not r.ok)
            passed += len(times)
            failed += fails
            if times:
                avg = sum(times) / len(times)
                mn, mx = min(times), max(times)
                stats = f"avg={_ft(avg)}  min={_ft(mn)}  max={_ft(mx)}"
                print(f"  {GREEN}PASS{RESET}  {name:<40} {stats}")
            if fails:
                print(f"  {RED}FAIL{RESET}  {name:<40} ({fails} failures)")

    print(f"{'─' * 65}")
    print(f"  total: {BOLD}{_ft(total_s)}{RESET}")

    # Cold start info from first iteration
    # The /docs probe is a readiness check, not a model request — skip it so
    # the cold-start metric reflects the first real inference call.
    first = next((r for r in all_iterations[0] if r.name != "GET /docs"), all_iterations[0][0])
    if first.ok:
        print(f"  first request: {BOLD}{_ft(first.wall_s)}{RESET} (includes cold start)")

    if n_iter > 1:
        warm = [r.wall_s for it in all_iterations[1:] for r in it if r.ok]
        if warm:
            print(f"  warm avg: {_ft(sum(warm) / len(warm))}")

    status = (
        f"{GREEN}{passed} passed{RESET}" if not failed else f"{GREEN}{passed} passed{RESET} {RED}{failed} failed{RESET}"
    )
    print(f"\n  {BOLD}{status}{RESET}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    endpoint = get_usage_flag("endpoint") or ""
    gpu = get_usage_flag("gpu") or None
    iterations = get_usage_int("iterations", 1)
    timeout = float(get_usage_int("timeout", 600))
    if iterations < 1:
        log_error("--iterations must be >= 1")
        return 1
    if timeout < 1:
        log_error("--timeout must be >= 1")
        return 1
    verbose = get_usage_bool("verbose")
    include_generate = get_usage_bool("generate")
    only_generate = get_usage_bool("only_generate")
    only_capacity_probe = get_usage_bool("only_capacity_probe")
    if only_generate:
        include_generate = True
    if only_capacity_probe:
        include_generate = True
        only_generate = True
    gpu_agnostic_chat_probe = get_usage_bool("gpu_agnostic_chat_probe") or only_capacity_probe
    check_docs = get_usage_bool("check_docs")
    skip_extract = get_usage_bool("skip_extract")
    rerank_model = get_usage_flag("rerank_model") or ""
    embeddings_determinism = get_usage_bool("embeddings_determinism")
    embeddings_batch = get_usage_int("embeddings_batch", 0)
    openai_embeddings = get_usage_bool("openai_embeddings") or embeddings_determinism or embeddings_batch > 0
    generate_streaming = get_usage_bool("generate_streaming")
    generate_model = get_usage_flag("generate_model") or ""
    generate_prompt = get_usage_flag("generate_prompt") or GENERATE_PROMPT
    generate_mode = (get_usage_flag("generate_mode") or "native").strip().lower()
    if generate_mode not in GENERATE_MODES:
        log_error("--generate-mode must be one of: native, chat, both")
        return 1
    generate_tokens_raw = get_usage_flag("generate_tokens") or "64"
    try:
        generate_tokens = int(generate_tokens_raw)
    except ValueError:
        log_error("--generate-tokens must be a positive integer")
        return 1
    if include_generate and generate_tokens < 1:
        log_error("--generate-tokens must be >= 1")
        return 1
    if include_generate and not generate_model:
        log_error("--generate-model is required when using --generate, --only-generate, or --only-capacity-probe")
        return 1
    models = {
        "encode": get_usage_flag("model") or "sentence-transformers/all-MiniLM-L6-v2",
        "score": get_usage_flag("score_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "extract": get_usage_flag("extract_model") or "urchade/gliner_small-v2.1",
        "generate": generate_model,
    }

    if not endpoint:
        log_error("Endpoint URL is required")
        return 1

    print(f"{BOLD}SIE Cluster Smoke Test{RESET}")
    print(f"  endpoint:   {endpoint}")
    print(f"  gpu:        {gpu or '(any)'}")
    print(f"  iterations: {iterations}")
    print(f"  timeout:    {timeout:.0f}s")
    print(f"  auth:       {'SIE_API_KEY' if os.environ.get('SIE_API_KEY') else '(none)'}")
    if rerank_model:
        print(f"  rerank:     {rerank_model} (OpenAI /v1/rerank)")
    if openai_embeddings:
        extras = [
            part
            for part, on in (
                ("determinism", embeddings_determinism),
                (f"batch={embeddings_batch}", embeddings_batch > 0),
            )
            if on
        ]
        note = f" (+{', '.join(extras)})" if extras else ""
        print(f"  embeddings: {models['encode']} (OpenAI /v1/embeddings){note}")
    if include_generate:
        streaming_note = " +streaming" if generate_streaming else ""
        print(f"  generate:   {models['generate']} ({generate_tokens} tokens, {generate_mode}{streaming_note})")
    if gpu_agnostic_chat_probe:
        probe_route = _gpu_agnostic_route(gpu)
        pool_note = f", pool={probe_route[:-1]}" if probe_route else ""
        print(f"  chat probe: gpu-agnostic{pool_note}, OpenAI HTTP not-ready contract")
    if only_capacity_probe:
        print("  mode:       capacity probe only")
    if only_generate:
        print("  mode:       generate only")
    if verbose:
        print("  verbose:    on")
    print(f"  {DIM}(will wait for capacity if cluster is scaling up){RESET}")

    api_key = os.environ.get("SIE_API_KEY")
    gpu_agnostic_route = _gpu_agnostic_route(gpu)
    client = SIEClient(endpoint, timeout_s=timeout, gpu=gpu, api_key=api_key)
    all_iterations: list[list[TestResult]] = []

    t_start = time.monotonic()
    for i in range(iterations):
        if iterations > 1:
            print(f"\n{BOLD}── Iteration {i + 1}/{iterations} ──{RESET}")

        results = run_iteration(
            client,
            endpoint,
            api_key,
            gpu_agnostic_route,
            models,
            timeout,
            verbose,
            include_generate,
            only_generate,
            generate_prompt,
            generate_tokens,
            generate_mode,
            gpu_agnostic_chat_probe,
            only_capacity_probe,
            check_docs,
            skip_extract,
            rerank_model,
            openai_embeddings,
            embeddings_determinism,
            embeddings_batch,
            generate_streaming,
        )
        all_iterations.append(results)

        for r in results:
            print(f"\n{BOLD}{CYAN}[{r.name}]{RESET}")
            for line in r.lines:
                print(line)

    total_s = time.monotonic() - t_start
    return print_summary(all_iterations, total_s)


if __name__ == "__main__":
    sys.exit(main())
