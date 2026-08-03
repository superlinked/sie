"""Mac-local OpenAI-compatible surfaces: ``/v1/chat/completions`` + ``/v1/rerank``.

These complete the local OpenAI surface for Apple-Silicon serving (alongside the
existing ``/v1/embeddings``), so any OpenAI/Cohere client or tool can hit the
local endpoint.

- ``/v1/chat/completions`` proxies the managed MLX generation subprocess
  (``mlx_lm.server``). Chat templating, streaming, and structured output come
  free from the child; we just resolve the model, ensure it is loaded, and
  stream-proxy. Only available where generation runs via MLX (Apple Silicon) —
  elsewhere it returns 501 (the production ingress is the Rust gateway).
- ``/v1/rerank`` wraps the in-process score adapter in the Cohere/OpenAI rerank
  shape (``{query, documents, top_n}`` -> ``{results: [{index, relevance_score}]}``).

These routes are mounted on the worker's own FastAPI app and are a local-dev /
single-node convenience, mirroring the existing direct ``/v1/generate`` and
``/v1/embeddings`` routes — no traffic reaches them in a real cluster.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sie_sdk.queue_types import denormalize_model_id

from sie_server.adapters.mlx.generation import MLXGenerationAdapter, normalize_mlx_seed
from sie_server.api.helpers import ModelStateChecker
from sie_server.api.options import resolve_runtime_options
from sie_server.api.score import score_usage_from_output
from sie_server.api.validation import validate_machine_profile_header, validate_signed_i64
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.score_cost import MAX_SCORE_ITEMS, build_score_prepared_items
from sie_server.core.timing import RequestTiming
from sie_server.observability.tracing import tracer
from sie_server.types.inputs import Item
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# Inter-chunk read timeout — bounded so a wedged child can't hang the request forever
# (this proxy is the direct ingress on Mac; there is no worker/abort layer behind it).
# Override via SIE_MLX_READ_TIMEOUT_S. Connect/write/pool stay short.
_PROXY_READ_TIMEOUT_S = float(os.environ.get("SIE_MLX_READ_TIMEOUT_S", "300"))
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=_PROXY_READ_TIMEOUT_S, write=10.0, pool=10.0)
# Cap the request body so a huge messages/documents array can't be buffered/forwarded
# unbounded (the gateway caps prod ingress; these local routes must cap themselves).
_MAX_BODY_BYTES = int(os.environ.get("SIE_CHAT_MAX_BODY_BYTES", str(8 * 1024 * 1024)))
# Compatibility requests project onto the same native score bound.
_MAX_RERANK_DOCS = MAX_SCORE_ITEMS


def _bad_request(message: str, *, param: str | None = None, code: str | None = None) -> HTTPException:
    detail: dict[str, Any] = {"code": code or ErrorCode.INVALID_INPUT.value, "message": message}
    if param is not None:
        detail["param"] = param
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _validate_mlx_seed(value: Any) -> int | None:
    """Validate the public signed-i64 seed and convert it for MLX's uint64 API."""
    try:
        validated = validate_signed_i64(value, param="seed")
    except ValueError as exc:
        raise _bad_request(str(exc), param="seed") from exc
    return None if validated is None else normalize_mlx_seed(validated)


async def _read_json_body(http_request: Request) -> dict[str, Any]:
    """Size-cap + parse the request body as a JSON object.

    These routes are direct local ingress (no gateway in front), so they bound the
    buffered body themselves: reject early on Content-Length, then on the real read.
    """
    clen = http_request.headers.get("content-length")
    if clen is not None and clen.isdigit() and int(clen) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": ErrorCode.INPUT_TOO_LONG.value, "message": "request body too large"},
        )
    raw = await http_request.body()
    if len(raw) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": ErrorCode.INPUT_TOO_LONG.value, "message": "request body too large"},
        )
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise _bad_request("request body must be a JSON object") from exc
    if not isinstance(body, dict):
        raise _bad_request("request body must be a JSON object")
    return body


def _unsupported_chat(model: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "unsupported_operation",
            "message": (
                f"/v1/chat/completions for '{model}' is only served by the local MLX "
                "generation backend (Apple Silicon). Use the Rust gateway in a cluster."
            ),
        },
    )


# -- /v1/chat/completions (proxy to the MLX subprocess) ----------------------


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> Response | StreamingResponse:
    """OpenAI-compatible chat completions, proxied to the local MLX subprocess.

    Resolves the requested model, ensures it is loaded, then forwards the body
    to the child ``mlx_lm.server``'s ``/v1/chat/completions`` (rewriting the
    ``model`` field to the child's served MLX repo so it does not try to hot-swap
    models). Supports both streaming (SSE pass-through) and non-streaming.
    """
    validate_machine_profile_header(x_machine_profile)

    body = await _read_json_body(http_request)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _bad_request("'model' must be a non-empty string", param="model")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise _bad_request("'messages' must be a non-empty array", param="messages")
    # Validate the fields this route interprets locally (the rest pass through to
    # mlx_lm.server, which validates them) — reject rather than coerce, matching the gateway
    # parser. `bool("false")` is True, so an unvalidated `stream` would silently stream; an
    # invalid `max_tokens` must 400 rather than be silently clamped to the model cap.
    stream_opt = body.get("stream")
    if stream_opt is not None and not isinstance(stream_opt, bool):
        raise _bad_request("'stream' must be a boolean", param="stream")
    max_tokens_opt = body.get("max_tokens")
    if max_tokens_opt is not None and (
        isinstance(max_tokens_opt, bool) or not isinstance(max_tokens_opt, int) or max_tokens_opt <= 0
    ):
        raise _bad_request("'max_tokens' must be a positive integer", param="max_tokens")
    mlx_seed = _validate_mlx_seed(body.get("seed"))

    registry = http_request.app.state.registry
    device = registry.device
    registry_key = denormalize_model_id(model)

    with tracer.start_as_current_span("chat_completions") as span:
        span.set_attribute("model", model)
        checker = ModelStateChecker(registry, registry_key, span)
        checker.check_exists()

        # Validate generation capability + backend BEFORE loading: a chat request for an
        # embedding/reranker model (or on a non-MLX/CUDA device) must fail fast instead of
        # kicking off a real model load/download that only 501s afterwards. Mirrors the
        # rerank route's tasks.score gate.
        config = registry.get_config(registry_key)
        gen_task = getattr(config.tasks, "generate", None)
        if gen_task is None:
            raise _bad_request(
                f"Model '{model}' does not support generation (no generate task). Use a generation model."
            )
        if str(device).startswith("cuda"):
            raise _unsupported_chat(model)

        checker.check_not_failed()
        checker.check_not_unloading()
        checker.check_not_loading()
        await checker.ensure_loaded(device)

        adapter = registry.get(registry_key)
        server_url = getattr(adapter, "server_url", None)
        if not isinstance(adapter, MLXGenerationAdapter) or server_url is None:
            # Defensive backstop — the capability/device gates above catch the common cases.
            raise _unsupported_chat(model)

        # Rewrite the model to the child's served repo so mlx_lm.server does not attempt to
        # load a different model per request, and clamp max_tokens to the model's output cap
        # so a huge value can't drive an unbounded generation / OOM (gen_task is non-None here).
        proxied = dict(body)
        proxied["model"] = adapter.mlx_repo
        if mlx_seed is not None:
            proxied["seed"] = mlx_seed
        cap = gen_task.max_output_tokens
        # max_tokens_opt is a positive int or None (validated above). Make the per-model cap
        # authoritative: clamp a valid value, and fill the cap in when omitted so the child's
        # own (large) default can't bypass it.
        proxied["max_tokens"] = min(max_tokens_opt, cap) if max_tokens_opt is not None else cap
        url = f"{server_url}/v1/chat/completions"
        stream = bool(stream_opt)  # validated above (bool or None)
        # Single-flight: the child is single-tenant; serialize with /v1/generate.
        slot = adapter.generation_slot()

        if stream:

            async def _proxy_sse() -> Any:
                async with slot:
                    try:
                        async with (
                            httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client,
                            client.stream("POST", url, json=proxied) as resp,
                        ):
                            if resp.status_code != status.HTTP_200_OK:
                                # The 200 text/event-stream status is already committed, so
                                # surface the child error as a terminal SSE event. Do NOT call
                                # aiter_raw() after aread() (raises httpx.StreamConsumed).
                                preview = (await resp.aread())[:500]
                                logger.error("MLX chat proxy error %d: %s", resp.status_code, preview)
                                err = json.dumps(
                                    {
                                        "error": {
                                            "code": "upstream_error",
                                            "message": preview.decode("utf-8", "replace"),
                                            "status": resp.status_code,
                                        }
                                    }
                                )
                                yield f"data: {err}\n\n".encode()
                                yield b"data: [DONE]\n\n"
                                return
                            async for chunk in resp.aiter_raw():
                                yield chunk
                    except httpx.HTTPError as exc:
                        # Child died / connection reset mid-stream — emit a terminal error
                        # event so the client doesn't get a silently-truncated body.
                        logger.warning("MLX chat proxy stream error: %s", exc)
                        # Generic client message — detail is logged above; don't leak
                        # the exception text to the client (CodeQL info-exposure).
                        err = json.dumps({"error": {"code": "upstream_error", "message": "upstream stream error"}})
                        yield f"data: {err}\n\n".encode()
                        yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _proxy_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with slot, httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
            resp = await client.post(url, json=proxied)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )


# -- /v1/rerank (Cohere/OpenAI shape over the score adapter) -----------------


@router.post("/rerank", response_model=None)
async def rerank(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse:
    """Cohere/OpenAI-style reranking backed by the in-process score adapter.

    Request: ``{model, query, documents: [str], top_n?, return_documents?}``.
    Response: ``{model, results: [{index, relevance_score, document?}], usage}``
    sorted by descending relevance.
    """
    validate_machine_profile_header(x_machine_profile)

    body = await _read_json_body(http_request)
    allowed_fields = {"model", "query", "documents", "top_n", "return_documents", "options"}
    unknown_fields = body.keys() - allowed_fields
    if unknown_fields:
        unknown = sorted(unknown_fields)[0]
        raise _bad_request(f"field '{unknown}' is not supported", param=unknown)

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _bad_request("'model' must be a non-empty string", param="model")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise _bad_request("'query' must be a non-empty string", param="query")
    documents = body.get("documents")
    if not isinstance(documents, list) or not documents or not all(isinstance(d, str) and d.strip() for d in documents):
        raise _bad_request("'documents' must be a non-empty array of strings", param="documents")
    if len(documents) > _MAX_RERANK_DOCS:
        raise _bad_request(f"'documents' exceeds the maximum of {_MAX_RERANK_DOCS} per request", param="documents")
    top_n = body.get("top_n")
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0):
        raise _bad_request("'top_n' must be a positive integer", param="top_n")
    return_documents = body.get("return_documents", False)
    if not isinstance(return_documents, bool):
        raise _bad_request("'return_documents' must be a boolean", param="return_documents")

    registry = http_request.app.state.registry
    device = registry.device
    registry_key = denormalize_model_id(model)

    with tracer.start_as_current_span("rerank") as span:
        span.set_attribute("model", model)
        span.set_attribute("batch_size", len(documents))
        checker = ModelStateChecker(registry, registry_key, span)
        checker.check_exists()

        config = registry.get_config(registry_key)
        if config.tasks.score is None:
            raise _bad_request(
                f"Model '{model}' does not support reranking (no score task). Use a reranker model.",
            )

        checker.check_not_failed()
        checker.check_not_unloading()
        checker.check_not_loading()
        await checker.ensure_loaded(device)

        query_item = Item(text=query)
        doc_items = [Item(id=str(i), text=str(doc)) for i, doc in enumerate(documents)]
        options_raw = body.get("options")
        if options_raw is not None and not isinstance(options_raw, dict):
            raise _bad_request("'options' must be an object", param="options")
        options = resolve_runtime_options(config, options_raw, span)
        instruction = options.get("instruction")

        timing = RequestTiming()
        timing.start_tokenization()
        prepared_items = build_score_prepared_items(query_item, doc_items)
        timing.end_tokenization()
        worker = await registry.start_worker(registry_key)
        future = await worker.submit_score(
            prepared_items=prepared_items,
            query=query_item,
            items=doc_items,
            instruction=instruction,
            options=options,
            timing=timing,
        )
        try:
            worker_result = await future
        except Exception as exc:
            logger.warning("rerank failed for %s", model, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "inference_error", "message": str(exc)},
            ) from exc

        score_output: ScoreOutput = worker_result.output
        scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]
        usage = score_usage_from_output(score_output)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "inference_error", "message": "score response missing authoritative usage"},
            )

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    results: list[dict[str, Any]] = []
    for index, score in ranked:
        entry: dict[str, Any] = {"index": index, "relevance_score": score}
        if return_documents:
            entry["document"] = {"text": documents[index]}
        results.append(entry)

    return JSONResponse(
        content={
            "model": getattr(config, "name", None) or registry_key,
            "results": results,
            "usage": usage,
        }
    )
