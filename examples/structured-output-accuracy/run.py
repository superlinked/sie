#!/usr/bin/env python3
"""Print or send the structured-output requests, one model at a time.

    uv run run.py --show e1 bf462440                       # the SIE request for one record, no network
    uv run run.py --show nhtsa nhtsa-11763784 --model claude-sonnet-5
    uv run run.py --record --set e1 --limit 20             # SIE, 20 records, needs SIE_API_KEY
    uv run run.py --record --set nhtsa --model gpt-6-luna  # a rival arm, needs OPENAI_API_KEY
    uv run score.py --calls runs                           # score what you recorded

Every request carries the same system prompt, user message and JSON Schema; only the vendor's
structured-output switch and the sampling settings differ:

    SIE        POST /v1/chat/completions, response_format json_schema strict, temperature 0,
               through sie_sdk.SIEClient; SIE_API_KEY, and SIE_BASE_URL for another SIE server
               (default https://api.superlinked.com)
    OpenAI     Chat Completions, response_format json_schema strict, reasoning_effort none,
               temperature 0; OPENAI_API_KEY
    Anthropic  Messages, output_config.format json_schema; ANTHROPIC_API_KEY. Sonnet 5 and 5.5 with
               thinking disabled (they reject temperature), Opus 5.5 at effort low (its thinking cannot
               be disabled), Haiku 4.5 at temperature 0

All calls stream, so the rows carry time to first token as well as time to the whole answer, with a
2,048-token output cap. A transient failure (429, 5xx, timeout, Anthropic's grammar-compilation rate
limit) is retried up to six times. A schema the vendor rejects is recorded as `rejected` and scores 0.

--sdk-transform passes an Anthropic arm's schema through the anthropic SDK's transform_schema first,
as its messages.parse does; the recorded NHTSA run has that variant because output_config.format
rejects the NHTSA schema as sent.

Rows go to runs/<set>/<model>.jsonl in the recorded run's format. A rerun skips records that
already have an answer, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import score

HERE = Path(__file__).resolve().parent
SIE_BASE_URL = "https://api.superlinked.com"
PATH = "/v1/chat/completions"
MAX_TOKENS = 2048
MAX_ATTEMPTS = 6
GRAMMAR_LIMIT = "Grammar compilation rate limit"
HTTP_TIMEOUT = 408
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
# Each model's thinking-off setting. Sonnet 5.5 rejects "disabled" and names "between_tools" instead.
THINKING_OFF = {"claude-sonnet-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


def provider_of(model: str, override: str | None) -> str:
    if override:
        return override
    if model == score.SIE_MODEL:
        return "sie"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    raise SystemExit(f"{model} is not one of the recorded arms; pass --provider")


def type_lists_to_any_of(schema: Any) -> Any:
    """`"type": ["integer", "null"]` as Pydantic writes `int | None`: an anyOf, which transform_schema accepts."""
    if isinstance(schema, list):
        return [type_lists_to_any_of(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: type_lists_to_any_of(v) for k, v in schema.items()}
    types = out.get("type")
    if isinstance(types, list):
        rest = {k: v for k, v in out.items() if k != "type"}
        variants = []
        for t in types:
            variant = {"type": t}
            if t != "null":
                variant |= {k: v for k, v in rest.items() if k not in {"description", "title"}}
            variants.append(variant)
        out = {k: v for k, v in rest.items() if k in {"description", "title"}} | {"anyOf": variants}
    return out


def anthropic_schema(schema: dict[str, Any], sdk_transform: bool) -> dict[str, Any]:
    if not sdk_transform:
        return schema
    # Optional: only this variant needs the anthropic SDK's private helper, the one messages.parse uses.
    # It cannot read a type list such as ["integer", "null"], so those become the anyOf Pydantic emits first.
    from anthropic.lib._parse._transform import transform_schema

    return transform_schema(type_lists_to_any_of(schema))


def request_body(model: str, provider: str, item: dict[str, Any], sdk_transform: bool = False) -> dict[str, Any]:
    """The request as each vendor's structured-output API takes it, before streaming is switched on."""
    if provider in {"sie", "openai"}:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}],
            "max_completion_tokens": MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": item["schema"], "strict": True},
            },
        }
        if provider == "openai" and model.startswith("gpt-"):
            body["reasoning_effort"] = "none"
        body["temperature"] = 0
        return body
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": item["system"],
        "messages": [{"role": "user", "content": item["user"]}],
        "output_config": {"format": {"type": "json_schema", "schema": anthropic_schema(item["schema"], sdk_transform)}},
    }
    if model in THINKING_OFF:
        body["thinking"] = THINKING_OFF[model]
    elif model == "claude-opus-5-5":
        body["output_config"]["effort"] = "low"
    else:
        body["temperature"] = 0
    return body


def wire_body(provider: str, body: dict[str, Any]) -> dict[str, Any]:
    """What goes over the wire: the same body with streaming switched on."""
    if provider == "anthropic":
        return body | {"stream": True}
    return body | {"stream": True, "stream_options": {"include_usage": True}}


class Backend:
    """One streamed call: text, token counts and timings."""

    def __init__(self, model: str, provider: str, base_url: str | None) -> None:
        self.model = model
        self.provider = provider
        if provider == "sie":
            from sie_sdk import SIEClient

            self.base_url = base_url or os.environ.get("SIE_BASE_URL", SIE_BASE_URL)
            self.client: Any = SIEClient(self.base_url, api_key=require("SIE_API_KEY"), timeout_s=600)
        elif provider == "openai":
            import openai

            self.base_url = base_url or "https://api.openai.com/v1"
            self.client = openai.OpenAI(
                api_key=require("OPENAI_API_KEY"), base_url=self.base_url, max_retries=0, timeout=300
            )
        else:
            import anthropic

            self.base_url = "https://api.anthropic.com"
            self.client = anthropic.Anthropic(api_key=require("ANTHROPIC_API_KEY"), max_retries=0, timeout=300)

    def call(self, body: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        first: float | None = None
        text = ""
        if self.provider == "sie":
            usage: dict[str, Any] = {}
            for chunk in self.client.stream_chat_completions(
                body["model"],
                body["messages"],
                max_completion_tokens=body["max_completion_tokens"],
                response_format=body["response_format"],
                temperature=body["temperature"],
                stream_options={"include_usage": True},
            ):
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta and first is None:
                        first = time.perf_counter() - start
                    text += delta
            return {
                "text": text,
                "tokens_in": int(usage.get("prompt_tokens") or 0),
                "tokens_cached": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "tokens_out": int(usage.get("completion_tokens") or 0),
                "tokens_reasoning": 0,
                "ttft_s": first,
                "total_s": time.perf_counter() - start,
            }
        if self.provider == "openai":
            stream = self.client.chat.completions.create(**wire_body("openai", body))
            usage_obj = None
            for chunk in stream:
                usage_obj = chunk.usage or usage_obj
                for choice in chunk.choices:
                    delta = choice.delta.content or ""
                    if delta and first is None:
                        first = time.perf_counter() - start
                    text += delta
            prompt = getattr(usage_obj, "prompt_tokens_details", None)
            completion = getattr(usage_obj, "completion_tokens_details", None)
            return {
                "text": text,
                "tokens_in": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "tokens_cached": getattr(prompt, "cached_tokens", 0) or 0,
                "tokens_out": getattr(usage_obj, "completion_tokens", 0) or 0,
                "tokens_reasoning": getattr(completion, "reasoning_tokens", 0) or 0,
                "ttft_s": first,
                "total_s": time.perf_counter() - start,
            }
        # The SDK's messages.stream takes no temperature argument, so it goes in the body as the API accepts it.
        params = dict(body)
        extra = {"temperature": params.pop("temperature")} if "temperature" in params else None
        with self.client.messages.stream(**params, extra_body=extra) as stream:
            for event in stream:
                if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                    if first is None:
                        first = time.perf_counter() - start
                    text += event.delta.text
            message = stream.get_final_message()
        usage = message.usage
        cache_read = usage.cache_read_input_tokens or 0
        return {
            "text": text,
            "tokens_in": usage.input_tokens + cache_read + (usage.cache_creation_input_tokens or 0),
            "tokens_cached": cache_read,
            "tokens_out": usage.output_tokens,
            "tokens_reasoning": 0,
            "ttft_s": first,
            "total_s": time.perf_counter() - start,
            "stop_reason": message.stop_reason,
        }


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name}. To check the published figures without a key, run score.py instead.")
    return value


def is_transient(error: Exception) -> bool:
    if GRAMMAR_LIMIT in str(error):
        return True
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in {HTTP_TIMEOUT, HTTP_TOO_MANY_REQUESTS} or status >= HTTP_SERVER_ERROR
    return True


def is_schema_rejection(message: str) -> bool:
    return "output_config.format.schema" in message or "Invalid schema for response_format" in message


def send(backend: Backend, rid: str, body: dict[str, Any]) -> dict[str, Any]:
    error: str | None = None
    result: dict[str, Any] | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = backend.call(body)
            error = None
            break
        except (TypeError, AttributeError, KeyError, NameError):
            # A bug in this script, not a failed call: stop rather than record it against the model.
            raise
        except Exception as exc:  # noqa: BLE001 - classified here, recorded when final
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if not is_transient(exc) or attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(61 if GRAMMAR_LIMIT in error else min(60, 2 ** (attempt + 1)))
    row: dict[str, Any] = {"id": rid, "model": backend.model, "at": datetime.now(UTC).isoformat(), "error": error}
    if error and is_schema_rejection(error):
        # A schema the API refuses is an answer that scores 0, not a transport failure to retry later.
        row |= {"error": None, "rejected": error, "text": "", "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0}
        row |= {"tokens_reasoning": 0, "ttft_s": None, "total_s": None}
    if result is not None:
        row |= result
    return row


def find(items: dict[str, dict[str, Any]], prefix: str) -> str:
    matches = [rid for rid in items if rid.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"{prefix} matches {len(matches)} records; give more of the id")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", nargs=2, metavar=("SET", "ID"), help="print one request and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="send the requests (needs the provider's key)")
    parser.add_argument("--set", choices=("e1", "nhtsa", "repeat"), default="e1")
    parser.add_argument("--model", default=score.SIE_MODEL, help=f"default {score.SIE_MODEL}")
    parser.add_argument("--provider", choices=("sie", "openai", "anthropic"), help="needed for a model not listed")
    parser.add_argument("--base-url", help="another endpoint for the sie or openai provider")
    parser.add_argument("--sdk-transform", action="store_true", help="Anthropic: schema through transform_schema")
    parser.add_argument("--limit", type=int, help="only the first N records of the set")
    parser.add_argument("--concurrency", type=int, default=6, help="requests in flight; the repeat set uses 1")
    parser.add_argument("--output", type=Path, help="default runs/<set>/<model>.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.show:
        set_name, prefix = args.show
        if set_name not in {"e1", "nhtsa", "repeat"}:
            raise SystemExit("the sets are e1, nhtsa and repeat")
        score.verify_evidence()
        items = score.items_for(set_name)
        provider = provider_of(args.model, args.provider)
        body = request_body(args.model, provider, items[find(items, prefix)], args.sdk_transform)
        url = {
            "sie": f"{SIE_BASE_URL}{PATH}",
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/messages",
        }[provider]
        print(
            json.dumps({"method": "POST", "url": url, "body": wire_body(provider, body)}, indent=2, ensure_ascii=False)
        )
        return 0
    if not args.record:
        raise SystemExit("pass --show SET ID to print a request, or --record to send them")

    score.verify_evidence()
    items = score.items_for(args.set)
    ids = list(items)[: args.limit] if args.limit else list(items)
    provider = provider_of(args.model, args.provider)
    if args.sdk_transform and provider != "anthropic":
        raise SystemExit("--sdk-transform applies to Anthropic arms only")
    stem = args.model.replace("/", "_") + ("__sdk-transform" if args.sdk_transform else "")
    output = args.output or HERE / "runs" / args.set / f"{stem}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {rid for rid, row in score.scored_rows(output).items() if row.get("error") is None}
        if output.exists()
        else set()
    )
    todo = [rid for rid in ids if rid not in done]

    backend = Backend(args.model, provider, args.base_url)
    print(f"{args.model} via {provider} at {backend.base_url}: {args.set}, {len(ids)} records, {len(todo)} to send")
    lock = Lock()
    failed = 0

    def one(rid: str) -> None:
        nonlocal failed
        row = send(backend, rid, request_body(args.model, provider, items[rid], args.sdk_transform))
        with lock, output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            failed += row["error"] is not None
            print(
                f"  {rid[:16]} {'error' if row['error'] else 'rejected' if row.get('rejected') else 'ok'}", flush=True
            )

    workers = 1 if args.set == "repeat" else max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    print(f"{len(todo)} sent, {failed} failed -> {output}")
    print("Next: uv run score.py --calls runs")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
