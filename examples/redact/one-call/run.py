"""Send one complete text item to pii-redact and mask the returned span union."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import httpx

MODEL = "superlinked/pii-redact"
PROFILE = "default"
DIRECTORY = Path(__file__).resolve().parent
SAVED_EVIDENCE = "saved composition replay; not a one-call HTTP recording"
EXTRACT_PATH = f"/v1/extract/{MODEL}"


class ExampleError(Exception):
    """An incomplete or invalid result must never become masked output."""


class SingleAttemptError(ExampleError):
    """The optional one-POST illustration has stopped."""


def load_input() -> str:
    source = json.loads((DIRECTORY / "source.json").read_text(encoding="utf-8"))
    data = (DIRECTORY / "input.txt").read_bytes()
    if hashlib.sha256(data).hexdigest() != source["input"]["sha256"]:
        raise ExampleError("input.txt does not match the pinned complete input")
    text = data.decode("utf-8")
    if len(data) != source["input"]["utf8_bytes"] or len(text) != source["input"]["code_points"]:
        raise ExampleError("input.txt has an unexpected length")
    return text


def logical_request(text: str) -> dict[str, Any]:
    return {
        "method": "POST",
        "path": EXTRACT_PATH,
        "body": {"items": [{"text": text}], "params": {"options": {"profile": PROFILE}}},
    }


def validate_entities(text: str, result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("error") is not None:
        raise ExampleError("extraction returned an error or an invalid item")
    entities = result.get("entities")
    if not isinstance(entities, list):
        raise ExampleError("extraction did not return an entities list")
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ExampleError(f"entity {index} is not an object")
        start, end = entity.get("start"), entity.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ExampleError(f"entity {index} has invalid code-point offsets")
        if entity.get("text") != text[start:end]:
            raise ExampleError(f"entity {index} does not match the original text at its offsets")
        if not isinstance(entity.get("label"), str) or not entity["label"]:
            raise ExampleError(f"entity {index} has no label")
    return entities


def merged_intervals(entities: list[dict[str, Any]]) -> list[list[int]]:
    intervals: list[list[int]] = []
    for start, end in sorted((entity["start"], entity["end"]) for entity in entities):
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return intervals


def render_result(text: str, result: Any, evidence: str) -> dict[str, Any]:
    entities = validate_entities(text, result)
    intervals = merged_intervals(entities)
    pieces: list[str] = []
    cursor = 0
    for start, end in intervals:
        pieces.extend((text[cursor:start], "█" * (end - start)))
        cursor = end
    pieces.append(text[cursor:])
    return {
        "evidence": evidence,
        "span_status": "spans returned" if entities else "no spans returned",
        "entities": entities,
        "merged_intervals": intervals,
        "masked_text": "".join(pieces),
    }


def validate_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError as exc:
        raise ExampleError("--url / SIE_URL must be an absolute HTTP(S) URL") from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ExampleError("--url / SIE_URL must be an absolute HTTP(S) URL")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ExampleError("put authentication in SIE_API_KEY; the URL must have no credentials, query or fragment")
    return url.rstrip("/")


class SingleAttempt:
    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        self.expected_path = parts.path + EXTRACT_PATH
        self.origin = (parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
        self.posts = 0
        self.last_status: int | None = None

    def on_request(self, request: httpx.Request) -> None:
        parts = urlsplit(str(request.url))
        origin = (parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
        if origin != self.origin:
            raise SingleAttemptError("single-attempt blocked a request to another origin")
        if request.method == "GET" and self.posts == 1:
            return  # The SDK may retrieve an already submitted result on this origin.
        if request.method != "POST" or parts.path != self.expected_path or parts.query or self.posts:
            raise SingleAttemptError("single-attempt blocked an unexpected request or second POST")
        self.posts += 1

    def on_response(self, response: httpx.Response) -> None:
        self.last_status = response.status_code
        if response.status_code >= 400:
            # Raising here preserves the known status before any SDK retry branch.
            raise SingleAttemptError(f"single-attempt stopped at HTTP {response.status_code}; no resend")


def live_extract(
    text: str,
    url: str,
    *,
    timeout_s: float = 900,
    no_auth: bool = False,
    single_attempt: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    # Optional dependencies: --show and --saved-composition use only the standard library.
    import httpx
    import msgpack
    from sie_sdk import Item, SIEClient

    url = str(httpx.URL(validate_url(url))).rstrip("/")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ExampleError("timeout must be a finite positive number of seconds")

    def check_wire_response(response: httpx.Response) -> None:
        if response.status_code != 200:
            return
        try:
            envelope = msgpack.unpackb(response.read(), raw=False)
        except (ValueError, msgpack.UnpackException) as exc:
            raise ExampleError("extraction returned malformed MessagePack") from exc
        if not isinstance(envelope, dict) or envelope.get("error") is not None:
            raise ExampleError("extraction returned an invalid envelope")
        items = envelope.get("items")
        if not isinstance(items, list) or len(items) != 1:
            raise ExampleError("extraction did not return exactly one item")
        validate_entities(text, items[0])

    guard = SingleAttempt(url) if single_attempt else None
    request_hooks = [guard.on_request] if guard else []
    response_hooks = ([guard.on_response] if guard else []) + [check_wire_response]
    with httpx.Client(
        base_url=url,
        follow_redirects=False,
        transport=transport if transport is not None else httpx.HTTPTransport(retries=0),
        timeout=httpx.Timeout(timeout_s),
        event_hooks={"request": request_hooks, "response": response_hooks},
    ) as http_client:
        with SIEClient(
            url,
            api_key="" if no_auth else os.environ.get("SIE_API_KEY", ""),
            timeout_s=timeout_s,
            http_client=http_client,
        ) as client:
            if single_attempt:
                result = client.extract(
                    MODEL,
                    Item(text=text),
                    options={"profile": PROFILE},
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=timeout_s,
                )
            else:
                result = client.extract(
                    MODEL, Item(text=text), options={"profile": PROFILE}, provision_timeout_s=timeout_s
                )
    validate_entities(text, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--show", action="store_true", help="print the logical request without dispatch")
    modes.add_argument("--saved-composition", action="store_true", help="mask the saved composition reference offline")
    parser.add_argument("--url", default=os.environ.get("SIE_URL"), help="SIE Cloud or self-hosted URL (or SIE_URL)")
    parser.add_argument("--no-auth", action="store_true", help="do not send the optional SIE_API_KEY")
    parser.add_argument(
        "--single-attempt", action="store_true", help="stop at the first HTTP failure; at most one POST"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=900, help="SDK provision budget and HTTP phase timeout"
    )
    args = parser.parse_args(argv)
    try:
        text = load_input()
        if args.show:
            output = {"evidence": "logical request display; no HTTP request sent", **logical_request(text)}
        elif args.saved_composition:
            source = json.loads((DIRECTORY / "source.json").read_text(encoding="utf-8"))
            output = render_result(text, {"entities": source["saved_composition"]["entities"]}, SAVED_EVIDENCE)
        else:
            if not args.url:
                raise ExampleError("provide --url or SIE_URL for a live request")
            result = live_extract(
                text,
                args.url,
                timeout_s=args.timeout_seconds,
                no_auth=args.no_auth,
                single_attempt=args.single_attempt,
            )
            output = render_result(text, result, "live SDK response")
    except ExampleError as exc:
        print(f"Failure: {exc}. No masked output produced.", file=sys.stderr)
        return 1
    except Exception as exc:
        # SDK/server/transport exception text can contain credential-bearing context.
        print(f"Failure: {type(exc).__name__}; no valid extraction result. No masked output produced.", file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
