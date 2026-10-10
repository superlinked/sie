"""Run every original DEV32 request once through the public SIE SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import httpx
from sie_sdk import SIEClient

SOURCE = json.loads((Path(__file__).parent / "source.json").read_text())
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def save_json(path: Path, value: Any) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_inputs(path: Path) -> list[dict]:
    raw = path.read_bytes()
    pin = SOURCE["files"]["inputs.json"]
    if len(raw) != pin["bytes"] or sha256(raw) != pin["sha256"]:
        raise ValueError("Inputs differ from the pinned public DEV32 source")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 32 or len({row["case_id"] for row in rows}) != 32:
        raise ValueError("The complete original 32-case frame is required")
    for row in rows:
        request = row["request"]
        if set(request) != {"messages", "response_format"} or sha256(canonical(request)) != row["request_sha256"]:
            raise ValueError("Original request wire changed")
        response_format = request["response_format"]
        if response_format.get("type") != "json_schema" or response_format["json_schema"].get("strict") is not True:
            raise ValueError("Original strict response schema is required")
    return rows


def request_body(case: dict, settings: dict) -> dict:
    body = {
        "model": settings["model"],
        **case["request"],
        "max_completion_tokens": settings["max_completion_tokens"],
        "temperature": settings["temperature"],
        "top_p": 0.95,
        "top_k": settings["top_k"],
    }
    if settings["thinking"] != "server":
        body["chat_template_kwargs"] = {"enable_thinking": settings["thinking"] == "on"}
    return body


def bounded_body(response: httpx.Response) -> bytes:
    if response.is_stream_consumed:
        body = response.content
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds the recorded byte bound")
        return body
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity", "gzip"}:
        raise ValueError("Unsupported response content encoding")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
    body = bytearray()
    encoded_bytes = 0
    for chunk in response.iter_raw(chunk_size=64 * 1024):
        encoded_bytes += len(chunk)
        if encoded_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("Encoded response exceeds the recorded byte bound")
        pending = chunk
        while pending:
            remaining = MAX_RESPONSE_BYTES - len(body)
            if decoder is None:
                decoded = pending
                pending = b""
            else:
                if decoder.eof:
                    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                decoded = decoder.decompress(pending, remaining + 1)
                pending = decoder.unused_data
            if len(decoded) > remaining:
                raise ValueError("Decoded response exceeds the recorded byte bound")
            body.extend(decoded)
    if decoder is not None and not decoder.eof:
        raise ValueError("Truncated gzip response")
    return bytes(body)


class OnePost(httpx.BaseTransport):
    """One physical chat POST; metadata/completion GETs keep separate evidence."""

    def __init__(self, folder: Path, url: str, expected: dict, inner: httpx.BaseTransport | None = None) -> None:
        self.folder = folder
        self.path = httpx.URL(url).path.rstrip("/") + "/v1/chat/completions"
        self.expected = expected
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        self.sends = 0
        self.post_status: int | None = None
        self.wire: list[dict] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if request.url.path != self.path or self.sends:
                raise RuntimeError("A case permits only one chat completion POST")
            raw_request = request.read()
            if json.loads(raw_request) != self.expected:
                raise ValueError("SDK request differs from the planned full source wire")
            save_new(self.folder / "request.json", raw_request)
            save_new(self.folder / "sent.json", canonical({"request_sha256": sha256(raw_request)}))
            self.sends = 1
        elif request.method != "GET":
            raise ValueError("Unexpected request method")
        response = self.inner.handle_request(request)
        try:
            body = bounded_body(response)
            name = f"wire-{len(self.wire):02d}-{request.method}.json"
            save_new(self.folder / name, body)
            self.wire.append(
                {
                    "method": request.method,
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "content_encoding": response.headers.get("content-encoding"),
                    "body_representation": "decoded",
                    "file": name,
                    "bytes": len(body),
                    "sha256": sha256(body),
                }
            )
            if request.method == "POST":
                self.post_status = response.status_code
            headers = response.headers.copy()
            for name in ("content-encoding", "content-length"):
                headers.pop(name, None)
            return httpx.Response(response.status_code, headers=headers, content=body)
        finally:
            response.close()

    def close(self) -> None:
        self.inner.close()


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Repeated output key")
        output[key] = value
    return output


def parsed_output(reply: dict, case: dict) -> tuple[dict, str]:
    choices = reply.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = choices[0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        raise TypeError("Completion has no JSON text")
    output = json.loads(content, object_pairs_hook=unique_object)
    schema = case["request"]["response_format"]["json_schema"]["schema"]
    fields = schema["properties"]
    if not isinstance(output, dict) or set(output) != set(fields):
        raise ValueError("Completion differs from the required field set")
    if any(not isinstance(value, str) or value not in fields[key]["enum"] for key, value in output.items()):
        raise ValueError("Completion contains an unsupported field value")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise ValueError("Completion did not finish normally")
    return output, finish_reason


def case_worker(case: dict, folder: Path, settings: dict, inner: httpx.BaseTransport | None = None) -> None:
    expected = request_body(case, settings)
    transport = OnePost(folder, settings["url"], expected, inner)
    terminal: dict = {"status": "attempt_status_unknown", "halt": True}
    reply = None
    started = time.monotonic()
    http = httpx.Client(
        base_url=settings["url"],
        headers={"Accept-Encoding": "gzip, identity"},
        transport=transport,
        follow_redirects=False,
        trust_env=False,
    )
    try:
        with SIEClient(
            settings["url"],
            api_key=os.environ.get("SIE_API_KEY") or None,
            timeout_s=settings["timeout_s"],
            http_client=http,
        ) as client:
            extra = (
                {"chat_template_kwargs": expected["chat_template_kwargs"]}
                if "chat_template_kwargs" in expected
                else None
            )
            reply = client.chat_completions(
                settings["model"],
                case["request"]["messages"],
                response_format=case["request"]["response_format"],
                max_completion_tokens=settings["max_completion_tokens"],
                temperature=settings["temperature"],
                top_p=0.95,
                top_k=settings["top_k"],
                extra_body=extra,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=settings["timeout_s"],
                read_timeout_s=settings["timeout_s"],
            )
            save_new(folder / "sdk.json", canonical(reply))
        terminal.update(usage=reply.get("usage"), returned_model=reply.get("model"), halt=False)
        choices = reply.get("choices")
        if isinstance(choices, list) and len(choices) == 1:
            terminal["finish_reason"] = choices[0].get("finish_reason")
        try:
            output, finish_reason = parsed_output(reply, case)
            terminal.update(status="ok", output=output, finish_reason=finish_reason)
        except (ValueError, TypeError, KeyError) as exc:
            terminal.update(status="failed", reason=type(exc).__name__, output=None)
    except Exception as exc:  # noqa: BLE001 - Record caller failures without exposing exception text.
        terminal["reason"] = type(exc).__name__
        if not transport.sends:
            terminal["status"] = "known_unattempted"
        elif transport.post_status is not None and transport.post_status != 303:
            terminal["status"] = "failed"
    finally:
        http.close()
    terminal.update(
        physical_sends=transport.sends,
        post_status=transport.post_status,
        wire=transport.wire,
        wall_s=time.monotonic() - started,
    )
    save_new(folder / "terminal.json", canonical(terminal))


def run_case(case: dict, folder: Path, settings: dict, *, worker=case_worker, context=None) -> dict:
    """Enforce the complete case wall window, then reap before another case."""
    ctx = context if context is not None else multiprocessing.get_context("spawn")
    process = ctx.Process(target=worker, args=(case, folder, settings))
    started = time.monotonic()
    try:
        process.start()
        process.join(max(0, started + settings["timeout_s"] - time.monotonic()))
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.is_alive():
                raise RuntimeError("Case process could not be reaped; do not dispatch again")
            process.join()
    terminal = folder / "terminal.json"
    if terminal.exists():
        result = json.loads(terminal.read_bytes())
    else:
        sent = (folder / "sent.json").exists()
        result = {
            "status": "attempt_status_unknown" if sent else "known_unattempted",
            "reason": "case_deadline_or_worker_exit",
            "physical_sends": int(sent),
            "halt": True,
        }
    result["case_wall_s"] = time.monotonic() - started
    return result


def run_frame(
    inputs: Path,
    out: Path,
    *,
    url: str,
    model: str,
    timeout_s: float = 1800,
    max_completion_tokens: int = 32768,
    thinking: str = "on",
    temperature: float = 1.0,
    top_k: int = 64,
    execute=run_case,
) -> list[dict]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 2**32 - 1:
        raise ValueError("Top-k must be an integer from 1 to 4294967295")
    if not math.isfinite(timeout_s) or timeout_s <= 0 or max_completion_tokens <= 0:
        raise ValueError("Positive finite case timeout and completion token cap are required")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("Temperature must be a finite number from 0 to 2")
    address = httpx.URL(url)
    if (
        address.scheme not in {"http", "https"}
        or not address.host
        or address.username
        or address.password
        or address.query
        or address.fragment
    ):
        raise ValueError("Use a HTTP(S) SIE base URL without credentials, query or fragment")
    if not model.strip() or thinking not in {"on", "off", "server"}:
        raise ValueError("Model and valid caller thinking setting are required")
    cases = load_inputs(inputs)
    settings = {
        "url": url.rstrip("/"),
        "model": model,
        "timeout_s": timeout_s,
        "max_completion_tokens": max_completion_tokens,
        "thinking": thinking,
        "temperature": float(temperature),
        "top_k": top_k,
    }
    out.mkdir(parents=True, exist_ok=False)
    sync_directory(out.parent)
    (out / "cases").mkdir()
    rows = [
        {
            "case_id": case["case_id"],
            "episode_id": case["episode_id"],
            "family": case["family"],
            "ordinal": i,
            "status": "known_unattempted",
            "reason": "not_started",
            "physical_sends": 0,
        }
        for i, case in enumerate(cases)
    ]
    save_new(
        out / "plan.json",
        canonical(
            {
                "source": SOURCE,
                "sie_url": str(address).rstrip("/"),
                "model": model,
                "timeout_s": timeout_s,
                "cases": 32,
                "decoding": {
                    k: v
                    for k, v in request_body(cases[0], settings).items()
                    if k not in {"messages", "response_format"}
                },
                "thinking": "Caller request only; actual model configuration may override it",
                "first_gate": "complete normal finish and original schema; no gold selection",
                "no_resume_or_retry": True,
            }
        ),
    )
    save_json(out / "responses.json", rows)
    for i, (case, row) in enumerate(zip(cases, rows, strict=True)):
        folder = out / "cases" / f"{i:02d}"
        folder.mkdir()
        save_new(
            folder / "intent.json", canonical({"case_id": case["case_id"], "request_sha256": case["request_sha256"]})
        )
        row.update(status="attempt_status_unknown", reason="durable_case_intent")
        save_json(out / "responses.json", rows)
        result = execute(case, folder, settings)
        row.update(result)
        save_json(out / "responses.json", rows)
        print(f"{i + 1}/32 {case['case_id']}: {row['status']}", flush=True)
        if row.get("halt") or (i == 0 and row["status"] != "ok"):
            for tail in rows[i + 1 :]:
                tail["reason"] = "first_wire_schema_gate_failed" if i == 0 else "stopped_before_send"
            save_json(out / "responses.json", rows)
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/inputs.json"))
    parser.add_argument("--out", type=Path, default=Path("run-output"))
    parser.add_argument("--url", default=os.environ.get("SIE_URL"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--thinking", choices=("on", "off", "server"), default="on")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("Set SIE_URL or pass --url")
    outcomes = run_frame(
        args.inputs,
        args.out,
        url=args.url,
        model=args.model,
        timeout_s=args.timeout,
        max_completion_tokens=args.max_completion_tokens,
        thinking=args.thinking,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return 1 if any(row.get("halt") for row in outcomes) or outcomes[0]["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
