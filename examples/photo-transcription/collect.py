"""Collect one native SIE transcription per frozen photo from a remote endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from fidelity import digest, load_packet


class PhysicalAttemptError(RuntimeError):
    """A native call failed or would exceed the once-only physical attempt."""


def collect(
    packet: Path,
    output: Path,
    model: str,
    base_url: str,
    instruction: str | None = None,
    max_new_tokens: int = 8192,
) -> dict[str, Any]:
    import httpx  # Optional: only native collection needs the SDK transport.
    from sie_sdk import SIEClient  # Optional for the offline scorer.

    output_limit = 12288 if model == "lightonai/LightOnOCR-3-4B" else 8192
    if not model or not 1 <= max_new_tokens <= output_limit:
        raise ValueError(f"A model and an output limit between 1 and {output_limit} are required")
    manifest, _ = load_packet(packet)
    output.mkdir(parents=True, exist_ok=False)
    calls: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    inference_path = httpx.URL(base_url).path.rstrip("/") + f"/v1/extract/{model}"
    wire_dir = output / "wire"
    wire_dir.mkdir()

    def request_hook(request: Any) -> None:
        if request.method == "POST" and request.url.path == inference_path:
            if current["attempts"]:
                raise PhysicalAttemptError("A second physical inference attempt is forbidden")
            current["attempts"] = 1
            payload = request.content
            (wire_dir / f"{current['index']:02d}-request.msgpack").write_bytes(payload)
            current["request_body_sha256"] = digest(payload)

    def response_hook(response: Any) -> None:
        if current.get("attempts") and response.request.method in {"POST", "GET"}:
            payload = response.read()
            hop = len(current["response_hops"])
            (wire_dir / f"{current['index']:02d}-response-{hop:02d}.bin").write_bytes(payload)
            current["response_body_sha256"] = digest(payload)
            current["http_status"] = response.status_code
            current["response_hops"].append(
                {
                    "method": response.request.method,
                    "http_status": response.status_code,
                    "body_sha256": digest(payload),
                }
            )
            # The SDK may retrieve a same-origin in-flight result with GET.
            # It validates that 303 destination; this never replays inference.
            if response.status_code != 303 and not 200 <= response.status_code < 300:
                raise PhysicalAttemptError(f"HTTP {response.status_code}; no automatic resubmission")

    transport = httpx.Client(
        base_url=base_url,
        timeout=180,
        follow_redirects=False,
        event_hooks={"request": [request_hook], "response": [response_hook]},
    )
    # Public SDK transport injection captures actual payloads without auth headers.
    # Reject error replies before the SDK's model-loading retry branch runs.
    with transport, SIEClient(base_url=base_url, timeout_s=180, http_client=transport) as client:
        stopped = False
        for index, source in enumerate(manifest["records"]):
            current = {
                "index": index,
                "case_id": source["id"],
                "image_sha256": source["input_image"]["sha256"],
                "requested_model": model,
                "attempts": 0,
                "status": "unattempted",
                "response_hops": [],
            }
            if not stopped:
                started = time.monotonic()
                try:
                    result = client.extract(
                        model,
                        {
                            "id": source["id"],
                            "images": [
                                {"data": (packet / source["input_image"]["path"]).read_bytes(), "format": "jpeg"}
                            ],
                        },
                        instruction=instruction,
                        options={"profile": "default", "max_new_tokens": max_new_tokens, "num_beams": 1},
                        wait_for_capacity=False,
                        max_oom_retries=0,
                    )
                    if result.get("id") != source["id"] or len(result["entities"]) != 1:
                        raise ValueError("Expected one OCR entity for the original item ID")
                    text = result["entities"][0]["text"]
                    if not isinstance(text, str):
                        raise ValueError("Native text must be a string")
                    current.update(
                        text=text,
                        transcription_sha256=digest(text.encode()),
                        returned_model=result.get("model"),
                        status="completed" if text.strip() else "empty",
                    )
                    (output / f"{index:02d}-decoded.json").write_text(json.dumps(result, indent=2) + "\n")
                except Exception as error:
                    current["status"] = "error" if current["attempts"] else "unattempted"
                    current["error_type"] = type(error).__name__
                    stopped = True
                current["client_elapsed_s"] = time.monotonic() - started
            calls.append(dict(current))
            (output / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls))
    review = [
        {
            "case_id": row["case_id"],
            "image_sha256": row["image_sha256"],
            "transcription_sha256": row["transcription_sha256"],
            "reviewer": "",
            "critical_outcomes": [""] * len(manifest["records"][row["index"]]["critical_values"]),
            "line_outcomes": [""] * len(manifest["records"][row["index"]]["reviewed_lines"]),
        }
        for row in calls
        if row["status"] == "completed"
    ]
    (output / "review-template.json").write_text(json.dumps(review, indent=2) + "\n")
    summary = {
        "frozen_photos": len(calls),
        "physical_inference_posts": sum(row["attempts"] for row in calls),
        "completed": sum(row["status"] == "completed" for row in calls),
        "stopped": stopped,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sie-url", default=os.environ.get("SIE_URL"))
    parser.add_argument("--instruction")
    parser.add_argument(
        "--max-new-tokens", type=int, default=8192, help="Up to 12288 for LightOnOCR-3-4B; up to 8192 for other models"
    )
    args = parser.parse_args()
    if not args.sie_url:
        parser.error("Provide --sie-url or SIE_URL for an existing remote deployment")
    print(
        json.dumps(
            collect(args.packet, args.output, args.model, args.sie_url, args.instruction, args.max_new_tokens), indent=2
        )
    )


if __name__ == "__main__":
    main()
