"""Read one document with the exact measured SIE profile, or show the request offline."""
# ruff: noqa: INP001 - Standalone example scripts are not a Python package.

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec"
SYSTEM = (
    "Extract data from the document image into the JSON schema. Copy values as they appear on the document. "
    "Use null for a field the document does not contain. Return only the JSON."
)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The study's shared schema conversion: required keys, nullable leaves."""
    if "$ref" in schema:
        raise ValueError("Schema references are unsupported; inline the referenced property schema")
    kind = schema.get("type")
    if kind == "enum" or (kind is None and "enum" in schema):
        kind = "string"
    if isinstance(kind, list):
        kind = next((value for value in kind if value != "null"), "string")
    output: dict[str, Any] = {}
    if schema.get("description"):
        output["description"] = schema["description"]
    if kind == "object":
        properties = {name: strict_schema(value) for name, value in (schema.get("properties") or {}).items()}
        return output | {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if kind == "array":
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {"type": "string"}
        output |= {"type": "array", "items": strict_schema(items)}
        return {"anyOf": [output, {"type": "null"}]}
    output["type"] = kind or "string"
    enum = schema.get("enum") or schema.get("emum")
    if enum:
        choices = [value for value in enum if value is not None]
        if not choices:
            raise ValueError("A string enum needs at least one non-null choice")
        if all(isinstance(value, str) for value in choices):
            output["enum"] = choices
    return {"anyOf": [output, {"type": "null"}]}


def media_type(data: bytes) -> str:
    """Identify the supported image format from its signature."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise ValueError("Use a PNG or JPEG document image")


def build_body(image: bytes, schema: dict[str, Any], *, show: bool = False) -> dict[str, Any]:
    """Build the study's one-image request, optionally replacing bytes with their hash."""
    if schema.get("type") != "object":
        raise ValueError("The document schema must have an object root")
    mime = media_type(image)
    encoded = f"<image sha256:{hashlib.sha256(image).hexdigest()}>" if show else base64.b64encode(image).decode()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]},
        ],
        "temperature": 0,
        "presence_penalty": 0,
        "max_completion_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "fields", "schema": strict_schema(schema), "strict": True},
        },
    }


def main() -> None:
    """Inspect a request offline or send one optional document trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--show", action="store_true", help="Print the request with image bytes replaced by a hash; no SDK or network"
    )
    args = parser.parse_args()
    body = build_body(args.image.read_bytes(), json.loads(args.schema.read_text(encoding="utf-8")), show=args.show)
    if args.show:
        print(json.dumps({"model": MODEL, **body}, indent=2))
        return
    from sie_sdk import SIEClient  # noqa: PLC0415 - Optional dependency; offline modes need only the standard library.

    with SIEClient(args.endpoint, api_key=os.environ.get("SIE_API_KEY"), read_timeout_s=300) as client:
        result = client.chat_completions(MODEL, **body, max_oom_retries=0, wait_for_capacity=False)
    choice = result["choices"][0]
    if choice.get("finish_reason") in {"length", "max_tokens"}:
        raise SystemExit("The output limit was reached; this is not a complete extraction")
    print(json.dumps(json.loads(choice["message"]["content"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
