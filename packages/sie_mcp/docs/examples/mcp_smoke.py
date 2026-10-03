# ruff: noqa: INP001 — standalone example script, not a package module.
"""End-to-end smoke for the SIE MCP edge, exactly as a Cowork connector calls it.

Connects to the remote streamable-HTTP MCP server with a connector-secret Bearer,
lists tools, and (optionally) calls docs_to_markdown on a local file.

Run it in the sie-mcp package env so the `mcp` client is importable:

    mise exec -- uv run --package sie-mcp python packages/sie_mcp/docs/examples/mcp_smoke.py \
        --url http://localhost:8088/mcp --secret 'replace-with-connector-secret' --list

    mise exec -- uv run --package sie-mcp python packages/sie_mcp/docs/examples/mcp_smoke.py \
        --url http://localhost:8088/mcp --secret 'replace-with-connector-secret' \
        --file packages/sie_mcp/docs/examples/sample.html --filename sample.html --engine docling
"""

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _unwrap(result: object) -> dict[str, Any]:
    """Return the tool's structured dict from a CallToolResult, tolerating shapes."""
    sc = getattr(result, "structuredContent", None)
    if sc:
        return sc
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"_raw_text": text}
    return {"_repr": repr(result)}


async def _run(args: argparse.Namespace, document_b64: str | None) -> int:
    headers = {"Authorization": f"Bearer {args.secret}"}
    async with (
        streamablehttp_client(args.url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print("TOOLS:", [t.name for t in tools.tools])
        if document_b64 is None:
            return 0

        fname = args.filename or args.file.rsplit("/", 1)[-1]
        print(f"\nCALL docs_to_markdown(filename={fname!r}, engine={args.engine!r})")
        result = await session.call_tool(
            "docs_to_markdown",
            {"document_base64": document_b64, "filename": fname, "engine": args.engine},
        )
        if getattr(result, "isError", False):
            print("TOOL ERROR:")
            print(json.dumps(_unwrap(result), indent=2)[:4000])
            return 1
        payload = _unwrap(result)
        md = payload.get("markdown", "")
        print("\n--- METADATA ---")
        print(json.dumps(payload.get("metadata", {}), indent=2)[:3000])
        print("\n--- MARKDOWN (first 1500 chars) ---")
        print(md[:1500])
        print(f"\n--- markdown length: {len(md)} chars ---")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8088/mcp")
    ap.add_argument("--secret", required=True)
    ap.add_argument("--list", action="store_true", help="list tools and exit")
    ap.add_argument("--file")
    ap.add_argument("--filename")
    ap.add_argument("--engine", default="docling")
    args = ap.parse_args()

    # Read the file synchronously here, outside the async client (avoids blocking
    # I/O inside the event loop).
    document_b64: str | None = None
    if args.file and not args.list:
        document_b64 = base64.b64encode(Path(args.file).read_bytes()).decode()

    return asyncio.run(_run(args, document_b64))


if __name__ == "__main__":
    sys.exit(main())
