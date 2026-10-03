---
name: parse-document
description: Convert a PDF, scan, image of a page, or office file to clean markdown through the connected Superlinked MCP edge, so the source document is not read into model context directly. Use when the user asks to read, parse, OCR, extract from, summarize, or answer questions about a document.
---

# Parse a Document with Superlinked MCP

Use the connected Superlinked MCP server to convert the source file to markdown
through the MCP `docs_to_markdown` tool, then work from the markdown artifact.

## Steps

1. Do not open, view, attach, or read the source document into context. Read its
   raw bytes only and base64-encode those bytes.

2. Call the Superlinked MCP `docs_to_markdown` tool with:
   - `document_base64`: the base64-encoded source bytes
   - `filename`: the original filename
   - `engine`: `auto` unless the user explicitly asks to force `docling` or
     `vl-ocr`
   - `ocr`: leave `false` unless forcing the `docling` OCR path

3. Write the returned markdown to `processed/<source-stem>.md`. Create
   `processed/` if it does not exist.

4. Your next message must begin with a short receipt, then answer from the
   artifact selectively:

   ```text
   Parsed <source> -> processed/<source-stem>.md (<chars> chars)
   ```

   If the MCP metadata includes token-saving figures, include those figures
   directly under the receipt. Keep estimates marked as estimates.

5. Answer the user's question from the artifact. Grep or read specific sections;
   do not read the whole markdown file unless it is small.

## Errors

If the MCP call fails, relay the tool error and do not retry blindly. For auth
or connector errors, tell the user to reinstall the generated MCP plugin pack or
rerun the generated `claude mcp add ... --header 'Authorization: Bearer ...'`
command from `INSTALL.md`.
