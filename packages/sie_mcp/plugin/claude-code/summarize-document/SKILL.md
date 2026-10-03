---
name: summarize-document
description: Summarize a long PDF, scan, office file, text file, or markdown file through the connected Superlinked MCP edge instead of reading the whole source into model context. Use when the user asks for a summary, overview, digest, or "what does this document say" about a large file.
---

# Summarize a Document with Superlinked MCP

Use the connected Superlinked MCP server to summarize the source on the SIE
cluster through the MCP `summarize_document` tool.

## Steps

1. Do not open, view, attach, or read the whole source document into context.

2. Call the Superlinked MCP `summarize_document` tool:
   - For PDF, scan, image-of-text, DOCX, PPTX, XLSX, or HTML inputs, read raw
     bytes only, base64-encode them, and pass `document_base64` plus `filename`.
   - For text or markdown inputs, read raw bytes only, base64-encode them, and
     pass `content_base64`.
   - Leave `engine` as `auto` unless the user explicitly asks to force a
     conversion engine.

3. Write the returned `summary` to `processed/<source-stem>.summary.md`.
   Create `processed/` if it does not exist.

4. Your next message must begin with a short receipt:

   ```text
   Summarized <source> -> processed/<source-stem>.summary.md (<summary_chars> chars)
   ```

   If metadata includes `token_savings_estimate`, include it as an estimate.

5. Present the summary from the artifact. If the user asks for exact quotes or
   detailed follow-up, use `parse-document` and inspect only relevant sections
   of the parsed markdown.

## Errors

If the MCP call fails, relay the tool error and do not retry blindly. For auth
or connector errors, tell the user to reinstall the generated MCP plugin pack or
rerun the generated `claude mcp add ... --header 'Authorization: Bearer ...'`
command from `INSTALL.md`.
