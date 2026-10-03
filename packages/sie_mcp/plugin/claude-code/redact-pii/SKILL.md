---
name: redact-pii
description: Redact personal data from a document through the connected Superlinked MCP edge before working with the content. Use when the user asks to redact, anonymize, scrub, de-identify, or remove PII/sensitive data from a document.
---

# Redact PII with Superlinked MCP

Use the connected Superlinked MCP server to detect and replace PII on the SIE
cluster through the MCP `redact_pii` tool.

## Hard Rules

- Do not open, view, attach, or read the source document into context before
  redaction.
- After redaction, continue the user's task using only the redacted artifact.
- The MCP tool intentionally does not return a placeholder-to-original map. Do
  not promise de-redaction from the MCP flow.

## Steps

1. Call the Superlinked MCP `redact_pii` tool:
   - For PDF, scan, image-of-text, DOCX, PPTX, XLSX, or HTML inputs, read raw
     bytes only, base64-encode them, and pass `document_base64` plus `filename`.
   - For text or markdown inputs, read raw bytes only, base64-encode them, and
     pass `content_base64`.
   - Pass `labels` only if the user wants a custom PII/sensitive-data set.
   - Leave `engine` as `auto` unless the user explicitly asks to force a
     conversion engine.

2. Write the returned `redacted_text` to
   `processed/<source-stem>.redacted.md`. Create `processed/` if it does not
   exist.

3. Your next message must begin with a short receipt:

   ```text
   Redacted <span_count> spans from <source> -> processed/<source-stem>.redacted.md
   ```

   Include the label counts and the note that detection is model-based and
   best-effort, not a certified scrubber.

4. Continue the user's actual task using only the redacted artifact.

## Errors

If the MCP call fails, relay the tool error and do not retry blindly. For auth
or connector errors, tell the user to reinstall the generated MCP plugin pack or
rerun the generated `claude mcp add ... --header 'Authorization: Bearer ...'`
command from `INSTALL.md`.
