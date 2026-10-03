---
name: extract-entities
description: Extract people, organizations, dates, amounts, or custom labels from a document through the connected Superlinked MCP edge, returning a compact table instead of reading the full document into context. Use when the user asks to list, extract, or tabulate entities from a file.
---

# Extract Entities with Superlinked MCP

Use the connected Superlinked MCP server to run zero-shot entity extraction on
the SIE cluster through the MCP `extract_entities` tool.

## Steps

1. Pick labels from the user's request. Lowercase nouns work best, for example:
   `person`, `organization`, `date`, `amount`, `contract term`.

2. Do not open, view, attach, or read the whole source document into context.

3. Call the Superlinked MCP `extract_entities` tool:
   - Pass `labels` as a list of label strings.
   - For PDF, scan, image-of-text, DOCX, PPTX, XLSX, or HTML inputs, read raw
     bytes only, base64-encode them, and pass `document_base64` plus `filename`.
   - For text or markdown inputs, read raw bytes only, base64-encode them, and
     pass `content_base64`.
   - Leave `engine` as `auto` unless the user explicitly asks to force a
     conversion engine.

4. Write the returned `markdown_table` to `processed/<source-stem>.entities.md`.
   Create `processed/` if it does not exist.

5. Your next message must begin with a short receipt:

   ```text
   Extracted <entity_count> entities from <source> -> processed/<source-stem>.entities.md
   ```

   Then present the entity table or the subset the user asked for.

## Errors

If the MCP call fails, relay the tool error and do not retry blindly. For auth
or connector errors, tell the user to reinstall the generated MCP plugin pack or
rerun the generated `claude mcp add ... --header 'Authorization: Bearer ...'`
command from `INSTALL.md`.
