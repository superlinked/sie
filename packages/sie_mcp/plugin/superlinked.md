# Install Superlinked document offload

`superlinked.md` is a router to each surface's **native** install path. There is no
drop-a-file auto-install — pick your surface below.

## Hosted cluster quick install

If a Superlinked operator gives you a hosted MCP endpoint plus a connector secret, you can
use the MCP-based document offload flow. Replace the example host with the endpoint the
operator provided, then build a small install pack from this repo:

```bash
mise run mcp-plugin-pack -- \
  --cluster-label my-sie-cluster \
  --mcp-url https://mcp.example.com/mcp
```

The command writes:

- `dist/superlinked-docs-plugin/INSTALL.md` — endpoint-specific install steps for Claude
  Code, claude.ai / desktop, and Cowork.
- `dist/superlinked-docs-plugin/superlinked-docs-skill.zip` — the claude.ai skill ZIP.
- `dist/superlinked-docs-plugin/claude-code/*/SKILL.md` — Claude Code skills in the
  local skill layout, including `parse-document`, `summarize-document`,
  `extract-entities`, and `redact-pii` skills backed by MCP tools.

No connector secret is written to the pack. Retrieve it from connector configuration or a
local secret store only when following the placeholder-based steps in `INSTALL.md`.
For local setup, load it into `SIE_MCP_CONNECTOR_SECRET` (using
`'replace-with-connector-secret'` only as the placeholder) and unset it after installation.
This singular name is a client-side shell variable used to construct the authorization
header; the MCP server instead reads the plural `SIE_MCP_CONNECTOR_SECRETS` runtime map.

The MCP redaction flow returns redacted text and counts, but not a local
placeholder-to-original map; do not use it when de-redaction is required.

For the full operator and user onboarding flow, including running a local edge,
smoke testing it, deploying it with Helm, and generating endpoint-specific install packs,
see `packages/sie_mcp/README.md`.

## Cowork (lead surface)

Install the **Superlinked plugin**, which provides this skill together with the remote MCP
connector. Configure the connector with your **connector secret** in the connector settings
(not in the chat).

Replace `<host>` below with your Superlinked deployment host (the address the MCP edge is
served at — e.g. the value of `SIE_MCP_HOST` for your cluster):

- MCP endpoint: `https://<host>/mcp`
- Auth header: `Authorization: Bearer <connector-secret>`
- Health check: `GET https://<host>/healthz`

## claude.ai

claude.ai is a **two-piece** install: a custom **connector** (the remote MCP) plus a
**skill ZIP** (the agent skill). Unlike Cowork, claude.ai connectors are OAuth-only — you
cannot paste a connector secret into a header — so the connector secret is entered during
an OAuth sign-in instead. Requires a plan that allows custom connectors and skills
(Pro / Max / Team / Enterprise) with code execution enabled.

Replace `<host>` with your Superlinked deployment host (the externally reachable origin of
the MCP edge — set `SIE_MCP_PUBLIC_URL` to pin it for OAuth metadata).

### 1. Connector (remote MCP + OAuth)

1. In claude.ai go to **Customize → Connectors → `+` → Add custom connector**.
2. Enter the remote MCP server URL: `https://<host>/mcp`. Click **Add**.
3. Click **Connect** and complete the OAuth sign-in. On the Superlinked authorize page,
   enter your **connector secret** (provided by your cluster operator). The edge maps the
   secret to your user identity — credentials are never pasted into the chat.

The OAuth metadata, registration, authorize, and token endpoints are served by the same
edge (`/.well-known/oauth-protected-resource`, `/.well-known/oauth-authorization-server`,
`/register`, `/authorize`, `/token`); the connector-secret check at `/mcp` is unchanged.

### 2. Skill ZIP

1. Build the ZIP: `mise run mcp-skill-zip` (writes `dist/superlinked-docs-skill.zip`).
2. In claude.ai go to **Customize → Skills → `+` → Create skill → Upload a skill** and
   upload the ZIP. The archive contains a single `superlinked-docs/SKILL.md` folder, the
   format claude.ai expects.

### File-landing

Generated markdown is written **in the sandbox**; export it by **Download** or **save to
Google Drive**, and optionally add the downloaded file to a **Project**'s knowledge to
reuse it across chats. The skill drives this flow (there is no persistent connected folder
as on Cowork).

### Operator notes

- Pin `SIE_MCP_PUBLIC_URL` to the externally reachable origin. It is advertised as the
  OAuth issuer and authorize/token endpoints, which is where users type their connector
  secret. Unpinned, metadata is served only when the request `Host` is loopback or listed in
  `SIE_MCP_ALLOWED_HOSTS`, and `X-Forwarded-Host` is never trusted. The Helm chart derives it
  from `mcpEdge.ingress.host`.
- Run the edge as a **single worker** (the default `mise run mcp-serve`): the OAuth
  authorization-code store is in-process, so a code issued on one worker cannot be
  redeemed on another.
- The OAuth redirect allowlist defaults to claude.ai's callback; override with
  `SIE_MCP_OAUTH_REDIRECT_URIS` (comma-separated). Set `SIE_MCP_OAUTH_ENABLED=0` to
  disable the bridge for Cowork-only deployments.
