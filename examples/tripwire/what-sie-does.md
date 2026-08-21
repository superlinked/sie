# What SIE does in Tripwire

Tripwire discovers AI skills and MCP servers, runs isolated safety scanners
(Modal + Cisco / Snyk / Tessl, etc.), stores findings in Supabase, and shows them
in one dashboard. **SIE is a post-scan tiered router**, not part of the core
scan path.

**Source of truth:**
[sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md) ·
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md) ·
[ADR-0016](https://github.com/neomatrix369/tripwire/blob/main/docs/adr/0016-tiered-router-sie-model-studio.md).

## SIE primitives

| Primitive | Role in this project |
|---|---|
| `generate` | Triage scanner findings via OpenAI-compatible chat (`/v1/chat/completions`) |

Default model override: `SIE_MODEL` (default `gen-4b`). Endpoint + API key from
`SIE_ENDPOINT` / `SIE_API_KEY`.

## Flow

```text
Discover → Scan (Modal / scanners) → Store (Supabase)
                                      │
                                      ▼
                         SIE triage (optional)
                                      │
                    escalate? ──yes──► Model Studio
                         │
                         ▼
                    Dashboard pathway strips
```

Router findings use `scanner_source=tiered_router` and are excluded from severity
rollups — triage is separate from scanner severity.

## vs Mock and Live without SIE

| Mode | Needs SIE? |
|---|---|
| Mock demo | No |
| Live scan + dashboard | No |
| Auto-route / `tripwire route` | Yes — `SIE_*` plus Model Studio keys (validated up front) |
| Model Studio second hop | Same keys; Alibaba called only when SIE escalates |

## Sample prototypes

- [prototypes/sie-studio](https://github.com/neomatrix369/tripwire/tree/main/prototypes/sie-studio) — list / generate against SIE
- [prototypes/model-studio](https://github.com/neomatrix369/tripwire/tree/main/prototypes/model-studio) — escalation sample

## Next

Wire keys and route: [SIE integration](./sie-integration.md). Problems:
[Troubleshooting](./troubleshooting.md).
