# Scan AI skills and MCP servers, then triage with SIE

> Tripwire is a metal detector for AI tools — discover, scan in isolation, review
> findings in one dashboard. Optional Superlinked SIE routes findings after Live scans.

This is an **external project guide**. The runnable app lives in
[neomatrix369/tripwire](https://github.com/neomatrix369/tripwire).
This folder is the SIE-facing onboarding surface: short pages here, full detail
in that repo.

**SIE primitives used:** `generate` (chat completions via OpenAI-compatible
`/v1/chat/completions` for post-scan triage). SIE is **optional** — Mock demo and
Live scans work without it; routing runs only when `SIE_*` keys are set.

## Who this is for

| You are… | Start here |
|---|---|
| New to SIE, found this in the gallery | [Getting started](./getting-started.md) → [SIE integration](./sie-integration.md) |
| New to Tripwire, want SIE triage | Same path — then [What SIE does here](./what-sie-does.md) |

Happy path for SIE: Mock demo first → Live scan (Supabase + Modal) → enable
hosted SIE → `tripwire route` (or auto-route after scan).

## Start here

1. [Getting started](./getting-started.md) — clone, Mock demo, then Live prerequisites
2. [SIE integration](./sie-integration.md) — `SIE_*` keys, verify, route a batch
3. [What SIE does here](./what-sie-does.md) — tiered router, Model Studio escalation
4. [Troubleshooting](./troubleshooting.md) — short FAQ + deep-links

**Canonical docs in the project:**
[QUICKSTART](https://github.com/neomatrix369/tripwire/blob/main/QUICKSTART.md) ·
[Tiered router / SIE](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/tiered-router-setup.md) ·
[docs hub](https://github.com/neomatrix369/tripwire/blob/main/docs/README.md)

## Ports cheat sheet

| Service | Port | Notes |
|---|---|---|
| Dashboard | `8765` | `node scripts/serve-dashboard.mjs` |
| SIE | Hosted | `https://api.superlinked.com` or EU endpoint — not self-hosted by default |

## Attribution

Built and maintained in
[neomatrix369/tripwire](https://github.com/neomatrix369/tripwire)
([license](https://github.com/neomatrix369/tripwire/blob/main/LICENSE)).
Architecture and deeper guides live in that repository.
