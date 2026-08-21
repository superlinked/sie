# SIE integration (tiered router)

Routing is **optional**. Without the keys below, auto-route warns and skips; the
scan itself still succeeds.

**Source of truth:**
[docs/user-guide/sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md).

Finish a Live scan path from [Getting started](./getting-started.md) before
relying on routing in the product UI.

## Happy path — hosted Superlinked gateway

1. Sign in at [console.superlinked.com](https://console.superlinked.com) → **Keys**.
2. Put values in the **repo-root** `.env` (product CLI does **not** load
   `prototypes/.env`):

```bash
SIE_ENDPOINT=https://api.superlinked.com
# EU: https://eu.api.superlinked.com
SIE_API_KEY=sk-sie-…
# optional
SIE_MODEL=gen-4b
```

3. For `tripwire route` / auto-route, also set Model Studio credentials in the
   **same** repo-root `.env`. Today `resolveRouteConfig()` validates
   `ALIBABA_OPENAI_BASE_URL` and `DASHSCOPE_API_KEY` up front (escalation still
   runs only when SIE signals conflict / unusual status / low confidence):

```bash
ALIBABA_OPENAI_BASE_URL=https://…/compatible-mode/v1
DASHSCOPE_API_KEY=sk-…
```

Key map:
[env-vars — tiered router](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md#optional--tiered-router-sie--model-studio).

## Verify SIE alone

The sample CLI does **not** read the repo-root `.env`. Put the same `SIE_*`
values in `prototypes/.env` or `prototypes/sie-studio/.env`, or export
`SIE_API_KEY` / `SIE_ENDPOINT` in the shell:

```bash
cd prototypes/sie-studio
python3 sie_studio.py list
python3 sie_studio.py generate "Reply with one word: ok" --model gen-4b
```

Sample CLI notes:
[prototypes/sie-studio](https://github.com/neomatrix369/tripwire/blob/main/prototypes/sie-studio/README.md).

## Route after a Live scan

```bash
tripwire scan ./fixtures/skills/safe-csv-cleaner   # auto-routes when router keys set
# or
tripwire route --batch-id <batch_id>
node scripts/serve-dashboard.mjs
```

Look for pathway strips (Scan → SIE → …) and filters (**Escalated** / **SIE-only**):
[reading-router-results.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/reading-router-results.md).

**First success:** a batch shows SIE routing strips in the Live dashboard (or
`tripwire route` completes without skipping for missing keys).

## Model Studio (required for route config; used on escalate)

Configure Part B in
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md)
before expecting pathway strips. Review billing/quotas first. Alibaba calls run
only when SIE escalates; missing MS keys still cause auto-route to skip today.

## Next

Understand the router design: [What SIE does here](./what-sie-does.md).
Stuck? [Troubleshooting](./troubleshooting.md).
