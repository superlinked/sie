# SIE integration (tiered router)

Routing is **optional**. Without the keys below, auto-route warns and skips; the
scan itself still succeeds.

**Upstream deep-links:**
[sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md) ·
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md).

Those project pages still describe Model Studio as “optional escalation.” Prefer
**this page** for routing prerequisites: current Tripwire
`resolveRouteConfig()` requires **both** `SIE_*` and Model Studio keys before
any route runs (SIE-only configs skip; no `routing_review` rows).

Finish a Live scan path from [Getting started](./getting-started.md) before
relying on routing in the product UI.

## Happy path — hosted Superlinked gateway

1. Sign in at [console.superlinked.com](https://console.superlinked.com) → **Keys**.
2. Put **all** of the following in the **repo-root** `.env` (product CLI does
   **not** load `prototypes/.env`):

```bash
SIE_ENDPOINT=https://api.superlinked.com
# EU: https://eu.api.superlinked.com
SIE_API_KEY=sk-sie-…
# optional model override
SIE_MODEL=gen-4b

# required today for tripwire route / auto-route (validated up front)
ALIBABA_OPENAI_BASE_URL=https://…/compatible-mode/v1
DASHSCOPE_API_KEY=sk-…
```

Escalation to Alibaba still runs only when SIE signals conflict, unusual status,
or low confidence — but the MS keys must be present for config resolution.

Key map:
[env-vars — tiered router](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md#optional--tiered-router-sie--model-studio)
(section title still says “optional”; treat MS keys as **required for routing**
until upstream docs catch up).

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

Follow Part B in
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md)
before expecting pathway strips (ignore any “optional” wording there for
routing). Review billing/quotas first. Alibaba calls run only when SIE
escalates; missing MS keys still cause auto-route to skip today.

## Next

Understand the router design: [What SIE does here](./what-sie-does.md).
Stuck? [Troubleshooting](./troubleshooting.md).
