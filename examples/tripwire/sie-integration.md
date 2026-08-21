# SIE integration (tiered router)

SIE is **optional**. Missing `SIE_*` keys causes auto-route to warn and skip; the
scan itself still succeeds.

**Source of truth:**
[docs/user-guide/sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md).

Finish a Live scan path from [Getting started](./getting-started.md) before
relying on routing in the product UI.

## Happy path — hosted Superlinked gateway

1. Sign in at [console.superlinked.com](https://console.superlinked.com) → **Keys**.
2. Put values in the **repo-root** `.env` (CLI does **not** load `prototypes/.env`):

```bash
SIE_ENDPOINT=https://api.superlinked.com
# EU: https://eu.api.superlinked.com
SIE_API_KEY=sk-sie-…
# optional
SIE_MODEL=gen-4b
```

Key map:
[env-vars — tiered router](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md#optional--tiered-router-sie--model-studio).

## Verify SIE alone

```bash
cd prototypes/sie-studio
python3 sie_studio.py list
python3 sie_studio.py generate "Reply with one word: ok" --model gen-4b
```

Sample CLI notes:
[prototypes/sie-studio](https://github.com/neomatrix369/tripwire/blob/main/prototypes/sie-studio/README.md).

## Route after a Live scan

```bash
tripwire scan ./fixtures/skills/safe-csv-cleaner   # auto-routes when SIE_* set
# or
tripwire route --batch-id <batch_id>
node scripts/serve-dashboard.mjs
```

Look for pathway strips (Scan → SIE → …) and filters (**Escalated** / **SIE-only**):
[reading-router-results.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/reading-router-results.md).

**First success:** a batch shows SIE routing strips in the Live dashboard (or
`tripwire route` completes without skipping for missing keys).

## Optional — Model Studio escalation

Alibaba Cloud Model Studio runs only when SIE signals conflict, unusual status,
or low confidence. Configure after Part A in
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md)
(`DASHSCOPE_*` / `ALIBABA_OPENAI_BASE_URL`). Review billing/quotas first.

## Next

Understand the router design: [What SIE does here](./what-sie-does.md).
Stuck? [Troubleshooting](./troubleshooting.md).
