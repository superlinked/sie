# Getting started (Mock first, then Live)

Goal: install the Tripwire CLI and see the Mock dashboard before adding cloud
accounts or SIE.

**Source of truth:**
[QUICKSTART.md](https://github.com/neomatrix369/tripwire/blob/main/QUICKSTART.md).

## Prerequisites

- Node.js **22**
- Python **3.12** (scanners / tooling)
- Git and npm
- For **Live** only: Modal CLI (`pip install modal`) before `./scripts/setup-modal.sh` —
  see [modal-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/modal-setup.md)

Details: [prerequisites.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/prerequisites.md).

## Mock demo (no cloud accounts)

```bash
git clone https://github.com/neomatrix369/tripwire.git
cd tripwire
cd cli && npm install && npm link && cd ..

tripwire scan --dry-discover ./fixtures/skills/safe-csv-cleaner
node scripts/serve-dashboard.mjs
```

Open **http://127.0.0.1:8765/** → choose **Mock (demo data)** in Guard.

More commands:
[setup-commands.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/setup-commands.md).

## Live path (before SIE)

SIE routing expects a completed **Live** scan batch. Minimum Viable Live:
**Supabase + Modal** (scanner vendor keys are optional and soft-skip if missing).

1. Accounts: [supabase-setup](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/supabase-setup.md) →
   [modal-setup](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/modal-setup.md)
2. Keys: [env-vars.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md)
3. Bootstrap:

```bash
cp .env.example .env
# fill SUPABASE_* and MODAL_TOKEN_* (and scanners you want)
tripwire setup
./scripts/setup-modal.sh
tripwire scan ./fixtures/skills/safe-csv-cleaner
node scripts/serve-dashboard.mjs
# Open Live (Supabase) in the dashboard
```

Full Live checklist:
[QUICKSTART — Live](https://github.com/neomatrix369/tripwire/blob/main/QUICKSTART.md#live-advanced).

## Next

Enable Superlinked routing: [SIE integration](./sie-integration.md).
