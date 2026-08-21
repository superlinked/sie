# Troubleshooting (SIE / router-focused)

Short FAQ for gallery readers. Full guides:
[sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md),
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md),
[setup-commands — when it fails](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/setup-commands.md#when-it-fails),
[env-vars.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md).

## Route warns and skips

`SIE_ENDPOINT` and/or `SIE_API_KEY` missing from **repo-root** `.env`. The CLI
does not load `prototypes/.env`. Scan can still succeed.

## `sie_studio.py` fails but `.env` looks fine

Confirm keys are in `prototypes/.env` for the sample CLI, or export the same
variables in the shell. Product routing still needs root `.env`.

## No pathway strips in the dashboard

- You are on **Mock** — switch to **Live (Supabase)** after a real scan
- Batch was never routed — run `tripwire route --batch-id …` or re-scan with
  `SIE_*` set
- Read filters: [reading-router-results.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/reading-router-results.md)

## Live scan “all clear” but a scanner was missing

Missing scanner keys **soft-skip** that engine — not a clean bill of health.
MVP Live only needs Supabase + Modal.

## Model Studio never runs

Expected unless SIE escalates. Configure Part B only after SIE works; check
`DASHSCOPE_*` / region endpoint in [model-studio-setup](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md).

## Still stuck?

1. [QUICKSTART](https://github.com/neomatrix369/tripwire/blob/main/QUICKSTART.md)  
2. [docs hub](https://github.com/neomatrix369/tripwire/blob/main/docs/README.md)  
3. Open an issue on
   [neomatrix369/tripwire](https://github.com/neomatrix369/tripwire/issues)
