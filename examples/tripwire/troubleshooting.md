# Troubleshooting (SIE / router-focused)

Short FAQ for gallery readers. Full guides:
[sie-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/sie-setup.md),
[model-studio-setup.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md),
[setup-commands — when it fails](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/setup-commands.md#when-it-fails),
[env-vars.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/env-vars.md).

## Route warns and skips

Missing `SIE_ENDPOINT` / `SIE_API_KEY` **or** `ALIBABA_OPENAI_BASE_URL` /
`DASHSCOPE_API_KEY` from **repo-root** `.env`. The product CLI does not load
`prototypes/.env`. Scan can still succeed; no `routing_review` rows are written.

## `sie_studio.py` fails but `.env` looks fine

Confirm keys are in `prototypes/.env` or `prototypes/sie-studio/.env` (or
exported in the shell). The sample CLI does not load repo-root `.env`. Product
routing still needs root `.env`.

## No pathway strips in the dashboard

- You are on **Mock** — switch to **Live (Supabase)** after a real scan
- Batch was never routed — run `tripwire route --batch-id …` or re-scan with
  router keys set (`SIE_*` and Model Studio)
- Read filters: [reading-router-results.md](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/reading-router-results.md)

## Live scan “all clear” but a scanner was missing

Missing scanner keys **soft-skip** that engine — not a clean bill of health.
MVP Live only needs Supabase + Modal.

## Model Studio never runs (no Alibaba calls)

Expected unless SIE escalates. Keys must still be present for route config; check
`DASHSCOPE_*` / region endpoint in [model-studio-setup](https://github.com/neomatrix369/tripwire/blob/main/docs/user-guide/model-studio-setup.md).

## Still stuck?

1. [QUICKSTART](https://github.com/neomatrix369/tripwire/blob/main/QUICKSTART.md)  
2. [docs hub](https://github.com/neomatrix369/tripwire/blob/main/docs/README.md)  
3. Open an issue on
   [neomatrix369/tripwire](https://github.com/neomatrix369/tripwire/issues)
