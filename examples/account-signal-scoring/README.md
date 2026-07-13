# account-signal-scoring

Turn a pile of account signals into **one score a team can act on** — with SIE
doing the ranking.

A CSM starts the day staring at a mess of signals: a Stripe downgrade here, an
API-usage cliff there, a seat cap about to hit somewhere else. This demo
collapses that into a single triage board. Each account's signals roll up into
one 0-100 score, and SIE ranks the account's *story* against a corpus of
past-outcome **playbooks** to pick the recommended play — then optionally drafts
the account brief with an LLM.

It uses three SIE primitives — `extract`, `encode`, and `score` — in one
round-trip, plus an optional `chat/completions` call for the brief. Everything
runs locally on CPU with `docker compose`; the LLM brief is opt-in.

> Contributed from a `{Tech: Europe}` London AI Hackathon project (codename
> "Rick") by the Attio-integration team. A cleaned-up, self-contained slice of a
> larger CRM churn-rescue + expansion product.

## What this demo actually shows

1. **Signals → one score (deterministic).** Each signal has a direction (risk
   vs. opportunity) and a severity weight. They roll up into a single score and
   a red/amber/green band. This part is plain, auditable arithmetic — you always
   know *why* an account is where it is.
2. **The account's story → the right play (SIE).** The score alone doesn't tell
   you what to *do*. So the account context is written as prose, `encode`d, and
   ranked (cosine → cross-encoder `score`) against a corpus of past outcomes
   ("champion left → renewed with new champion", "seat cap → expansion signed").
   The top-matched playbook drives the recommended play.
3. **A brief a human can send (optional LLM).** With `SIE_CHAT_MODEL` set, SIE
   drafts a summary / drivers / recommended-play brief via the OpenAI-compatible
   `/v1/chat/completions` endpoint, grounded in the matched playbook. Without it,
   a deterministic brief is written from the account's own data — so the board is
   always populated.

The UI streams every stage over SSE so you can watch extract → encode → score →
brief happen live.

## Run it locally

Requires Docker and Node 22+.

```bash
cd examples/account-signal-scoring
npm install
cp .env.example .env        # optional; defaults work out of the box
npm start                   # docker compose up -d + the UI server
```

`npm start` boots a local SIE server (CPU image, ~440 MB of models) and opens
the UI on <http://localhost:3044>. The first request builds the playbook index
(`data/playbook_index.json`) by encoding the corpus once; subsequent runs reuse
it.

Prefer the terminal? Rank every account into one board without the UI:

```bash
npm run score               # score all accounts, print the ranked boards
npm run score helix         # score a single account, verbose
```

### Turning on the LLM brief

The brief is deterministic by default so the demo stays CPU-only. To have SIE
draft it instead, point `SIE_CHAT_MODEL` at a generation model your cluster
serves (this needs the SIE generation bundle / a GPU — it is *not* in the
CPU compose file):

```bash
# .env
SIE_CHAT_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

The account panel header shows which mode you're in (`brief: deterministic` vs.
`brief: LLM (...)`).

## Specific things to try in the UI

| Account | Signals | Expected |
|---|---|---|
| **Pemberton & Co** | Stripe downgrade staged + NPS drop | **red**, top of risk board (cancellation short-circuits to 100) |
| **CloudScale Systems** | API usage down 41% + stale escalations | **red**, matches `usage_cliff_core_feature` |
| **Helix Robotics** | champion left + seats down 33% | **red/amber**, matches `champion_departed` |
| **Global Peak Inc** | 94% seat capacity, hard cap in ~8 days | **red** on the *expansion* board, matches `seat_capacity_expansion` |
| **Kestrel Bank** | renewed early + CSAT 9.6 | **green**, matches `early_renewal_advocate` |

Watch how two accounts with a similar *number* can get different plays: the
reranker matches each one to the playbook whose story fits.

## What the numbers in the UI mean

- **Signal score (0-100).** The deterministic roll-up. Risk signals add,
  opportunity signals subtract, each weighted by severity; a `usage_drop` is
  additionally scaled by its magnitude. An active `stripe_cancellation`
  short-circuits to 100 (red).
- **Reranker score (per playbook).** The cross-encoder (`BGE-reranker-base`)
  relevance of this account's summary to each shortlisted playbook. The top-3
  cosine candidates are reranked; the highest-scoring one (highlighted) drives
  the play.
- **ARR at stake.** The account's ARR — what a save or an expansion is worth.

## Model lineup

| Stage | Model | Size | Role |
|---|---|---|---|
| Extract | `urchade/gliner_multi-v2.1` | 280 MB | zero-shot NER on the account summary (display) |
| Encode | `sentence-transformers/all-MiniLM-L6-v2` | 80 MB | 384-dim dense encoder for cosine retrieval |
| Score | `BAAI/bge-reranker-base` | 280 MB | cross-encoder reranker on the top-K playbooks |
| Chat (optional) | e.g. `Qwen/Qwen3-4B-Instruct-2507` | — | drafts the account brief via `/v1/chat/completions` |

The first three live in the `default` CPU bundle and are preloaded by
`compose.yml`. The chat model is opt-in and needs the generation bundle / a GPU.

## SIE features used

- **`extract`** — GLiNER zero-shot NER surfaces typed entities from the account
  context.
- **`encode`** — MiniLM turns the account story into a dense vector; the playbook
  corpus is pre-encoded once (`npm run index`).
- **`score`** — the cross-encoder reranks the cosine shortlist so the *right*
  playbook wins, not just the lexically closest one.
- **`chat/completions`** *(optional)* — OpenAI-compatible endpoint drafts the
  brief, grounded in the matched playbook.

One cluster, one round-trip, four model roles — no separate model server per
task.

## What's in the box

```
account-signal-scoring/
├── compose.yml              # local SIE server (CPU, 3 models preloaded)
├── data/
│   ├── accounts.json        # 8 sample accounts with signals
│   └── playbooks.json       # 8 past-outcome playbooks (the corpus)
├── src/
│   ├── config.ts            # models, thresholds, paths
│   ├── types.ts             # Account, Signal, Playbook, AccountBrief
│   ├── signals.ts           # signal catalog + deterministic score roll-up
│   ├── score.ts             # extract → encode → cosine → rerank → brief
│   ├── brief.ts             # LLM brief (chat/completions) + deterministic fallback
│   ├── events.ts            # typed SSE events
│   ├── index-build.ts       # one-time playbook-corpus encode
│   └── cli.ts               # headless "rank the whole book" scorer
└── web/
    ├── server.ts            # tiny HTTP + SSE server
    └── public/              # vanilla-JS UI (no build step)
```

## Extend it

- **Bring your own signals.** Edit `data/accounts.json`, or wire the loader to
  your CRM (the original project pulled from Attio + Stripe). The signal catalog
  and weights live in `src/signals.ts`.
- **Grow the corpus.** Add rows to `data/playbooks.json` and re-run
  `npm run index`. More outcomes → sharper play matching.
- **Swap models.** Change `src/config.ts` — any encoder/reranker in the SIE
  catalog works with the same code.

## Honest scope and known limits

- The sample data is synthetic and small; the reranker signal is meaningful but
  the corpus is a demo corpus, not a trained ranker.
- The deterministic score is intentionally simple and tunable — it's a starting
  point, not a churn model. The point of the demo is the *ranking + play*
  matching that SIE adds on top.
- The LLM brief is best-effort: any error falls back to the deterministic writer
  so the board never breaks.

## Built with

- Original hackathon project ("Rick") by the Attio-integration team from the
  `{Tech: Europe}` London AI Hackathon.
- [SIE](https://github.com/superlinked/sie) — self-hosted inference for agents.
- Apache 2.0.
