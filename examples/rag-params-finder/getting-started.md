# Getting started (without SIE yet)

Goal: clone the external repo, start the local MongoDB stack, and open the
dashboard. Enable SIE in [SIE integration](./sie-integration.md) after this works.

**Source of truth:**
[QUICKSTART.md](https://github.com/neomatrix369/rag-params-finder/blob/main/QUICKSTART.md)
in the project repo.

## Prerequisites

- Git
- Docker Desktop (running)
- For host CLI later: Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

No Atlas account, Voyage key, or SIE credentials are required for this path.

## Clone and start (MongoDB local)

```bash
git clone https://github.com/neomatrix369/rag-params-finder.git
cd rag-params-finder
cp .env.example .env
./start-services.sh --mongodb-local
```

Open **http://localhost:5374**. The stack starts MongoDB Atlas Local, the API
server (`:8001`), and the dashboard. SIE is **not** started — that is intentional.

Verify the API:

```bash
curl -s http://localhost:8001/health
```

You should see the server healthy. With default `.env`, SIE reports as
`"sie": "disabled"`.

## Other storage paths

| Path | When | Doc |
|---|---|---|
| Atlas cloud | You already have `MONGODB_URI` | [QUICKSTART Path B](https://github.com/neomatrix369/rag-params-finder/blob/main/QUICKSTART.md) |
| Postgres / pgvector | Prefer Supabase or local Postgres | [Postgres setup](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/postgres-setup.md) |
| Manual two-terminal | No Docker for the app | [QUICKSTART Path C](https://github.com/neomatrix369/rag-params-finder/blob/main/QUICKSTART.md) |

Step-by-step install and first experiment:
[getting-started.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/getting-started.md).

## Next

Before the host CLI / SIE handoff, export the URI the startup script printed
(leave `.env` placeholders unchanged for Atlas Local):

```bash
# value also printed by ./start-services.sh --mongodb-local
export MONGODB_URI="mongodb://localhost:27017/rag_params_finder?directConnection=true"
```

Then wire SIE and run one sweep: [SIE integration](./sie-integration.md).
