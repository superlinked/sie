# SIE integration

SIE is **opt-in**. `./start-services.sh` and default Compose start the server +
dashboard only — they never start SIE.

**Source of truth:**
[docs/user-guide/sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

## Happy path — remote gateway (recommended)

Finish [Getting started](./getting-started.md) first so `:8001` is up and
`MONGODB_URI` is exported for the host CLI.

In the project `.env`:

```bash
SIE_ENABLED=true
SIE_ENDPOINT=https://your-sie-gateway.example.com
SIE_API_KEY=your_gateway_token
```

Reload the server so it picks up the new env (a plain restart is not enough for
Compose — env is baked in at container create time):

```bash
# Compose (typical after ./start-services.sh)
docker compose up -d --force-recreate server

# Host-run server instead: reload or restart uvicorn
```

Source `.env` into the **current shell** before gateway curls (editing the file
does not update existing variables):

```bash
set -a && source .env && set +a
```

### Readiness checks

**1. Gateway process alive** (`/healthz` ≠ model ready):

```bash
curl -H "Authorization: Bearer $SIE_API_KEY" "$SIE_ENDPOINT/healthz"
# → ok
```

**2. Model can encode** — wait for HTTP **200** (503 during warm-up is expected):

```bash
until curl -sf -o /dev/null -X POST "$SIE_ENDPOINT/v1/encode/BAAI/bge-m3" \
  -H "Authorization: Bearer $SIE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"items":[{"text":"readiness probe"}]}'; do
  echo "SIE encode not ready yet — waiting 10s..."
  sleep 10
done
```

**3. App sees SIE:**

```bash
curl -s http://localhost:8001/health
# → "sie":"reachable"
```

### First success

Provide an input PDF (`input_data/` is gitignored). Either copy a file to the
path expected by the example config, or point `data_paths` at an existing PDF:

```bash
mkdir -p input_data/pdfs
cp /path/to/your-document.pdf \
  input_data/pdfs/The_Federal_Pell_Grant_Program.pdf
# or edit data_paths in configs/mongodb/example-sie.yaml
```

Then run one sweep (CLI installed per project QUICKSTART):

```bash
rag-params-finder run --config configs/mongodb/example-sie.yaml
```

Config file:
[configs/mongodb/example-sie.yaml](https://github.com/neomatrix369/rag-params-finder/blob/main/configs/mongodb/example-sie.yaml).
On Postgres/Supabase use
[configs/supabase/example-sie.yaml](https://github.com/neomatrix369/rag-params-finder/blob/main/configs/supabase/example-sie.yaml)
instead.

Compare results in the dashboard at **http://localhost:5374**.

## Alternate — self-hosted Docker

Use when you have no remote gateway. Needs Docker, disk for model weights, and
usually `HF_TOKEN` on the **SIE container** (not for app routing).

Typical host endpoint:

```bash
SIE_ENABLED=true
SIE_ENDPOINT=http://localhost:8720
# SIE_API_KEY usually unset for local unauthenticated server
```

Full `docker run` flags, warm-up (wait for encode **200**, not only `/healthz`),
Apple Silicon notes, and Aim UI:
[Self-hosted Docker in sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

When the server runs in Docker and SIE on the host, use
`http://host.docker.internal:8720` as `SIE_ENDPOINT`.

## Env vars (same for remote and local)

| Variable | Role |
|---|---|
| `SIE_ENABLED` | Master on/off (default `false`) |
| `SIE_ENDPOINT` | HTTP base URL of the SIE gateway or local server |
| `SIE_API_KEY` | Bearer token when the gateway requires auth |

Details and smoke tests (including `POST /api/v1/sweep`):
[sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

## Next

Understand models and primitives: [What SIE does here](./what-sie-does.md).
Stuck? [Troubleshooting](./troubleshooting.md).
