# SIE integration

SIE is **opt-in**. `./start-services.sh` and default Compose start the server +
dashboard only — they never start SIE.

**Source of truth:**
[docs/user-guide/sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md).

## Happy path — remote gateway (recommended)

Finish [Getting started](./getting-started.md) first so `:8001` is up.

In the project `.env`:

```bash
SIE_ENABLED=true
SIE_ENDPOINT=https://your-sie-gateway.example.com
SIE_API_KEY=your_gateway_token
```

Restart or reload the rag-params-finder server after changing `.env`.

Check the gateway, then the app health:

```bash
curl -H "Authorization: Bearer $SIE_API_KEY" "$SIE_ENDPOINT/healthz"
curl -s http://localhost:8001/health
# → "sie":"reachable"
```

**First success:** `"sie":"reachable"`, then one sweep:

```bash
# from the rag-params-finder repo root, with CLI installed (see project QUICKSTART)
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
