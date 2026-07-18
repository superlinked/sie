# ABI course: managed SIE smoke test

This is the smallest possible course-participant path to managed SIE: one
deterministic text embedding request using only the Python standard library.
It prints a compact result with vector dimensions, a short numeric preview,
latency, request ID, usage, and any safe metering headers returned by the API.

> [!IMPORTANT]
> The code and mocked tests are complete, but this example has **not yet been
> validated against US production**. Live validation depends on the course
> model being present in the production catalog and route.

## Run from a clean environment

Python 3.12 is the only dependency. No packages need to be installed.

```bash
cd examples/abi-course
python3.12 -m venv .venv
source .venv/bin/activate

export SIE_API_KEY="<your course API key>"
export SIE_BASE_URL="https://api.superlinked.com"
python us_prod_smoke.py
```

Never paste an API key into this script, a notebook, command-line argument, or
source-controlled file. The script reads it only from `SIE_API_KEY`, sends it
as a bearer credential to the explicitly configured HTTPS origin, and never
prints or persists it. Redirects are refused to avoid forwarding that
credential elsewhere.

The default is `BAAI/bge-m3`, a small-input retrieval model in the proposed
course catalog. Override it only when the final catalog specifies another
model:

```bash
export SIE_MODEL="<exact model ID from the course catalog>"
python us_prod_smoke.py
```

A successful response is one JSON line shaped like this:

```json
{
  "client_latency_ms": 123.4,
  "dimensions": 1024,
  "l2_norm": 1.0,
  "model": "BAAI/bge-m3",
  "ok": true,
  "preview": [0.012345, -0.023456],
  "request_id": "<request ID>",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

Metering and server timing fields appear only when the service returns them.
Their presence must be checked during live production validation.

Expected failures are also one JSON line on stderr:

| Error | Meaning |
|---|---|
| `CONFIG_MISSING`, `CONFIG_INVALID` | Required environment setup is absent or unsafe |
| `AUTH_INVALID` | The key is missing, invalid, or inactive (`401`) |
| `CREDITS_EXHAUSTED` | The account lacks enough credits (`402`) |
| `ACCESS_FORBIDDEN` | The key cannot use the endpoint or model (`403`) |
| `MODEL_UNAVAILABLE` | The model is not available on the route (`404`) |
| `RATE_LIMITED` | The service rate limit was reached (`429`) |
| `SERVICE_ERROR` | The managed service returned a `5xx` |
| `NETWORK_ERROR`, `INVALID_RESPONSE` | The endpoint was unreachable or malformed |

## Test

The test suite uses only in-memory mocked responses and never contacts SIE:

```bash
python3.12 -m unittest discover -s tests -v
```

## Dependencies before calling this production-validated

- The catalog release must freeze the exact course model ID, revision, and
  serving recipe. The current default assumes `BAAI/bge-m3`.
- The production release must expose that model through a healthy US route.
- The production usage/credit metadata contract must be settled. This smoke
  reports compatible headers when present but does not assume an unsettled
  credit-cost header name.
- A fresh course account and key must then run this exact command from a clean
  environment, with request ID, latency, usage, credit deduction, and support
  recovery recorded.

The deeper course-project candidates are inventoried in
[CANDIDATES.md](./CANDIDATES.md); the third project remains deliberately open
until the course roster and production evidence are available.
