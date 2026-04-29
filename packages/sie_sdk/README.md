# SIE SDK

Python client SDK for the SIE inference server.

## Installation

```bash
pip install sie-sdk
```

## Quick Start

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Encode text
result = client.encode("BAAI/bge-m3", Item(text="Hello world"))
print(result.dense)  # [0.1, 0.2, ...]

# Score pairs
scores = client.score("cross-encoder", [
    (Item(text="query"), Item(text="document"))
])
```

## Handling resource exhaustion

The SDK automatically retries requests that the server signals as
transient — model still loading, scale-from-zero in progress, or **GPU
memory pressure (`RESOURCE_EXHAUSTED`)**. You don't have to write
retry logic for these.

### What happens by default

When the server's GPU runs out of memory mid-request, the worker first
attempts an internal recovery (clear cache → evict an idle sibling
model → recursively halve the batch). If that succeeds you get a normal
200 response — slightly slower than usual.

If recovery is exhausted, the server returns `503 RESOURCE_EXHAUSTED`
with a `Retry-After: 5` header. The SDK then retries with bounded
exponential backoff (5s → 10s → 20s, capped at 30s, max 3 attempts).
The first retry logs at WARNING so you can see it at default log
levels:

```text
WARNING sie_sdk.client.sync: Server resource exhausted, retrying in 5.0s (attempt 1/3, elapsed: 0.4s, timeout: 900.0s)
```

If all retries are exhausted, the SDK raises
`sie_sdk.client.errors.ResourceExhaustedError` (a subclass of
`ServerError`).

### Tuning the behaviour

| Parameter | Default | Effect |
|--|--|--|
| `max_oom_retries=N` | `3` | Cap on auto-retries. Pass `0` to fail fast. |
| `provision_timeout_s=T` | `900` (15 min) | Total wall-clock budget. OOM retries are clamped to the remaining budget — you'll never sleep past your timeout. |

### Examples

**Default (resilient) — recommended for most callers:**

```python
result = client.encode("BAAI/bge-m3", Item(text="Hello"))
# Auto-retries on RESOURCE_EXHAUSTED. May take up to ~35s extra
# if recovery + retries are needed.
```

**Fail-fast (CI tests, latency-critical hot paths):**

```python
from sie_sdk.client.errors import ResourceExhaustedError

try:
    result = client.encode(
        "BAAI/bge-m3",
        Item(text="Hello"),
        max_oom_retries=0,  # No retries; surface failure immediately
    )
except ResourceExhaustedError:
    # Server is under memory pressure — fall back to a smaller model,
    # batch later, or surface to the user.
    ...
```

**Tight wall-clock budget:**

```python
result = client.encode(
    "BAAI/bge-m3",
    Item(text="Hello"),
    provision_timeout_s=10.0,  # Total budget; OOM retries clamped to it
)
```

### What you'll see in your logs

| Server state | Client outcome | Log level |
|--|--|--|
| GPU OK | 200, normal latency | (none) |
| OOM, server-side recovery succeeds | 200, +1-3s latency | (none) |
| OOM, SDK retries succeed | 200, +5-35s latency | WARNING on 1st retry |
| OOM, SDK retries exhausted | `ResourceExhaustedError` | WARNING + traceback |

If you see frequent `Server resource exhausted, retrying...` warnings,
your cluster's GPU pool is undersized for the workload. Talk to the
operator running SIE — they have observability and tuning knobs
(`SIE_OOM_RECOVERY__*`) that aren't visible from the SDK side.

## License

Apache 2.0
