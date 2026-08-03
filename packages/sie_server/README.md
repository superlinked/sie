# SIE Server

GPU inference server for embeddings, reranking, and entity extraction.

## Features

- Multi-model serving with LRU eviction
- Token-based dynamic batching
- Hot reload model configs without restart
- Unified API: `encode()`, `score()`, `extract()`
- Prometheus metrics and OpenTelemetry tracing
- Reactive GPU OOM recovery + proactive idle eviction

## Installation

```bash
pip install sie-server
```

`sie-server` ships several model bundles, and a single environment can only hold one
`transformers` version — install the one your bundle needs:

- **Default bundle** (embeddings, reranking, extraction) — verified on `transformers` 4.x:

  ```bash
  pip install sie-server "transformers<5"
  ```

- **OCR-VLM bundle** (LightOnOCR, GLM-OCR) — requires `transformers` 5.x, and is served with `-b transformers5`:

  ```bash
  pip install sie-server "transformers>=5,<6"
  sie-server serve -b transformers5
  ```

## Quick Start

```bash
sie-server serve --port 8080 --device cuda:0
```

## Configuration

`sie-server` reads its config from `SIE_*` environment variables (Pydantic
`BaseSettings`). Common knobs:

### Memory & OOM resilience

| Env var | Default | Effect |
|--|--|--|
| `SIE_MEMORY_PRESSURE_THRESHOLD_PERCENT` | `95` | VRAM utilisation that triggers reactive LRU eviction by the pressure monitor. |
| `SIE_OOM_RECOVERY__ENABLED` | `true` | Master switch for reactive OOM recovery in the worker dispatch path (`cache_clear → evict_lru → split_batch`). |
| `SIE_OOM_RECOVERY__STRATEGY` | `cache_clear,evict_lru,split_batch` | Ordered recovery actions. Earlier actions tried first. |
| `SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH` | `4` | Cap on recursive batch halving (≤16 sub-batches). |
| `SIE_OOM_RECOVERY__EVICTION_LOCK_TIMEOUT_S` | `5.0` | Soft timeout when waiting for the registry's load-lock during recovery eviction. |
| `SIE_OOM_RECOVERY__RETRY_AFTER_S` | `5` | `Retry-After` header value on `RESOURCE_EXHAUSTED` responses. |
| `SIE_DISABLE_OOM_RECOVERY` | unset | Convenience kill switch (`1`/`true`/`yes`) for incident triage. Wins over `SIE_OOM_RECOVERY__ENABLED=true`. |
| `SIE_IDLE_EVICT_S` | unset (disabled) | Unload models that have been idle longer than this (seconds). Additive to the pressure monitor; helps free cold weights before pressure builds. |
| `SIE_OOM_NAK_DELAY_S` | `10.0` | Queue-mode only. NAK delay (seconds) for `RESOURCE_EXHAUSTED` work items so JetStream redelivers them after memory pressure has had a chance to clear. |

When OOM recovery is exhausted on a request, the server returns
`HTTP 503 RESOURCE_EXHAUSTED` with `Retry-After`. The Python SDK
auto-retries; see `packages/sie_sdk/README.md` for client-side controls.

### Batching & request handling

| Env var | Default | Effect |
|--|--|--|
| `SIE_MAX_BATCH_REQUESTS` | `64` | Maximum number of items per batched inference call. |
| `SIE_MAX_BATCH_WAIT_MS` | `10` | How long the batch-former waits for additional items before dispatching. |
| `SIE_MAX_CONCURRENT_REQUESTS` | `512` | Per-worker queue size; admission control returns `QUEUE_FULL` above this. |
| `SIE_MAX_LORAS_PER_MODEL` | `10` | Maximum concurrent LoRA adapters per base model. |

### Compute & precision

| Env var | Default | Effect |
|--|--|--|
| `SIE_DEFAULT_COMPUTE_PRECISION` | `float16` | One of `float16`, `bfloat16`, `float32`. |
| `SIE_ATTENTION_BACKEND` | `auto` | One of `auto`, `flash_attention_2`, `sdpa`, `eager`. |

### Diagnostics

| Env var | Default | Effect |
|--|--|--|
| `SIE_GRAMMAR_PREFLIGHT_DEBUG` | unset (off) | Enables the legacy worker-side Outlines preflight compile before each structured-output request. Off by default because SGLang is the production grammar authority. Use for diagnosing schema-rejection problems or slow compiles in a controlled environment; not recommended for production traffic. |

For nested settings (any field with `__`), the env-var format is
`SIE_<TOP>__<NESTED>=value`. The complete schema is in
`packages/sie_server/src/sie_server/config/engine.py`.

## Observability

The server emits the checked-in worker telemetry contract once through
OpenTelemetry/OTLP. The regional collector owns Prometheus exposition and
optional remote OTLP routing; the application does not expose `/metrics` or
maintain a second Prometheus registry. See `telemetry/contract.yaml` for exact
instrument names, dimensions, ownership, and histogram bounds.

## API

See the [API documentation](https://sie.dev/docs) for details.

## License

Apache 2.0
