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

### TensorRT-LLM generation

TensorRT-LLM buffers completion token IDs and emits the final text together only
after generation finishes and the terminal stream is verified, even with
`stream=True`. This preserves context-sensitive spacing and punctuation and
removes matched stop sequences consistently from returned text, completion-token
usage, and logprobs. Long completions delay the first visible text; existing
request timeouts still apply. Other adapters are unaffected.

### GLiClass usage

For GLiClass classification, `usage.input_tokens` counts each item's document
tokens plus the instruction and few-shot example texts sent with the request,
because the model encodes that text again for every item. Label names,
including the labels attached to examples, are not counted. With an
instruction or examples, an item's count is capped at the model's
`max_sequence_length` minus the label prompt, unless the document count alone
is already higher. An item refused because its document pushes the labels out
of the window returns a per-item `INPUT_TOO_LONG` error and counts nothing. A
request that sends no instruction or examples is counted exactly as before.

The instruction and each example text may be at most 2,048 characters, and
together with the example labels at most 8,192 characters. Up to 32 examples
are accepted, and they must leave room for the document in the model window.
Label names are refused when their total length exceeds 16 characters per
token of the window (8,192 characters for a 512-token model), more than any
label prompt can fit.

### GLiClass CUDA graphs

A GLiClass forward on a DeBERTa encoder launches about a thousand small GPU
kernels, so at small batch sizes the CPU that launches them is the bottleneck.
A CUDA graph records the kernel launches of one input shape once; a replay
launches them all in one call. Graphs are an operator setting, fixed when the
model loads, in the model profile:

```yaml
profiles:
  default:
    adapter_options:
      loadtime:
        cuda_graphs: bucketed
```

| Value | Shapes recorded | Scores |
|--|--|--|
| `off` (default) | none | eager |
| `exact` | each (batch, sequence length, label slots) seen twice | bit-identical to eager |
| `bucketed` | sequence lengths padded up to a multiple of 32 tokens (64 on 1,024-token models) | padding moves fp16 probabilities (see below) |

A request can send `options={"cuda_graphs": "off"}` to run eagerly on a model
loaded with graphs. It cannot turn graphs on: any other value is refused.

Padding is masked, but a longer sequence rounds fp16 sums differently, the same
kind of change batching requests together makes. We compared `bucketed` with
eager execution on the 384 CVE descriptions from `examples/typed-decisions`,
three questions each, asked one at a time, as separate groups and as joint
groups, plus 65 long documents. Each input was sent three times (long
documents twice), so that graphs were recorded and then replayed: 11,538
answers per model. A small change can still flip a near tie between the top
two labels:

| Model | Largest probability change | Top label changed | Shipped profile |
|--|--|--|--|
| `gliclass-small-v1.0` | 0.0039 | 3 answers | `off` |
| `gliclass-base-v1.0` | 0.0049 | none | `bucketed` |
| `gliclass-large-v1.0` | 0.0056 | none | `bucketed` |
| `gliclass-base-v3.0` | 0.0054 | 3 | `off` |
| `gliclass-large-v3.0` | 0.0093 | 3 | `off` |
| `gliclass-instruct-base-v1.0` | 0.0076 | 12 | `off` |
| `gliclass-instruct-large-v1.0` | 0.0098 | 18 | `off` |
| `opir-multitask-large-v1.0` | 0.0144 | none | `bucketed` |
| `gliclass-multilang-mini` (100 descriptions, no joint groups: 2,016 answers) | 0.0227 | 3 | `off` |

`exact` changed nothing.

The shipped profiles load with `bucketed` graphs only where no top label
changed: `gliclass-base-v1.0`, `gliclass-large-v1.0` and
`opir-multitask-large-v1.0`. Their probabilities can differ from eager
execution by up to the amounts above. To run one of them eagerly, set
`cuda_graphs: off` in its profile, or send `options={"cuda_graphs": "off"}` with
a request. The other models load with `off`.

Graphs apply on CUDA to the DeBERTa-based GLiClass models: the v1.0 models,
`gliclass-base-v3.0` and `gliclass-large-v3.0`, the base and large instruct
models, the Opir multitask models and `gliclass-multilang-mini`. The
ModernBERT-based models (the edge models, `gliclass-multilang-edge` and the
Opir edge models), CPU and MPS run eagerly with any value, and the load logs a
warning.

Nothing is recorded at load. A shape is recorded the first time a request
needs it (the second time in `exact` mode), and that request takes a little
over two eager forwards (76 ms against 33 ms on `gliclass-large-v1.0` on an
L4). A model keeps at most 64 graphs, least recently used first
out. A graph holds at most 2,048 tokens (batch times padded length); larger
forwards are bound by the GPU rather than by kernel launches and run eagerly.

**Memory, and other models on the same GPU.** Graph memory counts as device
memory in use, but it is not attributed to the model: under memory pressure
the server evicts whole models, least recently used first, which may be
another model. A model's graphs share one memory pool, and the driver keeps a
copy of each graph: about 8 MB for a `gliclass-large-v1.0` forward. On
`gliclass-large-v1.0` in `bucketed` mode, the 31 graphs recorded for the CVE
descriptions above took 555 MB of device memory: 490 MB for the pool and
the graphs, 60 MB cached on the recording stream (its cuBLAS workspace and one
warm-up row) and 5 MB of relative-position tables. The pool keeps what evicted
graphs used, so traffic with many shapes keeps growing it. The runner therefore
adds up the device memory its graphs hold (what each recording took, plus the
tables and buffers they read), and past 4% of the device's memory (900 MB on
an L4) it drops every graph, returns their memory to the device and records
again. Graphs are also released when the model unloads, and
when one of its forwards runs out of memory.

While a graph records, PyTorch's caching allocator does not free cached blocks
to satisfy other allocations, so another model on the same GPU that needs
memory in that window (about one forward) can run out of memory where it
otherwise would not. Recording is kept rare to limit this: one recording at a
time in the process, none while less than a tenth of the device's memory is
free, and per model at most 16 recordings at once, then one per 2 seconds. If a
recording itself runs out of memory, the request still gets its eager answer;
the model drops its graphs and records nothing for a minute.

Both limits are approximate. The free-memory check reads the device once,
before recording, so a model loading at the same moment can still meet one
recording. The memory budget is checked after each recording, so a model's
graphs can exceed it by one recording: up to about 330 MB, one graph of the
largest shape on `gliclass-large-v1.0`. On a GPU shared with other models,
leave memory headroom, or enable graphs only where the model has the GPU to
itself. Usage and billing do not change.

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
| `SIE_MAX_BATCH_WAIT_MS` | `15.0` | Initial value for the adaptive first-request batch timeout. At runtime the PI batching controller steers it between `SIE_ADAPTIVE_BATCHING__MIN_WAIT_MS` and `SIE_ADAPTIVE_BATCHING__MAX_WAIT_MS`, so it is a starting point rather than a fixed wait. |
| `SIE_MAX_CONCURRENT_REQUESTS` | `512` | Per-worker queue size; admission control returns `QUEUE_FULL` above this. |
| `SIE_MAX_LORAS_PER_MODEL` | `10` | Maximum concurrent LoRA adapters per base model. |
| `SIE_MAX_ITEM_TEXT_BYTES` | `2097152` (2 MiB) | Most bytes of UTF-8 one encode, score, or extract item may carry in its `text` and `metadata` together. A larger item is rejected with `INVALID_INPUT` (HTTP 400) before it is tokenized. Generation prompts are not items and are not affected. |

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
