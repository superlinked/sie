# @superlinked/sie-sdk

Official TypeScript SDK for SIE (Search Inference Engine). Async-only,
built on native `fetch`, works in Node.js (>= 22) and the browser.

## Installation

```bash
npm install @superlinked/sie-sdk
```

## Creating a client

```typescript
import { SIEClient } from "@superlinked/sie-sdk";

// Local server
const client = new SIEClient("http://localhost:8080");

// Managed SIE gateway: pass the gateway URL and your API key
// (sent as a Bearer token). Server-side code only — see below.
const managed = new SIEClient("https://your-gateway.example.com", {
  apiKey: "YOUR_API_KEY",
});
```

In Node.js, an omitted `baseUrl` or `apiKey` falls back to the `SIE_BASE_URL`
or `SIE_API_KEY` environment variable. An explicit value always wins;
`apiKey: ""` sends no credential.

> **Warning:** only pass `apiKey` in server-side code. Shipping it in a
> browser bundle exposes the bearer key to anyone who loads the page.
> For browser apps, route requests through a backend proxy that holds
> the key.

> **Cold-start continuations:** non-streaming requests through a managed
> gateway may receive a bounded Modal result continuation. Consuming it safely
> requires a server-side Fetch runtime (Node.js >= 22) that exposes the status
> and `Location` of a `redirect: "manual"` response. Browser Fetch exposes an
> opaque redirect instead, so the SDK fails closed; route these browser calls
> through a backend proxy as well.

## Encoding

```typescript
// Single item — result.dense is a Float32Array
const result = await client.encode("BAAI/bge-m3", { text: "Hello world" });
console.log(result.dense?.length); // 1024

// Batch — results come back in input order
const results = await client.encode("BAAI/bge-m3", [
  { text: "First document" },
  { text: "Second document" },
]);
```

## Scoring (reranking)

```typescript
const scored = await client.score(
  "BAAI/bge-reranker-v2-m3",
  { text: "What is machine learning?" },
  [
    { id: "doc-1", text: "Machine learning is a subfield of AI." },
    { id: "doc-2", text: "Python is a programming language." },
  ],
);
// Sorted by relevance (rank 0 = most relevant)
console.log(scored.scores[0].itemId, scored.scores[0].score);
```

## Generation

Text-only `generate` and `streamGenerate` prompts are raw continuation input:
the worker preserves them without rendering a chat template or applying
`enable_thinking` / `guardian_config`. Native requests with images render one
user turn. For chat, instruction-based structured output (`responseFormat`),
and guard checks, use `chatCompletions` or `streamChatCompletions` with messages
so the worker applies the served template settings. Operator settings take
precedence over request template kwargs.

Granite Guardian's shipped risk dimension is `harm`; requesting another
dimension in prompt text does not change it. Valid thresholded verdicts are
`Yes` (unsafe) and `No` (safe); missing or invalid verdicts fail with
`invalid_guard_verdict`. Never interpret an error or empty output as safe.
Reasoning remains private; a budget exhausted without usable output fails with
`empty_model_output`.

```typescript
// Aggregated result
const gen = await client.generate(
  "Qwen/Qwen3-4B-Instruct-2507",
  "Write a haiku about the sea.",
  { maxNewTokens: 64, temperature: 0.7 },
);
console.log(gen.text, gen.finishReason, gen.usage);

// Streaming (SSE) — chunks as they arrive (Node.js example;
// process.stdout is not available in the browser)
for await (const chunk of client.streamGenerate(
  "Qwen/Qwen3-4B-Instruct-2507",
  "Write a haiku about the sea.",
  { maxNewTokens: 64 },
)) {
  process.stdout.write(chunk.text_delta);
  if (chunk.done && chunk.ttft_ms !== undefined) {
    console.log(`\nTTFT: ${chunk.ttft_ms}ms`);
  }
}
```

## Error handling

All errors extend `SIEError` and are exported from the package root.
Server-reported errors carry `.code` and `.statusCode`, plus
`.requestId` (the gateway `x-sie-request-id`) when the gateway included
one in the response — check it before using it.

```typescript
import {
  ModelLoadingError,
  RequestError,
  ResourceExhaustedError,
  ServerError,
  SIEStreamError,
} from "@superlinked/sie-sdk";

try {
  const res = await client.encode("BAAI/bge-m3", { text: "Hello" });
} catch (error) {
  if (error instanceof RequestError) {
    console.error(`Bad request (${error.code}):`, error.message);
  } else if (error instanceof ServerError) {
    const correlation = error.requestId ? ` [request ${error.requestId}]` : "";
    console.error(`Server error (${error.code})${correlation}:`, error.message);
  }
}
```

Retry semantics for transient `503` codes:

- `PROVISIONING` — the cluster is scaling capacity from zero. Retried
  automatically while `waitForCapacity: true` (the default); with
  `waitForCapacity: false` it throws `ProvisioningError` immediately.
- `MODEL_LOADING` — the worker is cold-loading the model. The SDK
  retries automatically until `provisionTimeout` (default 900000 ms),
  then throws `ModelLoadingError`.
- `LORA_LOADING` — the requested LoRA adapter is still loading. Retried
  a bounded number of times, then throws `LoraLoadingError`.
- `RESOURCE_EXHAUSTED` — server-side GPU OOM. Retried with bounded
  exponential backoff; throws `ResourceExhaustedError` (a `ServerError`
  subclass) when retries run out.

Admission rejections happen before any work is dispatched, so every call
retries them within `provisionTimeout`, honouring `Retry-After` (delay
seconds or an HTTP date): `429 RATE_LIMIT`, `503 QUEUE_FULL`, and the
gateway's queue backpressure, a `503 QUEUE_UNAVAILABLE` that carries
`Retry-After`. A `QUEUE_UNAVAILABLE` without `Retry-After` is terminal.
[`packages/wire-fixtures/retry_classification.json`](../wire-fixtures/retry_classification.json)
lists how each response is classified; the TypeScript and Python SDKs
both test against it.

Each attempt waits up to `timeoutMs` (default 150000 ms, longer than the
gateway's default 120 s request deadline) for a response. A timed-out
request is never retried, because the server may still be processing it.
`generate` and `chatCompletions` accept a per-call `timeoutMs`; set it to
at least the model profile's `overall_timeout_s` for long generations.
Connection failures are classified by their cause: refused, unreachable
or reset connections are retried under `waitForCapacity`, while DNS name,
TLS and certificate failures fail immediately.

When a batch response carries fewer items than were sent, `encode` and
`extract` throw `IncompleteBatchError`. Its `results` holds the items
that did come back (match them by `id`), and `missingIds` names the
dropped items when every submitted item carried an `id`.

Streaming calls throw `SIEStreamError` for mid-stream error chunks —
the HTTP connection was healthy but the worker or gateway emitted an
error envelope partway through. Branch on `error.code` (for example
`empty_model_output`, a terminal generation that produced no visible
text, or `MODEL_OUTPUT_PARSE_ERROR`, a strict grammar whose finished
output did not conform) and, when `error.requestId` is present, use it
to correlate with gateway logs. For a validated `RESOURCE_EXHAUSTED` terminal,
`error.retryAfter` carries the operator hint in milliseconds; its presence
does not make retrying generation safe after output has already arrived.

## License

MIT
