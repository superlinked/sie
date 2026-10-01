# Wire-contract golden fixtures

Language-neutral golden fixtures for the shapes that cross the SIE wire and are
otherwise hand-maintained in several codebases (gateway `rs`, server `py`, SDK
`py`, SDK `ts`). Each implementation round-trips these fixtures in its own CI so
**drift is caught in CI, not production** — the parity promise becomes
executable.

## Files

- `model_state.json` — the canonical `ModelState` values (`available`,
  `loading`, `loaded`, `unloading`, `failed`).
- `request_usage.json` — how the response `usage` block is partitioned. The
  gateway is the only writer of `settled_charge_fields`; everything a consumer
  meters lives in `terminal_unit_fields`. The two sets are disjoint and their
  union is the WHOLE block, which is what lets a consumer reject an undeclared
  key — a silently renamed meter — without also rejecting the settled charge
  that legitimately rides alongside it. See issue #3063 for the failure this
  exists to prevent.
- `model_info.json` — the field set of one `/v1/models` entry, split into
  `typed` (an SDK `ModelInfo` MUST declare it) and `excluded` (an SDK MUST NOT
  declare it, with the reason recorded per key).
- `oss_payload_store.json` — the cross-language Alibaba OSS payload contract.
  The gateway writes `prefix/plain_key`, the queue retains `plain_key`, and the
  sidecar accepts that key or the exact `full_reference`; Python and both Rust
  binaries load the same fixture to prevent prefix drift.
- `retry_classification.json` — how the SDKs classify a non-2xx response to a
  buffered call: `retry` or `terminal` for the idempotent operations (encode,
  score, extract) and for generation, keyed by status, error code, and whether
  a usable `Retry-After` is present. It also pins how a `Retry-After` value is
  parsed (delay seconds or an HTTP date; a past date means retry now).
- `serving_disclosure.json` — remote serving: the `X-SIE-Remote: forbid`
  request header a caller sends to keep a request off remote upstreams, and the
  response headers that say which side served (`X-SIE-Served-By`,
  `X-SIE-Upstream`, `X-SIE-Fallback-Reason`, `X-SIE-Fallback-Error`), each with
  its allowed values or pattern. An SDK drops a value outside them.
- `worker_status.json` — the worker heartbeat the sidecar publishes on
  `sie.health.<worker_id>`: the exact `fields` it sends, the
  `omitted_when_empty` fields it leaves out when empty, and a fully populated
  `example`. A reader treats a missing field as its default, which is how an
  older worker's heartbeat without `unsupported_models` stays valid.

### Why `model_info.json` has two buckets

A "covered set" alone only catches fields an SDK forgot. It says nothing about
fields an SDK left out *on purpose*, so the next reader cannot tell an omission
from a decision — which is how `state`, `last_error`, `profiles` and
`pending_generation` sat undeclared in both SDKs for several releases. Listing
both buckets makes a new wire field fail the tests until someone consciously
puts it in one of them.

Today `excluded` holds only the OpenAI retrieve-model compat keys
(`id`/`object`/`created`/`owned_by`) that `GET /v1/models/{model}` merges in for
vanilla OpenAI clients.

## Adding a consumer

Point a test at the JSON and assert the implementation's enum/type matches the
fixture set. Current consumers:

- Python SDK — `packages/sie_sdk/tests/test_wire_contract.py` (asserts
  `typing.get_args(ModelState)` and `ModelInfo.__annotations__` match the
  fixtures, and that
  `RequestUsage`/`TERMINAL_UNIT_FIELDS`/`SETTLED_CHARGE_FIELDS` match
  `request_usage.json`), and
  `packages/sie_sdk/tests/client/test_retry_classification.py` (replays every
  `retry_classification.json` case through both clients against a local HTTP
  server). `test_wire_contract.py` also asserts the header names, values and
  patterns of `serving_disclosure.json` against the client constants and
  `RequestMetadata`.
- TypeScript SDK — `packages/sie_ts_sdk/tests/wireContract.test.ts` (asserts the
  runtime `MODEL_STATES` and `MODEL_INFO_WIRE_FIELDS` arrays — the single
  sources the `ModelState` type and `WireModelInfo` interface are checked
  against — match the fixtures), and
  `packages/sie_ts_sdk/tests/retryClassification.test.ts` (replays every
  `retry_classification.json` case through the client with a stubbed `fetch`),
  and `packages/sie_ts_sdk/tests/remoteServing.test.ts` (checks the runtime
  `SERVED_BY_VALUES`, `FALLBACK_REASONS` and header patterns against
  `serving_disclosure.json`, then sends and parses the headers through the
  client).
- Gateway — `packages/sie_gateway/src/handlers/serving_disclosure.rs`
  (asserts that the `X-SIE-Served-By` and `X-SIE-Upstream` names and the
  served-by values it emits are the ones `serving_disclosure.json` declares).
- Worker sidecar — `packages/sie_server_sidecar/src/health_publisher.rs`
  (the published key set equals `fields`, minus `omitted_when_empty` when those
  are empty) and gateway — `packages/sie_gateway/src/types/worker.rs` (the
  `example` and an older heartbeat without the omitted fields both parse) load
  `worker_status.json`.
- Downstream gateways assert that the members they inject are exactly
  `settled_charge_fields`, so a gateway that starts publishing a third field
  cannot reach production before every consumer has declared it.

The TS SDK carries extra tests the Python SDK does not need: its client re-maps
the `/v1/models` response (camelCasing three top-level keys), so a field can be
declared on the type and still never reach a caller. Those tests drive a fully
populated wire entry through `getModel`/`listModels`.

## Scope

This started as one slice (issue #1637): `ModelState` only. `request_usage.json`
(issue #3063) is the second, `model_info.json` (issue #3126) the third. Extend
the same directory with `ModelCapabilities` values, error codes, and status
messages as they are pinned. Codegen can come later if fixtures prove
insufficient.
