# OpenAI Decisions with SIE

SIE exposes `POST /v1/decisions` for its typed decision models, alongside
the TypeSafe-compatible `POST /v1/systemone` route. The OpenAI Python SDK
can call SIE by changing the base URL and model. This example pins SDK
3.26.0, which includes `client.decisions.create`.

Start a local model from the repository root:

```bash
mise run serve -- --models fastino/GLiNER2.5-Decide --host 127.0.0.1 --port 8080 --device cpu
```

Startup resolves the adapter dependencies. The first request downloads and
loads the pinned public checkpoint. Wait for `http://127.0.0.1:8080/readyz`,
then run:

```bash
mise exec -- uv run --frozen --project examples/decisions-local examples/decisions-local/decide.py
```

`conformance.py` checks all three answer types, question ordering, typed
choice values, text messages, the async client, and SDK error classes for
unknown models, unsupported inputs, and overflow. It allows up to five
minutes for a cold model to load; a ready worker can still return
`503 MODEL_LOADING` until the model is resident:

```bash
mise exec -- uv run --frozen --project examples/decisions-local examples/decisions-local/conformance.py
```

For an authenticated gateway, set `SIE_OPENAI_BASE_URL` to its URL with a
`/v1` suffix and `SIE_API_KEY` to your key. `SIE_DECISION_MODEL` selects a
different SIE decision model. This uses SIE inference and requires no
OpenAI account or OpenAI API key.

## Supported contract

- Text input or ordered text messages with user, developer, system, or
  assistant roles. Text content parts and assistant refusals are accepted.
- Ordered predicate, choice, and score questions. Answer order matches
  question order, including unnamed questions and repeated names.
- Choice values are strings or booleans; `true` and `"true"` remain distinct.
- Score answers include the probability of each ordered level and the
  expected zero-based level index.
- OpenAI error envelopes and retry headers, including a `400` error when
  a request exceeds a model's input window. SIE refuses overflow rather
  than silently truncating it.

The endpoint accepts up to 200 questions and 255 choices, but each model's
smaller prompt and task limits still apply. Scores support 1 to 10 levels.
For GLiNER2.5-Decide, all questions and input share a 512-token row, with
at most half available to question prompts; keep requests short or use a
larger decision model. See [System One model limits](../systemone-local/README.md#limits).

Images, audio, tool-call inputs, cache controls, and unknown request fields
return explicit errors. `safety_identifier` is accepted for client
compatibility but does not change model inference. SIE model IDs replace
OpenAI model IDs, and the probabilities and confidence values come from
the selected SIE model. This is API compatibility, not model equivalence.

Usage includes the OpenAI token-detail fields. Decision inference produces
no generated output tokens or cache/reasoning tokens. The direct Python
server reports native input usage; the gateway currently reports zero
input tokens because its extract response does not carry that measurement.

For the other SDK format, see the [TypeSafe example](../systemone-local/README.md).
The native SIE API remains available through `sie_sdk.SIEClient`.

## Verified run

On 2026-10-16, all 12 conformance checks passed with OpenAI SDK 3.26.0
against a local CPU server running the pinned `fastino/GLiNER2.5-Decide`
checkpoint in float32. The same server passed all 15 checks in the TypeSafe
SDK example. These checks verify the API contract and error handling, not
the model's classification accuracy.

The supported fields follow the [OpenAI Decisions API reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).
