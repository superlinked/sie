# Try a local System One model

## What this shows

SIE serves a System One-compatible API: `POST /v1/systemone` answers typed
`noul` (yes/no), `choice` (pick one) and `score` (ordinal) questions about a
text, and `GET /v1/models` lists the models with the fields the System One
model list carries. Code written with the TypeSafe SDK runs against SIE by
changing the SDK's base URL. Nothing else changes.

This directory has three pieces:

1. `decide.py`: ordinary TypeSafe Python SDK code. It asks three questions
   about a support ticket and prints the answers. Nothing in it names SIE.
2. `js/decide.mjs`: the same with the TypeSafe JS SDK.
3. `conformance.py`: runs the TypeSafe Python SDK, unchanged, through every
   SDK feature that reaches the server and reports each case as pass or fail.
   It covers each question type and builder, text, object and conversation
   states, questions without instructions, extra request fields,
   `models.list()`, the async client, and the SDK's error classes for an
   unknown model, an invalid question, too many score levels and a state that
   does not fit the model.

## Run it

Start SIE with a decision model. GLiNER2.5-Decide is a 340M-parameter
encoder, so a laptop CPU serves it:

```bash
# from the repository root, after ./tools/init.sh
mise run serve -- --models fastino/GLiNER2.5-Decide --port 8080 --device cpu
```

Wait until `curl -s localhost:8080/readyz` answers. The first start downloads
the model.

Point the TypeSafe SDK at it, then run the example:

```bash
cd examples/systemone-local
export TYPESAFE_BASE_URL=http://localhost:8080
export TYPESAFE_DEFAULT_MODEL=fastino/GLiNER2.5-Decide
export TYPESAFE_API_KEY=local   # any value: a local sie-server checks no keys

uv run decide.py
uv run conformance.py --json results.json
```

The JS SDK reads the same variables:

```bash
cd js && npm install && npm run decide
```

## Point it at SIE Serverless

Use the same three variables with the Serverless endpoint and your SIE key:

```bash
export TYPESAFE_BASE_URL=<your SIE Serverless URL>
export TYPESAFE_DEFAULT_MODEL=fastino/GLiNER2.5-Decide
export TYPESAFE_API_KEY=$SIE_API_KEY
uv run conformance.py --gateway --check-auth
```

`--gateway` allows the gateway's current `usage.input_tokens=0` placeholder
until native extract usage is available. Without it, the checker requires a
positive input-token count from a direct server. `--check-auth` separately
checks that an invalid key is refused.

A Serverless model that has gone cold takes minutes to load. SIE answers
`503` with a `Retry-After` of at most 60 seconds while it loads, and the
TypeSafe SDK retries twice by default (`max_retries=2` in `typesafe-sdk`
0.7.4), so a cold load outlasts its retries. Warm the model first, for example
with one call from the console playground.

## Limits

SIE's decision models read a fixed window per row. SIE refuses a request that
does not fit with `422` (the SDK's `TypeSafeUnprocessableEntityError`) and
never truncates the text silently:

| Model | Row | What shares the row |
|---|---|---|
| `fastino/GLiNER2.5-Decide` | 512 tokens | All of the request's questions (at most 256 tokens, half the row) and the state |
| `fastino/GLiNER2.5-Decide-1B` | 2,048 tokens | All of the questions (at most 512 tokens) and the state; use it for longer text |
| `fastino/GLiNER2.5-multi-Decide` | 2,048 tokens | Same as Decide-1B; multilingual |
| `convaiinnovations/laya-typed-decisions` | 1,024 tokens | One question and the state, per question |
| `convaiinnovations/laya-multilingual` | 1,024 tokens | Same; multilingual |
| `convaiinnovations/laya` | 512 tokens | Same |

Other differences from TypeSafe's hosted API:

- The model is an SIE model id; SIE has no `jev-latest`-style aliases. Set
  `TYPESAFE_DEFAULT_MODEL`, or pass `model=` per call.
- A score question takes 1 to 10 levels.
- `/v1/models` lists every model the server serves. `system_one` with a model
  that does not answer typed questions is refused with `422`.
- Confidences follow the System One definitions: for choice,
  `(p_max - 1/n) / (1 - 1/n)`; for score, one minus the spread around the most
  likely level relative to an even distribution.

## Test

```bash
uv run pytest -q
```

The tests run `conformance.py` against a stub that answers as SIE does, with
one knob per way to break it, so each case is shown to fail when it should.
They need no model and no network beyond installing the SDK.

## Recorded run

2026-10-14, local `sie-server` on an Apple M4 Max, CPU, serving
`fastino/GLiNER2.5-Decide`; `typesafe-sdk` 0.7.4 and `@typesafe-ai/sdk` 0.6.0,
unchanged. `conformance.py`: 15/15 passed (`--check-auth` not
run: a local server checks no keys).

| Case | Result | Seen |
|---|---|---|
| `models_list` | pass | 1 models; fastino/GLiNER2.5-Decide: release_date '2026-09-24' |
| `noul` | pass | noul=0.549 |
| `noul_criteria_only` | pass | no instructions sent; noul=0.689 |
| `choice` | pass | choice=billing confidence=0.632 |
| `score` | pass | score=1.999 confidence=0.998 |
| `dict_questions` | pass | noul, choice and score in one call |
| `object_state` | pass | object state; noul=0.646 |
| `conversation_state` | pass | conversation state; noul=0.977 |
| `extra_body` | pass | unknown request fields ignored |
| `envelope` | pass | model=fastino/GLiNER2.5-Decide input_tokens=22 |
| `async` | pass | AsyncTypeSafeClient noul=0.571 |
| `unknown_model` | pass | TypeSafeNotFoundError: POST http://localhost:8080/v1/systemone: 404 Model 'acme/no-such-model' not found |
| `invalid_question` | pass | TypeSafeUnprocessableEntityError: POST http://localhost:8080/v1/systemone: 422 question 'q': unknown type 'maybe'; use one of ['choice', 'noul', 'score'] |
| `too_many_levels` | pass | TypeSafeUnprocessableEntityError: POST http://localhost:8080/v1/systemone: 422 questions.q.criteria: score question 'q' has 11 levels; at most 10 are allowed |
| `state_too_long` | pass | TypeSafeUnprocessableEntityError: POST http://localhost:8080/v1/systemone: 422 state: The document does not fit in the 495 tokens the questions leave in the model's 512-token window (overflow_policy 'error') |

`decide.py` and `js/decide.mjs` printed the same answers:

```
model: fastino/GLiNER2.5-Decide
spam: P(yes) = 0.29
intent: billing (confidence 0.61)
urgency: 2.00 of 0-2 (confidence 0.99)
input tokens: 49
```

The probabilities are this model's; GLiNER2.5-Decide is weak on yes/no
questions in Superlinked's own benchmark, so read `spam` accordingly.
