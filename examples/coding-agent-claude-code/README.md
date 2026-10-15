# Run Claude Code on SIE through the Anthropic Messages API

## What this shows

Claude Code talks to models through the Anthropic Messages API. The SIE
gateway serves `POST /v1/messages`: it translates the request into a
`/v1/chat/completions` request, serves it on the chat route with all of that
route's limits, and translates the answer back, streamed or buffered. So Claude
Code reaches a model served by SIE by changing its base URL.

Anthropic does not support Claude Code with models other than Claude behind a
gateway. Nothing here is supported or endorsed by Anthropic. This example
records how Claude Code behaves against SIE's endpoint and checks that the
agent loop closes.

The pieces:

1. `recipes/cloud.sh` and `recipes/cluster.sh`, the environment Claude Code
   needs for SIE Cloud and for a self-hosted cluster's gateway.
2. The conformance check of the OpenCode example, run with `--api messages`.
   It drives `/v1/messages` the way Claude Code does and reports each case as
   pass, fail or recorded.
3. `run_gate.sh`, a seeded gate on the OpenCode example's fixture: a tiny
   package with two failing tests and the task "make the tests pass". Claude
   Code works a copy headless, then pytest judges the result.

`sie-server`, the single-node server, does not serve `/v1/messages`. This
example needs an SIE gateway: a cluster's or SIE Cloud's.

## Run it

```sh
export SIE_BASE_URL=https://...   # the gateway, without /v1
export SIE_API_KEY=...
source recipes/cloud.sh           # or recipes/cluster.sh, see below
claude
```

The cluster recipe also reads the model and its window, which must match what
the gateway's `GET /v1/models` declares for the model:

```sh
export SIE_MODEL=zai-org/GLM-5.3-Flash:relay
export SIE_CONTEXT_TOKENS=131072
export SIE_OUTPUT_TOKENS=32768
source recipes/cluster.sh
```

Claude Code's system prompt and tool schemas take tens of thousands of tokens,
so a profile with a window under about 64K tokens runs out early in a session.

The conformance check, from this directory:

```sh
uv run --project ../coding-agent-opencode python ../coding-agent-opencode/conformance.py \
    --api messages --base-url "$SIE_BASE_URL/v1" --model zai-org/GLM-5.3-Flash:relay
```

The seeded gate, which needs `claude` and `uv` on `PATH`:

```sh
./run_gate.sh            # SIE Cloud
./run_gate.sh cluster    # your gateway, with SIE_MODEL, SIE_CONTEXT_TOKENS and SIE_OUTPUT_TOKENS set
```

Neither recipe has been run against a live endpoint yet. The checks below were
run against the gateway's own tests and against a recording stub.

## The recipe

`recipes/cloud.sh` in full:

```sh
export ANTHROPIC_BASE_URL="$SIE_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$SIE_API_KEY"
export ANTHROPIC_MODEL="zai-org/GLM-5.3-Flash:relay"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS=1800000
```

- `ANTHROPIC_BASE_URL` is the gateway without `/v1`. Claude Code appends
  `/v1/messages`.
- `ANTHROPIC_AUTH_TOKEN` arrives as `Authorization: Bearer ...`, and
  Claude Code sends `ANTHROPIC_API_KEY` as `x-api-key`. The gateway reads
  either one; when a request carries both, it uses `Authorization` and
  ignores `x-api-key`.
- The model id is sent byte for byte, slash and colon included. The four
  model variables keep every request, background tasks and subagents
  included, on the SIE model. A request that names a model the gateway does
  not serve, such as a Claude model id, is answered with
  `404 not_found_error`.
- Claude Code assumes a 200K window for a model it does not know.
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` sets the profile's window, so Claude Code
  compacts the conversation before the server refuses it. The output cap must
  be at most the profile's `max_output_tokens`. A larger `max_tokens` is
  refused with `max_tokens: N > M, which is the maximum allowed number of
  output tokens for this model`.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` stops `context_management` and
  the experimental tool fields. SIE refuses `context_management` with a 400
  that names this variable, and Claude Code does not retry that refusal.
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` removes a block Claude Code otherwise adds
  to the system prompt. SIE passes system text through unchanged.
- `API_TIMEOUT_MS` is 30 minutes, so a long silent stretch, such as a model
  writing a large file in one tool call, does not end the request. SIE sends a
  `ping` event every 10 seconds while nothing else is due.

## What Claude Code puts on the wire

Recorded on 2026-10-15 with Claude Code 2.1.300, run headless (`claude -p`)
against a local stub that answered canned Messages replies. No model was
called. The scrubbed requests are in
[`conformance/anthropic_messages/claude_code_requests.json`](../../conformance/anthropic_messages/claude_code_requests.json),
and the gateway's tests translate every one of them.

- Every request goes to `POST /v1/messages?beta=true` with
  `anthropic-version: 2023-06-01` and an `anthropic-beta` list, which SIE
  accepts and ignores.
- The body carries `thinking: {"type": "adaptive"}`, which asks SIE for the
  model's reasoning as `thinking` blocks, `cache_control` on system and
  message blocks, and `metadata.user_id`. SIE accepts both and does nothing
  with them: prefix caching is automatic, and `metadata` is never logged or
  forwarded.
- The first request of a session carries `output_config.effort`. SIE refuses
  it with `output_config.effort: Extra inputs are not permitted`, and Claude
  Code sends the request again without effort and leaves it off for that
  model.
- Without `--bare`, Claude Code puts its environment notes into
  `role: "system"` turns in the middle of the conversation. SIE refuses those
  with `messages.N.role: role "system" is not supported in messages; use the
  top-level system parameter`, and Claude Code sends the request again with
  the same text as user-turn reminders, until `/clear` or `/compact`. So a
  session costs up to two refused requests at its start.
- An image on a model without image input is refused with a path such as
  `messages.2.content.0.tool_result.content.0.image: Model '...' does not
  support image input`. Claude Code replaces the image with a note and
  continues.
- A prompt over the window is refused with `prompt is too long: N tokens > M
  maximum`, the wording that starts Claude Code's compaction.
- SIE sends zero usage on `message_start` and the counts on `message_delta`.
  Claude Code reports the `message_delta` counts.
- With `--bare`, Claude Code declares three tools (`Bash`, `Edit`, `Read`) and
  a short system prompt, sends no system turns, and reads its key only from
  `ANTHROPIC_API_KEY` or an `apiKeyHelper`. The helper's value goes out as
  both `Authorization: Bearer` and `x-api-key`.
- Its `auto` permission mode asks the model to approve some tool calls in
  separate, non-streamed requests with `stop_sequences`. A model that answers
  them in another format gets the action blocked. The gate uses
  `--permission-mode acceptEdits`.

## What the Messages conformance check checks

| Case | Passes when |
|---|---|
| `streaming_tool_use` | a `tool_use` block opens with an id and the name `get_weather`, its `input_json_delta` pieces join into a JSON object with `city`, and `stop_reason` is `tool_use` |
| `parallel_tool_use` | one turn asked for Paris and Tokyo returns two or more `tool_use` blocks with distinct ids and both cities |
| `tool_result_round_trip` | after the assistant `tool_use` and a `tool_result` of 18 degrees, the answer carries 18, no further call, `stop_reason: end_turn` |
| `prompt_above_64k_tokens` | a 68,000-word prompt is accepted with over 65,536 input tokens and answered. When the declared window is smaller, the refusal and whether it uses Claude Code's compaction wording are recorded |
| `stop_reason_end_turn` | a one-word answer ends with `end_turn` |
| `stop_reason_tool_use` | the tool turn above ended with `tool_use` |
| `stop_reason_max_tokens` | `max_tokens: 8` on a long request ends with `max_tokens` |
| `usage_on_message_delta` | `message_start` carries a usage object and `message_delta` carries non-zero `input_tokens` and `output_tokens` |
| `thinking_blocks` | with `thinking` adaptive, reasoning comes back as `thinking` blocks streamed and buffered, never without asking, and no `<think>` tag leaks into text. A model that returns no reasoning is recorded |
| `tool_choice_any` | recorded: a forced call, or the refusal |
| `buffered_message` | a non-streamed request, which Claude Code falls back to after a broken stream, returns `end_turn` and usage |
| `inert_fields_accepted` | `metadata`, `cache_control` on system, message and tool blocks, and `anthropic-beta` are accepted |
| `effort_refusal` | `output_config.effort` is refused with exactly the wording Claude Code retries on |
| `system_turn_refusal` | a `role: "system"` turn is refused with wording Claude Code resends on |
| `unknown_model` | an unknown model answers `404 not_found_error` |
| `event_grammar` | every stream starts with `message_start`, opens one block at a time with indices from 0, and ends with `message_delta` then `message_stop`, or with one `error` event |
| `ping_events` | recorded: the `ping` events seen |
| `count_tokens` | recorded: the answer to `POST /v1/messages/count_tokens`, which SIE does not serve, so Claude Code estimates locally |

Busy refusals are retried as in the chat check: a `429` or `503`, or an
`overloaded_error` or `rate_limit_error` event before any output, with the
`Retry-After` wait, at most 4 times and 3 minutes.

## What result to expect

### The gate against a recording stub, recorded

`run_gate.sh` with Claude Code 2.1.300 and a stub on `SIE_BASE_URL` that
refuses `output_config.effort` as SIE does and otherwise plays a model that
reads the module, edits it and runs the tests, 2026-10-15:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.02s
== claude run, recipe cloud, up to 50 turns and 1800 s
== the run
{
  "turns": 4,
  "usage": {"input_tokens": 4000, "cache_read_input_tokens": 800, "output_tokens": 20},
  "tool_uses": ["Read", "Edit", "Bash"],
  "api_retries": 0,
  "problems": []
}
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.01s
GATE PASS (.../run-output/gate-20261015T160142Z-L4yhJy)
```

This checks the plumbing only: the recipe, `--bare` with the key helper, the
effort refusal and retry, the tool permissions and the verdict. It says
nothing about a model.

### Checked in this repository

- The gateway's tests replay the shared vectors in
  `conformance/anthropic_messages/` (request translation, buffered answer,
  event sequence, error mapping), translate every captured Claude Code
  request, serve `/v1/messages` through the real chat handler with a scripted
  transport, and check that any chat chunk sequence yields a well-formed event
  sequence.
- `tests/test_run_gate.py` runs `run_gate.sh` with a scripted stand-in for
  `claude`: a real fix, no change, a failed run, a run that ended in an error,
  read no input tokens or took too few turns, edited tests, a run past its
  deadline, both recipes, and the arguments and environment Claude Code is
  started with.
- The OpenCode example's `tests/test_messages_conformance.py` checks the
  `--api messages` checker against a scripted Messages stub.

```sh
uv run pytest
```

## What this does NOT establish

- **Not supported by Anthropic.** Anthropic does not support Claude Code with
  non-Claude models behind a gateway. A Claude Code release can change what it
  sends. The capture above is how to re-check a new version.
- **Not a live result yet.** Neither recipe has been run against a live
  endpoint, and no model has worked the fixture through `/v1/messages` here.
- **Not a coding benchmark.** The fixture is two failing tests in a
  twenty-line module.
- **Not the single-node server.** `sie-server` does not serve `/v1/messages`.
