# Run Claude Code on SIE through the Anthropic Messages API

## What this shows

Claude Code talks to models through the Anthropic Messages API. The SIE
gateway and `sie-server`, the single-node server, both serve
`POST /v1/messages`: each translates the request into a
`/v1/chat/completions` request, serves it on its own chat route with all of
that route's limits, and translates the answer back, streamed or buffered. So
Claude Code reaches a model served by SIE by changing its base URL.

Anthropic does not support Claude Code with models other than Claude behind a
gateway. Nothing here is supported or endorsed by Anthropic. This example
records how Claude Code behaves against SIE's endpoint and checks that the
agent loop closes.

The pieces:

1. `recipes/cloud.sh`, `recipes/cluster.sh` and `recipes/laptop.sh`, the
   environment Claude Code needs for SIE Cloud, for a self-hosted cluster's
   gateway and for `sie-server` on this machine.
2. The conformance check of the OpenCode example, run with `--api messages`.
   It drives `/v1/messages` the way Claude Code does and reports each case as
   pass, fail or recorded.
3. `run_gate.sh`, a seeded gate on the OpenCode example's fixture: a tiny
   package with two failing tests and the task "make the tests pass". Claude
   Code works a copy headless, then pytest judges the result.

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
./run_gate.sh laptop     # sie-server on this machine, see below
```

The cloud and cluster recipes have not been run against a live endpoint yet.
Their checks below were run against the gateway's own tests and against a
recording stub.

### On this machine

From a checkout of this repository, start `sie-server` with the generation
bundle:

```sh
mise run serve -- -b sglang           # generation bundle; MLX on Apple Silicon
```

Then, from this directory:

```sh
source recipes/laptop.sh
claude
```

`recipes/laptop.sh` points Claude Code at `http://localhost:8080` and names
`Qwen/Qwen3.5-4B:long-context`, the 32K-window profile of a 4B model that
declares tool calling, with its 4,096-token output cap. `SIE_BASE_URL`,
`SIE_API_KEY`, `SIE_MODEL`, `SIE_CONTEXT_TOKENS` and `SIE_OUTPUT_TOKENS`
override those. A server with authentication off accepts any key. With
authentication on, `SIE_API_KEY` is the server's token. A 32K window fits the
gate's `--bare` session, which declares three tools and a short system prompt.
An interactive session declares about twenty tools, so it reaches the window
sooner and Claude Code compacts it earlier.

The conformance check and the gate against it:

```sh
uv run --project ../coding-agent-opencode python ../coding-agent-opencode/conformance.py \
    --api messages --base-url http://localhost:8080/v1 --model Qwen/Qwen3.5-4B:long-context
./run_gate.sh laptop
```

Both were run on one Apple Silicon machine with the MLX backend; see
[the recorded result](#laptop-apple-silicon-recorded).

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
  Claude Code sends `ANTHROPIC_API_KEY` as `x-api-key`. The gateway and
  `sie-server` read either one; when both carry a token, they use
  `Authorization` and ignore `x-api-key`.
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

### Laptop, Apple Silicon, recorded

`Qwen/Qwen3.5-4B:long-context` on an Apple M3 Max with 48 GB under macOS
15.7, MLX backend (`mlx-community/Qwen3.5-4B-4bit`), `sie-server` from this
repository started with `mise run serve -- -b sglang --host 127.0.0.1`, Claude
Code 2.1.300, 2026-10-19.

The conformance check:

```
  streaming_tool_use       pass      1 tool_use: get_weather({"city": "Paris"}); 1 input_json_delta; stop_reason=tool_use
  parallel_tool_use        pass      2 tool_use in one turn: get_weather({"city": "Paris"}), get_weather({"city": "Tokyo"})
  tool_result_round_trip   pass      final answer used the tool result: 'The weather in Paris is currently overcast with a temperature of 18°C.'
  prompt_above_64k_tokens  pass      74847 prompt tokens accepted and answered; needle found
  stop_reason_end_turn     pass      stop_reason=end_turn after 'ready'
  stop_reason_tool_use     pass      stop_reason=tool_use on the tool turn
  stop_reason_max_tokens   pass      stop_reason=max_tokens at max_tokens=8 (output_tokens=8)
  usage_on_message_delta   pass      message_delta carries usage {'input_tokens': 4, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 16, 'output_tokens': 2}
  thinking_blocks          recorded  no thinking block came back; the model did not reason here or reasons privately
  tool_choice_any          recorded  refused: 400 invalid_request_error: tool_choice: 'tool_choice' must be 'auto' or 'none' on this device, which cannot force a tool call
  buffered_message         pass      stop_reason=end_turn, usage {'input_tokens': 20, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 2}
  inert_fields_accepted    pass      metadata, cache_control and anthropic-beta accepted
  effort_refusal           pass      400 invalid_request_error: output_config.effort: Extra inputs are not permitted
  system_turn_refusal      pass      400 invalid_request_error: messages.1.role: role "system" is not supported in messages; use the top-level system parameter
  unknown_model            pass      404 not_found_error: Model 'sie-conformance/no-such-model' not found
  event_grammar            pass      9 streams: message_start, one block at a time, message_delta then message_stop
  ping_events              recorded  9 ping events across the streams
  count_tokens             recorded  not served: 404; Claude Code then estimates locally

14 pass, 0 fail, 4 recorded
```

- The profile does not ask the model to reason, so no thinking came back.
- The MLX backend cannot force a tool call, so `tool_choice: any` is
  refused. Claude Code does not send it.
- The pings came during the 74,847-token prompt, which the server accepted
  although the profile declares a 32,768-token window. Set
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the declared window, as the recipe does,
  so Claude Code compacts before that.
- Prompt tokens served from the prefix cache come back as
  `cache_read_input_tokens`.

`./run_gate.sh laptop` passed three times out of three on the same server.
The first run, Claude Code's JSON event lines left out:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.02s
== claude run, recipe laptop, up to 50 turns and 1800 s
== the run
{
  "turns": 5,
  "duration_ms": 12443,
  "usage": {"input_tokens": 2688, "cache_read_input_tokens": 8858, "output_tokens": 513},
  "tool_uses": ["Bash", "Read", "Edit", "Bash"],
  "api_retries": 0,
  "problems": []
}
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.01s
GATE PASS (.../run-output/gate-20261019T105944Z-2izeas)
```

The other two took 5 turns each, in 12.1 s and 13.8 s. Each run's first
request carried `output_config.effort`, which the server refused with the
wording Claude Code retries on, and Claude Code sent it again without effort.
That refusal is not an API retry, so `api_retries` stays 0.

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
- `sie-server`'s tests replay the same vectors and captured requests, check
  that the server's own chat validation accepts every translation, and check
  the same event grammar. They also serve `/v1/messages` through the server's
  app and chat route on the weightless fake: answers that match the chat
  answer to the same conversation, buffered and streamed, thinking blocks
  only on request, refusals in the Anthropic envelope, `Authorization` and
  `x-api-key`, pings during a silent stretch, a failure after output, and a
  client that goes away.
- The OpenCode example's `tests/test_fake_engine.py` runs
  `conformance.py --api messages` against this repository's `sie-server` on
  the fake engine.
- `tests/test_run_gate.py` runs `run_gate.sh` with a scripted stand-in for
  `claude`: a real fix, no change, a failed run, a run that ended in an error,
  read no input tokens or took too few turns, edited tests, a run past its
  deadline, the three recipes, and the arguments and environment Claude Code
  is started with.
- The OpenCode example's `tests/test_messages_conformance.py` checks the
  `--api messages` checker against a scripted Messages stub.

```sh
uv run pytest
```

## What this does NOT establish

- **Not supported by Anthropic.** Anthropic does not support Claude Code with
  non-Claude models behind a gateway. A Claude Code release can change what it
  sends. The capture above is how to re-check a new version.
- **Not a live result yet.** The cloud and cluster recipes have not been run
  against a live endpoint.
- **Not a claim about every laptop.** The laptop recipe was run on one Apple
  Silicon machine with the MLX backend and one 4B model. It has not been run
  on an NVIDIA machine.
- **Not a coding benchmark.** The fixture is two failing tests in a
  twenty-line module.
- **The single-node server answers in its chat route's words.** `sie-server`
  translates the refusals its chat route makes. Its chat route does not
  report the token counts of a prompt that does not fit on every backend, so
  such a refusal can read `prompt is too long` without them. On a backend that
  refuses media, an image is refused for the whole request rather than the
  image block.
