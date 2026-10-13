# Run a coding agent on SIE with OpenCode

## What this shows

[OpenCode](https://opencode.ai) is an open-source coding agent. It talks to
any OpenAI-compatible chat endpoint through a custom provider built on
`@ai-sdk/openai-compatible`, which targets `/v1/chat/completions`. This
example points it at SIE three ways and checks that the agent loop holds up:

- a laptop running `sie-server` with a small tools-capable model,
- a self-hosted SIE cluster, reached through its gateway with a key,
- SIE Cloud, where `zai-org/GLM-5.3-Flash:relay` is served through SIE Relay.

Three pieces are in this directory:

1. `recipes/<laptop|cluster|cloud>/opencode.json`, one OpenCode config per
   setup.
2. `conformance.py`, an agent-loop conformance check. It drives
   `/v1/chat/completions` the way OpenCode does, streaming, with tools declared
   on every turn and tool results sent back as `role: tool` messages, and
   reports each case as pass, fail or recorded, with the values it saw.
3. `fixture/` and `run_gate.sh`, a seeded gate. The fixture is a tiny package
   with two failing tests and a one-line task, "make the tests pass". The
   script runs OpenCode on a copy non-interactively, then runs pytest on the
   result and prints the verdict. A run that changed the tests or
   `pyproject.toml`, or added a `conftest.py`, fails before pytest is asked,
   and the verdict run is pinned to the seeded `pyproject.toml` and `tests/`
   path, so a new `pytest.ini` or test directory does not count.

The conformance check uses the `openai` Python package rather than the SIE
SDK. The thing under test is the OpenAI-compatible wire surface that
OpenCode's provider speaks, so the reference client for that surface is the
right instrument.

## Run it

### Against SIE Cloud

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python conformance.py \
    --base-url https://api.superlinked.com/v1 --model zai-org/GLM-5.3-Flash:relay
```

Then copy `recipes/cloud/opencode.json` into a project and start `opencode`
there, or run the seeded gate, which needs only `opencode` and `uv` on `PATH`
and the key:

```sh
SIE_API_KEY=sk-sie-... ./run_gate.sh
```

### Against a laptop

On Apple Silicon or an NVIDIA machine, from a checkout of this repository:

```sh
mise run serve -- -b sglang           # generation bundle; MLX on Apple Silicon, SGLang on NVIDIA
```

Then, from this directory:

```sh
uv sync
uv run python conformance.py --model Qwen/Qwen3.5-4B:long-context
./run_gate.sh laptop
```

`Qwen/Qwen3.5-4B:long-context` is the 32K-window profile of a 4B model that
declares tool calling. Its bare name has an 8K window, which is smaller than
OpenCode's own system prompt plus its ten tool schemas, so the recipe names the
profile. On NVIDIA, `Qwen/Qwen3-4B-Instruct-2507` is the other small
tools-capable model in the catalog; the laptop recipe lists both.

### Against your cluster

Edit `recipes/cluster/opencode.json`: set `baseURL` to your gateway's
`/v1` and `limit.context` and `limit.output` to the window the cluster
advertises for the model in `/v1/models`. Then:

```sh
SIE_API_KEY=... uv run python conformance.py --base-url https://your-gateway/v1 --model zai-org/GLM-5.3-Flash
SIE_API_KEY=... ./run_gate.sh cluster
```

Every path writes a JSON report, `run-output/conformance.json` by default,
and prints a table. The exit code is 1 when any case fails.

## The OpenCode recipes

The three configs differ only in `baseURL`, the key and the model list. The
Cloud one in full:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "sie/zai-org/GLM-5.3-Flash:relay",
  "provider": {
    "sie": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "SIE Cloud",
      "options": {
        "baseURL": "https://api.superlinked.com/v1",
        "apiKey": "{env:SIE_API_KEY}",
        "timeout": 1800000,
        "headerTimeout": 600000,
        "chunkTimeout": 600000
      },
      "models": {
        "zai-org/GLM-5.3-Flash:relay": {
          "name": "GLM-5.3-Flash via SIE Relay",
          "limit": { "context": 131072, "output": 32768 },
          "tool_call": true,
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "include_reasoning": true }
        }
      }
    }
  }
}
```

- `npm` names the AI SDK provider package OpenCode installs for a custom
  provider. `@ai-sdk/openai-compatible` speaks `/v1/chat/completions`.
- `{env:SIE_API_KEY}` is OpenCode's own substitution; the key never sits in
  the file.
- `limit.context` and `limit.output` are what OpenCode budgets against. They
  match the window the `:relay` profile declares on SIE Cloud. OpenCode derives
  `max_tokens` from `limit.output` and sent 32000 for 32768.
- `reasoning: true` with `interleaved.field` tells OpenCode which field of a
  stream delta carries the model's thinking, so it can show it instead of
  hiding it. `options.include_reasoning` asks SIE for that field; it is off by
  default and a model that declares no reasoning answers it with nothing.
- `timeout`, `headerTimeout` and `chunkTimeout` bound the whole request, the
  wait for response headers and the gap between streamed chunks. A reasoning
  model can stream nothing for minutes while it thinks, so both of the last two
  are set above OpenCode's 300 s header default, and the whole-request bound is
  30 minutes, so a long think never looks like a hang.

### What OpenCode puts on the wire

Verified on 2026-10-13 with OpenCode 1.19.0 and a recording stub standing in
for the endpoint, rather than assumed from the docs:

- The model id is sent verbatim, colon included: `"model":
  "zai-org/GLM-5.3-Flash:relay"`. The `provider/model` selector splits on its
  first slash, so a model id can carry slashes and a colon.
- Per-model `options` are spread into the request body as top-level fields:
  the body carried `"include_reasoning": true`.
- `timeout`, `headerTimeout` and `chunkTimeout` never reach the body. OpenCode
  reads them off the provider options before building the request.
- Every agent turn is `stream: true` with `stream_options.include_usage:
  true`, `max_tokens`, `tool_choice: "auto"` and OpenCode's ten built-in tools
  (`bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`, `todowrite`,
  `webfetch`, `write`). No `temperature` and no `parallel_tool_calls` are sent.
  A separate, tool-less call names the session.
- The key arrives as `Authorization: Bearer ...`.

Because both the colon and the option forwarding held, the fallback of
sending options as per-model `headers` was not needed and is not in the
recipes.

## What `conformance.py` checks

| Case | Passes when |
|---|---|
| `tool_choice_auto` | the tools turn is accepted with `tool_choice: "auto"`, which OpenCode sends on every turn. When refused, the remaining tool cases run without the field so the rest of the loop is still measured |
| `streaming_tool_call_deltas` | the streamed deltas assemble into calls with an id, the name `get_weather`, JSON-object arguments carrying `city`, the first delta of each call carrying id and name, and `finish_reason: tool_calls` |
| `parallel_tool_calls` | one turn asked for Paris and Tokyo returns at least two calls with distinct ids and both cities |
| `tool_result_round_trip` | after the assistant `tool_calls` message and a `role: tool` result of 18 degrees, the final answer carries 18, no further call, `finish_reason: stop` |
| `prompt_above_64k_tokens` | a prompt of 60,000 filler words with a needle is accepted, `usage.prompt_tokens` is above 65,536, and an answer comes back; whether the needle was found is recorded, not required |
| `finish_reason_stop` | a one-word answer ends with `stop` |
| `finish_reason_tool_calls` | the tool-call turn above ended with `tool_calls` |
| `finish_reason_length` | `max_tokens: 8` on a long request ends with `length` |
| `usage_on_final_chunk` | with `stream_options.include_usage`, the last chunk carries `prompt_tokens` and `completion_tokens` |
| `include_reasoning` | with `include_reasoning: true` the request is accepted and no raw `<think>` tag leaks into content; without the flag no `reasoning_content` arrives. Whether reasoning deltas came back is recorded, since only a reasoning model sends them |
| `tool_choice_required` | recorded as `supported` when a call is forced or `refused` on a 4xx, neither a failure; accepted-but-ignored fails |

Every case keeps what it observed in the JSON report: the assembled calls, the
delta counts, the finish reason, the usage object, the error body, and the
`x-sie-served-by` header when the endpoint sends one.

A server that is still loading the model answers `503 MODEL_LOADING`, so the
check first waits for a tiny request to be served, up to 15 minutes, and
records how long that took.

## What result to expect

### Laptop, Apple Silicon, recorded

`Qwen/Qwen3.5-4B:long-context` on a 48 GB Apple Silicon machine, MLX backend,
`sie-server` from this repository at commit `b0c26b02`, 2026-10-13:

```
  tool_choice_auto            fail      OpenCode sends tool_choice=auto on every tools turn; refused: 400 unsupported_field: field 'tool_choice' is not supported
  streaming_tool_call_deltas  pass      1 call: get_weather({"city": "Paris"}); 1 deltas; finish_reason=tool_calls
  parallel_tool_calls         pass      2 calls in one turn: get_weather({"city": "Paris"}), get_weather({"city": "Tokyo"})
  tool_result_round_trip      pass      final answer used the tool result: 'The weather in Paris is currently overcast with a temperature of 18°C.'
  prompt_above_64k_tokens     pass      66047 prompt tokens accepted and answered; needle found
  finish_reason_stop          pass      finish_reason=stop after 'ready'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=8)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 2, 'prompt_tokens': 20, 'total_tokens': 22, 'prompt_tokens_details': {'cached_tokens': 0}}
  include_reasoning           fail      include_reasoning refused: 400 unsupported_field: field 'include_reasoning' is not supported
  tool_choice_required        recorded  refused: 400 unsupported_field: field 'tool_choice' is not supported

8 pass, 2 fail, 1 recorded
```

The two failures are the server's, not the model's, and both are worth
knowing before pointing OpenCode at a laptop:

- The MLX chat path refuses the `tool_choice` field outright, `auto`
  included. OpenCode sends `tool_choice: "auto"` on every turn, so on Apple
  Silicon its first turn ends in that 400. `./run_gate.sh laptop` on this
  machine stopped there, with the seeded tests still failing. The SGLang path
  on NVIDIA accepts `tool_choice`, so the laptop recipe works there today.
  Accepting `tool_choice: "auto"` on the MLX path is a server follow-up.
- `include_reasoning` is a newer addition to the chat contract than the server
  this ran against. A server that does not know the field refuses it with
  `unsupported_field`, so this case reads as fail until the server carries it.
  The recipes keep the option, since the Cloud endpoint is the one it is for.

The bare server also accepted and answered a 66,047-token prompt although the
catalog entry declares an 8,192-token window for this model. The declared
window is what a cluster advertises in `/v1/models` and what OpenCode's
`limit.context` should match.

The same run against the bare `Qwen/Qwen3.5-4B` (8K window) gave the same
table.

### SIE Cloud, GLM-5.3-Flash via SIE Relay

To be recorded on the deployed endpoint:

```
conformance table      pending
x-sie-served-by        pending (expected: relay)
seeded gate            pending
```

### BFCL through SIE Relay

The maintainers' benchmark harness, `sie-bench`, materialises the BFCL v3
`simple` split (400 rows) and a 200-row subsample of `multiple` from
`gorilla-llm/Berkeley-Function-Calling-Leaderboard` and scores each answer by
AST match against BFCL's accepted calls. The invocation against SIE Cloud:

```sh
sie-bench generate-prepare -t bfcl_simple -t bfcl_multiple
sie-bench generate-eval zai-org/GLM-5.3-Flash:relay \
    --base-url https://api.superlinked.com --api-key "$SIE_API_KEY" \
    -t bfcl_simple -t bfcl_multiple -o run-output/bfcl-glm-5.3-flash-relay.json
```

| Task | Rows | AST accuracy |
|---|---|---|
| `bfcl_simple` | 400 | pending |
| `bfcl_multiple` | 200 | pending |

## Checked without a GPU

`sie-server` ships a weightless `sie-fake` model for CI, but it has no
chat-template child: `/v1/chat/completions` answers `501
unsupported_operation` for it, checked the same day. So the conformance check
cannot run against `sie-fake`, and a fake engine that can script a
tool-calling chat turn is a follow-up.

What can run anywhere is `tests/test_conformance.py`. It starts a scripted
OpenAI-compatible stub that plays a conforming wire, then breaks one thing per
test: tool-call deltas without ids, one call where two were asked for,
`tool_choice: "required"` refused or silently ignored, usage missing, the long
prompt refused, `<think>` tags leaking into content, reasoning arriving without
the flag, `tool_choice: "auto"` refused, and a model that answers
`MODEL_LOADING` for its first requests. Each verdict is checked for what it
accepts and what it refuses:

```sh
uv run pytest
```

That is a check of the checker. The endpoint itself is only checked by running
`conformance.py` against it.

## What this does NOT establish

- **Not a coding benchmark.** The seeded fixture is two failing tests in a
  twenty-line module. Passing it shows that the loop closes, tools run and
  files land; it says nothing about harder tasks.
- **Not a tool-calling score.** The conformance cases use one weather tool.
  BFCL above is the measurement, and its numbers are pending.
- **Not a claim about every OpenCode version.** The wire facts were recorded
  with OpenCode 1.19.0. A later version can change what it sends; the
  recording stub approach in this README is how to re-check.
- **Not a claim about the laptop model's quality.** The 4B model passed the
  loop mechanics; whether it fixes your code is a different question.
- **Not a Cloud result yet.** The Cloud table, the served-by header, the gate
  run and the BFCL figures are placeholders until the run on the deployed
  endpoint is recorded here.
