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
   result and prints the verdict. A run where `opencode run` exits non-zero
   fails. A run that changed the tests or `pyproject.toml`, or added a
   `conftest.py`, fails before pytest is asked. The verdict run is pinned to
   the seeded `pyproject.toml` and `tests/` path and runs in a fresh
   environment outside the worked copy, so a new `pytest.ini`, test directory
   or anything planted in the copy's `.venv` does not count, and neither does
   a `PYTEST_ADDOPTS` or `PYTEST_PLUGINS` in the caller's environment. The
   OpenCode run is bounded: its build agent gets 50 steps (`GATE_STEPS`), and
   a run still going after 1,800 seconds (`GATE_DEADLINE_SECONDS`) is stopped
   and fails. OpenCode 1.19.0 treats the step budget as an instruction: at the
   last step it tells the model to answer in text but still declares the
   tools, so a model that keeps calling tools is stopped by the deadline.

The conformance check uses the `openai` Python package rather than the SIE
SDK. The thing under test is the OpenAI-compatible wire surface that
OpenCode's provider speaks, so the reference client for that surface is the
right instrument.

## Run it

### Against SIE Cloud

Set `SIE_BASE_URL` to the SIE Cloud endpoint your key belongs to, without
`/v1`, and check that `GET $SIE_BASE_URL/v1/models` lists
`zai-org/GLM-5.3-Flash:relay` there. The Cloud recipe reads both variables, so
it works unchanged against any endpoint that serves the model.

```sh
export SIE_BASE_URL=https://...   # your SIE Cloud endpoint
export SIE_API_KEY=sk-sie-...
uv sync
uv run python conformance.py --base-url "$SIE_BASE_URL/v1" --model zai-org/GLM-5.3-Flash:relay
```

Then copy `recipes/cloud/opencode.json` into a project and start `opencode`
there, or run the seeded gate, which needs only `opencode` and `uv` on `PATH`
and the two variables:

```sh
./run_gate.sh
```

### Against a laptop

From a checkout of this repository:

```sh
mise run serve -- -b sglang           # generation bundle; MLX on Apple Silicon
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
profile.

The laptop recipe has been run on Apple Silicon only, with the MLX backend and
this model: the conformance check bare and as `:long-context`, the seeded gate
with the recipe's `:long-context`. Both pass there (see
[the recorded result](#laptop-apple-silicon-recorded)). It has not been run on
an NVIDIA machine.

### Against your cluster

Set `limit.context` and `limit.output` in `recipes/cluster/opencode.json` to
the window your cluster advertises for the model in `/v1/models`. The recipe
reads `SIE_BASE_URL` and `SIE_API_KEY` like the Cloud one. Then:

```sh
export SIE_BASE_URL=https://your-gateway
export SIE_API_KEY=...
uv run python conformance.py --base-url "$SIE_BASE_URL/v1" --model zai-org/GLM-5.3-Flash
./run_gate.sh cluster
```

The cluster recipe is a template and has not been run. On a model whose
declared window is under 64K tokens, such as the bare `zai-org/GLM-5.3-Flash`
at 16,384, the 64K-plus prompt case is recorded rather than judged. A window
that size also runs out within a long agent session.

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
        "baseURL": "{env:SIE_BASE_URL}/v1",
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
- `{env:SIE_BASE_URL}` and `{env:SIE_API_KEY}` are OpenCode's own
  substitution; neither the endpoint nor the key sits in the file.
- `limit.context` and `limit.output` are what OpenCode budgets against. They
  match the window the `:relay` profile declares on SIE Cloud. OpenCode derives
  `max_tokens` from `limit.output` and sent 32000 for 32768.
- `options.include_reasoning` asks SIE to return the model's reasoning as
  `reasoning_content`. It is off by default, and a model that declares no
  reasoning answers it with nothing. `reasoning: true` declares the model as one
  that thinks.
- `interleaved.field: reasoning_content` makes OpenCode send each earlier
  assistant turn's reasoning back on that message as `reasoning_content`, an
  empty string when the turn had none, so the model sees its own earlier
  thinking on the next turn.
- `timeout`, `headerTimeout` and `chunkTimeout` bound the whole request, the
  wait for response headers and the gap between streamed chunks. A reasoning
  model can stream nothing for minutes while it thinks, so both of the last two
  are set above OpenCode's 300 s header default, and the whole-request bound is
  30 minutes, so a long think never looks like a hang.

### What OpenCode puts on the wire

Verified on 2026-10-13 with OpenCode 1.19.0 and a recording stub standing in
for the endpoint, rather than assumed from the docs:

- `{env:SIE_BASE_URL}/v1` in `baseURL` is substituted, and the requests
  arrived at `/v1/chat/completions` on the stub.
- The model id is sent verbatim, colon included: `"model":
  "zai-org/GLM-5.3-Flash:relay"`. The `provider/model` selector splits on its
  first slash, so a model id can carry slashes and a colon.
- Per-model `options` are spread into the request body as top-level fields:
  every request carried `"include_reasoning": true`.
- `timeout`, `headerTimeout` and `chunkTimeout` never reach the body. OpenCode
  reads them off the provider options before building the request.
- Every agent turn is `stream: true` with `stream_options.include_usage:
  true`, `max_tokens`, `tool_choice: "auto"` and OpenCode's ten built-in tools
  (`bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`, `todowrite`,
  `webfetch`, `write`). No `temperature` and no `parallel_tool_calls` are sent.
  A separate, tool-less call names the session.
- The assistant turn that made a tool call goes back with `content: ""` and,
  with the Cloud recipe, `reasoning_content` set to the reasoning streamed in
  that turn, or `""` when there was none. With the cluster recipe, which has no
  `interleaved` setting, `reasoning_content` goes back only for a turn that
  streamed some.
- The key arrives as `Authorization: Bearer ...`.

Because both the colon and the option forwarding held, the fallback of
sending options as per-model `headers` was not needed and is not in the
recipes.

## What `conformance.py` checks

When the endpoint accepts `include_reasoning`, every tools turn below carries
it and the round trip sends the first turn's reasoning back as
`reasoning_content`, which is the wire the Cloud recipe makes OpenCode send.
When the endpoint refuses the field, the tools turns go without it, as with
the laptop and cluster recipes. The report's `tools_turn_params` says which.

| Case | Passes when |
|---|---|
| `tool_choice_auto` | the tools turn is accepted with `tool_choice: "auto"`, which OpenCode sends on every turn. When the field is refused, the remaining tool cases run without it so the rest of the loop is still measured. When the turn fails for another reason, the case is recorded as not judged |
| `streaming_tool_call_deltas` | the streamed deltas assemble into calls with an id, the name `get_weather`, JSON-object arguments carrying `city`, the first delta of each call carrying id and name, and `finish_reason: tool_calls` |
| `parallel_tool_calls` | one turn asked for Paris and Tokyo returns at least two calls with distinct ids and both cities |
| `tool_result_round_trip` | after the assistant `tool_calls` message and a `role: tool` result of 18 degrees, the final answer carries 18, no further call, `finish_reason: stop` |
| `prompt_above_64k_tokens` | a prompt of 68,000 filler words with a needle is accepted, `usage.prompt_tokens` is above 65,536, and an answer comes back; whether the needle was found is recorded, not required. When the model's declared window in `/v1/models` is under 65,536, a refusal or an acceptance is recorded instead |
| `finish_reason_stop` | a one-word answer ends with `stop` |
| `finish_reason_tool_calls` | the tool-call turn above ended with `tool_calls` |
| `finish_reason_length` | `max_tokens: 8` on a long request ends with `length` |
| `usage_on_final_chunk` | with `stream_options.include_usage`, the last chunk carries `prompt_tokens` and `completion_tokens` |
| `include_reasoning` | the flag is accepted streamed and not streamed, no raw `<think>` tag leaks into content, and without the flag no `reasoning_content` arrives. When the model reasoned, the stream carries `delta.reasoning_content` and the non-streamed answer `message.reasoning_content`. A model reasoned when it returned reasoning, or when its completion tokens exceed the UTF-8 bytes of its visible text by more than 16, since a byte-level tokenizer spends at most one token per byte. It did not reason when no answer took more than 16 completion tokens. Between the two, whether it reasoned is unknown and the case is recorded, not judged |
| `tool_choice_required` | recorded as `supported` when a call is forced or `refused` on a 400 or 422, neither a failure; accepted-but-ignored fails, and so does any other error |

Every request asks for up to 4,096 tokens, except the `length` case, so a
model that thinks before it answers has room to finish. The 68,000-word
prompt counts at least one token per word on any tokenizer that does not merge
words: 71,431 with `cl100k_base`, 71,917 with `o200k_base`, 74,835 with
Qwen3.5's tokenizer and 71,432 with GLM-5.3-Flash's.

Every case keeps what it observed in the JSON report: the assembled calls, the
delta counts, the finish reason, the usage object, the error body, any
retries, and the `x-sie-served-by` header when the endpoint sends one.

A server that is still loading the model answers `503 MODEL_LOADING`, and a
gateway still starting a worker answers `503 provisioning`, so the check first
waits for a tiny request to be served, up to 15 minutes, and records how long
that took. A request refused as busy or rate-limited is retried: a `429` or
`503` whose code is `QUEUE_FULL`, `RESOURCE_EXHAUSTED` or
`rate_limit_exceeded`, or an error event with one of those codes that ends a
stream before any output. Each wait is the `Retry-After` header, or else the
error's `retry_after_s`, capped at 60 seconds. Without either, the wait is
5 seconds, doubling with each retry. A request is retried at most 4 times and
waits at most 3 minutes in all. Each retry is recorded with its error and
wait in the case's `observed`, and the case's `retries` counts them. A case
whose request is still refused as busy after that, or whose stream ends in a
busy error event after some output, has the status `capacity` rather than
`fail`. The summary counts it apart, and it also makes the exit code 1. Other
errors, including `MODEL_LOADING` or `provisioning` after the warm-up, are not
retried, and an error event in the middle of a stream is recorded as that
case's error.

## What result to expect

### Laptop, Apple Silicon, recorded

`Qwen/Qwen3.5-4B:long-context` on an Apple M3 Max with 48 GB under macOS
15.7, MLX backend (`mlx-community/Qwen3.5-4B-4bit`), `sie-server` from this
repository at commit `530dcde8` started with `mise run serve -- -b sglang`,
2026-10-13:

```
  tool_choice_auto            pass      tool_choice=auto accepted on the tools turn
  streaming_tool_call_deltas  pass      1 call: get_weather({"city": "Paris"}); 1 deltas; finish_reason=tool_calls
  parallel_tool_calls         pass      2 calls in one turn: get_weather({"city": "Paris"}), get_weather({"city": "Tokyo"})
  tool_result_round_trip      pass      final answer used the tool result: 'The weather in Paris is currently overcast with a temperature of 18°C.'
  prompt_above_64k_tokens     recorded  declared window 32768 is under 65536, yet 74847 prompt tokens were accepted
  finish_reason_stop          pass      finish_reason=stop after 'ready'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=8)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 2, 'prompt_tokens': 20, 'total_tokens': 22, 'prompt_tokens_details': {'cached_tokens': 16}}
  include_reasoning           pass      the model did not reason here (4 completion tokens for '9.9'); no reasoning to return
  tool_choice_required        recorded  refused: 400 unsupported_field: 'tool_choice' must be 'auto' or 'none' on this device, which cannot force a tool call

9 pass, 0 fail, 2 recorded
```

- The server accepted `include_reasoning`, so the tools turns carried it.
  This model did not reason on the probe, so there was no reasoning to
  return.
- The MLX backend accepts `tool_choice` as `auto` or `none` and refuses
  `required`, since it cannot force a call. OpenCode sends only `auto`.
- The server accepted and answered a 74,847-token prompt although this
  profile declares a 32,768-token window. The declared window is what a
  cluster advertises in `/v1/models` and what OpenCode's `limit.context`
  should match.

The same check against the bare `Qwen/Qwen3.5-4B` gave the same table, with
`declared window 8192` in the long-prompt line.

`./run_gate.sh laptop` on the same server, with OpenCode 1.19.0 and no global
OpenCode config, OpenCode's JSON event lines left out:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.01s
== opencode run, recipe laptop, up to 50 steps and 1800 s
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.00s
GATE PASS (.../run-output/gate-20261013T191942Z-KiK1JL)
```

OpenCode took 5 steps and 4 tool calls (`bash`, `read`, `edit`, `bash`), 8
seconds from its first event to its last. An earlier run on the same server
took 7 steps and 6 tool calls.

### SIE Cloud, GLM-5.3-Flash via SIE Relay, recorded

`zai-org/GLM-5.3-Flash:relay` on an SIE Cloud endpoint that serves it,
2026-10-14, from this repository at commit `93325c4b`:

```
model zai-org/GLM-5.3-Flash:relay at https://.../v1
served by remote

  tool_choice_auto            pass      tool_choice=auto accepted on the tools turn
  streaming_tool_call_deltas  pass      1 call: get_weather({"city": "Paris"}); 3 deltas; finish_reason=tool_calls
  parallel_tool_calls         pass      2 calls in one turn: get_weather({"city": "Paris"}), get_weather({"city": "Tokyo"})
  tool_result_round_trip      pass      final answer used the tool result: 'The current weather in Paris is **18°C** with **overcast** conditions.'
  prompt_above_64k_tokens     pass      71444 prompt tokens accepted and answered; needle found
  finish_reason_stop          pass      finish_reason=stop after 'ready'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=7)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 1, 'prompt_tokens': 20, 'total_tokens': 21, 'credits_charged': 1, ...}
  include_reasoning           pass      2 reasoning_content deltas, 14 chars; message.reasoning_content 14 chars; none without the flag
  tool_choice_required        fail      request failed: None RESOURCE_EXHAUSTED: the upstream is busy, please retry

10 pass, 1 fail, 0 recorded
```

- The one failure is the rate limit, not the feature. The check sends its
  cases back to back, and the `:relay` profile admits a bounded number of
  requests a minute. The refusal came as an error event in the stream, which
  the check at that commit did not retry. The same `tool_choice: "required"`
  request sent alone a few minutes later came back with `finish_reason:
  tool_calls` and the forced `get_weather` call.
- A direct chat answer from the same endpoint carried `x-sie-served-by:
  remote`, `x-sie-upstream: sie-relay` and `x-sie-credits-debited`, and the
  streamed answers' final usage carried `credits_charged`, so the requests are
  metered on SIE Cloud like any other.
- The 71,444-token prompt fits the 131,072-token window the `:relay` profile
  declares, which is the `limit.context` in the Cloud recipe.

`./run_gate.sh` against the same endpoint, with OpenCode 1.19.0 and no global
OpenCode config, OpenCode's JSON event lines left out:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.02s
== opencode run, recipe cloud, up to 50 steps and 1800 s
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.00s
GATE PASS (.../run-output/gate-20261014T092431Z-4eL32J)
```

OpenCode took 9 steps and 6 tool calls (`bash`, `bash`, `read`, `read`,
`edit`, `bash`), 21 seconds from its first event to its last, with at most
11,355 input tokens on a step. Its one edit to `invoice_totals/amounts.py`:

```diff
     cleaned = text.strip().replace("$", "").replace(",", "")
+    if cleaned.startswith("(") and cleaned.endswith(")"):
+        cleaned = "-" + cleaned[1:-1].strip()
     return Decimal(cleaned)
```

### BFCL through SIE Relay

The measurement of `zai-org/GLM-5.3-Flash:relay` on the Berkeley
Function Calling Leaderboard, BFCL v3, from
`gorilla-llm/Berkeley-Function-Calling-Leaderboard`: the full `simple` split
and a 200-row subsample of `multiple`, each answer scored by AST match against
BFCL's accepted calls, sent through SIE Cloud with the tools declared on the
request.

| Task | Rows | Correct | AST accuracy |
|---|---|---|---|
| `simple` | 400 | 368 | 0.920 |
| `multiple` | 200 | 184 | 0.920 |

Recorded 2026-10-14 on the same SIE Cloud endpoint, one request at a time
with an output cap of 8,192 tokens and `tool_choice: "auto"`.

- BFCL names many functions with dots, such as `math.factorial`.
  OpenAI-compatible APIs accept only letters, digits, `_` and `-` in a
  function name, and SIE Relay refuses the dotted form. So each dotted name
  was sent with `_` in place of the dot, and the model's calls were mapped
  back to BFCL's names before scoring, as BFCL's own harness does for
  OpenAI-style function calling.
- One of the 600 requests failed with a `502` and is scored as a miss.

## Checked without a GPU

`sie-server` ships a weightless `sie-fake` model for CI. Its
`sie-fake:chat` profile answers `/v1/chat/completions` from a script, not a
model. With tools declared, a turn that ends in a user message calls the
first tool, passing that message as every required argument. A turn that ends
in a `role: tool` message answers `Tool result: ` followed by the result. Any
other turn answers with hash-derived tokens. Every turn starts with one
sentence of reasoning, returned as `reasoning_content` only with
`include_reasoning`. A token is one word of the prompt or one streamed piece
of the answer, and the profile declares a 131,072-token window. So the
conformance check runs against a real `sie-server` without a GPU or weights:

```sh
mise run serve -- -b fake --device cpu      # from the repository root
uv run python conformance.py --model sie-fake:chat
```

```
  tool_choice_auto            pass      tool_choice=auto accepted on the tools turn
  streaming_tool_call_deltas  pass      1 call: get_weather({"city": "What is the weather in Paris right now? Use the get_weather tool."}); 14 deltas; finish_reason=tool_calls
  parallel_tool_calls         fail      1 tool call in the turn: get_weather({"city": "Get the current weather in Paris and in Tokyo. Call get_weather once per city, both in this turn."})
  tool_result_round_trip      pass      final answer used the tool result: 'Tool result: {"city": "Paris", "temperature_c": 18, "conditions": "overcast"}'
  prompt_above_64k_tokens     pass      68022 prompt tokens accepted and answered; needle not found
  finish_reason_stop          pass      finish_reason=stop after 'tok_b0437c31 tok_5aaf6099 tok_f3b6ad77 t'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=8)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 40, 'prompt_tokens': 7, 'total_tokens': 47}
  include_reasoning           pass      8 reasoning_content deltas, 40 chars; message.reasoning_content 40 chars; none without the flag
  tool_choice_required        recorded  supported: forced get_weather({"city": "Say hello."})

9 pass, 1 fail, 1 recorded
```

- A pass here checks SIE's single-node chat wire, not a model: the streamed
  call deltas, the tool result sent back, reasoning returned only on request,
  the finish reasons, the usage on the last chunk and the declared window.
- `parallel_tool_calls` fails because the fake makes one call per turn.
- The fake does not read the prompt, so the needle is not found, and the
  answer to "Reply with the single word: ready." is hash-derived tokens.

What runs anywhere, and in this repository's CI, is the test suite:

- `tests/test_fake_engine.py` starts this repository's `sie-server` with the
  fake bundle on the CPU and runs `conformance.py` against `sie-fake:chat`. It
  asserts the table above case by case, the failing `parallel_tool_calls`
  included, and what each case observed: the call's id and name on its first
  delta, the tool result in the answer, reasoning only with the flag and the
  same usage without it, the long prompt's size against the window, and
  `max_tokens: 8` spending exactly 8 tokens. It needs the repository's
  environment, from `mise run sync` at the root, and is skipped without it
  outside CI.
- `tests/test_conformance.py` starts a scripted OpenAI-compatible stub that
  plays a conforming wire, then breaks one thing per test: tool-call deltas
  without ids, one call where two were asked for, `tool_choice: "required"`
  refused, silently ignored or rate-limited, usage missing, the long prompt
  refused, a smaller declared window, `<think>` tags leaking into content,
  reasoning arriving without the flag, a reasoning model whose server drops
  the flag, a long answer that leaves the reasoning question open, reasoning
  missing from the non-streamed answer, `include_reasoning`
  or `tool_choice: "auto"` refused, busy refusals, an error event mid-stream,
  a `/v1/models` answered as a bare list, a model that thinks for thousands
  of tokens, and a model that answers `MODEL_LOADING` for its first requests.
  Each verdict is checked for what it accepts and what it refuses.
- `tests/test_run_gate.py` runs `run_gate.sh` with a scripted stand-in for
  `opencode`: a real fix, no change, a failed `opencode run`, edited tests, an
  environment planted in the worked copy, a caller's `PYTEST_ADDOPTS` or
  `PYTEST_PLUGINS` that would deselect the failing tests, the step budget
  handed to the build agent, a run that outlives its deadline, with and
  without a child that ignores TERM, and two runs started in the same second.

```sh
uv run pytest
```

`tests/test_conformance.py` and `tests/test_run_gate.py` check the checker,
and `tests/test_fake_engine.py` checks SIE's chat route with a scripted model.
A model's endpoint is only checked by running `conformance.py` against it.

## What this does NOT establish

- **Not a coding benchmark.** The seeded fixture is two failing tests in a
  twenty-line module. Passing it shows that the loop closes, tools run and
  files land; it says nothing about harder tasks.
- **Not a full BFCL score.** The conformance cases use one weather tool. The
  BFCL figures above cover the `simple` split and a 200-row `multiple`
  subsample, scored by AST match. BFCL's executable, parallel, relevance and
  multi-turn categories were not run.
- **Not a claim about every OpenCode version.** The wire facts were recorded
  with OpenCode 1.19.0. A later version can change what it sends; the
  recording stub approach in this README is how to re-check.
- **Not a claim about every laptop.** The laptop recipe was run on one Apple
  Silicon machine with one 4B model. NVIDIA and other models have not been
  run.
- **Not a measurement of SIE's own GPU serving.** Through SIE Relay the
  model runs on a partner's servers. SIE routes the request, checks the wire
  and meters it. A dedicated GPU deployment of the same model is a different
  serving path and has its own window and limits.
- **One endpoint, one day.** The Cloud results were recorded once, on
  2026-10-14.
