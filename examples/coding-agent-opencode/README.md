# Run a coding agent on SIE with OpenCode

## What this shows

[OpenCode](https://opencode.ai) is an open-source coding agent. It talks to
any OpenAI-compatible chat endpoint through a custom provider built on
`@ai-sdk/openai-compatible`, which targets `/v1/chat/completions`. This
example points it at SIE five ways and checks that the agent loop holds up:

- a laptop running `sie-server` with a small tools-capable model,
- one 80 GB NVIDIA GPU running `sie-server` with Qwen3.8-27B-FP8,
- a self-hosted `sie-server` that serves `zai-org/GLM-5.3-Flash` through an
  [OpenRouter](https://openrouter.ai) upstream, with your OpenRouter key,
- a self-hosted SIE cluster, reached through its gateway with a key,
- SIE Cloud, where `zai-org/GLM-5.3-Flash:relay` is served through SIE Relay.

Four pieces are in this directory:

1. `recipes/<laptop|gpu|openrouter|cluster|cloud>/opencode.json`, one
   OpenCode config per setup. The OpenRouter recipe also holds the server's
   `upstreams.yaml` and its `models/` directory.
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
4. `session_load.py`, a session load generator. It runs several agent-shaped
   sessions at once against one endpoint and reports, for each session count,
   the time to the first token on returning turns, the decode rate and the
   prompt tokens the server served from its cache.

The conformance check and the load generator use the `openai` Python package
rather than the SIE SDK. The thing under test is the OpenAI-compatible wire
surface that OpenCode's provider speaks, so the reference client for that
surface is the right instrument.

`conformance.py --api messages` checks the Anthropic Messages endpoint,
`/v1/messages`, the way Claude Code speaks it, reading the raw event stream.
[The Claude Code example](../coding-agent-claude-code/README.md) lists its
cases and runs Claude Code on this directory's fixture.

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

### On one GPU

On a Linux machine with one 80 GB NVIDIA GPU, start the server with
Qwen3.8-27B-FP8 loaded:

```sh
docker run --gpus all -p 127.0.0.1:8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  -v sie-kernel-cache:/app/.cache/sie-kernels \
  ghcr.io/superlinked/sie-server:latest-cuda13-sglang-cu130 \
  serve --host 0.0.0.0 --port 8080 --models-dir /app/models \
  --preload Qwen/Qwen3.8-27B-FP8:h100-256k-batch
```

Then, from this directory:

```sh
uv sync
uv run python conformance.py --model Qwen/Qwen3.8-27B-FP8:h100-256k-batch
./run_gate.sh gpu
uv run python session_load.py --model Qwen/Qwen3.8-27B-FP8:h100-256k-batch --sessions 1,2,4,6,8
```

and point OpenCode at it with `recipes/gpu/opencode.json`.

- `-p 127.0.0.1:8080:8080` publishes the port on this machine only.
  `sie-server` checks no key, and a coding agent sends it your source code, so
  the recipe keeps it off the network. To share one GPU with a team, serve it
  through the cluster gateway, which checks keys (the cluster recipe below).
- The two named volumes keep the weights and the compiled kernels across
  restarts. The first start downloads about 30 GB of weights.
- `--preload` loads the model before the server answers, so OpenCode's first
  turn does not meet `503 MODEL_LOADING`.
- `:h100-256k-batch` is the profile that fits an agent: a 262,144-token window,
  a 32,768-token output cap, up to 32 requests decoding at once, the
  `qwen3_coder` tool-call parser, no speculative decoding and no thinking. The
  recipe sets `limit.context` and `limit.output` to exactly those two numbers.
  OpenCode then sends `max_tokens: 32000`, and the server accepts a prompt of
  up to 230,144 tokens beside it.
- The recipe names one model. Each profile is its own engine launch, so asking
  the same server for a second profile unloads the first.
- The recipe sends `presence_penalty: 0` with every request. The profile's own
  default is 1.5, which penalises every token the answer has already used,
  and code repeats its identifiers. On BFCL, 0 scored higher on both routes
  below and lost no row that 1.5 got right.
- For a model that thinks before it answers, `Qwen/Qwen3.8-27B-FP8:thinking`
  is the same weights with reasoning on. Declare it with `"reasoning": true`,
  `"interleaved": {"field": "reasoning_content"}` and `"options":
  {"include_reasoning": true}`, as the Cloud recipe does. That profile runs one
  request at a time, and it was not recorded here.

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

### Self-hosted SIE with an OpenRouter upstream

This recipe runs the open-source `sie-server` on your machine and serves
`zai-org/GLM-5.3-Flash` as a remote-backed model: the server holds no weights
and sends each chat request on to OpenRouter's `z-ai/glm-5.3-flash`. OpenCode
talks only to the local server, and only the server needs your OpenRouter key.

From the root of a checkout of this repository:

```sh
export OPENROUTER_API_KEY=sk-or-...
mise exec -- uv run --frozen --project . --package sie-server sie-server serve \
  --models-dir examples/coding-agent-opencode/recipes/openrouter/models \
  --upstreams-file examples/coding-agent-opencode/recipes/openrouter/upstreams.yaml \
  --device cpu --host 127.0.0.1
```

Then, from this directory:

```sh
uv sync
uv run python conformance.py --base-url http://localhost:8080/v1 --model zai-org/GLM-5.3-Flash
./run_gate.sh openrouter
```

The server reads the key from its own environment each time it sends a
request, so it starts without one. When `OPENROUTER_API_KEY` is unset, it logs
a `WARNING` at startup, `requests to upstream 'openrouter' fail until its
credential is usable: environment variable OPENROUTER_API_KEY is not set`.
Export the key in the shell that starts the server. When OpenRouter answers
`429` or a `5xx` other than `501` and `505`, or cannot be reached, the server
answers `503` with `Retry-After`. When OpenRouter refuses the request with
`400` or `422`, the server answers `400 invalid_request`, `the upstream
refused the chat input`, and logs `upstream 'openrouter' refused a generation
request: the upstream refused the input (400)`, or `(422)`. A missing key, any
other refusal from OpenRouter and an answer the server does not accept answer
`502 upstream_error`, `upstream generation request failed`, before output.
After output, the stream ends with an error event instead. A request
the server refuses itself answers `400` before anything is sent. For a `502`
or an error event, the server log names the reason on a `WARNING` line,
`remote chat for zai-org/GLM-5.3-Flash failed: <reason>`, or `failed after
output: <reason>`:

| Reason | Cause |
|---|---|
| `the upstream credential is unavailable` | `OPENROUTER_API_KEY` is not set in the server's environment, so nothing was sent |
| `upstream answered 401` | OpenRouter refused the key |
| `upstream answered 402` | the OpenRouter account has no credits; GLM-5.3-Flash is a paid model on OpenRouter |
| `upstream answered 404` | OpenRouter found no endpoint for the request, for example under the account's provider or data-policy settings |
| `upstream answered` another status | OpenRouter refused the request with that status |
| `upstream sent an error instead of a chat answer` | OpenRouter accepted the request, then sent an error, as it does when the provider it picked fails |
| anything else | OpenRouter's answer broke a rule the server holds every upstream to, for example a call to a function the request did not declare, or usage above the request's `max_tokens` |

Pointed at OpenRouter directly, OpenCode would get the same model. Through SIE
it uses one API and one model name, `zai-org/GLM-5.3-Flash`, which the cluster
recipe uses too and SIE Cloud serves as its `:relay` profile, so the same
`conformance.py` and `run_gate.sh` check every setup, and the server adds the
upstream's rate cap and circuit breaker and its checks of tool calls and usage.

The two server files:

- `upstreams.yaml` defines the `openrouter` upstream at
  `https://openrouter.ai/api/v1`, offering `chat`. `api_key_secret` names the
  environment variable the server reads the key from when it sends a request.
  The key is in no file, and the recipe's `opencode.json` carries only a
  placeholder `apiKey` for the local server. The server listens on `127.0.0.1`
  because every request it serves spends that key.
- `rate_cap` allows 60 requests a minute and 4 at a time. `breaker` stops
  calls for 60 s after 5 calls in a row, within 30 s, found OpenRouter
  unavailable. A request either one refuses gets `503` with `Retry-After`. Both
  limits apply per server process.
- `strip_params` removes `chat_template_kwargs`, which the server adds from
  the model file and which only a local chat template reads.
- The upstream sets no OpenRouter provider preference. On 2026-10-14 all 33
  OpenRouter endpoints of `z-ai/glm-5.3-flash` listed every parameter the
  server forwards for OpenCode and `conformance.py` (`max_tokens`,
  `temperature`, `tools` and `tool_choice`), and OpenRouter sends a request
  with tools only to providers that support tools, so `require_parameters`
  would exclude no endpoint for them. An endpoint ignores a parameter it does
  not list, such as `parallel_tool_calls` (listed by 1 of the 33), `logprobs`
  (17) or `seed` (22); with `require_parameters` the request would go only to
  endpoints that list it. Of the 33, 11 report no support for `tool_choice:
  "required"`, and OpenRouter does not document whether it routes by that
  value.
- `models/zai-org__GLM-5.3-Flash.yaml` declares a remote-backed model
  (`remote_backed: true`, `routing: {policy: remote_only}`) with one profile on
  the `openrouter` upstream. It declares a 131,072-token window with up to
  32,768 tokens out, tool calling and text input. OpenRouter serves a larger
  window; the declared one is what `/v1/models` reports and what the server
  holds each answer's usage to. `chat_template_kwargs.enable_thinking: true`
  declares the model as one that reasons, which is what lets
  `include_reasoning` return its reasoning.

On this path OpenRouter parses the model's tool calls and reasoning. The server
checks that each streamed tool call arrives with an id, a function name and
arguments, that the calls keep to the request's `tools`, `tool_choice` and
`parallel_tool_calls`, that every answer finishes with exact usage inside the
declared window and the request's output cap, and reports that usage. OpenRouter
returns reasoning in a `reasoning` field; the server passes it on as
`reasoning_content` when the request sets `include_reasoning`, as the recipe's
`options` do, and drops it otherwise. OpenCode sends each earlier turn's
reasoning back as `reasoning_content`, which the server forwards and OpenRouter
accepts as an alias of `reasoning`. The server skips the SSE comment lines
OpenRouter sends while a request waits (`: OPENROUTER PROCESSING`), and turns
its closing usage event, which repeats the finished choice, into an OpenAI-form
usage chunk.

This recipe has been checked against a scripted stand-in for OpenRouter (see
[Checked without a GPU](#checked-without-a-gpu)). A successful run with a real
key is still [pending](#self-hosted-sie-with-an-openrouter-upstream-pending). To serve the
model from a cluster instead, define the upstream for the chart's remote worker
pool as
[Serving a model through a remote backend](../../packages/sie_server/REMOTE_BACKENDS.md#cluster-deployment)
describes.

### Against your cluster

`recipes/cluster/opencode.json` names `Qwen/Qwen3.8-27B-FP8:h100-256k-batch`,
the model and profile recorded on one GPU below, served by a cluster worker on
an 80 GB GPU. It also lists `zai-org/GLM-5.3-Flash` and
`Qwen/Qwen3-4B-Instruct-2507`. Set `limit.context` and `limit.output` for each
model to the window your cluster advertises for it in `/v1/models`. The recipe
reads `SIE_BASE_URL` and `SIE_API_KEY` like the Cloud one, and the gateway
checks the key. Then:

```sh
export SIE_BASE_URL=https://your-gateway
export SIE_API_KEY=...
uv run python conformance.py --base-url "$SIE_BASE_URL/v1" --model Qwen/Qwen3.8-27B-FP8:h100-256k-batch
./run_gate.sh cluster
```

The recipe's Qwen3.8 entry was run through this repository's gateway, queue
and worker on one GPU (see
[the recorded result](#the-same-gpu-through-the-cluster-gateway-recorded)).
The GLM and Qwen3-4B entries have not been run. On a model whose declared
window is under 64K tokens, such as the bare `zai-org/GLM-5.3-Flash` at
16,384, the 64K-plus prompt case is recorded rather than judged. A window
that size also runs out within a long agent session.

Every path writes a JSON report, `run-output/conformance.json` by default,
and prints a table. The exit code is 1 when any case fails.

## The OpenCode recipes

The configs differ only in `baseURL`, the key and the model list. The Cloud
one in full:

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
| `long_tool_call` | one `write_file` call asked for 1,320 numbered lines, 67,363 bytes of arguments, arrives whole: the arguments parse as a JSON object over 64 KiB, the turn ends with `tool_calls`, and the one call writes `notes/pangrams.txt` with every line as asked. The request sends `max_tokens: 32000`, as OpenCode does. A stream that breaks or arguments that do not parse fail. A model that answers in text, writes less than 64 KiB, writes another file or other lines, or runs out of `max_tokens` is recorded, and so is an endpoint that refuses `max_tokens: 32000`. The report keeps the argument size, how many lines came back as asked, the time to the first chunk, how long the stream ran and its longest gap between chunks |

Every request asks for up to 4,096 tokens, except the `length` case and the
long call, so a model that thinks before it answers has room to finish. The
long call takes several minutes on one GPU: about 20,000 tokens of output.
OpenCode's own `write` and `edit` calls carry whole files the same way, and a
server that sends nothing while a call is generated must not end the stream.
The laptop and SIE Cloud tables below were recorded before this case was
added. The 68,000-word
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
stream before any output. Each wait is the `Retry-After` header, in seconds or
until its future date, or else the error's `retry_after_s`, capped at 60
seconds. Without either, the wait is 5 seconds, doubling with each retry. A
request is retried at most 4 times and waits at most 3 minutes in all. Each retry is recorded with its error and
wait in the case's `observed`, and the case's `retries` counts them. A case
whose request is still refused as busy after that, or whose stream ends in a
busy error event after some output, has the status `capacity` rather than
`fail`. The summary counts it apart, and it also makes the exit code 1. Other
errors, including `MODEL_LOADING` or `provisioning` after the warm-up, are not
retried, and an error event in the middle of a stream is recorded as that
case's error.

## What `session_load.py` measures

A coding agent sends its whole transcript on every turn, so the number of
sessions one GPU holds depends on the KV cache and on whether the server reuses
the transcript's prefix from one turn to the next. `session_load.py` runs
agent-shaped sessions against one endpoint:

- Every session starts from one shared system prompt of `--prefix-tokens`
  (16,000 by default), standing in for an agent's instructions and tool
  descriptions.
- Each turn adds `--turn-tokens` (6,000) of new input, unique to the session,
  standing in for a file the agent read, and asks for an answer of
  `--output-tokens` (500). The answer goes back into the transcript as it was
  streamed, and the whole transcript is sent again on the next turn.
- After `--turns` (7) turns a session's prompt is about 61,000 tokens.
  `--turn-tokens 12000 --turns 9` grows it to about 128,000.
- For each count in `--sessions`, that many sessions start together and run
  their turns back to back, with no pause for tool calls in between, which is
  harder on the server than a real agent.
- No sampling fields are sent, so the profile's defaults apply, as they do to
  OpenCode's requests.

The input is sized with the endpoint's own token count of the shared prompt.
For each session count the report gives the decode rate after the first token
and the time to the first token on returning turns, at the median and at p90,
where the p90 decode rate is the rate nine turns in ten reached or beat. The
decode rate counts the tokens after the first streamed piece, estimated by
their share of the streamed text, over the time from the first piece to the
last. A model that thinks without returning its reasoning adds tokens the
text does not show, which raises that rate. A turn that ends without the
finish reason `stop` or `length` counts as failed, and answers that stopped
before `--output-tokens` are counted. It
also gives the share of returning-turn prompt tokens the server reported as
cached, when it reports them. A session count meets the bar when, at p90,
every session decodes at 30 tok/s or more and a returning turn starts within
3 seconds, with no failed turn. The bar was set before the runs below.

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

### One GPU, Qwen3.8-27B-FP8, recorded

`Qwen/Qwen3.8-27B-FP8:h100-256k-batch` on one NVIDIA H100 80GB HBM3 with 8 CPU
cores and 64 GiB of memory, 2026-10-15. `sie-server` ran from this repository
with the `sglang-cu130` bundle (SGLang 0.5.20), started as `sie-server serve
-b sglang-cu130 --preload Qwen/Qwen3.8-27B-FP8:h100-256k-batch`, in three
sessions on machines of that kind. Each result below names the commit of the
server it was recorded at; the checks are the ones in this directory. The
`docker run` line above starts the same server, profile and bundle from the
published image; the runs below did not use the image.

What the engine reported once loaded:

- a KV cache of 325,056 tokens, shared by every request on the GPU;
- 243 slots for the recurrent state of the model's linear-attention layers;
- up to 32 requests decoding at once;
- 28.5 GB of weights and 19.8 GB of KV cache.

The server answered its first request 197 s and 229 s after it started in two
new containers, with the weights already downloaded, and 76 s after a restart
in the same container, with the kernels already compiled.

The conformance check at commit `09b62a32`:

```
model Qwen/Qwen3.8-27B-FP8:h100-256k-batch at http://localhost:8080/v1
served by local

  tool_choice_auto            pass      tool_choice=auto accepted on the tools turn
  streaming_tool_call_deltas  pass      1 call: get_weather({"city": "Paris"}); 4 deltas; finish_reason=tool_calls
  parallel_tool_calls         pass      2 calls in one turn: get_weather({"city": "Paris"}), get_weather({"city": "Tokyo"})
  tool_result_round_trip      pass      final answer used the tool result: 'The weather in Paris right now is overcast with a temperature of 18°C.'
  prompt_above_64k_tokens     pass      74847 prompt tokens accepted and answered; needle found
  finish_reason_stop          pass      finish_reason=stop after 'ready'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=8)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 2, 'prompt_tokens': 20, 'total_tokens': 22, 'reasoning_tokens': 0}
  include_reasoning           pass      the model did not reason here (5 completion tokens for '9.11'); no reasoning to return
  tool_choice_required        recorded  supported: forced get_weather({"city": "Paris"})
  long_tool_call              pass      65.8 KiB of arguments in 1 call (1320 lines), streamed over 235 s, longest gap between chunks 235.0 s

11 pass, 0 fail, 1 recorded
```

- The long call came back with all 1,320 lines as asked: 19,840 completion
  tokens in 235 s, 84 tok/s. The engine's `qwen3_coder` parser streams a call's
  name and short arguments as they are generated but holds a long argument
  until it closes, so this stream sent no chunk for 235 s. `sie-server` writes
  an SSE comment every 15 s while the engine holds a tool call, which keeps a
  client's idle timer running without adding a chunk, and allows the call the
  profile's overall budget of 1,800 s. Before commit `00d7d732` it ended a
  stream that sent nothing for 300 s instead: at commit `461c5a58` the case
  passed with 243 s of silence, but an argument that took longer than 300 s to
  generate was cut off.
- This profile does not think, so `include_reasoning` returns nothing. Its
  answer, 9.11, is wrong; the case does not judge answers.
- The usage on this route carries no `cached_tokens`: the engine's cache report
  is off for this profile. The session load below shows the cache at work
  through the time to the first token.

`./run_gate.sh gpu` at commit `461c5a58`, with OpenCode 1.19.0 and no global
OpenCode config, before the recipe set `presence_penalty`, OpenCode's JSON
event lines left out:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.03s
== opencode run, recipe gpu, up to 50 steps and 1800 s
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.01s
GATE PASS (.../run-output/gate-20261015T162830Z-4g42xF)
```

Five runs in a row passed at the profile's `presence_penalty` of 1.5. Each
took 5 steps and 4 tool calls (`bash`, `read`, `edit`, `bash`), 8 to 9 seconds
from OpenCode's first event to its last, with at most 8,957 input tokens on a
step. Five more runs with the recipe's `presence_penalty: 0` also passed, with
5 or 6 steps, 4 to 8 tool calls and at most 9,723 input tokens, and so did
three more at commit `00d7d732`. The seeded gate does not tell the two
settings apart; BFCL below does.

`session_load.py` at commit `09b62a32`, with its defaults, sessions growing to
about 61,000 tokens, and with `--turn-tokens 12000 --turns 9`, growing to about
128,000:

| Sessions | Largest prompt | Decode p50 / p90, tok/s | Returning-turn first token p50 / p90 | Bar |
|---|---|---|---|---|
| 1 | 61,002 | 79.1 / 77.1 | 0.73 / 1.43 s | meets |
| 2 | 60,879 | 72.0 / 60.8 | 0.97 / 1.12 s | meets |
| 4 | 61,262 | 56.9 / 51.5 | 1.18 / 1.91 s | meets |
| 6 | 61,200 | 47.5 / 40.7 | 0.88 / 2.81 s | meets |
| 8 | 61,293 | 42.4 / 33.0 | 1.10 / 3.63 s | misses: first token |
| 1 | 127,910 | 74.6 / 70.6 | 1.61 / 2.10 s | meets |
| 2 | 128,251 | 65.4 / 47.0 | 2.08 / 2.39 s | meets |
| 3 | 128,183 | 57.9 / 36.4 | 2.03 / 2.87 s | meets |

- One H100 holds **6 sessions at about 61,000 tokens each, or 2 at about
  128,000**, within the bar, on this route and through the gateway below. No
  turn failed at any count.
- Three sessions at about 128,000 are at the edge. They met the bar here
  with 2.87 s at p90, missed it through the gateway below with 5.14 s, and
  missed it on this route in an earlier sweep at commit `461c5a58` with
  4.21 s. That sweep gave the same verdict as this one at every other count.
- The prefix cache carries a session from turn to turn. A returning turn with
  a 61,000-token prompt reached its first token in 0.73 s with one session,
  which is the time to prefill only the new input.
- The model often stopped before 500 tokens on later turns: answers averaged
  410 to 455 tokens a session count at 61,000 and 350 to 440 at 128,000. The
  recipes' `presence_penalty: 0` was not applied here.
- Six 61,000-token sessions fit a 325,056-token cache because the shared
  16,000-token system prompt is stored once.
- The profile's 32 requests decoding at once is a scheduler limit, not a
  session count.

BFCL v3 at commit `461c5a58`, with the harness and settings of
[the SIE Relay record](#bfcl-through-sie-relay): the full `simple` split and
the same 200-row `multiple` subsample, one request at a time, greedy decoding,
an output cap of 8,192 tokens, `tool_choice: "auto"`, and dotted function
names sent with `_`. Both routes ran on the same GPU, the cluster route as
[recorded below](#the-same-gpu-through-the-cluster-gateway-recorded):

| Route | `presence_penalty` | `simple`, 400 rows | `multiple`, 200 rows |
|---|---|---|---|
| Single server | 1.5, the profile's default | 359, 0.8975 | 182, 0.910 |
| Single server | 0, as the recipes set it | 370, 0.925 | 184, 0.920 |
| Cluster gateway | 1.5 | 370, 0.925 | 187, 0.935 |
| Cluster gateway | 0 | 378, 0.945 | 189, 0.945 |

- On each route every row that passed at 1.5 also passed at 0. Setting 0
  turned 13 failed rows into passes on the single server and 10 through the
  gateway, so the recipes set it.
- The two routes do not agree. They render the same tools turn to prompts of
  different length, 306 tokens on the single server and 295 through the
  gateway for the conformance check's weather turn, so their greedy answers
  differ, and the gateway's scored higher at both settings. Which rendering is
  closer to the model's training format was not established.

One real issue on this repository, run twice through the single-server
recipe at commit `461c5a58`:
[issue #717](https://github.com/superlinked/sie/issues/717), a
metering bug in the GLiNER adapters, on a checkout of commit `03dd3fd3`.
OpenCode was asked to fix it, add a regression test and run it, with a
100-step budget and no other help:

- At the profile's `presence_penalty` of 1.5: 100 steps and 100 tool calls in
  370 s, prompts up to 76,868 tokens, no failed request. After reading the
  code it re-ran one shell command over and over: from the 30th call on,
  every tenth call and each of the last four was that command. It made no
  edit.
- With `presence_penalty: 0`: 100 steps and 104 tool calls in 895 s, prompts
  up to 100,258 tokens, no failed request. It changed the adapter code and
  added a regression test, which still failed when the step budget ran out.
- Neither run fixed the issue. Both held a session past 75,000 tokens on one
  GPU without a serving error.

### The same GPU through the cluster gateway, recorded

The same model, profile and H100, served the way a cluster serves it: this
repository's `sie-gateway`, NATS, `sie-server-sidecar` and one `sie-server`
worker, on one machine, with the gateway on `127.0.0.1:31000`. The worker was
ready 100 s and 161 s after the stack started when the kernels were already
compiled, and 270 s after in a new container.

The conformance check at commit `09b62a32`:

```
model Qwen/Qwen3.8-27B-FP8:h100-256k-batch at http://127.0.0.1:31000/v1
served by local

  tool_choice_auto            pass      tool_choice=auto accepted on the tools turn
  streaming_tool_call_deltas  pass      1 call: get_weather({"city":"Paris"}); 2 deltas; finish_reason=tool_calls
  parallel_tool_calls         pass      2 calls in one turn: get_weather({"city":"Paris"}), get_weather({"city":"Tokyo"})
  tool_result_round_trip      pass      final answer used the tool result: "It's currently 18°C and overcast in Paris."
  prompt_above_64k_tokens     pass      74847 prompt tokens accepted and answered; needle found
  finish_reason_stop          pass      finish_reason=stop after 'ready'
  finish_reason_tool_calls    pass      finish_reason=tool_calls on the tool-call turn
  finish_reason_length        pass      finish_reason=length at max_tokens=8 (completion_tokens=8)
  usage_on_final_chunk        pass      final chunk carries usage {'completion_tokens': 2, 'prompt_tokens': 20, 'total_tokens': 22, 'prompt_tokens_details': {'cached_tokens': 0}}
  include_reasoning           pass      the model did not reason here (5 completion tokens for '9.11'); no reasoning to return
  tool_choice_required        recorded  supported: forced get_weather({"city":"Paris"})
  long_tool_call              pass      65.8 KiB of arguments in 1 call (1320 lines), streamed over 235 s, longest gap between chunks 0.1 s

11 pass, 0 fail, 1 recorded
```

- The long call came back whole: 19,840 completion tokens, all 1,320 lines as
  asked. The gateway passes on an empty chunk while the worker holds a call,
  so the stream never went quiet for more than 0.1 s. Before commit
  `00d7d732` the gateway ended a stream after 30 s without a chunk: at commit
  `461c5a58` the case failed after 30 s with `inter_chunk_timeout`, with no
  argument bytes delivered.
- This route reports `cached_tokens` in the usage, which the session load
  below reads.
- At commit `461c5a58`, one request with a 1.1 MB body, a 196,449-token
  prompt, was answered in 23 s.

The seeded gate, `./run_gate.sh cluster` with `SIE_BASE_URL` set to the
gateway, passed 5 of 5 runs at the profile's `presence_penalty` of 1.5, at
commit `461c5a58`, and 3 of 3 with the recipe's `presence_penalty: 0`, at
commit `00d7d732`. Each run took 5 steps and 4 or 5 tool calls, 7 to 9
seconds, with at most 9,240 input tokens on a step.

BFCL is in [the table above](#one-gpu-qwen38-27b-fp8-recorded).

`session_load.py` through the gateway at commit `09b62a32`:

| Sessions | Largest prompt | Decode p50 / p90, tok/s | Returning-turn first token p50 / p90 | Returning-turn prompt cached | Bar |
|---|---|---|---|---|---|
| 1 | 61,106 | 79.2 / 77.0 | 0.67 / 0.77 s | 86.3% | meets |
| 2 | 61,207 | 72.9 / 65.9 | 0.78 / 1.07 s | 86.3% | meets |
| 4 | 61,263 | 66.0 / 51.7 | 1.60 / 2.06 s | 86.3% | meets |
| 6 | 61,273 | 54.3 / 37.1 | 1.30 / 2.46 s | 86.2% | meets |
| 8 | 61,294 | 48.8 / 32.9 | 2.44 / 3.85 s | 86.3% | misses: first token |
| 1 | 127,560 | 74.5 / 70.4 | 1.58 / 2.00 s | 85.5% | meets |
| 2 | 127,991 | 67.7 / 50.7 | 1.82 / 2.26 s | 85.5% | meets |
| 3 | 128,187 | 59.0 / 36.8 | 3.18 / 5.14 s | 83.7% | misses: first token |

- The gateway gives the same capacity as the single server: 6 sessions at
  about 61,000 tokens and 2 at about 128,000 meet the bar on both. A sweep at
  commit `461c5a58` gave the same verdicts at the counts it ran, 1 to 8 at
  61,000 and 1 and 2 at 128,000.
- What a session sends again is served from the cache. On a returning turn the
  uncached part averaged 6,150 tokens at 61,000 and 12,150 at 128,000, the
  size of the turn's new input. With 3 sessions at 128,000 the three
  transcripts no longer fit the cache together, some turns had up to 37,307
  tokens to prefill again, and the first token came later.

The same real issue as above, once through the cluster recipe with
`presence_penalty: 0` at commit `00d7d732`: 100 steps and 104 tool calls in
751 s, prompts up to 139,712 tokens, no failed request. It spent most steps on
Python probes of the word splitter and made no edit.

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

### Self-hosted SIE with an OpenRouter upstream, pending

`zai-org/GLM-5.3-Flash` through OpenRouter's `z-ai/glm-5.3-flash`, with the
recipe's server files and a real OpenRouter key. `sie-server` from this
repository at commit pending, OpenCode version pending, recorded on date
pending.

| Check | Result |
|---|---|
| `tool_choice_auto` | pending |
| `streaming_tool_call_deltas` | pending |
| `parallel_tool_calls` | pending |
| `tool_result_round_trip` | pending |
| `prompt_above_64k_tokens` | pending |
| `finish_reason_stop` | pending |
| `finish_reason_tool_calls` | pending |
| `finish_reason_length` | pending |
| `usage_on_final_chunk` | pending |
| `include_reasoning` | pending |
| `tool_choice_required` | pending |
| `long_tool_call` | pending |
| `./run_gate.sh openrouter` | pending |

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
  long_tool_call              recorded  not run: max_tokens=32000 refused: 400 INVALID_INPUT: max_tokens (32000) exceeds model cap (4096)

9 pass, 1 fail, 2 recorded
```

- A pass here checks SIE's single-node chat wire, not a model: the streamed
  call deltas, the tool result sent back, reasoning returned only on request,
  the finish reasons, the usage on the last chunk and the declared window.
- `parallel_tool_calls` fails because the fake makes one call per turn.
- `long_tool_call` is not run because the profile caps output at 4,096
  tokens and refuses the 32,000 the call asks for.
- The fake does not read the prompt, so the needle is not found, and the
  answer to "Reply with the single word: ready." is hash-derived tokens.

What runs anywhere, and in this repository's CI, is the test suite:

- `tests/test_fake_engine.py` starts this repository's `sie-server` with the
  fake bundle on the CPU and runs `conformance.py` against `sie-fake:chat`. It
  asserts the table above case by case, the failing `parallel_tool_calls`
  included, and what each case observed: the call's id and name on its first
  delta, the tool result in the answer, reasoning only with the flag and the
  same usage without it, the long prompt's size against the window, and
  `max_tokens: 8` spending exactly 8 tokens. It also runs
  `conformance.py --api messages` against the same server, whose
  `/v1/messages` translates onto that chat route, and asserts that table. It
  needs the repository's environment, from `mise run sync` at the root, and is
  skipped without it outside CI.
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
  of tokens, a model that answers `MODEL_LOADING` for its first requests, and
  a long tool call that arrives whole, short, truncated, cut by an error
  event, ended by the output cap, refused for its `max_tokens`, or valid but
  for another file or with other lines. Each verdict
  is checked for what it accepts and what it refuses.
- `tests/test_session_load.py` runs `session_load.py` against a scripted stub
  that counts one token per word and reports a cached prefix: each session
  sends its whole transcript back with the answers as streamed, the input is
  sized with the endpoint's count, a failed turn or one without `stop` or
  `length` ends its session and fails the run, returned reasoning counts
  toward the first token and the decode window, short answers are counted,
  and the bar is judged at p90.
- `tests/test_messages_conformance.py` runs `conformance.py --api messages`
  against a scripted Anthropic Messages stub that plays a conforming event
  stream, then breaks one thing per test: usage missing from
  `message_delta`, a delta before its block, an accepted `output_config.effort`,
  a system-turn refusal without the wording Claude Code resends on, busy
  refusals and a loading model.
- `tests/test_run_gate.py` runs `run_gate.sh` with a scripted stand-in for
  `opencode`: a real fix, no change, a failed `opencode run`, edited tests, an
  environment planted in the worked copy, a caller's `PYTEST_ADDOPTS` or
  `PYTEST_PLUGINS` that would deselect the failing tests, the step budget
  handed to the build agent, a run that outlives its deadline, with and
  without a child that ignores TERM, two runs started in the same second, and
  the key check, which applies only to a recipe that reads `SIE_BASE_URL` and
  `SIE_API_KEY`.
- `tests/test_openrouter_recipe.py` starts this repository's `sie-server`
  with the OpenRouter recipe's `models/` directory and a copy of its
  `upstreams.yaml` pointed at a scripted stand-in for OpenRouter on
  `127.0.0.1`. The stand-in streams the way OpenRouter documents: SSE comment
  lines, reasoning in a `reasoning` field, tool-call deltas and a closing usage
  event that repeats the finished choice. `conformance.py` against the server
  gives 11 pass, 0 fail and 1 recorded (`tool_choice_required`, supported).
  The test pins each request the stand-in received, case by case: the model
  `z-ai/glm-5.3-flash`, the key from `OPENROUTER_API_KEY` as a bearer token,
  and exactly `messages`, `stream`, `max_tokens` and `temperature`, plus
  `stream_options` when streamed and `tools` and `tool_choice` on a tools
  turn. Asked for `require_parameters`, the stand-in answers `404` when no
  endpoint of the model lists every request field that is one of the
  parameter names in OpenRouter's public model list. That list does not hold
  `stream` or `stream_options`; OpenRouter documents the preference in terms of
  its parameters and does not say how it treats other fields.
  `tests/openrouter_endpoints.json` holds the endpoints' lists and those names
  as OpenRouter published them. The stand-in also answers `401`, `402`, `404`
  and a stream whose only event is an error, and the test checks that each
  comes back as a `502` whose reason the server log names. It needs the
  repository's environment, from `mise run sync` at the root, and is skipped
  without it outside CI.

```sh
uv run pytest
```

`tests/test_conformance.py`, `tests/test_session_load.py` and
`tests/test_run_gate.py` check the checkers,
`tests/test_fake_engine.py` checks SIE's chat and Messages routes with a
scripted model, and `tests/test_openrouter_recipe.py` checks the OpenRouter
recipe against a scripted stand-in for OpenRouter.
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
- **The SIE Relay results are not a measurement of SIE's own GPU serving.**
  Through SIE Relay the model runs on a partner's servers. SIE routes the
  request, checks the wire and meters it. The one-GPU results are SIE's own
  serving, of a different model.
- **One GPU, one machine, one day.** The one-GPU results were recorded on one
  H100 on 2026-10-15. Another card, driver or engine version can move every
  number, and sessions with pauses between turns, as real agents have, load
  the GPU less than `session_load.py` does.
- **Not a measure of how well Qwen3.8-27B codes.** BFCL scores single calls,
  the seeded gate is a twenty-line fix, and the one real issue was not fixed
  in either run.
- **Not yet a run against OpenRouter.** Until the result above is recorded,
  the OpenRouter recipe has been checked against a scripted stand-in only.
  Which provider OpenRouter routes a request to, and how that provider streams
  tool calls and reasoning, shows only in a real run. A provider without
  `tool_choice: "required"` support can answer the `tool_choice_required` case
  with text, which fails it, and a provider that reports more completion
  tokens than `max_tokens` allows fails `finish_reason_length`.
- **One endpoint, one day.** The Cloud results were recorded once, on
  2026-10-14.
