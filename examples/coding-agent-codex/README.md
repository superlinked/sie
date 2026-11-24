# Run Codex CLI on SIE through the OpenAI Responses API

## What this shows

Codex CLI talks to a custom model provider through the OpenAI Responses API
(`wire_api = "responses"`). The SIE gateway and `sie-server`, the single-node
server, both serve `POST /v1/responses`: each translates the request into a
`/v1/chat/completions` request, serves it on its own chat route with all of
that route's limits, and translates the answer back, streamed or buffered. So
Codex reaches a model served by SIE through a provider entry in its config.

Codex lets you configure other providers, but OpenAI does not endorse models
other than its own behind Codex. Nothing here is supported or endorsed by
OpenAI. This example records how Codex behaves against SIE's endpoint and
checks that the agent loop closes.

The pieces:

1. `recipes/cloud.sh`, `recipes/cluster.sh` and `recipes/laptop.sh`. Each
   writes the Codex config for SIE Cloud, for a self-hosted cluster's gateway
   or for `sie-server` on this machine, and defines `codex` as Codex CLI
   0.167.0 run through `npx`.
2. The conformance check of the OpenCode example, run with `--api responses`.
   It drives `/v1/responses` the way Codex does and reports each case as
   pass, fail or recorded.
3. `run_gate.sh`, a seeded gate on the OpenCode example's fixture: a tiny
   package with two failing tests and the task "make the tests pass". Codex
   works a copy headless, then pytest judges the result.

## Run it

```sh
export SIE_BASE_URL=https://...   # the gateway, without /v1
export SIE_API_KEY=...
source recipes/cloud.sh           # or recipes/cluster.sh, see below
codex
```

The cluster recipe also reads the model and its window, which must match what
the gateway's `GET /v1/models` declares for the model:

```sh
export SIE_MODEL=zai-org/GLM-5.3-Flash:relay
export SIE_CONTEXT_TOKENS=131072
source recipes/cluster.sh
```

Codex's first request is about 35 KB of JSON, most of it Codex's developer
instructions and four tool schemas, and every later request repeats it.

The conformance check, from this directory:

```sh
uv run --project ../coding-agent-opencode python ../coding-agent-opencode/conformance.py \
    --api responses --base-url "$SIE_BASE_URL/v1" --model zai-org/GLM-5.3-Flash:relay
```

The seeded gate, which needs `npx` and `uv` on `PATH`:

```sh
./run_gate.sh            # SIE Cloud
./run_gate.sh cluster    # your gateway, with SIE_MODEL and SIE_CONTEXT_TOKENS set
./run_gate.sh laptop     # sie-server on this machine, see below
```

No recipe has been run against a live endpoint yet. The recipes and the gate
were checked with the pinned Codex against a recording stub and against
`sie-server` on the weightless fake engine
([recorded below](#what-result-to-expect)).

### On this machine

From a checkout of this repository, start `sie-server` with the generation
bundle:

```sh
mise run serve -- -b sglang           # generation bundle; MLX on Apple Silicon
```

Then, from this directory:

```sh
source recipes/laptop.sh
codex
```

`recipes/laptop.sh` points Codex at `http://localhost:8080` and names
`Qwen/Qwen3.5-4B:long-context`, the 32K-window profile of a 4B model that
declares tool calling. `SIE_BASE_URL`, `SIE_API_KEY`, `SIE_MODEL` and
`SIE_CONTEXT_TOKENS` override those. A server with authentication off accepts
any key. With authentication on, `SIE_API_KEY` is the server's token.

The conformance check and the gate against it:

```sh
uv run --project ../coding-agent-opencode python ../coding-agent-opencode/conformance.py \
    --api responses --base-url http://localhost:8080/v1 --model Qwen/Qwen3.5-4B:long-context
./run_gate.sh laptop
```

## The recipe

`recipes/cloud.sh` in full:

```sh
export CODEX_HOME="${SIE_CODEX_HOME:-$HOME/.codex-sie}"
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<TOML
model = "zai-org/GLM-5.3-Flash:relay"
model_provider = "sie"
model_context_window = 131072
web_search = "disabled"

[features]
multi_agent = false
goals = false
plugins = false

[analytics]
enabled = false

[shell_environment_policy]
exclude = ["SIE_API_KEY"]

[model_providers.sie]
name = "SIE"
base_url = "$SIE_BASE_URL/v1"
env_key = "SIE_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 1800000
TOML
codex() { npx -y @openai/codex@0.167.0 "$@"; }
```

- The config goes in its own `CODEX_HOME`, `~/.codex-sie` unless
  `SIE_CODEX_HOME` names another directory, so the config, login and sessions
  in `~/.codex` are left alone. Sourcing the recipe again rewrites
  `config.toml`.
- `codex` runs Codex CLI 0.167.0, the version the facts below were recorded
  with, whatever version is installed.
- `wire_api = "responses"` is the wire Codex speaks to a custom provider.
  `base_url` ends in `/v1`, and Codex appends `/responses`. Codex reads the
  key from `SIE_API_KEY` and sends it as `Authorization: Bearer ...`; the key
  is not in the file.
- The model id is sent byte for byte, slash and colon included.
- Codex has no metadata for a model outside its catalog. It prints `Model
  metadata for ... not found. Defaulting to fallback metadata` on every run,
  and the gate keeps that line as a warning. `model_context_window` sets the
  window Codex budgets against, so Codex compacts the conversation before the
  server refuses it. In a stub run with a 6,000-token window, Codex compacted
  after a turn that reported 5,810 tokens and not after one that reported
  5,010. It compacted with an ordinary `POST /v1/responses` without tools,
  which SIE serves.
- With its defaults, Codex declares a hosted `web_search` tool and a
  `namespace` tool, `multi_agent_v1`. SIE serves function and custom tools
  only. It refuses both with a 400 that names `tools[i].type`, and Codex
  prints that error and exits non-zero on the first request.
  `web_search = "disabled"` and `multi_agent = false` leave them out. Web
  search is a hosted tool that the Responses server runs itself, and SIE runs
  none; the multi-agent tools come only as a `namespace`.
- `goals = false` leaves out three goal function tools that SIE would serve,
  so every request declares the four tools a coding session needs:
  `exec_command`, `write_stdin`, `request_user_input` and `view_image`.
- With its defaults, Codex also opened connections to `chatgpt.com`,
  `ab.chatgpt.com`, `github.com` and `api.github.com`, seen through a recording
  proxy. With `plugins = false` and `analytics.enabled = false` it opened none
  but SIE's.
- Codex passes its environment to the commands the model runs, and
  `SIE_API_KEY` went with it. `shell_environment_policy.exclude` keeps the key
  out of those commands; Codex itself still sends it.
- Codex ends a stream that sends no event for `stream_idle_timeout_ms`, 300
  seconds by default, and sends the request again. SIE keeps the model's
  reasoning private on `/v1/responses`, so a model that thinks for minutes
  streams nothing until it answers. SSE comment lines, such as the keep-alive
  `sie-server` sends every 15 seconds, did not reset Codex's timer in a stub
  run. The recipe allows 30 minutes, like the Claude Code example's
  `API_TIMEOUT_MS`.
- Codex sends no output cap, so the chat route's default for the model
  applies. On `sie-server` that is the profile's `max_output_tokens`.

## What Codex puts on the wire

Recorded on 2026-11-24 with Codex CLI 0.167.0, run headless (`codex exec`)
against a local stub that answered scripted Responses streams. No model was
called. The scrubbed requests are in
[`conformance/openai_responses/codex_requests.json`](../../conformance/openai_responses/codex_requests.json),
and the gateway's and `sie-server`'s tests translate every one of them.

- Every request goes to `POST /v1/responses` with `stream: true`,
  `store: false`, `include: ["reasoning.encrypted_content"]`,
  `reasoning: {"summary": "auto"}`, `tool_choice: "auto"`,
  `parallel_tool_calls: true`, a `prompt_cache_key` and `client_metadata`.
  SIE accepts all of them. `prompt_cache_key` goes on to the chat route; the
  rest have no effect. No `max_output_tokens` and no `temperature` are sent.
- Codex's instructions arrive as `developer` messages at the head of `input`,
  then a user message with the environment, then the task. SIE joins the
  leading `developer` messages into one system message.
- Each turn sends the whole conversation again: the `function_call` and
  `function_call_output` items, with Codex's ids, which SIE ignores. Codex
  sends no `previous_response_id`, which SIE refuses because it stores no
  responses.
- `view_image` returns the image as an `input_image` part with a base64
  `data:` URL inside the `function_call_output`. SIE moves it into a user
  message after the tool message. A model without image input refuses that
  with a 400, and Codex ends the turn on a 400.
- Codex retried a `503` while the fake engine's model loaded, with no event in
  its output. A dropped stream comes back as an `error` event
  `Reconnecting... n/m (...)` before the retry.
- Without `< /dev/null`, `codex exec` waits to read more of the prompt from
  standard input.
- With `--sandbox workspace-write`, the model's commands run without network
  and can write only in the working directory and the temporary directory.
  So `uv run pytest` there needs an environment synced beforehand and a uv
  cache inside the working directory, and the gate prepares both. The
  commands run in a login shell, so they use the `uv` your login shell finds.
  On macOS, Homebrew's uv 0.8.24 panicked inside the sandbox; uv 0.5.31 and
  0.12.10 ran there.

To check another Codex version, change the version in the recipe, run
`codex exec --json --skip-git-repo-check --sandbox workspace-write "<task>" < /dev/null`
with `SIE_BASE_URL` pointing at a server that records each request body and
answers with a Responses stream, and compare the requests with
`codex_requests.json`: the tool types, the top-level fields and the input
items. Then run the conformance check and the gate.

## What the Responses conformance check checks

| Case | Passes when |
|---|---|
| `plain_answer` | a streamed one-word answer ends `completed` with `output_text` and usage carrying `input_tokens` and `output_tokens` |
| `function_call` | with Codex's fields and a developer message, a `function_call` item opens with a `call_id` and the name `get_weather`, its argument deltas join into a JSON object with `city`, its `output_item.done` carries the same arguments, and the response ends `completed` |
| `tool_output_round_trip` | after the `function_call` and a `function_call_output` of 18 degrees, as Codex sends them, the answer carries 18, no further call, `completed` |
| `image_in_tool_output` | a `view_image` call whose output is an `input_image` `data:` URL, as Codex sends it, is answered. When the chat route refuses the image, for a model without image input, the refusal is recorded. A refusal at an `input` field, the translation's own, fails |
| `buffered_response` | a non-streamed request returns an `object: "response"` that is `completed`, with `output_text` and usage |
| `hosted_tool_refusal` | a `web_search` tool beside a function tool is refused with a 400 at `tools[1].type`, the refusal Codex prints with its default tools |
| `previous_response_id_refusal` | `previous_response_id` is refused with a 400 at that field |
| `event_order` | every stream starts with `response.created` and `response.in_progress`, numbers its events from 0 without a gap, carries each event's name as its `type`, opens one item at a time at the next `output_index`, sends each item's events while it is open, and ends with one terminal event that closes every item, or with `response.failed` |

Busy refusals are retried as in the chat check: a `429` or `503` whose code
is `QUEUE_FULL`, `RESOURCE_EXHAUSTED` or `rate_limit_exceeded`, or a
`response.failed` with one of those codes before any output, with the
`Retry-After` wait, at most 4 times and 3 minutes. A model still loading is
waited for first, up to 15 minutes.

## What result to expect

### The gate against a recording stub, recorded

`run_gate.sh` with an installed Codex CLI 0.167.0 behind a stand-in for
`npx -y @openai/codex@0.167.0`, and a stub on `SIE_BASE_URL` that plays a
model that reads the module,
prints `SIE_API_KEY`, rewrites the module and runs the tests, on macOS,
2026-11-24. The test command was sent as a non-login shell, so it used the
`uv` the gate ran with. Codex's JSON event lines are left out:

```
== before: the seeded tests
FAILED tests/test_amounts.py::test_credit_in_parentheses_is_negative - decima...
FAILED tests/test_amounts.py::test_total_nets_credits - decimal.InvalidOperat...
2 failed, 1 passed in 0.02s
== the worked copy's environment, for the tests Codex runs in its sandbox
== codex run, recipe cloud, up to 1800 s
Reading additional input from stdin...
== the run
{
  "commands": 4,
  "items": [
    "error",
    "command_execution",
    "command_execution",
    "command_execution",
    "command_execution",
    "agent_message"
  ],
  "usage": {
    "input_tokens": 500,
    "cached_input_tokens": 0,
    "output_tokens": 50
  },
  "reconnects": 0,
  "warnings": [
    "Model metadata for `zai-org/GLM-5.3-Flash:relay` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."
  ],
  "unreadable_lines": 0,
  "problems": []
}
== after: the seeded tests must be untouched
== after: pytest in a fresh environment, pinned to the seeded config and tests
...                                                                      [100%]
3 passed in 0.01s
GATE PASS (.../run-output/gate-20261124T141421Z-p8VUOB)
```

The `printenv SIE_API_KEY` command printed `SIE_API_KEY is not set`, every
request arrived with the key as a bearer token and declared only the four
function tools, and a recording proxy set as Codex's `HTTPS_PROXY` saw no
connection. The `uv run
pytest -q` the stub asked for ran in the sandbox and reported `3 passed`. This
checks the plumbing only: the recipe, the pinned Codex, the sandbox and the
verdict. It says nothing about a model.

### Codex against `sie-server` on the fake engine, recorded

`sie-server` from this repository with the fake bundle on the CPU (`mise run
serve -- -b fake --device cpu`), then `./run_gate.sh laptop` with
`SIE_BASE_URL=http://127.0.0.1:18900`, `SIE_MODEL=sie-fake:chat` and
`SIE_CONTEXT_TOKENS=131072`, the same day. The fake answers from a script: with
tools declared it calls the first one, `exec_command`, passing the last user
message, the task text, as the command. After the command's output it answers
`Tool result: ` and that output.

The first request was answered `503` while the model loaded, and Codex sent
it again. Every later request was answered `200`. The run finished cleanly in
one command, with 8,957 input tokens over two requests, and the gate then
failed on the seeded tests, 2 failed and 1 passed, because the fake changes
nothing. This is how Codex's real requests fare on SIE's translation and chat
route, not a model.

The conformance check against the same server:

```
  plain_answer                  pass      status=completed after 'tok_b0437c31 tok_5aaf6099 tok_f3b6ad77 t'; usage {'input_tokens': 7, 'input_tokens_details': {'cached_tokens': 0}, 'output_tokens': 40, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': 47}
  function_call                 pass      1 function_call: get_weather({"city": "What is the weather in Paris right now? Use the get_weather tool."}); 13 arguments deltas; status=completed
  tool_output_round_trip        pass      final answer used the tool output: 'Tool result: {"city": "Paris", "temperature_c": 18, "conditions": "overcast"}'
  image_in_tool_output          recorded  the model's chat route refused the image: 400 unsupported_field: Model 'sie-fake:chat' does not support image input
  buffered_response             pass      status=completed, usage {'input_tokens': 7, 'input_tokens_details': {'cached_tokens': 0}, 'output_tokens': 40, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': 47}
  hosted_tool_refusal           pass      400 unsupported_field: tools[1].type: 'web_search' tools are not supported; send function or custom tools
  previous_response_id_refusal  pass      400 unsupported_field: previous_response_id: SIE does not store responses; send the full conversation in input
  event_order                   pass      3 streams: created, in_progress, one item at a time, sequence numbers from 0, one terminal event last

7 pass, 0 fail, 1 recorded
```

The fake takes no image input, so the image is refused by the chat route and
recorded.

### Checked in this repository

- The gateway's and `sie-server`'s tests replay the shared vectors in
  `conformance/openai_responses/` (request translation, buffered answer,
  event sequence, error mapping) and translate or refuse every captured Codex
  request as its vector says.
- The OpenCode example's `tests/test_fake_engine.py` runs
  `conformance.py --api responses` against this repository's `sie-server` on
  the fake engine and asserts the table above.
- The OpenCode example's `tests/test_responses_conformance.py` checks the
  `--api responses` checker against a scripted Responses stub that plays a
  conforming stream, then breaks one thing per test: an accepted hosted tool,
  a refusal at another field, an image refused by the translation, done
  arguments that differ from the streamed ones, a gap in the sequence numbers,
  busy refusals and a loading model.
- `tests/test_run_gate.py` runs `run_gate.sh` with a scripted stand-in for
  `npx`: a real fix, no change, a failed run, a failed turn, an error event,
  no input tokens, no command, no turn completed, edited tests, a run past its
  deadline, the three recipes, the config each writes, and the arguments and
  environment Codex is started with. It also sources a recipe the way a user
  does and checks that it writes its own `CODEX_HOME` and runs the pinned
  Codex.

```sh
uv run pytest
```

## What this does NOT establish

- **Not supported or endorsed by OpenAI.** OpenAI does not endorse models
  other than its own behind Codex. A Codex release can change what it sends,
  which is why the recipe pins 0.167.0. The capture above is how to re-check a
  new version.
- **No live run yet.** Neither SIE Cloud, a cluster nor a real model on a
  laptop has run the gate or the conformance check. The results above come
  from a recording stub and the weightless fake engine.
- **Not a coding benchmark.** The fixture is two failing tests in a
  twenty-line module.
- **Compaction seen only against a stub.** Codex 0.167.0 compacts with an
  ordinary `POST /v1/responses` request without tools, which SIE serves. That
  was checked with a 6,000-token window against a stub, not in a long live
  session.
- **Images need a model that reads them.** On a model without image input, a
  `view_image` call is refused, and Codex ends the turn.
- **One operating system.** The sandbox and the stub runs were on macOS.
  Codex's Linux sandbox has not been run with this gate.
