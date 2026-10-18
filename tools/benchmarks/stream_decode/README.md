# Single-stream decode benchmark

This records fixed-length, serial chat completions through `sie_sdk.SIEClient`
and SIE's actual `/v1/chat/completions` proxy. It addresses the measurement part
of [#435](https://github.com/superlinked/sie/issues/435). It does not yet contain
a qualified GPU result or justify a performance change.

The dedicated benchmark server observes SGLang 0.5.20 inside its child process.
It chains SIE's production startup hooks, then observes
`TokenizerManager._tokenize_one_request` and `generate_request`. Native SIE
`adapter.generate` instrumentation cannot observe this chat route.

The observer retains actual prepared input IDs, effective sampling, raw output
events and cumulative completion counts. SDK chunk IDs must match the backend
request ID. Prompt, model, source manifest, clock domain and fixed trial count
must also match. Each call writes one initial occupancy record; events stay in
memory until stream completion or cancellation. No per-event file writes or
retokenization occur. This instrumentation adds host
overhead and must be identical in every compared arm.

## Run in a dedicated, bounded Linux GPU environment

Use a frozen public checkpoint, server configuration and environment. Preload
the model and verify readiness before admitting the client. Own the server's
process group and enforce an external absolute deadline, including teardown.
The benchmark wrapper does not allocate GPUs or supervise server lifetime.
Use public synthetic prompts: raw requests and outputs are retained as evidence.

From a clean checkout with the pinned CUDA13/SGLang bundle installed:

```sh
mise exec -- python -m tools.benchmarks.stream_decode.server \
  --trace-directory /tmp/decode-a-server --cohort decode-a -- \
  serve --host 127.0.0.1 --port 8080 --device cuda:0 \
  --models-dir /tmp/frozen-model-configs \
  --models Qwen/Qwen3.8-27B-FP8 --preload Qwen/Qwen3.8-27B-FP8

# In another shell, after model readiness and observer installation:
mise exec -- python -m tools.benchmarks.stream_decode.client \
  --url http://127.0.0.1:8080 --model Qwen/Qwen3.8-27B-FP8 \
  --prompt /tmp/fixed-public-prompt.txt \
  --server-manifest /tmp/decode-a-server/server.json \
  --output /tmp/decode-a-client --completion-tokens 128 \
  --warmups 2 --repeats 6 --request-seconds 120 --total-seconds 1200

mise exec -- python -m tools.benchmarks.stream_decode.analysis \
  --client-directory /tmp/decode-a-client \
  --server-directory /tmp/decode-a-server \
  --prompt-tokens 670 --completion-tokens 128 \
  --output /tmp/decode-a-analysis.json
```

The prompt-token count above is an example: freeze the actual chat-template
token count before the experiment. A mismatch invalidates the cohort. Use a
new directory and cohort for each planned arm; retain failed attempts. The
client requests greedy sampling, a fixed seed, reasoning off, and equal
minimum/maximum output tokens. Operator template settings remain authoritative;
analysis rejects a run that enabled thinking or changed these settings.

Each request runs in a fresh child with its own wall deadline, plus bounded
parent termination/reaping. HTTP proxies and remote fallback are disabled.
There is no retry after an admitted failed stream. Client construction and
post-request file writes are outside the client timing interval.

## Interpret the measurements

Client TTFT is the time until the first visible text delta. Total latency runs
from request iteration to completed stream. Completion tokens divided by total
latency includes TTFT and is reported separately.

Decode rate uses actual cumulative completion counts and host observation
timestamps before chat formatting: `(last_count - first_count) / elapsed`.
After-first-token rate is available only when the first observed count is
exactly one. SGLang may coalesce outputs, particularly with speculative
decoding; a first count above one produces only an after-first-observation rate.
An SSE event is never assumed to represent one token. These timestamps are not
GPU kernel timings. Kernel profiling belongs in separate fixed calls so profiler
overhead cannot silently enter timing measurements.

The analyzer rejects incomplete streams, retries, wrong lengths, mismatched
IDs/text, malformed or regressing counts, missing usage, changed prepared inputs,
overlapping trials and extra backend requests inside the measured schedule.
Native startup and health generations pass through unchanged; their intervals
are retained, and any overlap with the measured schedule invalidates the run.
Engine cleanup may finish after the SDK receives its terminal response; all
recorded engine events must still lie inside both applicable timing intervals.

CPU controls exercise the actual SIE CUDA chat route with a synthetic child,
the real SDK over local HTTP, concurrent observation/cancellation, startup-hook
restoration, child deadlines, and negative evidence cases. They do not establish
GPU correctness, kernel performance or model quality.

```sh
mise run test -- tools/benchmarks/stream_decode
mise exec -- ruff check tools/benchmarks/stream_decode
mise exec -- ruff format --check tools/benchmarks/stream_decode
```

Reviewed upstream boundary: [SGLang 0.5.20 TokenizerManager](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/managers/tokenizer_manager.py)
and [chat formatting](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/entrypoints/openai/serving_chat.py),
including [sampling normalization](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/sampling/sampling_params.py).
All three installed sources must match the checked-in hashes before observation.
The client requires a matching observer installation record before sending requests.
