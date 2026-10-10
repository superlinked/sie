# Run the fixed Workflow DEV32 source through a SIE URL

This runner sends all 32 original structured workflow requests through
`sie_sdk.SIEClient.chat_completions`. Supply a compatible SIE URL and a model
that accepts the original messages and strict JSON schema. A hosted endpoint
or your own server works; the runner requires no particular deployment.

The [public source](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/6ca14001c2b6287473d89da72e549702a9b9343c/typed-decisions/2026-10-09/gemma31b-fresh-workflow32-source-v2)
contains eight episodes, four dependent snapshots per episode, four policy
families and 208 required answers. The fixed requests contain the full policy,
state, questions and schema. The [upstream source](https://github.com/JacobLinCool/StreamDecisionBench/tree/253c741573559c8167863f172001e478575b0dde)
and MIT notices are pinned in the source packet. `source.json` fixes the public
revision, file lengths and checksums; `fetch.py` verifies them.

**This is a development set.** Eight semantic source-ancestry exclusions
remain unresolved. Its eight episodes do not establish performance on unseen
families, population equivalence or a held-out confirmation result. This set
is separate from the historical CVE and LocalLLaMA examples in the parent
directory.

## Fetch and run

From this directory, with Python 3.12 and `uv`:

```bash
uv sync --frozen
uv run --frozen python fetch.py
export SIE_URL=http://localhost:8080
uv run --frozen python run.py \
  --model google/gemma-4-31B-it \
  --thinking on \
  --out run-output
```

Use your endpoint in `SIE_URL`, or pass `--url`. Set `SIE_API_KEY` only if the
endpoint requires authentication. Choose a model available on that endpoint;
the example model name does not require a specific machine profile. The URL
may include a base path. Credentials, query strings and fragments belong
outside the URL.

The default settings are a total completion-token limit of 32,768, temperature
1, top-p 0.95, top-k 64 and a 1,800-second wall window per case. The completion
budget covers reasoning as well as the final JSON. `--thinking on` requests
`chat_template_kwargs.enable_thinking=true`; the server's model configuration
may override that request. The runner records caller settings without claiming
to verify actual server configuration. Use `--thinking off` or
`--thinking server` to record a different configuration; the latter omits the
template override. `--max-completion-tokens` and `--timeout` are explicit
overrides and remain visible in `plan.json`. Results with different settings
are different runs.

Every case runs once, in source order, with no truncation, tools, retry,
synthetic probe or warmup. The first ordinary case gates continuation on a
complete normal finish with valid JSON matching the original schema. A wrong
but structurally valid answer passes that gate: gold is never used to choose
which cases run. Later complete capped, empty or invalid model outputs remain
one failed case and the independent cases continue. An unresolved send,
transport/admission error or persistence failure stops new sends; all original
32 cases remain in the ledger, including the untouched tail.

Use a fresh output directory. The runner refuses an existing directory and
has no resume mode. Each case has a durable exclusive intent and pre-POST
fence. The transport permits only one physical completion POST; SDK metadata
and completion-result GETs retain separate evidence. The parent process
enforces the complete case deadline and reaps its caller before continuing.
If interrupted, preserve the directory and treat consumed requests without a
complete outcome as unknown; do not automatically resend them.

## Outputs and the original scorer

`run-output/responses.json` retains the fixed 32-case denominator and is
compatible with the original source scorer. `plan.json` records the source,
validated base URL and caller settings. Each `cases/NN/` directory retains its
intent, actual request, bounded decoded HTTP-response bodies, full SDK response,
usage and terminal outcome when available. HTTP evidence records the original
content encoding separately; response bodies are decoded once. No credential
or authorization header is recorded. The full SDK
result preserves reported reasoning/usage rather than estimating cost from
visible JSON. Encoded and decoded HTTP responses are each limited to 16 MiB;
gzip decompression is bounded as it produces output. Exceeding either bound
leaves an unresolved attempt and stops continuation.

For **your new rerun**, invoke the fetched, unchanged scorer:

```bash
uv run --frozen python data/score.py run-output/responses.json > run-output/score.json
```

The primary metric requires exact agreement on **every required field**,
including inactive and unknown fields. Its useful-fit gates are at least
29/32 joint records, 8/8 controls, 7/8 complete episodes and at least one of
the two complete episodes in each family. Missing, failed, capped and invalid
outcomes remain failures in the original denominator. Active-branch
application correctness is a separately reported sensitivity metric and never
replaces these gates. Four snapshots from one episode are dependent; do not
treat 32 rows as 32 independent episodes.

The [published provider comparison](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/34b46099a7fd1642f6bbe61a3b1a652204a762f8/typed-decisions/2026-10-10/gemma31b-dev32-provider-results-v1)
uses the same pinned source and primary scorer: GPT-6 Luna with reasoning
effort `none` scored 11/32 joint records; GPT-6.1 Sol with effort `low` scored
31/32. One known Luna HTTP 400 remains a failed original case. Those are saved
provider results, not numbers produced by this runner or a native SIE quality
claim. The linked packet preserves the exact provider settings and full
denominators.

Case times here are client wall times and include the chosen endpoint and
network. Record server revision, model configuration and caller location
separately when comparing latency; do not infer native model latency or a
retail price from these fields alone.

## Local validation

The tests use mocked HTTP responses and local child-process controls. They
require no key, model, provider or running SIE server:

```bash
uv run --frozen python -m unittest discover -s tests
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```
