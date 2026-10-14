# Read a full app screenshot into typed fields

Gemma 4 31B on SIE reads a Kanboard task detail screen into a strict JSON schema.
One call returns the task's title, priority, dates and other summary fields,
plus three ordered subtasks with assignees and time-tracking strings. The
recorded answer passes all **26 of 26** original field checks and the complete
schema. This is one fictional fixture, not a population accuracy estimate.

[View the full original screenshot](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/77a6f4402054447580e55cd60c82ddfcd9813497/screenshot/2026-10-11/fixed-public-multi-app-fixtures-v2/cases/KB01/image.png).

## Verify the recording without a model or API key

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
cd examples/screenshot-fields
uv sync --frozen
uv run --frozen python evidence.py fetch
uv run --frozen python evidence.py verify
uv run --frozen python score.py
```

Fetch downloads a fixed list of public files anonymously. Each file's URL has an
immutable dataset revision; `pins.json` records its byte length and SHA-256.
Existing files with the wrong digest fail verification instead of being silently
replaced. After the first download, verification and scoring are offline.

Expected result: `checks: 26`, `passed: 26`, `fully_right: true`,
`schema_valid: true`. Scoring authenticates and imports the original published
scorer, including its text normalization and extra-row penalty. It uses the
original KB01 gold, with no post-hoc field exclusions. Schema validation also
rejects wrong types and undeclared fields. A truncated or invalid answer fails.

The complete published HTTP response is in
`evidence/native/KB01-response.json`, and its SDK representation is in
`evidence/native/KB01-sdk-response.json`. Both retain all answer fields and usage.
The public packet removed only the top-level response ID; its manifest records
that removed JSON path and the original raw-response digest. The example does
not replace either file with a clipped or reformatted answer.

## Send the same inputs to your SIE endpoint

Configure a SIE endpoint with
`google/gemma-4-31B-it:h100-96k-hires-thinking-budget512-no-spec` first. The endpoint
must have that model and profile available. Set `SIE_API_KEY` if it requires
authentication; leave it unset for an unauthenticated local deployment.

```sh
uv run --frozen python run.py --endpoint http://localhost:8080
uv run --frozen python score.py --response run-output/response.json
```

This command deliberately makes an inference request. `run.py` verifies the
original 1440 × 1000 PNG, the entire instruction including its final newline,
and the complete schema before calling `sie_sdk.SIEClient.chat_completions` once.
The system message carries the instruction; the user message carries the full
PNG data URI. The strict schema is supplied separately in `response_format`,
named `fixture_kb01`. The client sends a 32,768-token cap, temperature 0 and
presence penalty 0, with no application retry loop or fallback model. The SDK
may retry pre-execution admission or model-loading signals; it does not retry
a generation after a post-dispatch timeout.

A fresh complete response is saved separately in `run-output/response.json`.
An existing output file stops the command before dispatch. A new endpoint,
checkpoint, runtime or profile may return different fields; the recorded 26/26
result does not promise identical output or timing on every deployment.

## Request equivalence and provenance

The example pins released `sie-sdk==0.10.1`. The native recording used the
0.10.0 CUDA13 server image with public source overlay
[`ab4c9962`](https://github.com/superlinked/sie/tree/ab4c9962d5c0a411b7832f0b7b50cc8ff8d57566),
model revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, and the exact profile above.
The [native packet](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/c718fe42a8e914e122ffe8e5e20ac6f5a81bd6f9/screenshot-request-correction/2026-10-14/native/README.md)
contains the full request, response and runtime receipt.

The outgoing request is transport-equivalent to that recording after one
explicit alias: this example sends `max_completion_tokens`, while the recording
uses `max_tokens`. The pinned server's
[`_prepare_chat_body`](https://github.com/superlinked/sie/blob/ab4c9962d5c0a411b7832f0b7b50cc8ff8d57566/packages/sie_server/src/sie_server/api/openai_local.py#L762-L778)
resolves both to the same cap. An offline test uses the real released SDK and
`httpx.MockTransport` to compare every other request value, including the full
image, schema and prompt. Mock responses are test fixtures, not new measurements.

The [Screenshot Mining task page](https://superlinked.com/screenshot-mining)
has a separate 100-screen cost and field-agreement comparison. Its
[current sources](https://superlinked.com/reference/screenshot-mining/multiapp100/SOURCES.md)
and [immutable audit methodology](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/18e8c9ae848f55f5a5b2039bd79065b80b9c00ef/screenshot-request-correction/2026-10-14/audit/README.md)
explain that study, including preserved stock scores and sensitivity analyses.
This single-call example does not reproduce those aggregate costs or scores.
The older [12-screen example](../screenshot-mining) remains a separate recording.

## Tests

Fetch once using the commands above, then run credential-free checks:

```sh
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Tests block external downloads and mock SDK transport. They cover altered and
missing evidence, fixed download URLs, the full recorded request and response,
incorrect field values, missing and extra rows, schema errors and HTTP failures.
They make no model calls.

## Source notices

The screenshot uses a fictional Kanboard workflow. Kanboard is copyright
Frédéric Guillot and contributors, under MIT. Font Awesome is by Dave Gandy and
contributors, with its font and supporting-code notices retained in the corpus.
Read the pinned
[attributions](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/77a6f4402054447580e55cd60c82ddfcd9813497/screenshot/2026-10-11/fixed-public-multi-app-fixtures-v2/ATTRIBUTIONS.md),
[rights record](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/77a6f4402054447580e55cd60c82ddfcd9813497/screenshot/2026-10-11/fixed-public-multi-app-fixtures-v2/RIGHTS.json)
and [original notices](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/77a6f4402054447580e55cd60c82ddfcd9813497/screenshot/2026-10-11/fixed-public-multi-app-fixtures-v2/licenses).
Software names identify the source applications; this example makes no upstream
endorsement claim or additional trademark grant.
