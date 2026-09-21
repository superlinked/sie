# Check text for a planted instruction before your agent acts on it

The runnable example behind [superlinked.com/guardrails](https://superlinked.com/guardrails).

## What this shows

Twelve inputs an agent could plausibly be handed, six carrying a planted
instruction and six ordinary, scored by two guard models on
`https://api.superlinked.com`:

| model | role |
|---|---|
| `fastino/gliguard-LLMGuardrails-300M` | the model the page runs |
| `ibm-granite/granite-guardian-3.0-2b` | the larger guard, for comparison |

The texts are verbatim from BIPIA, InjecAgent, LLMail-Inject, AgentDojo, a
deepset prompt-injection set, a NIST document and XSTest.

The page publishes one figure:

> Across all 12 recorded inputs, of which 8 are shown on this page, GLiGuard
> flagged 4 of 6 planted instructions and passed 4 of 6 ordinary messages.

`score.py` re-derives 12, 4 of 6 and 4 of 6 from the recorded responses,
offline, and prints all twelve rows including the two misses and the two false
alarms.

Four of twelve wrong is the honest result and it is on the page. This example
exists so the number can be checked rather than believed.

### Why the page runs the 300M model and not the 2B one

On these same twelve inputs, recorded in the same run:

| | planted instructions flagged | ordinary messages passed | right |
|---|---|---|---|
| GLiGuard 300M | 4 of 6 | 4 of 6 | **8 of 12** |
| Granite Guardian 2B | 5 of 6 | 2 of 6 | **7 of 12** |

Granite catches one more planted instruction and raises a false alarm on four
of the six ordinary messages. That is the behaviour its published figures
describe rather than a surprise: the served model catalog records recall 0.97 at
precision 0.16 for this model on ToxicChat under the risk it is served with. A
recall number is not an accuracy number, and on a set that is half ordinary
traffic the gap shows up as false alarms. `score.py` asserts both models'
counts, so a re-record that moves either one fails rather than quietly changing
a published sentence.

Twelve inputs decide nothing about either model in general. See
"What this does NOT establish".

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces both models' figures offline
python3 run.py --check   # rebuilds all 48 recorded requests from the inputs
```

These three need nothing installed: they are standard library only, and none
of them needs an API key, a Hugging Face token or any inference spend.
`fetch.py` pins a commit SHA, not `main`, and checks every downloaded file
against a digest. A file that is missing, unreachable or that fails its digest
stops the run with a message and writes nothing, so a partial tree can never be
scored as though it were whole. It replaces `--dest` wholesale, so it refuses to
touch anything without the `.sie-evidence` marker it writes, and it swaps the
new directory in by rename rather than deleting the old one first.

To call the API yourself. This is the only part that needs the SDK, and the
only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py --record --out run-output/calls.json
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into
`main()`, so `--check` and `--show` keep working on a bare `python3` with
nothing installed. `python3 run.py --show <id>` prints a request without
sending it. Before spending anything, `--record` reads `GET /v1/models` and
stops if either model is served at a revision other than the one this example
publishes; `--allow-revision-mismatch` records anyway, for your own comparison
rather than to reproduce the published figures.

## What to expect

```
GLiGuard 300M       8 of 12 right   (4 of 6 flagged, 4 of 6 passed)
Granite Guardian 2B 7 of 12 right   (5 of 6 flagged, 2 of 6 passed)
```

with a row per input for each model, naming every miss and every false alarm.
`score.py` exits nonzero if any of those counts fails to reproduce.
`run.py --check` prints `48 of 48 recorded requests rebuilt from the inputs and
matched`, and fails if a call is missing, recorded twice or implied by no input.

## What is in the dataset

```
guardrails/
  inputs/inputs.json   12 inputs: verbatim text, expected verdict, source, licence, digests
  calls.json           48 calls: request, response, status, timing, one file
  manifest.json        endpoint, models, revisions, run window, digests
```

Four calls were recorded per input, all 48 kept:

| call | what it sends |
|---|---|
| `granite-harm` | `ibm-granite/granite-guardian-3.0-2b` through `/v1/chat/completions` |
| `gliguard-jailbreak` | 12 published jailbreak labels, `multi_label`, threshold 0 |
| `gliguard-prompt-safety` | the served default, no params |
| `gliguard-snippet` | safe/unsafe under `prompt_safety`. The page's verdicts come from this one |

### Two things worth knowing about the Granite call

**It goes through `/v1/chat/completions`, not `/v1/generate`.** `/v1/generate`
passes raw input with no chat template, so a guard model's risk template never
runs. The 2026-09-15 recording of this task did that and eleven of its twelve
Granite verdicts came back as empty text. Those recordings are superseded by
this revision, and `manifest.json` names the one it supersedes and why.

**It sends no `chat_template_kwargs`.** SIE Cloud serves this model under one
risk dimension, `harm`, fixed in the served model catalog. A per-request
`chat_template_kwargs.guardian_config.risk_name` is accepted and validated by
the gateway and then discarded by the worker, which applies the catalog value.
Measured two ways on 2026-09-21: the twelve inputs sent under `jailbreak` and
under the default returned identical verdicts and identical `prompt_tokens`, and
a 120-character risk name rendered a prompt of exactly the same length as the
default. So the request sends nothing that the server ignores, and the call is
named for the risk actually applied.

### Two revisions per call, under two names

`recorded.model_revision` is the weights revision, read from `GET /v1/models`.
`recorded.served_model_revision_header` is the `X-SIE-Model-Revision` response
header, which is the deployment's execution bundle digest and is the same value
for every model that deployment serves. The superseded revision recorded the
header alone, under the name `model_revision`, so two models with different
weights appeared to share a revision.

## What this does NOT establish

- **Nothing about either model's accuracy in general.** Twelve inputs is a
  demonstration. 8 of 12 and 7 of 12 are the results on these twelve, and a
  one-input gap between two models over twelve inputs is not a ranking.
- **Nothing about the other two GLiGuard calls per input.** `score.py` reads
  `gliguard-snippet` and `granite-harm`. The other 24 recorded calls are in the
  dataset and are scored by nothing here.
- **Nothing about Granite under another risk dimension.** SIE Cloud serves only
  `harm` for this model, so that is the only dimension these twelve were scored
  under. A deployment serving a different one could get a different answer.
- **Nothing about "8 shown on this page".** That is a display decision made in
  sie-web. `score.py` scores all 12 and does not check the display count.
- **Nothing about a threshold.** Each verdict here is a top label or a one-word
  completion, not a score cut. A production gate would pick a threshold from its
  own costs, and the Granite figures above move a long way with one.
- **A fresh `--record` run records less than the archive.** For the three
  `client.extract` calls the SDK returns the per-item result rather than the
  server's envelope, so `--record` rebuilds the envelope around it; the
  `client.chat_completions` call returns the SDK's own result. Neither carries
  response headers. Each entry carries a `shape` field saying which it is.
- **No tamper resistance.** The digests catch a truncated or corrupted
  download. They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/guardrails/`, which is what its CI tests
read. Nothing binds the two copies together, so they can drift; as of this
revision sie-web still holds the superseded 2026-09-15 run.
