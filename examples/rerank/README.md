# Rank by a relevance rule you write in the call

The runnable example behind [superlinked.com/rerank](https://superlinked.com/rerank).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/rerank/SOURCES.md).

## What this shows

24 questions, each sent to `Qwen/Qwen3-Reranker-4B` against the same four
candidate passages, five times: once with no instruction, and once under each of
four relevance rules.

Every candidate is a verbatim abstract from the Federal Register. Each case pairs
a rulemaking's **final rule** with the **proposed rule** published under the same
title, plus a final rule and a proposed rule from a second docket of the same
agency on a neighbouring subject. The query is the shared docket title, so it
names the subject and never says whether the rule is in force. Only the
instruction can choose between a regulation already adopted and a proposal that
is not.

Ground truth is the Federal Register's own `type` field. Nothing in this example
decides which document is in force.

## What the run found

```
With no relevance rule in the call, the document at rank 1 was
  a regulation already in force        24/24
  a proposal not yet adopted            0/24

rule stated in the call              right document   right kind
  in-force                                  16/24        19/24
  proposed                                  20/24        21/24
  in-force-positive                         21/24        23/24  (published)
  proposed-positive                         16/24        17/24  (published)
```

Two things, both worth reading twice.

The reranker has a prior. Asked a question with no relevance rule attached, it
returned a regulation already in force in **all 24** questions and a pending
proposal in **none**. That is not a length or a keyword effect: the proposal
abstracts are the longer ones on average, and query word overlap is tied between
the two in 18 of the 24 cases.

Stating the rule overrides the prior. Ask for the proposal and one leads in 17 of
24, up from 0. The two published rules put a **different** document first in 16
of the 24 questions, over an identical candidate set and an identical query.

Stating the rule the model already follows does not help. `in-force-positive`
scores 21 of 24 against a 23 of 24 baseline. An instruction is worth writing when
your definition of relevant differs from the model's, not when it agrees.

## Why four rules, and which two the page shows

The first recording used rules ending in a negative clause ("a change that has
only been proposed is not relevant, however closely it matches the subject").
That pair scored 36 of 48 summed over both directions, and the "in force"
direction came out *worse* than no instruction at all, 16 of 24 against 23.

A second wording was registered before it was run, in the two `-positive` rules,
which state only what to retrieve and never what to exclude. The decision rule
was fixed in advance: publish whichever pair scores higher summed over both
directions. The positive pair took it, 37 of 48 against 36 — by a single case,
which is well inside the noise of a 24-case set, and the README says so rather
than dressing one point up as a finding.

All four arms are in `calls.json`. The pair the page does not display is
recomputable from the same file.

## Run it

The code is in this repository. The inputs and the recorded responses are in the
public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces every published figure offline
python3 run.py --check   # rebuilds all 120 recorded calls from the inputs
python3 -m unittest discover -s tests
```

These need nothing installed. They are standard library only, and none of them
needs an API key, a Hugging Face token or any inference spend. `fetch.py` pins a
commit SHA rather than `main`, and checks every downloaded file against a digest
before the scorer sees it. It replaces `--dest` wholesale, so it refuses to touch
anything without the `.sie-evidence` marker it writes, and it swaps the new
directory in by rename rather than deleting the old one first.

To record against SIE Cloud yourself. This is the only part that needs the SDK,
and the only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py --record
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into `main()`,
so `--check` and `--show` keep working on a bare `python3` with nothing
installed. `python3 run.py --show air-plan-approval-pennsylvania` prints one
case's five calls without sending any of them.

## How the figures are derived

The two sides of every comparison come from different places. The document each
rule asks for was derived from the Federal Register's `type` field and written
into `inputs/cases.json` before the run. The ranks come out of the recorded
responses in `calls.json`. The figures `score.py` asserts against come from a
third place again: the page, pinned in `score.py`'s own source, so editing the
fetched evidence alone will not satisfy it.

- **fail closed**: a missing call, a non-200 status, a response scoring a
  different candidate set than the inputs list, or a recording made against
  another model revision stops the run rather than producing a figure.
- **absent data is a failure, never a skip**. A case with no recorded call does
  not quietly leave the denominator.
- **a count is not a set**: the check compares scored item identities against the
  candidate list, so a duplicated row does not pass on row count alone.

`tests/test_example.py` tamper-tests each of those. The sharpest one trades two
responses between arms of the same case: every digest, every count and every
rebuilt request envelope survives it, `run.py --check` still returns 0, and only
the figures pinned in `score.py` refuse it.

## What is in the dataset

```
rerank/
  inputs/cases.json  24 cases: the query, the agency, the two dockets, the four
                     candidate abstracts with their Federal Register type,
                     document number and URL, and the document each rule asks
                     for; plus the four rule texts
  calls.json         120 calls: the SDK call envelope including the instruction
                     sent, the response, the status and the latency, one file
  manifest.json      the endpoint, the model and its revision, the deployment
                     revision, the run date and a digest for every file
```

## What this does NOT establish

- **24 questions are a demonstration, not a benchmark.** No figure here
  generalises to your corpus, and one document type axis is not a study of
  instruction following.
- **The corpus is one kind of hard.** Adopted-versus-proposed is a distinction
  the passages themselves express. A rule that depends on metadata the passage
  never states does not work this way: two earlier designs, one keyed on which
  RFC in a lineage is current and one on an IETF document's standards-track
  status, came out at or near chance, because an abstract does not say whether
  it has since been obsoleted. An instruction can only apply a rule the text
  supports.
- **One case is not attributable.** The 37-to-36 margin between the two wordings
  is a single question out of 48.
- **The scores are not probabilities.** SIE's score response exposes `item_id`,
  `score` and `rank`. The model's internal logits are not part of the public API
  response, so nothing here reconstructs them.
- **No tamper resistance.** The digests catch a truncated or corrupted download.
  They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/rerank/`, which is what its CI tests read.
The two copies hold the same recorded responses. Nothing binds them together, so
they can drift.
