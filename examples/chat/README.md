# Answer a question from a passage, and quote the sentence you used

## What this shows

Twelve passages from twelve different subject areas go to SIE Cloud, two
questions each, one `/v1/chat/completions` call per question. One question the
passage answers; one it does not. Every reply comes back as two lines: a short
answer, and the sentence from the passage the answer rests on.

The point is that the reply can be checked before anything ships. Four checks,
written before the run, score all 24 answers: the shape of the reply, whether
the quoted sentence is really in the passage, whether the answer matches the
benchmark's reference answer, and whether it stayed under 25 words.

The run is already recorded. The 25 requests and the exact responses they
returned live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive the
published numbers with **no API key and no inference spend**. Those are the
same bytes behind the figures on
[superlinked.com/chat](https://superlinked.com/chat).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `Qwen/Qwen3.8-27B-FP8`
- Endpoint: `https://api.superlinked.com/v1/chat/completions`
- Served deployment revision: `8bd714204e67a1c6c81f84b0dc486b6a6e96e943c42ff488f6b3cbf936e07955`, the `X-SIE-Model-Revision` every recorded call carries. Models served together share this value, so it names the deployment rather than the weights.
- SIE server version 0.7.3
- Run date 2026-09-21

| Check | Passes when |
|---|---|
| `format` | the reply is exactly two lines, the first starting `Answer: ` and the second `Quote: ` |
| `quote` | the quoted text appears in the passage character for character, or is exactly `none` when the answer is `Not stated in the passage.` |
| `reference` | the answer contains one of SQuAD's reference answers, or, for a question the passage cannot answer, is exactly `Not stated in the passage.` |
| `length` | the answer is 25 words or fewer |

## Run it

Download the recorded run, then score it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Look at a request without sending it:

```sh
python3 run.py --show oxygen__answerable
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

## What result to expect

`score.py` prints a line per question and ends with:

```
format                   24/24
quote                    24/24
reference                20/24
length                   24/24

15 of 24 answers cited a sentence, and every one of those is in its passage
9 of 24 declined and cited nothing
20 of 24 answers passed all four checks
```

The `quote` check reads 24/24 because it also passes on an abstention, where
the recorded quote is the literal `none`. Fifteen answers cited a sentence and
all fifteen are in their passage character for character; the other nine quoted
nothing, which is what the passage not answering the question is supposed to
look like.

Those are the figures the task page publishes. The page displays four of the
twelve passages, one in the hero and three in the proof grid; all twelve are in
`calls.json` here and all twelve are scored. A thirteenth passage carries the
page's playground call. It is fetched and rescored with the rest, printed with
`(playground, not counted)`, and counted in no total.

### The four answers that miss the reference

`score.py` fails on any integrity problem first: a missing input, a case with
no recorded call, a call nothing pins, a response that does not match its
digest, a pinned input that no longer rebuilds its recorded request, or a
served revision the manifest does not name. Once all of that passes, the
scoring exit status is decided by one thing: whether any answer quoted text
that is not in its passage, because that is what the page headlines. `reference` is 20 of 24, and here is every miss, so
nobody has to take a summary's word for what they are.

One is a model error:

- `complexity-theory__unanswerable` asks for *three* basic primary resources
  used to gauge complexity. The passage names two. The model answered "Time,
  storage, and communication are three basic primary resources used to gauge
  complexity." and quoted "quantifying the amount of resources needed to solve
  them, such as time and storage". The quote is real, which is why it passes
  the `quote` check; it names two resources and does not support the answer
  above it. A genuine citation under a claim it cannot carry is the failure
  mode worth knowing about.

Three differ from the reference in wording rather than in substance:

- `geology__unanswerable`: "The passage does not specify the exact ocean
  locations where volcano arcs are found." It declines, in its own phrasing
  rather than the fixed line.
- `amazon-rainforest__unanswerable`: "The passage states the Amazon rainforest
  covers most of the Amazon basin in South America, not Central America." It
  rejects the question's premise rather than using the fixed line.
- `ctenophora__answerable`: "Only 100 to 150 species of ctenophores have been
  validated according to the passage." The reference is `100–150`; this writes
  the same range in words.

The `wording` and `model error` split is a reading of the text, not something a
check decides. `score.py` counts four misses and takes no position on them.

## What this does NOT establish

- **Not that the answers are true.** `reference` compares against SQuAD's
  reference answers, which are spans other people annotated. An answer that
  contains the reference span and surrounds it with a wrong claim still passes,
  and no check reads the answer for meaning.
- **Not that a quote supports its answer.** `quote` establishes only that the
  quoted text is in the passage. `complexity-theory__unanswerable` above is
  exactly the case where that is true and the answer is still wrong.
- **Not a hallucination rate.** One run of 24 questions on one day. Sampling
  defaults were used and no question was repeated, so a rerun can return
  different wording and a different score. Treat these as what this recording
  shows, not as a measured rate.
- **Not a SQuAD benchmark result.** Twelve passages selected by the rule in
  `inputs/cases.json` are not the SQuAD 2.0 development set, and these numbers
  are not comparable with published SQuAD scores.
- **Not a claim about your documents.** All twelve passages are encyclopedia
  prose of a similar register. A corpus of tickets, manuals or filings may
  score differently.
- **Not bound to the website.** The same recordings back the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/chat/`, which
  is what that repository's CI checks. Nothing automatically ties the two
  copies together, so they could drift.
- **Not a guarantee the dataset is unchanged.** `fetch.py` pins a dataset
  revision rather than `main`, so a later upload cannot silently change what
  you score. It does not prove the revision holds what it held yesterday.

## Inputs

`evidence/inputs/cases.json` pins twelve passages from the development set of
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), with the SHA-256 of
each passage, the question ids, and SQuAD's own reference answers. The upstream
file `dev-v2.0.json` is pinned there by digest and is not copied into the
dataset. SQuAD 2.0 is CC BY-SA 4.0 and its passages are paragraphs from English
Wikipedia.

Selection, fixed before any call: twelve development-set articles on distinct,
non-political subjects; for each, one paragraph between 70 and 190 words
carrying both an answerable and an unanswerable question, and one question of
each kind that is well formed and whose reference answer is a short unambiguous
string.

`prompt.py` builds the request the model sees from that file. `run.py` and
`score.py` both read it from there, so the request the scorer rebuilds is the
request the runner sends. `score.py` verifies each passage against its digest,
rebuilds every recorded request from the pinned passage and question, and
checks each recorded response against its `response_sha256` and the served
model revision the manifest names. A passage edited to match a recomputed
digest still fails, because the rebuilt request no longer equals the recorded
one.

`score.py` fails rather than skipping. A missing input, a case with no recorded
call, a call nothing pins, a response that does not match its digest, or a call
whose served revision the manifest does not name all exit non-zero. Nothing is
scored around a missing input.
