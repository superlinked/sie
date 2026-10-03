# Extract the entity types you name, with no model to train

## What this shows

Seven public NER test sets, each with its own entity types: five CrossNER
domains (AI, literature, music, politics, science) and the MIT Movie and MIT
Restaurant query sets. Between them they have 85 types, from "programming
language" and "astronomical object" to "average rating" and "restaurant name".
None of them is PERSON, ORG or LOC alone.

Every arm gets the same job: here is a sentence, here are this set's types, return
every entity of those types. SIE's GLiNER model takes the types as `labels` in
the request and returns character offsets. The LLMs get the types in a prompt and
a strict JSON schema, and return strings; a string is mapped back to every place
it occurs in the sentence. AWS Comprehend and spaCy cannot be told the types,
so their own fixed types were mapped onto each set's types by a map fitted on the
dev split to help them as much as possible, then frozen before they saw the test.

Scoring is exact: an entity counts when its start, its end and its type all
match the gold. A string an LLM returns that is not in the sentence at all counts
as a false positive.

The sample is 652 test sentences, 2,468 gold entities, about 350 per set, drawn
with a fixed seed. The protocol was registered before the first request.

| Arm | Exact-span F1 | Price per 1M characters, real-time |
|---|---|---|
| SIE GLiNER BioMed Large | 65.7% | $0.030 |
| SIE GLiNER Multi | 55.2% | $0.024 |
| GPT-6 Luna | 66.1% | $0.335 |
| GPT-6 Sol | 74.3% | $6.553 |
| Claude Haiku 4.5 | 63.3% | $5.485 |
| SIE Qwen3.8 27B | 54.3% | $1.342 |
| AWS Comprehend | 39.0% | $0.250 |
| spaCy en_core_web_trf | 35.0% | self-hosted |

Every price is one a synchronous request pays, on the tokens and characters
each arm used: the LLMs at their list price per token, AWS Comprehend at its
cheapest volume tier, SIE at its per-token rate. OpenAI and Anthropic also sell
a batch tier; it is not used here. GLiNER BioMed Large has no published SIE
Cloud rate yet, so it is priced at a proposed $0.15 per million input tokens,
which `score.py` reads from `manifest.json`.

GLiNER BioMed Large is a general GLiNER checkpoint despite its name: it won a
screen of 19 GLiNER-family models on the sets' dev splits, at a threshold of 0.8
chosen there too.

The run is already recorded. Every sentence, every arm's response to it, the two
frozen type maps and the study's own results file live in the public
HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive every
figure above with **no API key and no inference spend**. They are the figures on
[superlinked.com/named-entities](https://superlinked.com/named-entities), whose
sources are in its
[SOURCES.md](https://superlinked.com/reference/named-entities/SOURCES.md).

- SIE model: `Ihor/gliner-biomed-large-v1.0`, request threshold 0.8, served by
  `sie-server` 0.9.0 on one NVIDIA L4
- LLMs: GPT-6 Luna and GPT-6 Sol with reasoning off, Claude Haiku 4.5 with no
  extended thinking, and SIE's Qwen3.8 27B, all at temperature 0 with a strict
  JSON schema whose `type` field is an enum of the set's types
- AWS Comprehend `DetectEntities` in us-east-1; spaCy `en_core_web_trf` 3.8.0
- Recorded 2026-09-30. Two of Qwen3.8 27B's sets were sent twice. The first
  pass of politics lost 4 of its 60 requests to a concurrency limit on the API
  key, and the first pass of MIT Movie lost 2 of 158 to a transient 503. Under
  the protocol a set with any failed request is void, so both were rerun in
  full. Only the reruns are recorded and scored.

## Run it

Download the recorded run, then score it. Both steps are standard library only:

```sh
python3 fetch.py
python3 score.py
```

Look at a request without sending it:

```sh
python3 run.py --show music-main-0
```

Run the SIE arm yourself against any SIE server that serves the model, for
example one you start with `sie-server serve --models Ihor/gliner-biomed-large-v1.0`:

```sh
uv sync
SIE_URL=http://localhost:8080 uv run python run.py
python3 score.py --rows run-output
```

## What result to expect

`score.py` prints every arm's F1, precision, recall, macro average over the seven
sets, the strings not found in the text and the real-time price, then each
arm's price as a multiple of SIE's (and, for the LLMs, what their output tokens
alone cost), then the paired
differences against SIE's GLiNER arm, and ends with:

```
Real-time price as a multiple of SIE GLiNER BioMed Large's:
  SIE GLiNER Multi                     0.8x
  GPT-6 Luna                          11.2x   output tokens alone 6.5x
  GPT-6 Sol                          218.8x   output tokens alone 125.6x
  Claude Haiku 4.5                   183.1x   output tokens alone 78.6x
  SIE Qwen3.8 27B                     44.8x   output tokens alone 37.6x
  AWS Comprehend                       8.3x

SIE GLiNER BioMed Large minus each arm, pooled F1 points, 95% interval (2000 paired resamples):
  SIE GLiNER Multi                  +10.5  [+8.0, +13.2]
  GPT-6 Luna                         -0.4  [-2.7, +2.0]
  GPT-6 Sol                          -8.5  [-10.7, -6.3]
  Claude Haiku 4.5                   +2.4  [-0.0, +4.9]
  SIE Qwen3.8 27B                   +11.4  [+8.3, +14.7]
  AWS Comprehend                    +26.7  [+23.4, +30.0]
  spaCy en_core_web_trf             +30.7  [+27.1, +34.3]

Every figure matches the study's results.json.
```

It fails rather than skipping: a file whose SHA-256 differs from the manifest, a
type map that changed after it was frozen, a sentence with no recorded row, or
any figure that differs from the study's `results.json` exits non-zero.

## What this does NOT establish

- **Not your documents.** These are single sentences from Wikipedia and
  voice-search queries. Long documents, tables and your own type names can move
  every arm.
- **Not a verdict on Comprehend or spaCy at their own job.** They were asked for
  types they do not have. On PERSON, ORGANIZATION and LOCATION alone they are
  strong; this measures what happens when the types are yours.
- **Not Comprehend Custom.** A Comprehend recognizer trained on your own
  annotations was not measured.
- **Not the LLMs at their best.** No few-shot examples, no retries on a bad
  answer, reasoning off. A longer prompt or a reasoning setting may score higher,
  at a higher price.
- **Not latency.** Nothing here measures response time.
- **Not a guarantee the dataset is unchanged.** `fetch.py` pins a dataset
  revision rather than `main`; it does not prove the revision holds what it held
  when this was written.

## Inputs

`evidence/data/main/` holds the 652 test sentences, one file per set, each with
its gold spans as character offsets into the text (tokens joined by single
spaces). `evidence/data/dev/` and `pilot/` hold the dev sentences the type maps
and the threshold were chosen on and the pilot the harness was checked on.
`evidence/data/SOURCES.json` pins the SHA-256 of every source file.

- CrossNER, github.com/zliucr/CrossNER at commit
  `2e7ba2a7798c961e3f29fbc51252c5a8d40224bf`, MIT licence.
- MIT Movie (`engtrain.bio`, `engtest.bio`) and MIT Restaurant, published by the
  MIT Spoken Language Systems group for research use.

`evidence/rows/<arm>/` holds each arm's recorded response to each sentence;
`evidence/maps/` the frozen type maps for Comprehend and spaCy with their
digests; `evidence/results.json` the study's own figures; and
`evidence/page-documents/` SIE's and Comprehend's responses on the four documents
the web page shows.
