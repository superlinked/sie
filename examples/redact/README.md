# Get the exact character offsets to mask

The runnable example behind [superlinked.com/redact](https://superlinked.com/redact).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/redact/SOURCES.md).

## What this shows

Twelve documents sent to `urchade/gliner_multi_pii-v1` and to
`numind/NuNER_Zero` on `https://api.superlinked.com`, each request naming the
ID types that document uses as plain strings. No training set and no model per
type. The response gives back exact character offsets, so the surrounding text
stays readable.

The documents are CFPB and CMS published sample forms and synthetic records
from NVIDIA's Nemotron-PII and Gretel's synthetic PII finance dataset, the
latter two carrying their publishers' own gold PII annotations.

The page masks personal data in four steps, and `score.py` re-derives each one
from the recorded responses, offline:

| step | published PII spans masked |
|---|---|
| One call to `gliner_multi_pii-v1`, whole document | 26 of 45 |
| Split what runs past the model's input window | 33 of 45 |
| Union a second call to `NuNER_Zero` | 41 of 45 |
| Mask every later mention of a name already found | 45 of 45 |

Both models are in the same task's catalog, so the second call is one more
model id on the same endpoint. Both were recorded on all twelve documents in
the same run, and the dataset has held all 24 calls since the page was first
published.

Two more figures it reproduces: 42 of the 49 masks land on a published gold
span, and 0 of the 29 currency amounts across the ten recorded documents are
masked.

The second step is why the window matters. The model reads the first 384 words
and punctuation marks and the reply gives no truncation signal, so a document
longer than that has to be split by the caller, with each part's starting
offset added back to the spans it returns.

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces every published figure offline
python3 run.py --check   # rebuilds all 24 recorded requests from the inputs
python3 score.py --floor-0   # the same figures with the confidence floor removed
```

The last one is why the 0.6 floor is not doing the work. Five of the run's 168
spans fall below it: `ID #`, `File #` and `MIC #`, each covering a field's
printed label and no value, and the given name `Annibale` twice, which the
first model also returned at the same offsets inside `Annibale Caboto`. Every
figure above is unchanged with the floor removed, the 42 of 49 included: the
three field-label spans are on the Closing Disclosure, which carries no gold
spans and so is not one of the six documents the precision figure is measured
over, and the two `Annibale` spans merge into masks the first model already
produced.

These three need nothing installed: they are standard library only, and none
of them needs an API key, a Hugging Face token or any inference spend.
`fetch.py` pins a commit SHA, not `main`, and checks every downloaded file
against a digest. It replaces `--dest` wholesale, so it refuses to touch
anything without the `.sie-evidence` marker it writes, and it swaps the new
directory in by rename rather than deleting the old one first.

To call the API yourself. This is the only part that needs the SDK, and the
only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py --record --out run-output/calls.json
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into
`main()`, so `--check` and `--show` keep working on a bare `python3` with
nothing installed. `python3 run.py --show <id>` prints a request without
sending it.

## What to expect

```
benchmark documents scored: 6
  nemotron_insurance_claims_log        6 of 6 in one call, 6 composed
  nemotron_insurance_application       7 of 7 in one call, 7 composed
  gretel_customer_support_log          2 of 11 in one call, 11 composed
  gretel_it_support_ticket             3 of 11 in one call, 11 composed
  gretel_policyholder_report           4 of 5 in one call, 5 composed
  gretel_german_health_claim           4 of 5 in one call, 5 composed

published PII spans masked, by step:
  26 of 45  one call to urchade/gliner_multi_pii-v1, whole document
  33 of 45  splitting what runs past the model's input window
  41 of 45  unioning a second call to numind/NuNER_Zero
  45 of 45  masking every later mention of a name already found

42 of 49 masks land on a published gold span
7 fall outside them, so neither benchmark can confirm or refute those

the propagation step added 6 masks: Annibale, Paul

0 of 29 currency amounts masked across the 10 recorded documents

564-word chat, 11 published spans: 2 masked by one call, 11 by the composition
  the model reads the first 384 words, up to character 813
  7 of those spans sit past that point, and the second request starts at character 812

Reproduced: 26, 33, 41 and 45 of 45, 42 of 49 on gold, and 0 of 29 amounts masked.
```

`score.py` exits nonzero if any figure fails to reproduce. `run.py --check`
prints `24 of 24 recorded requests rebuilt from the inputs and matched`, and
fails if a call is missing, recorded twice or implied by no case.

## How the figures are derived

The two sides of every comparison come from different places. The gold spans
are the benchmark publishers' own annotations, carried in `inputs/cases.json`.
The masks are read out of the recorded API responses in `calls.json`. Neither
is derived from the other.

- **covered**: a gold span counts as masked only when the union of returned
  spans covers every one of its characters, so two overlapping returned spans
  are never counted twice.
- **in the gold schema**: a returned span counts toward the 25 only when its
  label is one the gold annotation can express.
- **on gold**: such a span counts toward the 23 when it overlaps any gold span.
- **564 and 384**: counted with GLiNER's own word rule, against the recorded
  window boundary character offset, rather than taken from the page.

## What is in the dataset

```
redact/
  inputs/cases.json   12 documents: text, requested labels, word counts, sources,
                      and for the 6 benchmark documents the published gold PII spans
  calls.json          24 calls: request, response, status, timing, one file
  manifest.json       endpoint, models, served revision, run window, digests
```

The 24 calls were 48 separate JSON files in sie-web. Merging them changed no
byte of any request or response.

## What this does NOT establish

- **The 42 of 49 is agreement, not correctness.** Both benchmarks annotate
  only a subset of each document, so a mask outside their gold spans is
  unjudged rather than wrong. The seven unjudged masks are two claim numbers, a
  payment card's last four digits, a doctor's practice address, a medical
  condition, and two further mentions of a name a model had already returned.
- **Nothing about recall on your documents.** Twelve documents from four
  publishers is a demonstration.
- **Nothing about the 384-word window being a fixed property.** It is what
  these recordings show for these models. Measure it for the model you deploy.
- **Nothing about either model alone being enough.** Chunked, on its own,
  `gliner_multi_pii-v1` reaches 33 of 45 and `NuNER_Zero` 37 of 45. The 41 is
  the union of the two, and the 45 needs the propagation step as well.
- **Nothing about which documents the page shows.** Every figure here is over
  the recorded set: the four steps and the 42 of 49 over the six documents
  carrying gold spans, and the 0 of 29 amounts over all ten recorded documents
  that are not a second chunk. Which of them a page renders is the page's
  decision and nothing here can read it.
- **A fresh `--record` run records less than the archive.** `client.extract`
  returns the per-item result rather than the server's envelope, and surfaces
  no response headers, so `--record` rebuilds the envelope around that item and
  each entry carries a `shape` field saying so. `score.py` reads the published
  `calls.json`.
- **No tamper resistance.** The digests catch a truncated or corrupted
  download. They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/redact/`, which is what its CI tests read.
The two copies hold the same recorded responses. Nothing binds them together,
so they can drift.
