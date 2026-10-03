# Read a form into typed fields, then send the answer back for a check

Eight scanned documents, one call each to `Qwen/Qwen3.8-27B-FP8` on SIE Cloud
with a strict JSON-schema grammar, plus a ninth playground call on one of them.
Then a second call over each of those eight answers. The extractions behind
[superlinked.com/doc-field-extraction](https://superlinked.com/doc-field-extraction),
whose sources are in its
[SOURCES.md](https://superlinked.com/reference/doc-field-extraction/SOURCES.md).

180 of 223 expected field values came back exactly on the first call. A second
call that sees the schema and the first answer, and never sees the page, took
that to 204: it fixed 24 and broke none of the 180. Every one of the eight
ticked and empty checkboxes on the two FAA release certificates read correctly,
and all 17 replies across both passes were valid against their schema on the
first try. A receipt photographed sideways produced schema-valid JSON with
invented numbers, and the second call could not touch it, which is the most
useful pair of results in the set.

The interesting part is which arm won. Three were fixed in writing before any of
those 24 calls, and the two that lost are why this is worth reading:

| arm | what it sends | fields exact | fixed | broke |
|---|---|---|---|---|
| first call alone | image + schema | 180 of 223 | | |
| A1, one call | image + a schema with a description on every field | 174 of 223 | 10 | 16 |
| **A2, second call** | **schema + the first answer, no image** | **204 of 223** | **24** | **0** |
| A3, second call | image + schema + the first answer | 185 of 223 | 5 | 0 |

Writing better field descriptions made it worse. Handing the second call the
page image dropped it from 24 fixes to 5. The second call helps because it
answers a different question from the first, not a harder version of the same
one: the first reads the page, the second checks each value against the field it
was written into.

## Where the evidence lives

The code is here. The document images and the recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you
cannot verify this by cloning alone: clone, fetch, then score. What you do not
need is an API key, or a cent of inference spend, to re-derive the numbers.

```
doc-field-extraction/
  inputs/inputs.json     schemas and expected values, fixed before the first call
  inputs/images/         the eight page images, 2,936,346 bytes
  calls.json             9 entries: request, response, status, timing
  manifest.json          endpoint, model id, served revision, run date, sources
  diagnostics/           two calls kept out of every total (see below)
```

The second pass is the exception and lives here, in the repository:

```
second-pass/
  PRE-REGISTRATION.md    the three arms and the decision rule, before any call
  experiment.json        each arm's prompt and schemas
  calls.json             24 entries: three arms over the eight documents
```

It is in git rather than in the dataset because none of it is large. A1 and A3
do send the page image; what the recorded requests store in its place is a
placeholder naming the path and SHA-256 the first pass pinned, so these files
carry no image bytes and `python3 score.py` scores both passes after a single
`fetch.py`. A2 is the arm that sends no image at all, and that turns out to be
why it wins.

## Run it

Download the recorded run, then score it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect a per-document line, then the figures the page publishes:

```
180 of 223 fields exact across all 8 recorded documents, in 9 calls
8 of 8 ticked and empty boxes read correctly
9 of 9 replies valid against the schema, first call

Second call, 2026-09-22. Three arms, one pre-registered rule:
  bar: 195 of 223                                and at most 5 broken
  a1  One call, described schema       174 of 223   fixed  10   broke  16   below the bar
  a2  Second call, no image            204 of 223   fixed  24   broke   0   CLEARS
  a3  Second call, with the image      185 of 223   fixed   5   broke   0   below the bar

204 of 223 fields exact after the second call, up from 180: 24 fixed, 0 broken
17 of 17 replies valid against the schema, first try, across both passes
```

`score.py` exits non-zero if that is not what it computes, rather than printing
whatever it found.

Look at a request without sending it:

```sh
python3 run.py --show faa-8130-3-export-flap
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence. Your
extractions will differ from the recorded ones: nothing pins a seed, and the
served model revision moves.

## How a field is scored

Every leaf value in a document's `expected` object is one field. Array rows are
compared by index; a missing row scores each of its fields as a miss, and extra
rows are reported and not scored. Strings compare after NFKC normalization, case
folding, straightened quotes, hyphenated dashes, and removal of whitespace and
commas, with one trailing period ignored. Numbers compare within 1e-9; integers,
booleans and enums compare exactly. `inputs.json` records those rules and they
were fixed before the run.

`score.py` checks the bytes before it scores them. The checks fall into two
layers, and they catch different attacks.

**The metadata files.** `calls.json`, `manifest.json` and `inputs/inputs.json`
are hashed whole, before anything is parsed, against object ids pinned in this
repository beside the dataset revision. That is the layer that defeats a
coordinated edit: every other digest here travels inside those three files, so
someone who changes a recorded response and recomputes the `response_sha256`
sitting beside it satisfies all of them, and the run still prints the published
figure. Only a value pinned outside the evidence sees that. These are the ids
HuggingFace publishes for the revision, so you can check them by hand:

```sh
curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/2d733ecb8b270fbb154975776e7f5a4315330f36/doc-field-extraction?recursive=true"
```

**The image files.** No image id is pinned here, so the object ids above would
not notice a modified image. Two digests cover that instead: each file is
hashed against the digest `calls.json` records for the call that sent it, and
that digest is held against the `image_sha256` the case registered in
`inputs.json`. The first catches an image edited while the metadata is left
untouched. The second catches an image that hashes correctly for the call
carrying it but is not the one the case pinned, so one document cannot be
scored against another's schema and expected values. Editing an image *and* the
digests that describe it means editing the metadata, which is the first layer
again.

The remaining checks are consistency within the evidence: the scored digest of
`inputs.json` against the one the run recorded, every request and response
record against its own, and each entry's `slug` against `<case>__<call>`, so an
edit to one field cannot have the evidence validated as one document and the
reply scored against another's schema. A recorded call that no case reaches is
a failure too rather than a silent pass. Anything missing, or that any check
disagrees about, is a failure and nothing is scored; it is never skipped past.

**The second pass.** Its file is pinned by living in this repository, so any
change to it shows in the diff. On top of that, `score.py` rebuilds all 24
request bodies from `second-pass/experiment.json`, `inputs.json` and the first
pass's own replies, and compares each with the request recorded on disk. That is
the check with teeth, and the one the digests in `second-pass/calls.json` cannot
do: those travel with the records they describe, so they catch a corrupted file
and nothing else. The expected 24 slugs come from the arms and the registered
cases, so a call that is missing, duplicated or implied by no arm fails there
too. The arms are also scored by the same `compare` and `schema_errors` this
file already uses for the first pass, against the same unedited `expected`
values, and every fixed and broken count is paired field by field against the
first pass this script has already verified.

Two calls sit in `diagnostics/` and are counted in nothing: one exploratory call
on the FAA rebuilt fuel control certificate, and a superseded playground call
whose schema asked for `payment_terms` and got the printed "NET 30" followed by
boilerplate that is not on the invoice. The field was replaced with `currency`
and the call re-run; both calls are kept.

## The inputs are not ours

Every document is third-party. None were made by Superlinked, and none are
AI-generated. Four are U.S. federal publications in the public domain: two FAA
Form 8130-3 figures from Order 8130.21H, an OSHA Form 300 example log whose
names and cases are OSHA's own illustrative entries, and page 3 of a NIST
certificate of analysis for SRM 1155a. Two are Wikimedia Commons scans held to
be ineligible for copyright: a Wolters Kluwer Health invoice and a USPS CN22
customs label. One is a CC0 1.0 Greek electricity bill and one a CC BY-SA 4.0
receipt photograph, both from Wikimedia Commons and attributed to the uploader.
`manifest.json` records for each document its title, publisher, licence, source
URL and how the page image was derived from it. Licences vary by source and have
not been cleared for reuse beyond quotation here; treat the provenance record as
the starting point for that, not as a clearance.

## What this does not establish

- Not a benchmark. Eight documents chosen to span aviation certificates,
  regulatory logs, metrology tables, invoices, customs labels, utility bills and
  a receipt photograph is a demonstration, not a measurement.
- Not a claim about your documents. 180 of 223, and 204 after the second call,
  is the score on these eight. It ranges from 16 of 16 on both FAA certificates
  to 4 of 11 on the sideways receipt, which the second call does not move.
- The zero is not a guarantee. "Broke none of the 180" is what happened in this
  recorded run. Nothing pins a seed, so a rerun can break one.
- Three arms is not an ablation of the design space. It is the smallest set that
  answers the two questions worth asking before adding a call: would a better
  schema do it instead, and does the second call need the page.
- Schema-valid does not mean correct. All 17 replies across both passes
  validated; the receipt's did so while inventing a store address, a total and
  an item count, and the second call passed that reply through untouched. That
  is the gap this example is most useful for showing.
- Not reproducible against the live API. These are recordings. A rerun goes
  through a different served revision and unfixed sampling, so it will differ.
- The page's proof grid shows three of the eight documents, the hero shows a
  fourth, the FAA rebuilt fuel control certificate, and the playground a fifth,
  the Wolters Kluwer invoice. Its line under the grid counts all three surfaces
  and says so. All eight are here, and so are the two diagnostics calls.
- Each proof card prints six rows of a longer answer. The card's own footer
  gives the score over every field of that document, and this example holds all
  of them.
