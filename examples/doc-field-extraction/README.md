# Read a form into typed fields

Eight scanned documents, one call each to `Qwen/Qwen3.8-27B-FP8` on SIE Cloud
with a strict JSON-schema grammar, plus a ninth playground call on one of them.
The extractions behind
[superlinked.com/doc-field-extraction](https://superlinked.com/doc-field-extraction).

180 of 223 expected field values came back exactly. Every one of the eight
ticked and empty checkboxes on the two FAA release certificates read correctly,
and all nine replies were valid against their schema on the first call. A
receipt photographed sideways produced schema-valid JSON with invented numbers,
which is the most useful failure in the set.

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

## Run it

Download the recorded run, then score it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect a per-document line, then the three figures the page publishes:

```
180 of 223 fields exact across all 8 recorded documents, in 9 calls
8 of 8 ticked and empty boxes read correctly
9 of 9 replies valid against the schema, first call
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

`score.py` checks the bytes before it scores them. It recomputes the scored
digest of `inputs.json`, the digest of every request and response record, and
the SHA-256 of every image against the digest recorded for the call that sent
it. A file that is missing or does not match is a failure and nothing is scored;
it is never skipped past.

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
- Not a claim about your documents. 180 of 223 is the score on these eight, and
  it ranges from 16 of 16 on one FAA certificate to 4 of 11 on the sideways
  receipt.
- Schema-valid does not mean correct. All nine replies validated; the receipt's
  did so while inventing a store address, a total and an item count. That is the
  gap this example is most useful for showing.
- Not reproducible against the live API. These are recordings. A rerun goes
  through a different served revision and unfixed sampling, so it will differ.
- The page's proof grid shows three of the eight documents, the hero repeats
  fields from one of those three, and the playground shows a fourth, the Wolters
  Kluwer invoice. The page's own line, "the 5 documents not shown", counts the
  grid. All eight are here, and so are the two diagnostics calls.
