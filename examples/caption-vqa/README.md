# Ask a question about an image

Twelve photographs and engineering drawings, one specific question each, one
call per question to `Qwen/Qwen3.8-27B-FP8` on SIE Cloud. The answers behind
[superlinked.com/caption-vqa](https://superlinked.com/caption-vqa), whose sources
are in its [SOURCES.md](https://superlinked.com/reference/caption-vqa/SOURCES.md).

Ten of the twelve answers matched. The two that did not are both needle gauges,
where the model read the dial wrong: a fire hose test gauge at 400 psi came back
as 350, and a Mir station pressure gauge near 761 mm Hg came back as 130.
Printed labels, hazmat placards and wiring drawings answered exactly.

## Where the evidence lives

The code is here. The images and the recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you
cannot verify this by cloning alone: clone, fetch, then score. What you do not
need is an API key, or a cent of inference spend, to re-derive the number.

```
caption-vqa/
  inputs/inputs.json     questions, expected answers and scoring patterns, all
                         fixed before the first model call
  inputs/images/         the twelve images, 2,082,155 bytes
  calls.json             12 entries: request, response, status, timing
  manifest.json          endpoint, model id, served revision, run date, sources
  diagnostics/probe/     one exploratory call, scored by nothing (see below)
```

## Run it

Download the recorded run, then score it. Both steps are standard library
only, so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect `10 of 12 answers matched`, and a per-case line for each question
showing the answer the model gave. `score.py` exits non-zero if that is not what
it computes, rather than printing whatever it found.

Look at a request without sending it:

```sh
python3 run.py --show fire-hose-gauge
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence. Your
answers will differ from the recorded ones: nothing pins a seed, and the served
model revision moves.

## How a case is scored

Take the first non-empty line of the returned text, then apply the regular
expressions `inputs.json` registered before any model call. A case passes only
when every one of its checks passes. Scoring is deterministic string matching,
so no model judges another model.

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
curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/6d90772866ff2272526f6b386dc2db1bc6837e28/caption-vqa?recursive=true"
```

**The image files.** No image id is pinned here, so the object ids above would
not notice a modified image. Two digests cover that instead: each file is
hashed against the digest `calls.json` records for the call that sent it, and
that digest is held against the `image_sha256` the case registered in
`inputs.json`. The first catches an image edited while the metadata is left
untouched. The second catches an image that hashes correctly for the call
carrying it but is not the one the case pinned, so one case's photograph cannot
be scored against another's question. Editing an image *and* the digests that
describe it means editing the metadata, which is the first layer again.

The remaining checks are consistency within the evidence: `inputs.json` against
the digest the run recorded, every request and response record against its own,
and each entry's `slug` against its own `case`, so an edit to one field cannot
have the evidence validated as one case and the answer scored against another's
patterns. A case that registers no checks, and a recorded call that no case
reaches, are failures too rather than silent passes. Anything missing, or that
any check disagrees about, is a failure and nothing is scored; it is never
skipped past.

The prompt ends with "Start your reply with the answer in one sentence." That
sentence was added after a single exploratory call, kept in
`diagnostics/probe/`, in which the bare question produced visible step-by-step
reasoning and a pattern matched a scale label inside it. The questions, expected
answers and patterns are otherwise unchanged from the pre-registration. The
probe is not one of the twelve and is scored by nothing.

## The inputs are not ours

Every image is third-party. None were made by Superlinked, and none are
AI-generated. Seven are in the public domain: five U.S. government works, from
DVIDS, the Library of Congress, OSTI and a NASA project report, and two released
by their authors. Five are Creative Commons: three CC BY 2.0, one CC BY 4.0 and
one CC0 1.0, from Flickr and Wikimedia Commons, each attributed to a named
photographer. `manifest.json` records for each image its title, author, licence,
the page it came from and the original publisher URL. Licences vary by source
and have not been cleared for reuse beyond quotation here; treat the provenance
record as the starting point for that, not as a clearance.

## What this does not establish

- Not a benchmark. Twelve questions chosen to span printed text, placards,
  schematics and analogue dials is a demonstration, not a measurement, and 10 of
  12 carries no useful confidence interval.
- Not a claim about your images. Nothing here says the model reads your
  equipment photographs at this rate.
- Not reproducible against the live API. These are recordings. A rerun goes
  through a different served revision and unfixed sampling, so it will differ.
- Two of the twelve answers are wrong, both analogue needle gauges. Both are
  scored here and `score.py` prints each beside the value its question expected.
- Not a description of what the task page shows. All twelve cases are scored
  here and all twelve are inside the 10 of 12. Which of them the page features,
  and where it reports a total, are the page's decisions and its own
  `SOURCES.md` records them.

  This bullet used to name how many cases the proof grid draws and where the two
  wrong answers appear, and it was wrong twice. It first said five grid cards,
  which was true at the time. It was then rewritten to describe a grid of three
  with both wrong answers printed underneath, taken from a page change that had
  not shipped, while the page went on showing five. The page has since been
  reselected again and reports no total over the twelve at all, so the
  replacement was wrong in a second way.

  Every run stayed green through both, because nothing in this example reads the
  page and it never could. The claim is removed rather than corrected a third
  time. The figure this script and the page's evidence share is 10 of 12, and
  that has not moved.
