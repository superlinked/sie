# Grade a product photo with labels you write, and no training set

Sixteen VisA photographs of cashews, fryum wheels, snack tubes and gum pellets,
scored against two plain-English labels each by
`Qwen/Qwen3-VL-Reranker-2B` and `google/siglip2-base-patch16-224`. The scores
behind [superlinked.com/image-classify](https://superlinked.com/image-classify),
whose sources are in its
[SOURCES.md](https://superlinked.com/reference/image-classify/SOURCES.md).

Over all sixteen, the reranker flagged 8 of the 8 damaged pieces and passed 7 of
the 8 whole ones. The sixteenth, a fryum wheel, scored 0.562 on both labels, so
it was neither flagged nor passed. SigLIP 2 base, the cheaper model the page
puts beside it, sorted 12 of 16: it passed 2 damaged pieces and flagged 2 whole
ones.

The page publishes twelve of those sixteen. The tie says this model and this
pair did not separate one photograph; it does not say whether the labels, the
model or the two together are responsible, and one tied photo cannot settle
that. The page does not need the answer, because it is read by someone about to
write labels of their own and a pair that comes back undecided on one of its own
photographs is not the pair to show them. So the whole product comes off the
page and off every figure on it, all four of its photographs rather than only
the one that tied. Over the remaining twelve the reranker flagged 6 of 6 and
passed 6 of 6, and SigLIP 2 base sorted 9 of 12. `score.py` prints and checks
both sets.

A first run put four fine-grained grade labels on sixteen other VisA
photographs, of cashews and fryum wheels only, asking each model to name the
defect rather than spot one. Both got 4 of 16. That run is here too, on photos
the pass-or-reject run deliberately does not reuse, and it is why the page
frames the task as pass or reject.

## Where the evidence lives

The code is here. The recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you cannot
verify this by cloning alone: clone, fetch, then score. What you do not need is
an API key, or a cent of inference spend, to re-derive the numbers.

```
image-classify/
  inputs/01-grades.json       the four-label run: photos, labels, expected class
  inputs/02-pass-reject.json  the two-label run the page publishes
  inputs/display/             the page's WebP renditions of the sixteen
                              pass-or-reject photos, 375,166 bytes
  calls.json                  64 entries: request, response, status, timing
  manifest.json               endpoint, model ids, served revision, run date,
                              sources
```

The photographs themselves are not here. They are VisA originals inside a 20 GB
Amazon Science tar, pinned by SHA-256 and by their path inside it. The WebP
files under `inputs/display/` are the page's renditions, different bytes, and
the manifest says so rather than letting them read as the input. `score.py`
hashes all sixteen against the digests the manifest pins anyway: they change no
figure, but a missing or altered one means the bundle is incomplete, and it is
reported as a failure that says so. Every file the scorer opens is checked for
existence first, so a partial fetch is a named failure rather than a traceback.

## Run it

Download the recorded run, then score it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect the sixteen score pairs, then:

```
Across all 16 recorded photos the reranker flagged 8 of 8 damaged pieces and passed 7 of 8 whole ones
  it scored 0.562 on both labels for fryum-intact-003, so that photo is neither flagged nor passed
SigLIP 2 base sorted 12 of 16: it passed 2 damaged pieces and flagged 2 whole ones

The page leaves the fryum off, all 4 of its photos, and publishes the other 12: the reranker flagged 6 of 6 damaged pieces and passed 6 of 6 whole ones, with 0 ties
SigLIP 2 base sorted 9 of 12 of those: it passed 2 damaged pieces and flagged 1 whole one
```

`score.py` also checks all sixteen recorded score pairs, in both directions so
that a recorded pair cannot go unpinned, that the run's one tie is on the
product `TIED_PRODUCT` names, and the first run's 4 of 16. It exits non-zero if
any of them is not what it computes.

It used to check the order the playground's output panel lists two labels in.
That is a claim about which case a page surface displays, which is the same
thing `PAGE_PAIRS` was and which nothing here can verify, so it is gone with
the rest. The scores of that photograph are still checked, in `RECORDED_PAIRS`
along with the other fifteen.

Look at a request without sending it, and check that this runner is the one
that sent them:

```sh
python3 run.py --show cashew-damaged-014
python3 run.py --check-requests    # 64 of 64 rebuilt identically
```

Both work without a single photograph, because the stored request carries the
digest rather than the bytes.

Send the calls yourself, which needs a key, an extracted VisA copy and credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --images /path/to/VisA --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence, and
refuses to send any photograph whose bytes do not hash to the digest the
pre-registered inputs file pins.

## How a photo is scored

The reranker returns one score per label for the image, so a photo is flagged
when the damaged label scores strictly above the whole label, passed when the
whole label scores strictly above, and a tie when they are equal. A tie counts
as neither.

SigLIP is not a reranker and returns no scores. Its recorded response holds the
raw image vector and one vector per label, and `score.py` L2-normalizes them
and takes the cosine itself. That matters: the 12 of 16 is re-derived from the
embeddings rather than read back from the verdict the original run wrote down
beside them, so the recording and the number are two things rather than one.

The labels and the expected class for every photo come from the pre-registered
inputs files, which name the VisA image-level annotation each one follows and
the rule by which the photos were selected.

## One digest that no longer matches, and why

Both inputs files were edited after their runs: a hand-entered
`pre_registered_utc` was wrong, because it postdated the run it claimed to
precede, and a note explaining the correction was added. So neither file hashes
to the `inputs_sha256` its run recorded, and no amount of reverting the
timestamp brings it back, since the note did not exist at run time either.

That is recorded rather than smoothed over. `manifest.json` carries both
digests, `score.py` prints the mismatch every time it runs, and it checks the
substance instead of trusting it: every case still registered must have been
run, against the photograph it names, and every recorded call must still be
registered. An inputs file edited after packaging fails. All 64 request and
response digests match exactly.

## The photographs are not ours

Every photograph comes from VisA (Visual Anomaly), published by Amazon Science
under CC BY 4.0 with the ECCV 2022 paper by Zou, Jeong, Pemula, Zhang and
Dabeer. Superlinked made none of them and none are AI-generated.
`manifest.json` records the dataset citation, licence, source URL and tar
digest, and for each photo its path inside the tar and the SHA-256 of the exact
bytes sent. Licences vary by source and have not been cleared for reuse beyond
quotation here; treat the provenance record as the starting point for that, not
as a clearance.

## What this does not establish

- Not a benchmark. Sixteen photographs across four products is a
  demonstration, not a measurement, and VisA has thousands.
- Not a claim about your photographs. These are studio shots of small objects
  on a plain background, which is the easy end of visual inspection.
- The two-label result is the one that works. The same models got 4 of 16 on a
  separate set of sixteen photographs when asked which of four defects they
  were looking at, and that run is scored here beside the published one.
- One photo of the sixteen tied exactly. A tie is not a pass, and a production
  threshold would have to decide what happens to it. The page's answer was to
  stop recommending that label pair, which is a choice about the demonstration
  rather than a result: this scorer keeps the tie and all sixteen photographs,
  and the page's own SOURCES.md prints both tables.
- SigLIP's 12 of 16 is a cosine over pooled embeddings, which is a different
  operation from the reranker's, not a weaker version of it. The page compares
  price and latency alongside.
- Not reproducible against the live API. These are recordings, and a rerun goes
  through a different served revision.
- All sixteen photographs are scored here, and both sets of figures, the whole
  run and the run without the fryum, are printed and checked. Which photographs
  the page displays is the page's decision and is recorded in its own
  `SOURCES.md`.

  This paragraph used to name them, and the block it described held eight score
  pairs "the page prints". Both went stale the moment the board was reselected,
  and nothing here can check either, so the pairs block now holds all sixteen
  and the claim is gone rather than corrected.
