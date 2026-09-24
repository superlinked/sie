# Put a box on the object an agent names

Nine photographs of warehouses, shop shelves and loading docks, sent to
`IDEA-Research/grounding-dino-base` and `google/owlv2-base-patch16-ensemble`
with nothing but a list of label strings. The detections behind
[superlinked.com/detect](https://superlinked.com/detect).

Grounding DINO returned 57 boxes across the nine photographs. 55 sit on the
object the request asked for and 2 do not: one box labelled `price tag` spans
the whole soft-drink shelf, and one labelled `pallet jack` sits on a different
machine. That precision figure is what the page publishes.

Recall is the other half, and the page states no figure for it. Those 55 boxes
cover 52 of 88 hand-counted objects, with 3 of them landing on an object another
box had already found. The gap is not evenly spread: `person` came back 8 of 8
and 3 of 3, while `price tag` came back 0 of 11 and `yellow price sign` 1 of 7.
`score.py` checks the 52 of 88 against the results table in the page's
[SOURCES.md](https://superlinked.com/reference/detect/SOURCES.md), which is
where a reader finds it.

## Where the evidence lives

The code is here. The photographs and the recorded calls are in the
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence)
dataset on Hugging Face, pinned to a revision in `fetch.py`. So you cannot
verify this by cloning alone: clone, fetch, then score. What you do not need is
an API key, or a cent of inference spend, to re-derive the number.

```
detect/
  inputs/inputs.json     the labels sent and a hand count of every visible
                         instance, registered before any model call
  inputs/images/         the nine photographs as sent, 2,862,730 bytes
  box-review.json        one verdict per returned box, assigned by hand
                         after the run
  calls.json             18 entries: request, response, status, timing
  manifest.json          endpoint, model ids, served revision, run date, sources
```

Each request record carries a descriptor naming the image file, its SHA-256 and
its byte length in place of the base64 it sent. `score.py` hashes the stored
file against that descriptor and against the digest `inputs.json` pinned before
the run, and fails if either disagrees. Every file it opens is checked for
existence first, so a partial fetch is a named failure saying what can no longer
be checked, never a traceback and never a comparison that quietly does not
happen.

## Run it

Download the recorded run, then score it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Expect a per-photo line, then:

```
Across all 9 recorded photos, 55 of 57 returned boxes sit on the object the request asked for and 2 do not
Those 55 boxes cover 52 of 88 hand-counted objects, with 3 more on objects already found
```

`score.py` also checks the four per-photo figures pinned in `PAGE_PER_PHOTO`,
and that every box on those four is a first box on the object its label names,
which is what makes a boxes count and a found count the same number there. It
exits non-zero if any of that, or the headline, is not what it computes.

Look at a request without sending it, and check that this runner is the one
that sent them:

```sh
python3 run.py --show javits-pallet-jacks
python3 run.py --check-requests    # 18 of 18 rebuilt identically
python3 run.py --verify-images     # 9 of 9 photographs verified
```

`--check-requests` rebuilds every recorded request with the `run.py` in this
directory and compares the canonical digests. It is the reason the runner is
worth shipping: it shows the file here sends what the recording says was sent,
rather than merely resembling it.

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

`run.py` writes into `run-output/`, never over the downloaded evidence. Your
boxes will differ from the recorded ones: the served model revision moves. It
writes no `box-review.json`, because a verdict per box is a hand judgement, not
something a rerun produces.

## How a box is scored

`score.py` reads two things that one edit cannot move together. The recorded
responses say what boxes came back. `box-review.json` gives each box one
verdict, assigned by eye after the run: `hit` is the first box on a counted
object, `duplicate` is a further box on an object already boxed, and `wrong` is
a box on something that is not its label. Before counting anything, `score.py`
matches every reviewed box against a box actually in the response, by label,
score and rectangle, and fails if a box is reviewed twice, left unreviewed, or
is not in the response at all. Identities are compared rather than counted, so
swapping one box for a copy of another does not pass.

The headline is then `hit + duplicate` over all boxes, coverage is `hit` over
the hand counts, and the per-label `found` figures stated in the review have to
equal the hits the verdicts produce.

The hand counts come from `inputs.json` and were registered before the first
model call, with three candidate photographs dropped beforehand because their
counts were ambiguous. `score.py` cannot edit them.

## The inputs are not ours

Every photograph is third-party. Five are public domain: four U.S. federal
works, from the Marine Corps, FEMA, the USDA and the Army, and one released by
its author. The other four are CC BY images from Wikimedia Commons and Flickr,
three under 2.0 and one under 4.0. None were made by Superlinked and none are
AI-generated. `manifest.json` records for each one its title, creator, licence,
the Commons or Flickr page, the URL of the original and the SHA-256 of both the
original and the JPEG actually sent, along with the resize rule between them.
Licences vary by source and have not been cleared for reuse beyond quotation
here; treat the provenance record as the starting point for that, not as a
clearance.

## What this does not establish

- Not a benchmark. Nine photographs chosen to span warehouses, retail shelves
  and loading docks is a demonstration, not a measurement.
- Not a claim about your photographs. 55 of 57 is the precision of the boxes
  that came back on these nine, and it says nothing about the 36 counted
  objects no box was drawn on.
- The hand counts and the per-box verdicts are human judgements, made by one
  person. They were registered before the run and reviewed after it
  respectively, and `score.py` binds the second to the recorded responses, but
  neither is an independent ground truth.
- The OWLv2 calls are recorded and verified here, and are in no published
  figure. Nothing on the page or in this scorer compares the two models.
- Not reproducible against the live API. These are recordings, and a rerun goes
  through a different served revision.
- All nine photographs are scored here. Which of them the page displays, and on
  which surface, is the page's decision and is recorded in its own `SOURCES.md`.

  This paragraph used to say which photographs the page draws. It was wrong for
  weeks, naming six proof cards after the page had gone to three, and every run
  stayed green throughout, because nothing in this example can reach the page.
  The claim is removed rather than restated a third time. The numbers this
  script and the page share are 55 of 57 and the four per-photo figures, and a
  tamper confirms what that leaves: swapping a pinned photograph for one the
  page does not display, with its correct figures, exits 0.
