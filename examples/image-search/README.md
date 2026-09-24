# Search a picture catalogue with words it was never tagged with

The runnable example behind [superlinked.com/image-search](https://superlinked.com/image-search).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/image-search/SOURCES.md).

## What this shows

50 public-domain photographs from the Metropolitan Museum of Art, encoded by
`google/siglip-so400m-patch14-384` and ranked against 24 text requests that
name a colour, a material and an object.

```
the object                       2 of 24   median rank 5.5
colour and object                7 of 24   median rank 3
material and object              6 of 24   median rank 2
colour, material and object     13 of 24   median rank 1
```

**The catalogue carries no colour.** Every photograph is a museum record with an
`objectName` and a `medium`, and nothing in either says blue or brown. The
colour written beside each photograph here is ground truth for scoring, assigned
by looking at the picture; it is never sent to the model and never indexed. The
model sees pixels and the request sees words, and the only thing joining them is
that SigLIP puts both in one 1152-dimensional space.

Each request is written four ways, from the bare object up to all three
attributes:

```
a vase
a blue vase
a porcelain vase
a blue porcelain vase
```

Scoring compares where the photograph matching all three lands under each form.
That makes "the whole request wins" a measurement rather than an assertion.

## Where the attributes come from, exactly

`material` is the first vocabulary word that appears LITERALLY in the Met's own
`medium`, with `stoneware` checked before `porcelain` so a "Porcelaneous
stoneware" reads as the stoneware the Met calls it. A photograph whose `medium`
names none of the vocabulary is dropped rather than inferred.

That rule is stricter than the one this example started with, and it cost the
headline figure. An earlier catalogue labelled six stoneware pieces
`earthenware` and two `Copper alloy` plates `brass`, and scored 15 of 24 on the
complete request. This one scores 13. The lower number is the published one,
because the higher one rested on eight labels the museum record does not
support.

`category` is the Met's `objectName` through a synonym table: its Dish is a
plate, its Ewer and Pitcher are jugs, its Beaker is a cup, its Tea caddy is a
box. Both values are in `inputs/catalogue.json`.

## Why the catalogue is built the way it is

A request only counts if first place has to be taken from a near miss. So a
(colour, material, object) triple becomes a request only when the catalogue also
holds a photograph matching each PAIR of the three:

- the same colour and material on another object
- the same colour and object in another material
- the same material and object in another colour

`score.py` re-checks that for every request before it derives a figure, and
refuses the whole run if any request has lost a near miss. It also refuses a
catalogue holding two photographs with the same triple, because then "the
photograph matching all three" is not one photograph and no figure over it means
anything.

## Run it

The code is in this repository. The photographs, the requests and the recorded
responses are in the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py   # downloads the pinned revision into evidence/
python3 score.py   # reproduces every published figure offline
```

Both are standard library only. Neither needs an API key, a Hugging Face token
or any inference spend. `fetch.py` pins a dataset commit rather than `main`, and
checks every file, each photograph included, against a digest from a manifest
its own source pins.

To record against SIE Cloud yourself. This is the only part that needs the SDK,
and the only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into `main()`,
so `--show` keeps working on a bare `python3` with nothing installed.

## How the figures are derived

Two sides, two sources. The photograph each request is looking for was fixed in
`inputs/queries.json` before any call, by a rule over the catalogue's attributes
that reads no score. The ranking comes out of the recorded vectors in
`calls.json`. The figures `score.py` asserts against come from a third place
again: the page, pinned in `score.py`'s own source, so editing the fetched
evidence alone will not satisfy it.

Everything fails rather than skips. A missing file, a photograph whose bytes no
longer match their digest, a response that does not match its
`response_sha256`, a request the pinned inputs do not rebuild, a photograph or a
request with no recorded vector, a vector returned twice, a declared width that
disagrees with its own values, or a non-finite value all exit non-zero.

## What is in the dataset

```
image-search/
  inputs/catalogue.json  every photograph: its file, digest, byte length,
                         dimensions, its colour, material and object, and the
                         Met record each attribute came from
  inputs/*.jpg           the photographs themselves, exactly the bytes encoded
  inputs/queries.json    the requests, each naming a colour, a material, an
                         object and the photograph matching all three
  calls.json             one entry per encode call: the request, the returned
                         vectors, the status, the served deployment revision
                         and the round-trip time
  manifest.json          endpoint, model and its revision, run date, and a
                         digest for every file above
```

## What this does NOT establish

- **This is a demonstration, not a benchmark.** No figure here generalises to
  your catalogue, and a set of this size is not a retrieval evaluation.
- **Colour is a judgement.** Object and material come from the Met's own
  cataloguing. Colour was assigned by looking at each photograph, and a reader
  who disagrees with one can see the picture and say so. It is recorded per
  photograph, with `colour_source` saying exactly that.
- **Museum objects are not product photography.** Lighting, background and
  framing are consistent in a way a real catalogue's are not.
- **The displayed pictures are not the encoded ones.** The website serves
  smaller renditions so the page stays light. The bytes SIE read are the files
  in the dataset, and their digests are what `score.py` checks.
- **No tamper resistance.** The digests catch a truncated or corrupted
  download. They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/image-search/`, which is what its CI tests
read. The two copies hold the same recorded responses. Nothing binds them
together, so they can drift.
