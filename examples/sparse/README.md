# Expand a query into the terms a sparse index can match

The runnable example behind [superlinked.com/sparse-embeddings](https://superlinked.com/sparse-embeddings).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/sparse/SOURCES.md).

## What this shows

Thirteen texts encoded to sparse vectors by two models on
`https://api.superlinked.com`, 26 calls in all:

- `prithivida/Splade_PP_en_v2`, which adds terms the text never used
- `BAAI/bge-m3:sparse`, which does not

The texts are BANKING77 customer messages, CPSC recall notices and Amazon ESCI
queries and product listings.

The page publishes no single headline number. It publishes concrete figures,
and `score.py` re-derives every one of them from the recorded responses,
offline.

Hero, the query "color switching led lights" against an LED bulb listing:

| displayed | value |
|---|---|
| Sparse match score | 13.998 |
| Words in both texts | 5.342 |
| Terms SPLADE added | 8.656 |
| shared terms | 7 listed, "11 more shared terms", so 18 |

Three proof cards, each as "N terms returned, A added by the model":

| text | SPLADE | bge-m3 sparse |
|---|---|---|
| `b77-card-arrival` | 46 of 56 | 0 of 15 |
| `esci-query-chrome-notebook` | 23 of 25 | 0 of 3 |
| `cpsc-25431-ladder` | 127 of 144 | 0 of 19 |

The line under that grid is a claim about the whole run rather than about the
three cards, so `score.py` checks it over every recorded text:

| displayed | value |
|---|---|
| recorded texts | 13, in 26 calls |
| SPLADE added | between 23 and 127 terms the text never used |
| bge-m3 sparse added | none, on all 13 |

The ten texts the page does not show are recorded all the same, and the
run-level figures above are derived from all thirteen. Which cards the grid
displays is a display choice; the run is not, and the two are checked
separately because they fail separately.

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces every published figure offline
python3 run.py --check   # rebuilds all 26 recorded requests from the inputs
```

These three need nothing installed: they are standard library only, and none
of them needs an API key, a Hugging Face token, any inference spend or any
model weights. `fetch.py` pins a commit SHA, not `main`, and checks every
downloaded file against a digest. It replaces `--dest` wholesale, so it
refuses to touch anything without the `.sie-evidence` marker it writes, and
it swaps the new directory in by rename rather than deleting the old one
first.

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

`score.py` prints the added and active term counts for all 26 calls, then the
run-level figures, then the three published cards, then:

```
Hero pair color-switching-to-color-changing-bulb, model prithivida/Splade_PP_en_v2
  sparse match score   13.998   (13.997640416026115)
  words in both texts  5.342
  terms SPLADE added   8.656
  shared terms         18   (7 listed on the page, 11 more)

Reproduced: the hero pair 13.998 = 5.342 + 8.656 over 18 shared terms;
the added-of-active counts on the 3 cards the proof grid displays;
and, over all 13 recorded texts, that SPLADE added between 23
and 127 terms while bge-m3 sparse added none.
```

It exits nonzero if any figure fails to reproduce, including the decomposition:
5.342 and 8.656 have to add up to 13.998, checked as integer thousandths so
float representation cannot paper over a gap.

`run.py --check` prints `26 of 26 recorded requests rebuilt from the inputs and
matched`, and fails if a call is missing, recorded twice or implied by no input.

## How each number is derived

| figure | derivation |
|---|---|
| active terms | the number of nonzero weights in the recorded response |
| added terms | terms whose token is absent from the model's own tokenization of the input |
| match score | dot product of the two recorded sparse vectors |
| words in both texts | the part of that dot product from terms appearing literally in both texts |
| terms the model added | the remainder of the score |

Mapping a token index back to a string needs the model's tokenizer, so that
mapping was recorded at run time into `derived/decoded/`. `score.py` does not
take it on trust: it re-counts active terms and added terms from the response
and fails if the decoded record disagrees.

A card lists a handful of rows beside a count like "144 terms", and both are
right. The count is every returned dimension. The list is a selection the page
makes out of them, and a dimension is not always a word: SPLADE scores entries
of the BERT uncased vocabulary, some of which are word pieces such as `##book`.
Which rows a card shows is the page's decision and can change without any count
moving, so this example reproduces the counts and not the selection.

## What is in the dataset

```
sparse/
  inputs/inputs.json                     13 texts with sources, plus the 2 scored pairs
  calls.json                             26 calls: request, response, status, timing, one file
  derived/decoded/<model>/<input>.json   token indices decoded to strings, recorded at run time
  derived/decoded/<model>/pairs.json     dot products of the recorded vectors for the 2 pairs
  manifest.json                          endpoint, models, Hugging Face revisions, digests
```

The 26 calls were 52 separate JSON files in sie-web. Merging them changed no
byte of any request or response.

## What this does NOT establish

- **Nothing about retrieval quality.** A higher dot product is not a better
  search result. Nothing here measures recall or precision on a ranked list.
- **Nothing about bge-m3 sparse being worse.** It adds no terms by design.
  "0 of 15" is what that model does, not a failure.
- **Nothing about the three cards being representative.** They were chosen to
  span the recorded range of added terms and to cover three kinds of text.
  `score.py` checks that their figures are the recorded ones and that the
  run-level range holds over all thirteen. It does not check that a reader
  seeing three cards would draw the same conclusion as one seeing thirteen.
- **Nothing about the tokenizer mapping if `derived/decoded/` is wrong.** The
  counts are re-derived from the responses and cross-checked, but the token
  strings themselves are taken from the recorded decode. Re-deriving them
  needs the tokenizer files, which this example deliberately does not download.
- **Nothing about a fresh run.** The responses were recorded on 2026-09-15
  against `prithivida/Splade_PP_en_v2` at Hugging Face revision
  `f0d4aa214dcb60c274052a52c0497535e3aec64c` and `BAAI/bge-m3` at
  `5617a9f61b028005a4858fdac845db406aefb181`.
- **A fresh `--record` run records less than the archive.** `client.encode`
  returns the per-item result with the vector as numpy arrays, and surfaces no
  response headers, so `--record` rebuilds the envelope, converts the arrays to
  lists and records no `dims` or `dtype`. Each entry carries a `shape` field
  saying so. `score.py` reads the published `calls.json`.
- **No tamper resistance.** The digests catch a truncated or corrupted
  download. They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/sparse/`, which is what its CI tests read.
The two copies hold the same recorded responses. Nothing binds them together,
so they can drift.
