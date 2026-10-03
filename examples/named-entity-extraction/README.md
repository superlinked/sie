# Extract custom entities from primary sources

The runnable example behind
[superlinked.com/named-entities](https://superlinked.com/named-entities). That
page's sources are in its [SOURCES.md](https://superlinked.com/reference/named-entities/SOURCES.md).

## What this shows

GLiNER accepts its label set at request time. This example sends one model,
`urchade/gliner_multi-v2.1`, four different label sets over financial,
healthcare, rail-safety and legal text. No training set, no model per type, six
labels per request.

The inputs are verbatim excerpts from the SEC, CMS, NTSB and the Supreme Court.
Each one records a source URL, a locator inside that document and the SHA-256 of
its text in `inputs/cases.json`, and nothing here is synthetic or paraphrased.

`score.py` re-derives what the page publishes from the recorded responses,
offline:

| label set | spans returned | required anchors | requested, never returned |
|---|---|---|---|
| financial, SEC Form 10-K/A | 24 | 8 of 8 | |
| healthcare, CMS L1851 claim | 9 | 5 of 5 | `payment action` |
| rail safety, NTSB alerts | 13 | 8 of 8 | `recipient` |
| legal, Coinbase v. Suski | 7 | 7 of 7 | |

53 spans across the four, and 28 of them were named before the run with their
exact offsets. The page shows 5, 6, 5 and 5 of those spans and renders the rest
as a "+19 more", "+3 more", "+8 more" and "+2 more" counter. That cap is a
display decision in sie-web, computed there from the same totals printed above;
the recordings hold all 53 either way.

The two empty labels stay visible. Asking for `payment action` on a claim denial
and `recipient` on a detector alert returned nothing, and the page says so on
those two cards. The CMS response also keeps the model's incorrect
`proof of delivery` span under `missing documentation`, which is not a required
anchor and is left in because it makes label and threshold tradeoffs easier to
inspect.

## Run it

The code is in this repository. The inputs and the recorded responses are in the
public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is no longer enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces every published figure offline
python3 run.py --check   # rebuilds all four recorded calls from the inputs
python3 -m unittest discover -s tests
```

These need nothing installed. They are standard library only, and none of them
needs an API key, a Hugging Face token or any inference spend. `fetch.py` pins a
commit SHA rather than `main`, and checks every downloaded file against a digest
before the scorer sees it. It replaces `--dest` wholesale, so it refuses to touch
anything without the `.sie-evidence` marker it writes, and it swaps the new
directory in by rename rather than deleting the old one first.

To call GLiNER yourself. This is the only part that needs the SDK, and the only
command here that spends anything:

```sh
uv sync
SIE_BASE_URL=https://your-cluster.example SIE_API_KEY=... \
  uv run python run.py --record --out run-output/calls.json
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into `main()`,
so `--check` and `--show` keep working on a bare `python3` with nothing
installed. `python3 run.py --show ntsb_detector_alert` prints a call without
sending it.

## What to expect

```
4 cases, model urchade/gliner_multi-v2.1

  sec_filing_amendment     6 labels   24 spans   8 of 8 required anchors matched
  cms_lower_limb_orthosis  6 labels    9 spans   5 of 5 required anchors matched   no span for: payment action
  ntsb_detector_alert      6 labels   13 spans   8 of 8 required anchors matched   no span for: recipient
  scotus_two_contracts     6 labels    7 spans   7 of 7 required anchors matched

53 spans returned across the 4 cases, every one inside a requested label
28 of 28 required anchors matched at the offsets registered before the run
every one reproduces its own text from input_text[start:end], or this script would have stopped above
the page displays 5, 6, 5 and 5 of them and counts the remaining 19, 3, 8 and 2. That cap is a display decision in sie-web, computed there from these same totals.

Reproduced: 24, 9, 13 and 7 spans, 53 in total, and 28 of 28 required anchors.
```

`score.py` exits nonzero if any figure fails to reproduce. `run.py --check`
prints `4 of 4 recorded calls rebuilt from the inputs and matched`, and fails if
a call is missing, recorded twice or implied by no case.

## How the figures are derived

The two sides of every comparison come from different places. The 28 required
anchors and every excerpt digest were registered in `inputs/cases.json` before
the run. The spans are read out of the recorded responses in `calls.json`.
Neither is derived from the other.

- **the offset contract**: every returned span must use a requested label, carry
  a finite score in [0, 1] and reproduce its own text from
  `input_text[start:end]`. This holds for all 53 spans, not only the 28 anchors,
  and a span that fails it stops the run.
- **exact excerpts**: each case's text is hashed and compared with the digest
  recorded beside it and with the canonical digest in `inputs/sources.json`.
  Rewriting a passage and its own declared digest together is not enough; the
  canonical entry has to agree as well.
- **an empty label is a finding, not an absence**: `score.py` subtracts the
  labels that came back from the labels that were asked for, so a label going
  quiet is printed rather than passed over.

## What is in the dataset

```
named-entity-extraction/
  inputs/cases.json    4 cases: the verbatim excerpt, its digest and locator,
                       the six requested labels, and the exact spans registered
                       before the run as required anchors
  inputs/sources.json  the four source documents, with a canonical digest and
                       locator for every excerpt drawn from them
  calls.json           4 calls: the SDK call envelope, the response, the status
                       and the latency, one file
  manifest.json        the server release and commit, the model revision, the
                       hardware, the run window and a digest for every file
```

The four calls were eight separate JSON files under `verified-run/` in this
repository. Merging them changed no byte of any envelope or response.

## What this does NOT establish

- **Four excerpts are a demonstration.** Nothing here measures recall on your
  documents, and no score generalises.
- **28 anchors is not 53 spans.** The anchors are what somebody registered as
  having to be found. The other 25 spans are unjudged, neither confirmed nor
  refuted by anything here.
- **Nothing about the two empty labels beyond these two documents.** Ask for
  `recipient` on other rail text and it may well come back.
- **The saved latencies are provenance, not timings.** The first call of the run
  includes model loading, at 21.1 seconds against 0.1 to 0.4 for the rest.
- **No endpoint is recorded.** The run predates this example recording one. The
  manifest names public SIE v0.6.23 at commit `9d6ca6b0`, running on an L4 in
  Modal on 2026-07-24, which is what was known.
- **No tamper resistance.** The digests catch a truncated or corrupted download.
  They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/named-entities/`, which is what its CI tests
read. The two copies hold the same recorded responses. Nothing binds them
together, so they can drift.
