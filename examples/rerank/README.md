# Rank exact primary-source passages

The runnable example behind [superlinked.com/rerank](https://superlinked.com/rerank).

## What this shows

Four questions, each sent to `Qwen/Qwen3-Reranker-4B` with a handful of
candidate passages, where every candidate is a verbatim excerpt from an official
document:

- Pathward Financial's Form 10-K/A filed with the SEC
- CMS's published Lower Limb Orthoses claim example
- the NTSB's East Palestine illustrated digest
- the Supreme Court's Coinbase v. Suski opinion

The questions are authored evaluation prompts. The passages are not. Every
candidate carries its source URL, its locator inside that document and the
SHA-256 of its text, in `inputs/cases.json` and `inputs/sources.json`, and
nothing here is synthetic or paraphrased.

The page publishes a score pair per case, at three decimals, and `score.py`
re-derives all four from the recorded responses offline:

| case | first | closest other candidate |
|---|---|---|
| NTSB detector alerts | `ntsb-salem-noncritical` 1.000 | `ntsb-help-desk-process` 0.992 |
| Coinbase v. Suski | `scotus-two-contract-holding` 1.000 | `scotus-one-contract-rule` 0.439 |
| CMS L1851 claim | `cms-missing-documentation` 1.000 | `cms-seven-month-evidence` 0.816 |
| SEC filing amendment | `sec-non-reliance` 1.000 | `sec-amendment-scope` 0.107 |

The 0.992 is the interesting one. That near-match names the Wayside Help Desk
and the alert process, so it answers half the question, and the passage that
answers both halves still comes first.

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

To call a reranker yourself. This is the only part that needs the SDK, and the
only command here that spends anything:

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
4 cases, model Qwen/Qwen3-Reranker-4B

  sec_filing_amendment     4 candidates   sec-non-reliance             1.000   next sec-amendment-scope          0.107
  cms_lower_limb_orthosis  5 candidates   cms-missing-documentation    1.000   next cms-seven-month-evidence     0.816
  ntsb_detector_alert      4 candidates   ntsb-salem-noncritical       1.000   next ntsb-help-desk-process       0.992
  scotus_two_contracts     4 candidates   scotus-two-contract-holding  1.000   next scotus-one-contract-rule     0.439

4 of 4 cases rank the expected primary-source passage first
17 candidates scored exactly once across the 4 cases

Reproduced the four score pairs the page prints: 1.000/0.992, 1.000/0.439, 1.000/0.816, 1.000/0.107
```

`score.py` exits nonzero if any figure fails to reproduce. `run.py --check`
prints `4 of 4 recorded calls rebuilt from the inputs and matched`, and fails if
a call is missing, recorded twice or implied by no case.

## How the figures are derived

The two sides of every comparison come from different places. The expected top
candidate was registered in `inputs/cases.json` before the run. The ranks and
scores are read out of the recorded responses in `calls.json`. The four score
pairs `score.py` asserts against come from a third place again: the page.

- **exact excerpts**: every candidate's text is hashed and compared with the
  digest recorded beside it and with the canonical digest in
  `inputs/sources.json`. Rewriting a passage and its own declared digest
  together is not enough; the canonical entry has to agree as well.
- **fail closed**: a wrong model id, a wrong query id, a missing or duplicated
  candidate, an incomplete rank sequence, or a score that is not a finite number
  stops the run rather than producing a figure.
- **three decimals**: the page rounds with `toFixed(3)` and so does `score.py`.
  The unrounded values are in `calls.json`.

## What is in the dataset

```
rerank/
  inputs/cases.json    4 cases: the query, its provenance, the candidate
                       passages with locators and digests, and the passage
                       registered as the expected first result
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

- **Four questions are a demonstration.** No score here generalises to your
  corpus, and 17 candidates is not a benchmark.
- **The 1.000 is not a probability.** SIE's score response exposes `item_id`,
  `score`, `rank` and token usage. The model's internal yes/no logits are not
  part of the public API response, so nothing here reconstructs them.
- **The saved latencies are provenance, not timings.** The first call of the run
  includes model loading, at 29.3 seconds against 0.1 to 0.2 for the rest.
- **No endpoint is recorded.** The run predates this example recording one. The
  manifest names public SIE v0.6.23 at commit `9d6ca6b0`, running on an L4 in
  Modal on 2026-07-24, which is what was known.
- **No tamper resistance.** The digests catch a truncated or corrupted download.
  They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/rerank/`, which is what its CI tests read.
The two copies hold the same recorded responses. Nothing binds them together, so
they can drift.
