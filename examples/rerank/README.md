# Rank primary-source passages

This example asks a reranker to select the passage that answers a precise
question. The candidates are verbatim excerpts from four official sources:

- Pathward Financial's Form 10-K/A filed with the SEC
- CMS's published Lower Limb Orthoses claim example
- the NTSB's East Palestine illustrated digest
- the Supreme Court's Coinbase v. Suski opinion

The questions are authored evaluation prompts. Every candidate records its
source URL, locator, and SHA-256 digest in `data/cases.json` and
`data/sources.json`. No candidate is synthetic or paraphrased.

## Run it

Start public SIE with `Qwen/Qwen3-Reranker-4B` enabled, then install the SDK:

```sh
uv sync
```

Run all four cases against a local server:

```sh
uv run python run.py --output run-output/results.json
```

Run one case against a hosted cluster:

```sh
SIE_BASE_URL=https://your-cluster.example \
SIE_API_KEY=your-key \
uv run python run.py \
  --case sec_filing_amendment \
  --output run-output/sec-restatement.json
```

The command stops with a nonzero exit code if input hashes differ, a candidate
is missing, ranks are incomplete, a score is invalid, or the expected
primary-source passage does not rank first.

## Inspect the recorded run

`verified-run/requests/` contains the audit envelopes produced by the runner.
Each envelope records the SDK arguments plus source IDs and excerpt hashes used
for provenance. Those provenance fields are not part of the SIE wire payload.
The unmodified public-SIE responses are in `verified-run/raw/`.
`verified-run/evaluation.json` records the expected top passage for each
source; `verified-run/manifest.json` pins the server, model revision, hardware,
timing, and every artifact checksum.

The first recorded request includes model loading time. Treat the saved
latencies as provenance, not a benchmark.

## Test offline

```sh
uv run python -m unittest discover -s tests -v
```

The tests do not call a server.
