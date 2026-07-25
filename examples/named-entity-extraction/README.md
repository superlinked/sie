# Extract custom entities from primary sources

GLiNER accepts labels at request time. This example sends the same model four
different label sets for financial, healthcare, rail-safety, and legal text.
The inputs are verbatim excerpts from the SEC, CMS, NTSB, and Supreme Court.

Each excerpt records a source URL, document locator, and SHA-256 digest in
`data/cases.json`. The text is neither synthetic nor paraphrased.

## Run it

Start public SIE with `urchade/gliner_multi-v2.1` enabled, then install the SDK:

```sh
uv sync
```

Run every source against a local server:

```sh
uv run python run.py --output run-output/results.json
```

Run only the NTSB source against a hosted cluster:

```sh
SIE_BASE_URL=https://your-cluster.example \
SIE_API_KEY=your-key \
uv run python run.py \
  --case ntsb_detector_alert \
  --output run-output/ntsb-bearing-alert.json
```

The runner verifies each committed excerpt hash before it calls SIE. It then
checks that every returned span uses a requested label, has a finite score, and
exactly matches `input_text[start:end]`. Each case also requires the exact
spans featured on the task page. Any mismatch or missing required span stops
the run.

## Inspect the recorded run

`verified-run/requests/` contains the audit envelopes produced by the runner.
Each envelope records the SDK arguments plus source IDs and excerpt hashes used
for provenance. Those provenance fields are not part of the SIE wire payload.
`verified-run/raw/` contains the unmodified response for each source.

The raw CMS response intentionally retains the model's incorrect
`proof of delivery` span under the `missing documentation` label. It is not a
required anchor. Keeping it visible makes label and threshold tradeoffs easier
to inspect.

`verified-run/manifest.json` pins public SIE `v0.6.23`, the GLiNER revision,
hardware, request timing, and artifact checksums. The first request includes
model loading time, so the saved latency is provenance rather than a benchmark.

## Test offline

```sh
uv run python -m unittest discover -s tests -v
```

The tests validate all 53 recorded spans and 28 required anchors without
calling a server.
