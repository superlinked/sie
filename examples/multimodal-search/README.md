# Search images with text

This example embeds the text query `a red leather handbag` and six photographs
with the same SigLIP model. It computes cosine similarity locally and returns
the image ranking.

The candidate set makes the result meaningful:

- a red leather handbag, the intended match
- red shoes, which match the color but not the object
- a black handbag, which matches the object but not the color
- a green backpack, black camera, and blue running shoe

Every photograph is real. Three are CC0 files; the red handbag has a
No Copyright, United States rights statement. The black handbag is CC BY 4.0,
and the red shoes are CC BY 3.0. Exact attribution, source links, dimensions,
byte lengths, and checksums are in `data/sources.json`.

## Run it

Start public SIE with `google/siglip-so400m-patch14-384` enabled, then install
the SDK:

```sh
uv sync
```

Run against a local server:

```sh
uv run python run.py --output run-output/results.json
```

Run against a hosted cluster:

```sh
SIE_BASE_URL=https://your-cluster.example \
SIE_API_KEY=your-key \
uv run python run.py --output run-output/results.json
```

The script verifies every image byte before making a request. It requires one
1,152-dimensional embedding for the query and each image, rejects non-finite
vectors, recomputes cosine similarity, and fails if the red leather handbag
does not rank first.

## Inspect the recorded run

`verified-run/raw/` contains all seven recorded vectors from public SIE. The
saved evaluation is derived from those vectors. It is never substituted for a
live response.

`verified-run/requests/` contains audit envelopes. They record the SDK inputs
plus local filenames, byte lengths, and hashes used for provenance. The SDK
encodes the image bytes in the SIE wire payload.

`verified-run/manifest.json` pins public SIE `v0.6.23`, commit
`9d6ca6b00f788b6ab19f8d6dc9506e1b31dad2f0`, model revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`, NVIDIA L4 hardware, input
checksums, and artifact checksums. The recorded query latency includes model
loading, so it is not a benchmark.

## Test offline

```sh
uv run python -m unittest discover -s tests -v
```

The tests recompute the six recorded cosine scores from the full vectors.
