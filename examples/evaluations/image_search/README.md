# Compare direct image search with caption search

This example reproduces the Image Search pilot recorded on 6 October 2026:
20 licensed catalogue and archive photos, eight frozen query families, and
five retrieval recipes. It reconstructs all 800 cosine scores from the full
vectors, checks all 40 rankings, and recalculates the model API costs.

The code also records new native SIE vectors against a compatible endpoint
you supply. It sends the prepared JPEG bytes directly, indexes photos in two
batches of ten, and encodes all eight queries separately with `is_query=True`.
Cosine ranking happens in the client.

## Reproduce the recording

Run these commands from this directory:

```sh
python3 fetch.py
python3 score.py
```

Both commands use only the Python standard library. Fetch downloads the
[public evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/3ca626712b086958963cfa5741b3993e1d81614e/evaluations/2026-10-06/image-search-0fe605cb2953dd92)
at commit `3ca626712b086958963cfa5741b3993e1d81614e`. The archive, its manifest,
every retained file and the corrected cost companion are checked against
pinned SHA256 digests. Scoring needs no API key and makes no inference calls.

| Recipe | First result matches every requested detail |
| --- | ---: |
| SIE SigLIP 384 | 8 of 8 |
| SIE SigLIP2 base 224 | 7 of 8 |
| Voyage multimodal 3.5 | 8 of 8 |
| GPT-6 Luna captions + small embeddings | 7 of 8 |
| GPT-6.1 Sol captions + large embeddings | 8 of 8 |

These are eight purposive query families. The 160 query/photo relevance
grades describe the shared candidate set; they are not 160 independent
trials. Grades were frozen before output inspection and reviewed using AI
vision. This pilot demonstrates the recorded cases; it does not estimate
quality across a population of catalogues.

For the blue-shoe query, SigLIP 384 ranks the blue shoe first. Luna caption
search ranks the white shoe first. The saved caption of the blue shoe does
mention its colour, so this case does not establish that captioning omitted
colour. Read the full captions and rankings in the downloaded evidence.

## Recalculate the costs

`score.py` independently prices the retained provider usage and applies the
published SIE image and query units. The workload is one initial index of
100,000 photos plus a million short queries, using this pilot's prepared
image-size and query-token distribution.

| Recipe | Initial photo index | Million queries | Total model API forecast |
| --- | ---: | ---: | ---: |
| SIE SigLIP 384 | $2.32 | $1.98 | $4.30 |
| SIE SigLIP2 base 224 | $1.43 | $1.47 | $2.90 |
| Voyage multimodal 3.5 | $11.73 | $1.08 | $12.81 |
| Luna captions + small embeddings | $6.88 | $0.20 | $7.08 |
| Sol captions + large embeddings | $160.25 | $1.30 | $161.55 |

Caption generation is priced once per indexed image. SIE query billing uses
the recorded adapter's fixed 64 padded tokens per query. The public tariff
and meter links are in `cost-projection.json`. These are list-price forecasts
as of 6 October 2026, without Batch discounts. Shared vector storage,
ranking and application costs are excluded equally. They are not invoices.

## Record a new native run

Install the standalone project's locked dependencies, then inspect the plan:

```sh
uv sync --frozen
uv run python run.py
```

To send the three native Encode requests, set your endpoint and explicitly
choose live mode. Supply `SIE_API_KEY` when your endpoint requires it.

```sh
SIE_URL=https://your-sie.example \
SIE_API_KEY=your-key \
uv run python run.py --live --output runs/my-siglip384.json
```

The default model is `google/siglip-so400m-patch14-384`, checkpoint
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`, at its default 384-pixel, float16
serving profile with unnormalised 1,152-dimensional dense output. Enable that
profile on your SIE server. The script verifies vector dimensions and finite
values; it does not attest the endpoint's checkpoint. It requests float32
output and disables capacity waiting and out-of-memory retries.

`--arm sie-siglip2-base224` selects the other recorded native profile. Every
new result is scored against the frozen gold, without forcing it to equal
the saved result. Endpoint URLs and API keys are not written to the output.

## Provider settings and rights

The evidence contains complete captions, vectors, usage, source pins and
byte-preserving request templates. `inputs/original-context/arms.json` records
the exact provider settings:

- Voyage multimodal 3.5: 1,024 dimensions, float output, `document` indexing,
  `query` search, and truncation disabled.
- GPT-6 Luna: the frozen general photo-search caption prompt, high-detail
  images, no reasoning, 512 output-token limit; text-embedding-3-small at
  1,536 dimensions.
- GPT-6.1 Sol: the same caption prompt and image detail, low reasoning,
  2,048 output-token limit; text-embedding-3-large at 3,072 dimensions.

The Python live client here runs the native arms. The saved provider request
templates and full usage remain available for reproducing those arms.

Photographs were aspect-preserving prepared JPEGs with a maximum 512-pixel
long edge. No photograph is resized or re-encoded by this example.
`inputs/original-context/ATTRIBUTION.md` and `licences.json` retain each
creator, source and licence, including CC BY, CC BY-SA and public-domain
works. Preserve those notices when redistributing the images.

The published archive is a derived semantic export of selected recorded
values. Its hashes bind those published bytes; they do not authenticate
omitted raw provider envelopes, invoices or deployment state.

## Test offline

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```
