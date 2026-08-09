# One shelf photo yields reviewable stockout evidence

This example finds a shelf gap and reads the notice and shelf label next to it. The input is one public 4032 by 3024 supermarket shelf photo. The output keeps DINO boxes, DINO-derived OCR crops, raw LightOnOCR responses, timing, and seven evidence lines together in one run directory.

The image is case `042` from the [Humans in the Loop Supermarket Shelves Dataset](https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset). It shows an empty Panadol facing, a printed out-of-stock notice, and a shelf label.

## Verify the recorded run

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run verify-retail-records
```

The command verifies every checked-in checksum. It rebuilds both the pinned
Modal-direct fixture and the August 9, 2026, prod-US evidence from raw detector
and OCR responses, and verifies the production runtime model and request IDs.

## Rerun through an SIE API

Set an SIE endpoint and process the full shelf image:

```bash
export SIE_BASE_URL="https://your-sie-endpoint.example"
export SIE_API_KEY="your-key"
uv run audit-retail-shelf --run-id cloud-042
```

The command writes these files to `runs/cloud-042/`:

- `raw/grounding-dino.json` and two raw LightOnOCR responses
- `crops/`, containing the upper notice and lower shelf-label regions created from DINO boxes
- `selection.json`, with every geometry decision and source coordinate
- `evidence.json`, with the selected gap, shelf-label box, and OCR lines
- `evaluation.json`, with the fail-closed geometry and OCR checks
- `manifest.json`, with endpoint, model IDs, source hash, output hashes, and diagnostic timing

It fails if DINO returns no usable gap, if two aligned text candidates cannot be found, or if OCR does not return three upper lines and four distinct lower lines.

For a self-hosted GPU deployment, the equivalent model command is:

```bash
sie-server serve \
  --models IDEA-Research/grounding-dino-base,lightonai/LightOnOCR-2-1B
```

## Exact launch pins

| Step | Model | Hugging Face revision | Recorded compute |
|---|---|---|---|
| Detect gap and text regions | `IDEA-Research/grounding-dino-base` | `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | NVIDIA L4 |
| Read the two derived crops | `lightonai/LightOnOCR-2-1B` | `c97bd377f04481830395218fa8951df9deaba756` | NVIDIA L4 |

Both catalog entries declare Apache-2.0. The revisions live in `retail_shelf_audit/config.py`.

## Geometry chooses the OCR inputs

The shelf contains several Panadol products and price labels. Full-frame OCR cannot prove which label belongs to the empty facing.

The runner uses four fixed rules:

1. Drop full-width and very short `empty shelf space` boxes, then take the highest-scoring remaining gap.
2. Collect `price tag` boxes that horizontally overlap the gap and sit in its vertical band. DINO labels the printed notice as a price tag, so it enters this candidate set without a status prompt.
3. Deduplicate boxes at intersection-over-union `>= 0.7`, then choose the strongest vertically aligned pair. The upper box is the notice; the lower box is the shelf label.
4. Expand each box by 20% horizontally and 25% vertically, resize 3x, and OCR with `max_new_tokens=64`. Keep four unique lower lines and three upper lines.

The runtime selector never searches for a known product, SKU, price, or sign phrase. Known case-042 strings live only in the offline verifier and tests.

## Recorded results and limits

The direct Modal records under `recorded/modal-direct/` use the pinned launch checkpoints and matching adapter settings. They are checkpoint evidence. They are not an SIE Cloud API run.

- DINO returned an `empty shelf space` box at `[2043.8, 2137.0, 623.6, 402.4]` with score `0.274157`.
- The geometry pass selected an upper `price tag` box at `[2300.1, 2318.7, 335.8, 244.5]` for the notice and a lower one at `[2235.5, 2559.4, 399.2, 186.3]` for the shelf label.
- LightOnOCR read the two derived crops in `5270.533 ms` on an NVIDIA L4. The first useful lines contain `Panadol Child`, `5-12Yrs Elixir 100ml`, `101760`, `10⁹⁹`, and the three-line out-of-stock notice.

DINO called the notice a `price tag`; the evidence keeps that model label unchanged. The upper/lower geometry and OCR text make the record reviewable.

The source dataset labels `Product` and `Price` boxes. It has no empty-gap ground truth, so this record is visual case evidence, not an accuracy measurement.

See [SOURCES.md](./SOURCES.md) for license terms and [source-manifest.json](./source-manifest.json) for checksums.
