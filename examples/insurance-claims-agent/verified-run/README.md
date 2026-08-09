# Verified run

This directory records a successful run against FEMA Flood Insurance Appeal
Decision B8 and the Standard Flood Insurance Policy Dwelling Form.

- Run ID: `prod-us-20260809-corrected`
- Run date: August 9, 2026
- Runtime: SIE Cloud prod-US at `https://api.superlinked.com`
- Evaluation: all ten factual checks passed

The run used `docling`, `fastino/gliner2-large-v1`,
`Qwen/Qwen3-Reranker-4B`, and `Qwen/Qwen3.5-4B`.

`manifest.json` records the models, timings, and SHA-256 hash of every recorded
artifact. `source-manifest.json` pins the publisher URLs and source hashes.
`markdown/` and `raw/` preserve every parsed document and model response. The
other files keep the extracted facts, reranked policy passages, structured
review, and deterministic evaluation.

The example also includes the two downloaded FEMA PDFs, so the verified run can
be reproduced if a publisher URL changes.
