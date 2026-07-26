# Verified run

This directory records a successful run against FEMA Flood Insurance Appeal
Decision B8 and the Standard Flood Insurance Policy Dwelling Form.

- Run ID: `verified-20260726b`
- Run date: July 26, 2026
- Hardware: one NVIDIA L4
- Runtime: SIE on Modal
- Evaluation: all ten factual checks passed

The run used `docling`, `fastino/gliner2-large-v1`,
`BAAI/bge-reranker-v2-m3`, and `Qwen/Qwen3.5-4B:no-spec`.

`manifest.json` records the models and timings. `source-manifest.json` pins the
publisher URLs and SHA-256 hashes. `markdown/` and `raw/` preserve every parsed
document and model response. The other files keep the extracted facts, reranked
policy passages, structured review, and deterministic evaluation.

The example also includes the two downloaded FEMA PDFs, so the verified run can
be reproduced if a publisher URL changes.
