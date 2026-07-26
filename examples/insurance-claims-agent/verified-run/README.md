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
publisher URLs and SHA-256 hashes. The other files preserve the extracted
facts, reranked policy passages, structured review, raw reranker response, and
deterministic evaluation.

The source PDFs are downloaded from FEMA at run time. They are not copied into
this repository.
