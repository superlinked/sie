# Source documents

The example downloads every source at run time. No complete third-party PDF is
in this repository, and only the FEMA form, a work of the U.S. federal
government, is redistributed at all.

## NVIDIA Q4 FY2025 CFO Commentary

- Document: https://investor.nvidia.com/files/doc_financials/2025/Q425/Q4FY25-CFO-Commentary.pdf
- Publisher page: https://investor.nvidia.com/financial-info/quarterly-results/default.aspx
- Use: financial tables, headings, footnotes, and eight-page reading order
- Rights basis: public investor document. The publisher retains copyright.
  Use short attributed excerpts for commentary; do not redistribute the full PDF.

## SiriusPoint Q1 2025 Investor Presentation

- Document: https://s27.q4cdn.com/660241321/files/doc_financials/2025/q1/Q1-2025-SPNT-Investor-Presentation_vFinal.pdf
- Publisher page: https://investors.siriuspt.com/financials/quarterly-results/default.aspx
- Use: slide reading order, charts, labels, and financial metrics
- Rights basis: public investor presentation. The publisher retains copyright.
  Use short attributed excerpts for commentary; do not redistribute the full PDF.

## Docling Technical Report

- Document: https://arxiv.org/pdf/2408.09869
- Record: https://arxiv.org/abs/2408.09869
- Citation: Christoph Auer et al., “Docling Technical Report,” arXiv:2408.09869
- Use: two-column academic layout, figures, references, and a comparison table
- Rights basis: the authors retain copyright. Fetch the paper from arXiv and
  use short attributed excerpts.

## FEMA Hermit's Peak/Calf Canyon Proof of Loss Form

- Document: https://www.fema.gov/sites/default/files/documents/fema_hpcc-proof-of-loss-form-english-exp-11.30.2026.pdf
- Publisher page: https://www.fema.gov/flood-insurance/find-form/underwriting
- Form: FEMA FF-206-FY-21-119
- Use: dense form fields, labels, instructions, and signature lines
- Rights basis: work of the U.S. federal government under 17 U.S.C. § 105.
  Do not imply FEMA or DHS endorsement. Government seals and insignia may have
  separate restrictions.
- Reproducibility: the exact government PDF travels with the Hugging Face
  dataset as a fallback, because FEMA may reject requests from datacenter IP
  addresses. `python3 fetch.py` puts it at
  `data/inputs/fema-proof-of-loss-form.pdf`, and the manifest still records the
  official publisher URL.

The recorded run's provenance is `data/inputs/sources.json`, downloaded by
`python3 fetch.py`: the acquired size, SHA-256, rights note and retrieval path
for every file. `uv run fetch-documents` writes the same shape for the copies it
downloads, at `pdfs/manifest.json`.
