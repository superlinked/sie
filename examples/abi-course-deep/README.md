# ABI inference course: deep-example pack

This directory turns three existing SIE examples into bounded course projects.
It does not claim that their managed-cloud model roster or budget has been
validated yet.

The selected projects are:

1. [`contract-review-agent`](../contract-review-agent) — the principal
   ingestion, retrieval, reranking, grounded-generation, extraction, and
   evaluation project.
2. [`vision-doc-rag`](../vision-doc-rag), with bounded fixtures from
   [`document-ocr`](../document-ocr) — the multimodal document project.
3. [`retrieval-ablation`](../retrieval-ablation) — the quality/latency/credit
   experiment.

`retrieval-ablation` is the third project because it already has explicit
qrels and NDCG/Recall evaluation and lets a student change one subsystem
(reranking) while holding the dataset and first-stage retrieval constant.
`document-ocr` is valuable, but it is a pipeline stage rather than a separate
retrieval/evaluation project, so this pack uses its redistributable synthetic
images in the vision project.

## Release status

The offline preparation paths are ready to run. Managed US-production
instructions remain gated on:

- the launch catalog and exact model IDs/revisions from
  `superlinked/sie-internal#1943`;
- a live US-production run that records cold and warm latency, request IDs,
  usage metadata, and settled credits;
- a conservative per-student experiment budget derived from those settled
  measurements.

Those gates are deliberately represented by `null` values and
`live_measurement_required` states in [`course-pack.json`](course-pack.json).
Do not replace them with estimates.

## Clean setup

Use Python 3.12 and the committed lock:

```bash
cd examples/abi-course-deep
uv sync --frozen
uv run python -m unittest discover -s tests
```

The tests are offline. They validate fixture hashes, sample caps, output
contracts, unresolved live gates, and secret-safe configuration. They do not
contact SIE, AWS, Modal, Hugging Face, or a vector database.

Never put `SIE_API_KEY` in a source file, notebook cell, command-line
argument, or output artifact. Export it in the shell only:

```bash
export SIE_API_KEY="..."
```

## Project 1: bounded contract review

Prepare three fictional contracts without network access, then review only the
MSA:

```bash
cd ../contract-review-agent
uv sync --frozen
uv run make-sample
uv run review --contract acme-msa
```

What to change:

- one role-to-model mapping at a time in `config.yaml`;
- `search.top_k_candidates` or `search.top_k_results`, one at a time;
- the review instruction, while keeping the contract fixed.

What to measure:

- required expected facts found for the synthetic MSA;
- unsupported facts and citation/grounding failures;
- cold and warm latency for each ledger stage;
- request IDs and usage/settled credits when the public response or account
  ledger exposes them.

Expected output shape:

- one structured review with document type, parties, dates, renewal terms,
  governing law, execution status, obligations, risk flags, and a
  recommendation;
- one ordered stage ledger with model, SIE function, warm-up time, call
  latency, input size, and output size;
- a live measurement record whose credit fields stay `null` until settled
  values have been retrieved.

Course gate: every model role in `config.yaml` must be replaced by the
catalog-frozen ID/revision and exercised on US production. The current IDs are
candidates, not a promise of managed availability.

## Project 2: three-page vision/document RAG

Prepare exactly three Apache-2.0 synthetic document images already committed
to this repository:

```bash
cd ../abi-course-deep
uv run python scripts/prepare_vision_sample.py
cd ../vision-doc-rag
uv run --project ../abi-course-deep python python/ingest.py
uv run --project ../abi-course-deep python python/search.py
```

The preparation command verifies SHA-256 hashes before copying the receipt,
invoice, and slide into `vision-doc-rag/data/pages/course/`, then writes the
page manifest consumed by the existing ingest script. It never downloads a
PDF.

What to change:

- OCR off versus the catalog-approved OCR model;
- visual reranking off versus on;
- one catalog-approved retriever at a time;
- `top_k_candidates`, while keeping the three pages and queries fixed.

What to measure:

- correct page in the top result for receipt-total, invoice-due-date, and
  roadmap-slide queries;
- citation tenant, fixture filename, and page number;
- OCR completeness versus visual-only retrieval quality;
- cold/warm stage latency, request IDs, usage, and settled credits.

Expected output shape:

- query and tenant;
- optional short answer;
- ranked results with page ID, tenant, fixture citation, and per-stage scores;
- timings by stage;
- a separate measurement record with nullable request IDs, usage, and settled
  credits.

Course gate: do not edit `config.yaml` to imply that a retriever, reranker,
answer model, or OCR model is in the managed catalog until #1943 freezes it.
Run the OCR comparison only after the production-ready SGLang OCR default is
named and validated.

## Project 3: bounded retrieval ablation

The classroom path uses 12 synthetic passages, six queries, and four
candidates per query. It has no Turbopuffer or dataset download dependency.

Run the offline contract first:

```bash
cd ../abi-course-deep
uv run python scripts/course_ablation.py
```

After the catalog freeze, run the two live conditions by supplying the exact
approved IDs explicitly:

```bash
export SIE_BASE_URL="https://<approved-us-production-route>"
export SIE_API_KEY="..."

uv run python scripts/course_ablation.py \
  --live \
  --embed-model "<catalog-frozen-embedding-id>" \
  --rerank-model "<catalog-frozen-reranker-id>"
```

The script never prints the endpoint or key and never writes responses or
credentials to disk.

What to change:

- reranking off versus on (the primary controlled ablation);
- candidate count from 2 to 4;
- one catalog-approved embedding or reranker ID at a time.

What to measure:

- NDCG@3 and Recall@3 against the committed qrels;
- total and per-condition warm/cold latency;
- number of SIE requests and request IDs when exposed;
- usage metadata and settled credits from the authoritative account ledger.

Expected output shape:

```json
{
  "schema_version": "abi-course-ablation/v1",
  "mode": "offline-contract | live",
  "dataset": {"documents": 12, "queries": 6},
  "conditions": [
    {
      "name": "dense | dense-plus-rerank | lexical-contract",
      "models": [],
      "metrics": {"ndcg_at_3": 0.0, "recall_at_3": 0.0},
      "latency_ms": 0.0,
      "request_count": 0,
      "request_ids": [],
      "usage": null,
      "settled_credits": null
    }
  ],
  "measurement_gate": "offline_only | live_measurement_required"
}
```

The offline numbers test the evaluator and output contract only. They are not
model-quality, latency, or cost evidence.

## Course-freeze checklist

Before presenting any of these as managed-US-production exercises:

1. Fill every `model_gate.roles[].model_id` and `revision` in
   `course-pack.json` from the approved #1943 catalog.
2. Exercise the exact clean setup and bounded sample path on a reviewed course
   account.
3. Record one cold run and at least three warm runs per condition.
4. Capture request IDs and authoritative settled credits without storing a
   key.
5. Set a per-run and per-student budget from the observed upper bound.
6. Re-run the offline contract tests and the live endpoint smoke.
