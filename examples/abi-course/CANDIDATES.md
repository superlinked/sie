# ABI course deep-example inventory

This inventory maps existing examples to the proposed managed course-model
roster. It is an implementation starting point, not a final choice of three
projects.

## 1. Contract review agent

Start from [`../contract-review-agent`](../contract-review-agent).

What is already strong:

- a realistic investigator/synthesizer flow with OCR, visual reasoning,
  retrieval, reranking, entity extraction, SQL, guardrails, and generation;
- CUAD contracts (CC BY 4.0) plus three synthetic offline contracts;
- models already aligned with the proposed roster include Qwen3.5-4B,
  Qwen3.6-27B, Granite Guardian, LightOnOCR, BGE-M3, and Qwen3 Reranker 4B;
- per-model request and timing observability is already built in.

Work required for the course:

- change `urchade/gliner_large-v2.1` to the roster target
  `urchade/gliner_multi-v2.1`, subject to catalog validation;
- add a bounded, repeatable quality and citation evaluation rather than
  treating the observability ledger as an evaluation;
- pin a reproducible clean-environment dependency lock;
- add a small managed-cloud fixture and budget, while preserving the offline
  synthetic path.

This is the strongest current candidate for the contract-analysis project.

## 2. Vision document RAG

Start from [`../vision-doc-rag`](../vision-doc-rag), borrowing the recognition
stage from [`../document-ocr`](../document-ocr).

What is already strong:

- multi-tenant page-image ingestion, tenant filtering, negative examples,
  page citations, visual retrieval, optional visual reranking, and a vision
  answer model;
- Qwen3.5-4B and Qwen3-VL-Reranker-2B already align with the proposed roster;
- the document OCR example already exercises roster candidates LightOnOCR and
  GLiNER Multi.

Work required for the course:

- replace current `vidore/colqwen2.5-v0.2` retrieval, which is not in the
  proposed roster, with a validated course-catalog vision/retrieval path;
- add OCR as an explicit comparison or fallback. The current example
  intentionally excludes OCR from ranking, so it does not yet teach the
  requested OCR-versus-vision tradeoff;
- add a small licensed fixture, citation-quality checks, latency, and credit
  accounting;
- prove the complete managed route in US production.

This is the strongest current candidate for the multimodal document project.

## 3. Retrieval quality/cost ablation

Start from [`../retrieval-ablation`](../retrieval-ablation).

What is already strong:

- real queries and qrels with NDCG and Recall evaluation;
- explicit dense, multivector, hybrid, and reranking comparisons;
- a benchmark-shaped result that naturally teaches quality/latency/cost
  tradeoffs.

What must change:

- the full 1,854-query/2,942-page benchmark and external Turbopuffer dependency
  are too large for a first course run;
- current leading configurations use Jina and Mixedbread models outside the
  proposed roster;
- create a small deterministic subset comparing roster-backed BGE-M3 dense
  versus multivector output and a roster Qwen reranker;
- report quality, latency, request IDs, token/image units, and credit cost from
  the settled production contract.

This is the best roster-aligned third-project candidate today, but selection
should remain open until the roster has quality and production evidence.

## Alternative: regulatory RAG

[`../regulatory-rag`](../regulatory-rag) is compact and technically
distinctive: it has 12 passages, a custom LoRA encoder, and a token-pruning
adapter. It currently depends on custom images and bespoke models outside the
managed course roster, so it is better kept as a self-hosted extension unless
the course explicitly chooses fine-tuning over a managed-cloud ablation.
