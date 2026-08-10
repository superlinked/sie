# Contract review with the OpenAI Agents SDK, on one SIE cluster

A multi-agent contract reviewer built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) where **every model call is served by SIE**. No `api.openai.com`; managed calls are metered by SIE's native primitives. An **investigator** agent follows a required tool sequence to gather grounded facts, then a **synthesizer** agent turns them into a structured review. Each step runs on the **right model from the SIE catalog**: a fast triage model, a vision model that reads the scanned signature page, a reasoning sub-agent for clause risk, a text-to-SQL specialist, an OCR model, embedding and reranker models for clause search, a zero-shot entity extractor, and a safety guardrail. Ten specialized jobs, one cluster.

This is the "one cluster powers every model your agent calls" idea from the [SIE landing page](https://superlinked.com), made real and runnable.

## The catalog: the right model for each job

Every value below is a real model in the [SIE catalog](https://superlinked.com/models). Swap any line in `config.yaml` to try another; nothing else changes.

| Role in the agent | SIE model | SIE function |
|---|---|---|
| Triage: classify the document type | `Qwen/Qwen3.5-4B` | generate |
| **Orchestrator**: plan, call tools, assemble the review | `Qwen/Qwen3.6-27B` (non-thinking) | generate + tools + JSON schema |
| Vision: read the scanned signature page | `Qwen/Qwen3.5-4B` | generate + image |
| Reasoning sub-agent: bounded grounded clause-risk analysis | `Qwen/Qwen3.6-27B` | generate |
| Text-to-SQL: query the obligations DB | `Qwen/Qwen3.5-4B` (raw-prompt specialists use `sql.mode: prompt`) | generate |
| Guardrail: safety / prompt-injection | `ibm-granite/granite-guardian-3.0-2b` (alias `guard`) | generate |
| OCR: scanned page to markdown | `lightonai/LightOnOCR-2-1B` | extract |
| Clause search: dense embeddings | `BAAI/bge-m3` | encode |
| Clause rerank: cross-encoder | `Qwen/Qwen3-Reranker-4B` | score |
| Entity extraction: parties, dates, amounts | `fastino/gliner2-large-v1` | extract |

## How it works

The Agents SDK accepts any model that implements its model interface. This
example binds that interface to `SIEAsyncClient.generate` in
`contract_review_agent/native_model.py`:

```python
Agent(
    name="Risk Analyst",
    model=SIENativeModel(
        "Qwen/Qwen3.5-4B",
        sie_client,
        provision_timeout_s=900,
    ),
)
```

For each required investigation step, the adapter emits the configured tool
call directly and binds its fixed query or question. This prevents a model from
skipping a required source or copying large clause payloads into arguments.
After the required sequence, SIE performs the final unstructured investigator
turn through its native generation primitive. The example passes the
investigator's final grounded findings, rather than its raw tool messages, into
a separate synthesizer turn. Only that synthesizer uses the agent's declared
Pydantic schema for structured final output. The example never calls an
OpenAI-compatible endpoint or sends data to `api.openai.com`. The text-to-SQL
tool may execute model-generated SQL only after enforcing one SELECT statement;
it never executes generated Python or shell code. The SDK normalizes slash-form
catalog IDs into wire-safe paths, but the example never substitutes one catalog
model for another.

The core flow is an investigator plus a synthesizer with a reasoning sub-agent.
For the checked-in published example, source-specific publication gates assemble
the final narrative from the validated signature evidence, source clauses,
structured risks, and exact SQL rows if the investigator's prose is incomplete:

1. An **investigator** (on `Qwen3.6-27B`) with seven tools and **no** structured `output_type`, so it can't short-circuit to a hallucinated answer. It must call tools to learn anything about the contract:
   - `classify_document` (triage) · `read_signature_page` (vision) · `analyze_clause_risks` (delegates to the reasoning **sub-agent**): generative LLMs
   - `ocr_signature_page` · `extract_entities` (`extract`), `search_clauses` (`encode` + `score`), `query_obligations_db` (`generate`): retrieval and extraction
   - a `granite-guardian` **input guardrail** screens the request first and fails closed if the guard model is unavailable.
2. A **synthesizer** (structured `output_type=ContractReview`, no tools) turns the investigator's grounded findings into the final review (parties, dates, governing law, executed?, key obligations, risk flags with severity plus redlines, recommendation) via SIE's JSON-schema-constrained generation. Complete validated SQL rows and structured clause risks are retained as evidence appendices, so narrative assembly cannot silently drop either specialist's result.

> Two agents instead of one: a single structured-output agent tends to emit the schema immediately and skip the tools, sometimes hallucinating the fields. Splitting "gather with tools" from "format the result" keeps the fan-out real and the output grounded.

## Run it

You need Python 3.12 and a **GPU-backed SIE deployment**. The generative models run on SIE's generation bundle (CUDA), so the `latest-cpu-default` image can't serve them.

```bash
# 1. SIE on a local NVIDIA GPU, or point SIE_CLUSTER_URL / SIE_API_KEY at a managed GPU cluster.
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-default

cd examples/contract-review-agent
cp .env.example .env          # edit SIE_CLUSTER_URL / SIE_API_KEY if not localhost
uv sync --frozen

# 2. Fetch a handful of real contracts from CUAD (CC BY 4.0). Downloads a ~18 MB archive once.
uv run fetch-contracts                 # or: uv run make-sample  (offline synthetic contracts)

# 3. Review the first contract and watch the model fan-out.
uv run review                          # uv run review --list   to see available contracts
uv run review --contract <slug>        # review a specific one
uv run review --run-id local           # also write a reproducible evidence bundle
```

> **GPU sizing.** The orchestrator, structured synthesizer, and bounded clause analyst run on `Qwen/Qwen3.6-27B` (non-thinking), so they need an H100 or RTX PRO 6000. The latency-sensitive roles (`triage`, `vision`, `sql`) run on `Qwen/Qwen3.5-4B`. A cold cluster pays a one-time load per model on first use; the agent retries the "still provisioning" responses under `cluster.provision_timeout_s`. Keep bundles warm (`minReplicas: 1`) to skip the wait. A required model or tool failure is printed with the partial ledger and exits nonzero.

## What you'll see

`uv run review` prints the model catalog, runs the agent, then prints the
structured review **plus a per-model observability ledger**. Each row carries
the step's model, native SIE primitive, total latency, data sent, and available
throughput. Every primitive uses the SDK's governed capacity wait. Try
`--instruction "..."` to change the ask, or feed the guardrail a malicious
prompt to watch `granite-guardian` trip the tripwire.

`verified-run/` contains the August 10, 2026, prod-US record. Its manifest pins
the endpoint, run date, input hashes, models, artifact hashes, and diagnostic
wall time. `api-calls.json` preserves requested and runtime model IDs, request
IDs, rate-book versions, execution identities, debited credits, and stable
workflow stages. `source-evidence.json` records the exact full-contract section excerpt behind each published risk flag, plus its hash. The evaluation requires the ordered tool and agent
sequence, including each stage's configured model and native primitive, and
permits at most one structured synthesis repair; historical bundles may include
at most one findings-audit pass. The publisher itself normalizes line endings and trailing whitespace only. Current runs require an exact safe
guardrail verdict and atomically discard any rejected bundle. Historical ledger
entries therefore remain evidence of what the endpoint returned rather than
being rewritten to satisfy newer validation.

## Swapping models (the point of the catalog)

`config.yaml` maps each role to a model id. Change a string, rerun, no code edits:

```yaml
models:
  sql: "defog/sqlcoder-7b-2"                  # raw specialist; set sql.mode: prompt
  ocr: "opendatalab/MinerU2.5-Pro-2604-1.2B"  # try a different OCR model
```

Alternatively, resolve roles **server-side** with SIE's gateway aliases. Set `SIE_GATEWAY_MODEL_ALIASES='{"vision":"Qwen/Qwen3.5-4B","ocr":"lightonai/LightOnOCR-2-1B"}'` and reference `vision` / `ocr` (the built-ins `code`, `sql`, `guard` already ship).

## Data

The default corpus is **[CUAD](https://www.atticusprojectai.org/cuad/)** (Contract Understanding Atticus Dataset): 510 real commercial contracts filed with the SEC, released by The Atticus Project under **CC BY 4.0**. `fetch-contracts` downloads CUAD's ~18 MB archive once (from the [Atticus Project repo](https://github.com/TheAtticusProject/cuad)), parses the SQuAD-format contract text, writes a curated handful as the corpus, renders the contract's signature block to an image for the OCR/vision step (falling back to the document tail when no signature marker exists), and seeds a small SQLite obligations database that references the contracts pulled.

> CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. arXiv:2103.06268. Licensed CC BY 4.0.

`uv run make-sample` builds a fully synthetic, offline input corpus (an Acme MSA, an NDA, and an SOW) so the demo runs with no network.

## Notes

- Agent planning, tool selection, structured output, vision questions, and text-to-SQL all use native `generate`.
- `sql.mode: instruct` builds one role-labelled raw prompt; `sql.mode: prompt` sends a specialist's raw template. Both use the same native primitive.
- This is a demo of inference orchestration, **not legal advice**.

Apache-2.0, like the rest of SIE.
