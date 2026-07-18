# Contract review with the OpenAI Agents SDK, on one SIE cluster

> ABI course: use the bounded synthetic path and release gates in
> [`../abi-course-deep`](../abi-course-deep). The model IDs below are example
> configuration until the managed course catalog is frozen and live-validated.

A multi-agent contract reviewer built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) where **every model call is served by SIE**. No `api.openai.com`, no per-token bill. An **investigator** agent autonomously calls tools to gather grounded facts, then a **synthesizer** agent turns them into a structured review. Each step runs on the **right model from the SIE catalog**: a fast triage model, a vision model that reads the scanned signature page, a reasoning sub-agent for clause risk, a text-to-SQL specialist, an OCR model, embedding and reranker models for clause search, a zero-shot entity extractor, and a safety guardrail. Ten specialized jobs, one cluster.

This is the "one cluster powers every model your agent calls" idea from the [SIE landing page](https://superlinked.com), made real and runnable.

## The catalog: the right model for each job

Every value below is a real model in the [SIE catalog](https://superlinked.com/models). Swap any line in `config.yaml` to try another; nothing else changes.

| Role in the agent | SIE model | SIE function |
|---|---|---|
| Triage: classify the document type | `Qwen/Qwen3.5-4B` | chat |
| **Orchestrator**: plan, call tools, assemble the review | `Qwen/Qwen3.6-27B` (64K, non-thinking) | chat + tools + JSON schema |
| Vision: read the scanned signature page | `Qwen/Qwen3.5-4B` | chat + image |
| Reasoning sub-agent: clause-risk analysis | `Qwen/Qwen3.5-4B` | chat |
| Text-to-SQL: query the obligations DB | `Qwen/Qwen3.5-4B` (or `defog/sqlcoder-7b-2` with `sql.mode: completions`) | chat |
| Guardrail: safety / prompt-injection | `ibm-granite/granite-guardian-3.0-2b` (alias `guard`) | chat |
| OCR: scanned page to markdown | `lightonai/LightOnOCR-2-1B` | extract |
| Clause search: dense embeddings | `BAAI/bge-m3` | encode |
| Clause rerank: cross-encoder | `Qwen/Qwen3-Reranker-4B` | score |
| Entity extraction: parties, dates, amounts | `urchade/gliner_large-v2.1` | extract |

## How it works

The whole trick is one idea: **the Agents SDK speaks the OpenAI wire protocol, and SIE serves an OpenAI-compatible `/v1` endpoint.** So we point the SDK at SIE and force chat completions (`contract_review_agent/runtime.py`):

```python
client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
set_default_openai_client(client)        # every agent talks to SIE...
set_default_openai_api("chat_completions")  # ...over chat completions, not the Responses API...
set_tracing_disabled(True)               # ...and we never phone home with traces.
```

After that, each `Agent` just names the SIE model it should run on:

```python
Agent(name="Risk Analyst", model=OpenAIChatCompletionsModel("Qwen/Qwen3.5-4B", openai_client=client), ...)
```

The flow is **two agents**:

1. An **investigator** (on `Qwen3.6-27B`) with seven tools and **no** structured `output_type`, so it can't short-circuit to a hallucinated answer. It must call tools to learn anything about the contract:
   - `classify_document` (triage) · `read_signature_page` (vision) · `analyze_clause_risks` (delegates to the reasoning **sub-agent**): generative LLMs
   - `ocr_signature_page` · `extract_entities` (`extract`), `search_clauses` (`encode` + `score`), `query_obligations_db` (`chat`): retrieval and extraction
   - a `granite-guardian` **input guardrail** screens the request first (and fails open, logged, if the guard model is unavailable).
2. A **synthesizer** (structured `output_type=ContractReview`, no tools) turns the investigator's grounded findings into the final review (parties, dates, governing law, executed?, key obligations, risk flags with severity plus redlines, recommendation) via SIE's JSON-schema-constrained generation.

> Two agents instead of one: a single structured-output agent tends to emit the schema immediately and skip the tools, sometimes hallucinating the fields. Splitting "gather with tools" from "format the result" keeps the fan-out real and the output grounded.

## Run it

You need Python 3.12 and a **GPU-backed SIE deployment**. The generative models run on SIE's generation bundle (CUDA), so the `latest-cpu-default` image can't serve them.

```bash
# 1. SIE on a local NVIDIA GPU, or point SIE_CLUSTER_URL / SIE_API_KEY at a managed GPU cluster.
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-default

cd examples/contract-review-agent
cp .env.example .env          # edit SIE_CLUSTER_URL / SIE_API_KEY if not localhost
uv sync

# 2. Fetch a handful of real contracts from CUAD (CC BY 4.0). Downloads a ~18 MB archive once.
uv run fetch-contracts                 # or: uv run make-sample  (offline synthetic contracts)

# 3. Review the first contract and watch the model fan-out.
uv run review                          # uv run review --list   to see available contracts
uv run review --contract <slug>        # review a specific one
```

> **GPU sizing.** The brain (`orchestrator`, which also does the structured synthesis) runs on `Qwen/Qwen3.6-27B` (64K, non-thinking), so it needs an H100 or RTX PRO 6000. The shorter-context roles (`triage`, `vision`, `reasoning`, `sql`) run on `Qwen/Qwen3.5-4B`. A cold cluster pays a one-time load per model on first use; the agent retries the "still provisioning" responses under `cluster.provision_timeout_s`. Keep bundles warm (`minReplicas: 1`) to skip the wait. Any model the cluster can't serve degrades gracefully, logged in the ledger, instead of failing the run.

## What you'll see

`uv run review` prints the model catalog, runs the agent, then prints the structured review **plus a per-model observability ledger**. Each row carries the step's model, SIE function, **cold-start warm-up**, warm latency, data sent, and **warm throughput (tokens/s)**, so you can watch one cluster fan a single request across the catalog and see how each model performed. (Warm-up is shown separately from throughput for the generative calls; the `encode`/`score`/`extract` calls go through the SIE SDK, which provisions internally, so those show total latency.) Try `--instruction "..."` to change the ask, or feed the guardrail a malicious prompt to watch `granite-guardian` trip the tripwire.

## Swapping models (the point of the catalog)

`config.yaml` maps each role to a model id. Change a string, rerun, no code edits:

```yaml
models:
  sql: "defog/sqlcoder-7b-2"                  # a text-to-SQL specialist (also set sql.mode: completions)
  ocr: "opendatalab/MinerU2.5-Pro-2604-1.2B"  # try a different OCR model
```

Alternatively, resolve roles **server-side** with SIE's gateway aliases. Set `SIE_GATEWAY_MODEL_ALIASES='{"vision":"Qwen/Qwen3.5-4B","ocr":"lightonai/LightOnOCR-2-1B"}'` and reference `vision` / `ocr` (the built-ins `code`, `sql`, `guard` already ship).

## Data

The default corpus is **[CUAD](https://www.atticusprojectai.org/cuad/)** (Contract Understanding Atticus Dataset): 510 real commercial contracts filed with the SEC, released by The Atticus Project under **CC BY 4.0**. `fetch-contracts` downloads CUAD's ~18 MB archive once (from the [Atticus Project repo](https://github.com/TheAtticusProject/cuad)), parses the SQuAD-format contract text, writes a curated handful as the corpus, renders one page to an image for the OCR/vision step, and seeds a small SQLite obligations database that references the contracts pulled.

> CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. arXiv:2103.06268. Licensed CC BY 4.0.

`uv run make-sample` builds a fully synthetic, offline alternative (an Acme MSA, an NDA, and an SOW) so the demo runs with no network.

## Notes

- Chat completions, tool calling, JSON-schema structured output, vision, and `/v1/completions` (set `sql.mode: completions` to route SQL through a completion-only model like `sqlcoder`) are all served over SIE's OpenAI-compatible API.
- Text-to-SQL runs on `Qwen/Qwen3.5-4B` over chat; to demo the `/v1/completions` path instead, set `sql: "defog/sqlcoder-7b-2"` with `sql.mode: completions`.
- This is a demo of inference orchestration, **not legal advice**.

Apache-2.0, like the rest of SIE.
