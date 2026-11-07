# Build source-backed memory with SIE and Cognee in TypeScript

These examples connect Cognee's persistent memory to SIE models. Retrieve the
original text, read a document image, rerank evidence for a typed decision, and
promote a session correction into durable memory. Each script writes its actual
results to an ignored `runs/` directory so you can inspect what it used.

They use the [SIE TypeScript integration](../../../integrations/sie_ts_cognee/README.md)
and the official `@cognee/cognee-ts` 0.2.0 Node.js bindings. The bindings run
Cognee's Rust engine inside your process. No Python server is needed.

**Compatibility limit:** Cognee TypeScript 0.2.0 omits an explicit output-token
budget on structured graph-extraction requests. Responses can exceed an
endpoint's default budget and fail even for ordinary documents. A fix in the
upstream request path is required before these examples are ready for general
use with SIE Serverless.

## Run

Use Node.js 22 or newer on a platform with Cognee's prebuilt native bindings.
Linux x64 is tested. From the repository root:

```bash
mise exec -- pnpm install --frozen-lockfile
mise run ts -- build
cd examples/cognee/typescript
cp .env.example .env
```

Set `SIE_API_KEY` in `.env`, then fetch the pinned public inputs:

```bash
pnpm fetch-data
pnpm memory
pnpm documents
pnpm decide
```

The first model call may wait for SIE Serverless to start a worker. Set
`SIE_COGNEE_NAMESPACE` to a new value to create a fresh memory. State lives in
`.state/<namespace>`; retain the complete directory, including its provenance
manifests. Do not reuse the Python examples' state directory.

The inputs are six synthetic CC0 files from
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/487cdc3bd0480659afd7ee88329449e6a6c8655d/cognee/workshop-starter/2026-11-06/v1).
The fetcher verifies the pinned manifest, file sizes and SHA-256 checksums. The
files are downloaded into the shared ignored `../data/` directory; no dataset
is committed here.

## Four primitives

| Script | What it exposes | Extend it into |
| --- | --- | --- |
| `01_memory.ts` | Original source chunks, stable IDs and source metadata | A searchable project notebook or agent knowledge base |
| `02_read_documents.ts` | Actual OCR text, source image and page identity, retrieved chunks | Document intake with traceable evidence |
| `03_decide.ts` | Retrieval candidates, reranking scores, the exact evidence packet and typed choice probabilities | Triage, readiness checks or review queues |
| `04_learn.ts` | A session correction and its durable retrieval evidence in a new process | Agents that carry explicit guidance into later sessions |

`pnpm memory -- --graph` also shows model-derived graph context. It is kept
separate from the source chunks used for decisions.

`pnpm documents -- path/to/page.png` accepts one PNG, JPEG or WebP raster page.
Render PDF pages before calling the page helper; this example does not render
PDFs. The output reports the OCR model's actual text. Source identity does not
make OCR correct: inspect uncertain scans before using their text as evidence.

The decision example asks whether the projector passed its image check. It
retrieves the source records, reranks them, and sends one selected chunk to a
small decision model. `confirmed`, `failed` and `unknown` remain distinct
outcomes. If the evidence exceeds the configured bound, select fewer sources
or request review. No text is silently cut off.

## Carry guidance into a new session

Run the teaching step:

```bash
pnpm learn
```

It generates a unique project and approval code, records an explicit rule in a
Cognee session, adds feedback, and calls `improve()`. It then indexes the
persisted session record and prints a second command:

```bash
pnpm learn -- --verify-run "runs/.../04-learn-run-directory"
```

Run that command in a new process with the same namespace. The new session
starts empty. Its retrieval query names the project but does not contain the
expected code. Verification checks that an actual permanent source chunk has
the correct dataset, document, session tag and authored rule.

`sessionTransferVerified` reports whether the raw guidance is retrievable.
`distilledLessonVerified` separately reports an accepted lesson published by
Cognee's optional distillation stage. Distillation is not guaranteed; an
`improve()` result alone proves neither outcome. These derived records live in
a separate learning dataset and are not mixed into the original-source evidence
used by the decision example.

## Models and endpoints

Model IDs, bounds and questions live in `config.json`:

| Role | Default model |
| --- | --- |
| Generation and graph extraction | `zai-org/GLM-5.3-Flash:relay` |
| Embeddings | `BAAI/bge-m3` — 1,024 dimensions |
| OCR | `lightonai/LightOnOCR-2-1B` |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` |
| Typed decisions | `fastino/GLiNER2.5-Decide` |

The TypeScript SDK currently needs a non-Qwen generation model for SIE. This
is separate from the native reranking call. The Python examples use a different
generation transport and retain their Qwen default.

The `:relay` generation profile sends inference through SIE Serverless to a
remote provider. To run locally, choose a model/profile served by your local
SIE instance and set `SIE_LLM_BASE_URL`, `SIE_EMBEDDING_BASE_URL`,
`SIE_OCR_BASE_URL`, `SIE_RERANK_BASE_URL` and `SIE_DECISION_BASE_URL` as needed.
Set a role's key explicitly when its endpoint has a different origin; the
shared key is never forwarded to a different origin automatically.

Your application and memory stay on your machine. SIE Local model workers run
on that same machine, SIE Cluster workers run in your cloud, and SIE Serverless
workers run in a third-party cloud. Fully local inference requires all five
roles to be local; changing only the generation endpoint is not sufficient.

## Checks

From the repository root:

```bash
mise run ts -- build
mise exec -- pnpm --filter @superlinked/sie-cognee test
mise exec -- pnpm --filter sie-cognee-typescript-examples test
```

The integration tests exercise the real published Cognee bindings against a
local deterministic HTTP fixture. They use no credentials or model downloads.
