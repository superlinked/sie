# SIE for Cognee

Combine Cognee's persistent document and conversation memory with SIE models
for generation, embeddings, OCR, source reranking and typed decisions. This
integration configures Cognee 1.6.7's supported LiteLLM and compatible embedding
transports and provides native SIE helpers for OCR, reranking and decisions.

Choose an endpoint for each role: SIE Local runs models on your machine,
SIE Cluster runs workers in your cloud, and SIE Serverless runs hosted models
in a third-party cloud. Cognee's databases and session cache stay in the
directory you select. Serverless receives inference inputs; fully local
inference requires every model role to run locally.

Install from a source checkout:

```bash
uv venv --python 3.12
uv pip install ./packages/sie_sdk ./integrations/sie_cognee
```

See [the runnable memory, document, decision and learning examples](../../examples/cognee/README.md)
for scripts, synthetic inputs and deployment choices.

## Configure before importing Cognee

```python
import asyncio
import os
from pathlib import Path

from sie_cognee.config import CogneeSettings, Endpoint, configure
from sie_cognee.documents import SourceDocument
from sie_cognee.retrieval import retrieve_chunks

configure(CogneeSettings(
    llm=Endpoint(os.environ["SIE_LLM_URL"], os.getenv("SIE_LLM_KEY")),
    embedding=Endpoint(os.environ["SIE_EMBEDDING_URL"], os.getenv("SIE_EMBEDDING_KEY")),
    llm_model="Qwen/Qwen3.5-4B",
    embedding_model="BAAI/bge-m3",
    embedding_dimensions=1024,
    storage_root=Path(".local/cognee/example-v1"),
))

import cognee

async def main():
    source = SourceDocument(
        "Synthetic note: the projector is confirmed for 2026-11-14.",
        source_id="equipment-2026-11-14",
        filename="equipment.txt",
    )
    await cognee.remember(source.to_data_item(), dataset_name="example-v1", self_improvement=False)
    for chunk in await retrieve_chunks("Is the projector confirmed?", dataset="example-v1"):
        print(chunk.chunk_id, chunk.text, chunk.metadata)

asyncio.run(main())
```

The `.local/` directory is ignored by this repository. Use an ignored storage
directory in your application too.

`configure` sets the supported `custom` LLM provider with
`openai/<SIE model ID>` and `openai_compatible` embeddings with the raw model
ID. Pass raw SIE identifiers to both model settings, including any vendor
namespace such as `openai/`; `configure` adds the transport prefix separately.
Each endpoint gets exactly one final `/v1`. Generation and embeddings can
use separate endpoints and keys; native OCR/rerank/decision calls use a
separately constructed `sie_sdk.SIEClient`. With a loopback URL only, a missing
compatible API key becomes the harmless placeholder `sie-local-no-key`.
Remote compatible generation/embedding endpoints require an explicit key.
Native SDK endpoint authentication follows the configured service.
Credentials are excluded from `Endpoint` representations.

Generation uses native JSON schemas for Cognee's graph extraction and lesson
distillation. The helper registers the selected SIE model's schema capability
through LiteLLM's public `register_model` API; it does not add a new Cognee
provider or change pricing. Use a generation model that supports SIE structured
output. For an endpoint without that capability, set `native_json_schema=False`
to use Cognee's prompted JSON mode.

`configure` sets Cognee's model endpoints, SQLite/LanceDB/Ladybug storage,
local SQLite session cache and logs. It explicitly sets `GRAPH_EXTRACTOR=llm`
and disables mock embeddings, avoiding an accidental local model fallback.
Inherited stage-specific LLM routes, fallback routes, API versions or
`LLM_ARGS` are rejected; unset them before configuration. Environment settings
are applied before importing Cognee, then its public configuration setters are
called. Start a fresh process to change configuration. Do not initialize other
Cognee clients before this helper.

Use a dedicated empty storage directory. The helper creates an ownership
marker and refuses to reuse unowned nonempty storage or a different embedding
model/dimension pair. It never prunes an existing Cognee installation. Use a
new directory and dataset namespace after a vector model/revision change.
Delete only that dedicated directory when you deliberately want to reset the
experiment. Dataset and session names are caller-owned.

## Inspectable source evidence

- `SourceDocument.to_data_item()` always sets `literal_text=True`: an OCR
  string that looks like a URL or filename is stored as text. Metadata includes
  source ID, basename, page number and SHA-256 of the literal text.
- `ocr_page(client, model=..., image=..., source_id=..., filename=...,
  page_number=...)` extracts one image through native SIE and rejects empty or
  malformed text. It defaults to a 10 MB input limit. Render PDF pages and set
  page-count/pixel limits in the calling application.
- `retrieve_chunks(query, dataset=..., top_k=...)` explicitly requests
  `SearchType.CHUNKS`. `normalize_chunks` accepts released Cognee chunk objects
  or their serialized dictionaries, preserves metadata and raw payloads, and
  rejects graph completions or unidentified/duplicate chunks.
- `rerank_chunks(client, query, chunks, model=..., top_k=...)` preserves source
  IDs, requires exactly one finite score per candidate, and keeps original
  ordering for equal scores. Results have `.chunk` and `.score`.
- `evidence_packet(query, chunks, max_chars=...)` renders exactly the packet to
  inspect and send. It raises `EvidenceOverflowError` instead of truncating.
- `decide(client, packet, model=..., choices=..., instructions=...,
  max_chars=...)` requires a `review` or `unknown` choice, sends
  `overflow_policy=error`, and validates the returned choice and probabilities.
  Results have `.choice` and `.probabilities`; these are model preferences,
  not correctness guarantees.

Import helpers from their explicit submodules: `documents`, `retrieval` and
`decision`. Character limits are application budgets, not token estimates.
The decision model also counts schema/instructions against its context window;
handle a server overflow by selecting fewer sources or requesting review.
Native helpers default to a 180-second provisioning budget; configure the
client's request timeout too. Compatible Cognee transports own their timeouts
and retries. Cold model provisioning can take minutes.

Generated graph assertions can add facts absent from source text. Use source
chunks for decisions and display graph context separately. Cognee may not
include every document's external metadata in a chunk response; the preserved
`raw` and metadata IDs let callers join to their own source manifest.

## Explicit learning

Set `auto_feedback=True` before importing Cognee to enable sequential session
analysis and caching. Use Cognee's public `recall(..., session_id=...)`,
`session.get_session`, `session.add_feedback` and `session.distill_session`
APIs directly. An authored correction must produce a gated session entry;
`add_feedback` alone need not create one. Inspect distillation's status and
actual accepted lesson documents, then query a new input in a fresh session.
A rejected lesson, empty result or unchanged decision remains a visible outcome.

## Tests

```bash
mise run lint
mise run test-integrations -- --python-only
```

The credential-free suite tests configuration against installed Cognee,
remember/recall over deterministic loopback HTTP fixtures, literal provenance,
score alignment, invalid outputs and overflow. It never loads model weights.
The opt-in `tests/test_integration.py` uses only the existing real-model lane's
BGE-M3 embedding and Jina reranker. Full OCR, generation and learning runs
require configured endpoints.
