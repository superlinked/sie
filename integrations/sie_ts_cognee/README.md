# SIE for Cognee in TypeScript

Use Cognee's persistent memory with SIE generation and embeddings, then use
SIE OCR, reranking and decision models on the same source material. The Node.js
integration uses the published `@cognee/cognee-ts` Rust bindings directly.

The package provides:

- `createCognee`: explicit model endpoints and a dedicated local storage directory.
- `SourceDocument` and `SourceMemory`: literal text ingestion, source identity and
  retrieval of original document chunks.
- `ocrPage`: one raster page to text with filename and page provenance.
- `rerankChunks`, `evidencePacket` and `decide`: source selection and bounded
  typed decisions with probabilities.

See [the four TypeScript examples](../../examples/cognee/typescript/README.md)
for memory, document ingestion, decisions and session learning.

**Compatibility limit:** Cognee TypeScript 0.2.0 omits an explicit output-token
budget on structured graph-extraction requests. Responses can exceed an
endpoint's default budget and fail even for ordinary documents. A fix in the
upstream request path is required before these examples are ready for general
use with SIE Serverless.

## Set up

Use Node.js 22 or newer. From a source checkout:

```bash
mise exec -- pnpm install --frozen-lockfile
mise run ts -- build
```

This integration pins `@cognee/cognee-ts` 0.2.0. Its prebuilt native bindings
support Linux x64/arm64 with glibc, macOS arm64 and Windows x64. The integration
is tested on Linux x64. It runs in Node.js; it is not a browser library.

## Store and retrieve source evidence

Configure before importing Cognee elsewhere. Importing `@superlinked/sie-cognee`
itself does not initialize the native runtime or download models.

```typescript
import {
  createCognee,
  SourceDocument,
  SourceMemory,
} from "@superlinked/sie-cognee";

const endpoint = {
  baseUrl: "https://api.superlinked.com",
  apiKey: process.env.SIE_API_KEY,
};
const storageRoot = ".local/cognee/typescript-example";
const cognee = await createCognee({
  llm: endpoint,
  embedding: endpoint,
  llmModel: "zai-org/GLM-5.3-Flash:relay",
  embeddingModel: "BAAI/bge-m3",
  embeddingDimensions: 1024,
  storageRoot,
});

try {
  await cognee.warm();
  const memory = new SourceMemory(cognee, {
    dataset: "equipment",
    storageRoot,
  });
  await memory.add(new SourceDocument({
    text: "The projector passed its image check.",
    sourceId: "CHECK-01",
    filename: "projector-check.txt",
  }));
  await memory.build();
  console.log(await memory.retrieve("Did the projector pass?"));
} finally {
  // Shutdown is process-wide; call it after every Cognee handle has finished.
  (await import("@cognee/cognee-ts")).shutdown();
}
```

Generation uses Cognee's custom OpenAI-compatible provider; embeddings use its
OpenAI-compatible embedding provider. Endpoint paths receive one final `/v1`,
and raw SIE model IDs are preserved. Each role can use a separate endpoint and
key. A missing key is accepted only for exact loopback hosts.

Cognee 0.2.0's Qwen generation requests include options that SIE does not accept.
Use a non-Qwen model with tool-calling support for generation with this pin.
Qwen rerankers and other models called through the native SIE SDK are unaffected.
The example's `:relay` generation profile uses a remote provider through SIE
Serverless; choose a locally served model/profile for fully local inference.

`createCognee` disables Cognee's automatic dotenv loading and product analytics,
selects explicit generation/embedding routes, and sets SQLite, graph, vector,
cache and log paths below `storageRoot`. Load your application's environment
before calling it. Inherited mock, fallback and extra-request settings are
rejected. Initialize a fresh process to change the runtime configuration.

Use an empty dedicated storage directory. An ownership marker prevents reuse
with a different embedding model, endpoint or dimension count. Python and
TypeScript Cognee storage layouts are separate; do not share their directories.
The integration never prunes existing Cognee memory.

## Preserve provenance

The published TypeScript SDK does not accept external metadata on text ingestion.
`SourceMemory` therefore keeps an explicit local manifest mapping returned
Cognee document IDs to source ID, filename, page number and SHA-256 of the
original text. It verifies the dataset and document IDs on every retrieved
chunk. Identical text from multiple sources retains every recorded source.
Keep this manifest with the Cognee databases when moving or backing up memory.

Only `CHUNKS` results with a `DocumentChunk` payload are evidence. Graph answers,
session replies and learning-generated documents are separate outputs. Unknown
provenance is an error; the integration does not guess a source from similar
text. A literal source string that resembles a URL is stored as text.

## Rerank and decide

Pass a configured `SIEClient` to `ocrPage`, `rerankChunks` or `decide`; native
calls can use different endpoints from Cognee's generation and embeddings.
`rerankChunks` requires one finite score for every candidate ID and preserves
source metadata. `evidencePacket` renders the exact text sent to `decide` and
refuses overflow rather than truncating it. Its character limit is a client
bound; the server separately enforces its token limit.

Decisions require an `unknown` or `review` choice. Returned probabilities must
cover exactly the requested labels, sum to one and agree with the selected
label. Probabilities describe model preferences, not guarantees of correctness.

## Choose where models run

Your Node.js application and Cognee databases run on your machine. SIE Local
runs model workers on that same machine. SIE Cluster runs them in your cloud.
SIE Serverless runs them in a third-party cloud and receives inference inputs.
Fully local inference requires **every** model role to use a local endpoint and
locally served model profile.
