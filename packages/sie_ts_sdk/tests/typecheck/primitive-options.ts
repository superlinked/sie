import type { SIEClient } from "../../src/client.js";
import type {
  ChatCompletionRequest,
  EncodeOptions,
  ExtractOptions,
  ScoreOptions,
} from "../../src/types.js";

declare const client: SIEClient;
declare const image: Uint8Array;

// Extract takes no options for models without labels (OCR, speech, documents).
void client.extract("lightonai/LightOnOCR-2-1B", { images: [image] });
void client.extract("lightonai/LightOnOCR-2-1B", [{ images: [image] }]);

// Labels are one of several ways to say what to extract.
const typedDecisions: ExtractOptions = {
  outputSchema: { handoff: { type: "noul", instructions: "Should a person take over?" } },
  instruction: "Triage the support ticket.",
  adapterOptions: { overflow_policy: "truncate_text" },
};
void client.extract("fastino/GLiNER2.5-Decide", { text: "Get me a person." }, typedDecisions);

const entities: ExtractOptions = { labels: ["person"], threshold: 0.75 };
void client.extract("urchade/gliner_multi-v2.1", { text: "Tim Cook leads Apple." }, entities);

// Encode and score accept the runtime options the Python SDK names `options`.
const encodeOptions: EncodeOptions = {
  outputTypes: ["multivector"],
  isQuery: false,
  adapterOptions: { profile: "long_context" },
};
void client.encode("lightonai/GTE-ModernColBERT-v1", { text: "a long document" }, encodeOptions);

const scoreOptions: ScoreOptions = {
  instruction: "Rank passages that answer the question",
  adapterOptions: { profile: "default" },
};
void client.score("Qwen/Qwen3-Reranker-4B", { text: "query" }, [{ text: "doc" }], scoreOptions);

// Results name the model that served them.
void client.encode("BAAI/bge-m3", { text: "hello" }).then((result) => {
  const model: string | undefined = result.model;
  return model;
});
void client.extract("urchade/gliner_multi-v2.1", { text: "hello" }).then((result) => {
  const model: string | undefined = result.model;
  return model;
});

// The gateway's schema_in_prompt chat field is part of the request type.
const chat: ChatCompletionRequest = {
  model: "Qwen/Qwen3-4B-Instruct-2507",
  messages: [{ role: "user", content: "Extract the invoice fields." }],
  response_format: { type: "json_schema", json_schema: { name: "invoice", schema: {} } },
  schema_in_prompt: false,
};
void client.chatCompletions(chat);
