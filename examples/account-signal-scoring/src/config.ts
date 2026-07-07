export const config = {
  sieUrl: process.env.SIE_URL ?? "http://localhost:8080",
  sieApiKey: process.env.SIE_API_KEY,

  models: {
    extractor: "urchade/gliner_multi-v2.1",
    encoder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "BAAI/bge-reranker-base",
  },

  // Optional generation model for the LLM brief (via /v1/chat/completions).
  // Empty -> deterministic brief, so the demo stays CPU-only by default.
  chatModel: process.env.SIE_CHAT_MODEL ?? "",

  // Entity types GLiNER surfaces from the account context (display only).
  extractLabels: ["company", "person", "job_title", "product_metric", "money"],

  rerank: {
    // How many playbooks to shortlist by cosine before the cross-encoder reranks.
    topK: 3,
  },

  paths: {
    accounts: "data/accounts.json",
    playbooks: "data/playbooks.json",
    index: "data/playbook_index.json",
  },

  port: Number(process.env.PORT ?? 3044),
} as const;
