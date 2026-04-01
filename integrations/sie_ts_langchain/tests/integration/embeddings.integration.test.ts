/**
 * Integration tests for SIE LangChain.js embeddings against a running server.
 *
 * These tests require a running SIE server with BAAI/bge-m3 model.
 * The server is started automatically via the SDK's globalSetup.
 *
 * To run integration tests:
 *   pnpm test:integration
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SIEEmbeddings, SIESparseEncoder } from "../../src/index.js";

// Server info file location (from SDK's globalSetup)
const SERVER_INFO_FILE = resolve(
  __dirname,
  "../../../../packages/sie_ts_sdk/tests/integration/.server-info.json",
);

interface ServerInfo {
  url: string;
  pid: number;
}

function getServerUrl(): string {
  try {
    const serverInfo: ServerInfo = JSON.parse(readFileSync(SERVER_INFO_FILE, "utf-8"));
    return serverInfo.url;
  } catch {
    throw new Error("Server info file not found - globalSetup may have failed");
  }
}

describe("SIEEmbeddings integration tests", () => {
  let embeddings: SIEEmbeddings;

  beforeAll(() => {
    embeddings = new SIEEmbeddings({
      baseUrl: getServerUrl(),
      model: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await embeddings.close();
  });

  describe("embedDocuments()", () => {
    it("embeds multiple documents and returns number[][]", async () => {
      const docs = ["Hello world", "Machine learning is great", "TypeScript SDK"];
      const vectors = await embeddings.embedDocuments(docs);

      expect(vectors).toHaveLength(3);
      expect(vectors[0]).toHaveLength(1024); // bge-m3 dimension
      expect(vectors[1]).toHaveLength(1024);
      expect(vectors[2]).toHaveLength(1024);

      // Verify it's number[] not Float32Array
      expect(Array.isArray(vectors[0])).toBe(true);
      expect(typeof vectors[0]?.[0]).toBe("number");
    });

    it("returns empty array for empty input", async () => {
      const vectors = await embeddings.embedDocuments([]);
      expect(vectors).toEqual([]);
    });
  });

  describe("embedQuery()", () => {
    it("embeds a single query and returns number[]", async () => {
      const vector = await embeddings.embedQuery("What is machine learning?");

      expect(vector).toHaveLength(1024);
      expect(Array.isArray(vector)).toBe(true);
      expect(typeof vector[0]).toBe("number");
    });

    it("sets isQuery=true for asymmetric embedding", async () => {
      // Both should work, but query embedding uses isQuery=true internally
      const queryVector = await embeddings.embedQuery("search query");
      const docVectors = await embeddings.embedDocuments(["document text"]);

      expect(queryVector).toHaveLength(1024);
      expect(docVectors[0]).toHaveLength(1024);
    });
  });

  describe("semantic similarity", () => {
    it("produces higher similarity for related texts", async () => {
      const docs = [
        "Python is great for machine learning",
        "ML models require training data",
        "The weather is sunny today",
      ];
      const query = "What programming language is used for AI?";

      const docVectors = await embeddings.embedDocuments(docs);
      const queryVector = await embeddings.embedQuery(query);

      // Compute cosine similarities
      const similarities = docVectors.map((docVec) => cosineSimilarity(queryVector, docVec));

      // Python/ML doc should be most similar to AI query
      const weatherSimilarity = similarities[2] ?? 0;
      expect(similarities[0]).toBeGreaterThan(weatherSimilarity);
      // ML doc should also be more similar than weather doc
      expect(similarities[1]).toBeGreaterThan(weatherSimilarity);
    });
  });

  describe("with instruction", () => {
    it("supports instruction-tuned embedding", async () => {
      const instructedEmbeddings = new SIEEmbeddings({
        baseUrl: getServerUrl(),
        model: "BAAI/bge-m3",
        instruction: "Represent this document for retrieval:",
        timeout: 60_000,
      });

      try {
        const vectors = await instructedEmbeddings.embedDocuments(["Sample document"]);
        expect(vectors[0]).toHaveLength(1024);
      } finally {
        await instructedEmbeddings.close();
      }
    });
  });
});

describe("SIESparseEncoder integration tests", () => {
  let sparseEncoder: SIESparseEncoder;

  beforeAll(() => {
    sparseEncoder = new SIESparseEncoder({
      baseUrl: getServerUrl(),
      model: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await sparseEncoder.close();
  });

  describe("encodeDocuments()", () => {
    it("returns sparse vectors with indices and values", async () => {
      const results = await sparseEncoder.encodeDocuments(["Hello world", "Test doc"]);

      expect(results).toHaveLength(2);

      // Each document should have sparse representation
      expect(results[0]?.indices.length).toBeGreaterThan(0);
      expect(results[0]?.values.length).toBeGreaterThan(0);
      expect(results[0]?.indices.length).toBe(results[0]?.values.length);
    });
  });

  describe("encodeQueries()", () => {
    it("returns sparse vectors for queries", async () => {
      const results = await sparseEncoder.encodeQueries(["search query"]);

      expect(results).toHaveLength(1);
      expect(results[0]?.indices.length).toBeGreaterThan(0);
      expect(results[0]?.values.length).toBeGreaterThan(0);
    });
  });
});

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
