/**
 * Integration tests for SIE LlamaIndex.TS embeddings against a running server.
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
import { SIEEmbedding, SIESparseEmbeddingFunction } from "../../src/index.js";

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

describe("SIEEmbedding integration tests", () => {
  let embedding: SIEEmbedding;

  beforeAll(() => {
    embedding = new SIEEmbedding({
      baseUrl: getServerUrl(),
      modelName: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await embedding.close();
  });

  describe("getTextEmbedding()", () => {
    it("embeds a single text and returns number[]", async () => {
      const vector = await embedding.getTextEmbedding("Hello world");

      expect(vector).toHaveLength(1024); // bge-m3 dimension
      expect(Array.isArray(vector)).toBe(true);
      expect(typeof vector[0]).toBe("number");
    });
  });

  describe("getTextEmbeddings()", () => {
    it("embeds multiple texts and returns number[][]", async () => {
      const texts = ["Hello world", "Machine learning is great", "TypeScript SDK"];
      const vectors = await embedding.getTextEmbeddings(texts);

      expect(vectors).toHaveLength(3);
      expect(vectors[0]).toHaveLength(1024);
      expect(vectors[1]).toHaveLength(1024);
      expect(vectors[2]).toHaveLength(1024);

      // Verify it's number[] not Float32Array
      expect(Array.isArray(vectors[0])).toBe(true);
      expect(typeof vectors[0]?.[0]).toBe("number");
    });

    it("returns empty array for empty input", async () => {
      const vectors = await embedding.getTextEmbeddings([]);
      expect(vectors).toEqual([]);
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

      const docVectors = await embedding.getTextEmbeddings(docs);
      const queryVector = await embedding.getTextEmbedding(query);

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
      const instructedEmbedding = new SIEEmbedding({
        baseUrl: getServerUrl(),
        modelName: "BAAI/bge-m3",
        instruction: "Represent this document for retrieval:",
        timeout: 60_000,
      });

      try {
        const vector = await instructedEmbedding.getTextEmbedding("Sample document");
        expect(vector).toHaveLength(1024);
      } finally {
        await instructedEmbedding.close();
      }
    });
  });
});

describe("SIESparseEmbeddingFunction integration tests", () => {
  let sparseEmbed: SIESparseEmbeddingFunction;

  beforeAll(() => {
    sparseEmbed = new SIESparseEmbeddingFunction({
      baseUrl: getServerUrl(),
      modelName: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await sparseEmbed.close();
  });

  describe("encodeDocuments()", () => {
    it("returns tuple of [indices[][], values[][]]", async () => {
      const [indices, values] = await sparseEmbed.encodeDocuments(["Hello world", "Test doc"]);

      expect(indices).toHaveLength(2);
      expect(values).toHaveLength(2);

      // Each document should have sparse representation
      expect(indices[0]?.length).toBeGreaterThan(0);
      expect(values[0]?.length).toBeGreaterThan(0);
      expect(indices[0]?.length).toBe(values[0]?.length);
    });

    it("returns empty arrays for empty input", async () => {
      const [indices, values] = await sparseEmbed.encodeDocuments([]);
      expect(indices).toEqual([]);
      expect(values).toEqual([]);
    });
  });

  describe("encodeQueries()", () => {
    it("returns sparse vectors for queries", async () => {
      const [indices, values] = await sparseEmbed.encodeQueries(["search query"]);

      expect(indices).toHaveLength(1);
      expect(values).toHaveLength(1);
      expect(indices[0]?.length).toBeGreaterThan(0);
      expect(values[0]?.length).toBeGreaterThan(0);
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
