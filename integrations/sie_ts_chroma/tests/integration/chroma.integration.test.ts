/**
 * Integration tests for SIE ChromaDB embeddings against a running server.
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
import { SIEEmbeddingFunction, SIESparseEmbeddingFunction } from "../../src/index.js";

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

describe("SIEEmbeddingFunction integration tests", () => {
  let embeddingFunction: SIEEmbeddingFunction;

  beforeAll(() => {
    embeddingFunction = new SIEEmbeddingFunction({
      baseUrl: getServerUrl(),
      model: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await embeddingFunction.close();
  });

  describe("generate()", () => {
    it("generates embeddings for texts and returns number[][]", async () => {
      const texts = ["Hello world", "Machine learning is great", "TypeScript SDK"];
      const embeddings = await embeddingFunction.generate(texts);

      expect(embeddings).toHaveLength(3);
      expect(embeddings[0]).toHaveLength(1024); // bge-m3 dimension
      expect(embeddings[1]).toHaveLength(1024);
      expect(embeddings[2]).toHaveLength(1024);

      // Verify it's number[] not Float32Array
      expect(Array.isArray(embeddings[0])).toBe(true);
      expect(typeof embeddings[0]?.[0]).toBe("number");
    });

    it("returns empty array for empty input", async () => {
      const embeddings = await embeddingFunction.generate([]);
      expect(embeddings).toEqual([]);
    });

    it("generates single embedding correctly", async () => {
      const embeddings = await embeddingFunction.generate(["Single document"]);

      expect(embeddings).toHaveLength(1);
      expect(embeddings[0]).toHaveLength(1024);
    });
  });

  describe("semantic similarity", () => {
    it("produces higher similarity for related texts", async () => {
      const docs = [
        "Python is great for machine learning",
        "ML models require training data",
        "The weather is sunny today",
      ];
      const query = ["What programming language is used for AI?"];

      const docEmbeddings = await embeddingFunction.generate(docs);
      const queryEmbeddings = await embeddingFunction.generate(query);

      const queryVec = queryEmbeddings[0];
      if (!queryVec) throw new Error("Query embedding missing");

      // Compute cosine similarities
      const similarities = docEmbeddings.map((docVec) => cosineSimilarity(queryVec, docVec));

      // Python/ML doc should be most similar to AI query
      const weatherSimilarity = similarities[2] ?? 0;
      expect(similarities[0]).toBeGreaterThan(weatherSimilarity);
      // ML doc should also be more similar than weather doc
      expect(similarities[1]).toBeGreaterThan(weatherSimilarity);
    });
  });
});

describe("SIESparseEmbeddingFunction integration tests", () => {
  let sparseEmbed: SIESparseEmbeddingFunction;

  beforeAll(() => {
    sparseEmbed = new SIESparseEmbeddingFunction({
      baseUrl: getServerUrl(),
      model: "BAAI/bge-m3",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await sparseEmbed.close();
  });

  describe("generate()", () => {
    it("returns sparse embeddings with indices and values", async () => {
      const embeddings = await sparseEmbed.generate(["Hello world", "Test document"]);

      expect(embeddings).toHaveLength(2);

      // Each document should have sparse representation
      expect(embeddings[0]?.indices.length).toBeGreaterThan(0);
      expect(embeddings[0]?.values.length).toBeGreaterThan(0);
      expect(embeddings[0]?.indices.length).toBe(embeddings[0]?.values.length);
    });

    it("returns empty array for empty input", async () => {
      const embeddings = await sparseEmbed.generate([]);
      expect(embeddings).toEqual([]);
    });
  });

  describe("generateAsDict()", () => {
    it("returns sparse embeddings as dict format", async () => {
      const dictEmbeddings = await sparseEmbed.generateAsDict(["Hello world"]);

      expect(dictEmbeddings).toHaveLength(1);
      const dict = dictEmbeddings[0];
      expect(dict).toBeDefined();

      // Should have token IDs as keys and weights as values
      const keys = Object.keys(dict ?? {});
      expect(keys.length).toBeGreaterThan(0);

      // All values should be numbers
      for (const key of keys) {
        const numKey = Number.parseInt(key, 10);
        expect(typeof dict?.[numKey]).toBe("number");
      }
    });

    it("handles multiple texts", async () => {
      const dictEmbeddings = await sparseEmbed.generateAsDict(["Hello", "World"]);

      expect(dictEmbeddings).toHaveLength(2);
      expect(Object.keys(dictEmbeddings[0] ?? {}).length).toBeGreaterThan(0);
      expect(Object.keys(dictEmbeddings[1] ?? {}).length).toBeGreaterThan(0);
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
