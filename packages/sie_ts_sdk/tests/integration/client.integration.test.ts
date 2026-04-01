/**
 * Integration tests for SIE TypeScript SDK against a running server.
 *
 * These tests require a running SIE server with BAAI/bge-m3 model.
 * The server is started automatically via globalSetup.ts.
 *
 * To run integration tests:
 *   mise run ts test-integration
 *
 * These tests mirror the Python SDK integration tests in:
 *   packages/sie_sdk/tests/test_integration.py
 */

import { readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SIEClient } from "../../src/client.js";
import { SERVER_INFO_FILE, type ServerInfo } from "./globalSetup.js";

// Server URL is read from file (written by globalSetup)
function getServerUrl(): string {
  try {
    const serverInfo: ServerInfo = JSON.parse(readFileSync(SERVER_INFO_FILE, "utf-8"));
    return serverInfo.url;
  } catch {
    throw new Error("Server info file not found - globalSetup may have failed");
  }
}

describe("SIEClient integration tests", () => {
  let client: SIEClient;

  beforeAll(() => {
    client = new SIEClient(getServerUrl(), { timeout: 60_000 });
  });

  afterAll(async () => {
    await client.close();
  });

  describe("listModels()", () => {
    it("can list models from running server", async () => {
      const models = await client.listModels();

      expect(models.length).toBeGreaterThan(0);

      // Check structure
      const model = models[0];
      expect(model).toBeDefined();
      expect(model?.name).toBeDefined();
      expect(model?.outputs).toBeDefined();
    });
  });

  describe("encode() - single item", () => {
    it("can encode a single item and get back Float32Array", async () => {
      const result = await client.encode("BAAI/bge-m3", { text: "Hello world" });

      // Should be a single result, not array
      expect(result).toBeDefined();
      expect(result.dense).toBeDefined();
      expect(result.dense).toBeInstanceOf(Float32Array);
      expect(result.dense?.length).toBe(1024); // bge-m3 dimension
    });

    it("echoes item ID in result", async () => {
      const result = await client.encode("BAAI/bge-m3", { id: "doc-123", text: "Test" });

      expect(result.id).toBe("doc-123");
    });
  });

  describe("encode() - batch", () => {
    it("can encode multiple items and get list of results", async () => {
      const results = await client.encode("BAAI/bge-m3", [
        { text: "Hello" },
        { text: "World" },
        { text: "Test" },
      ]);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result.dense).toBeDefined();
        expect(result.dense?.length).toBe(1024);
      }
    });

    it("preserves item IDs in batch results", async () => {
      const results = await client.encode("BAAI/bge-m3", [
        { id: "doc-1", text: "First" },
        { id: "doc-2", text: "Second" },
      ]);

      expect(results).toHaveLength(2);
      expect(results[0]?.id).toBe("doc-1");
      expect(results[1]?.id).toBe("doc-2");
    });
  });

  describe("encode() - output types", () => {
    it("can request all output types from BAAI/bge-m3", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputTypes: ["dense", "sparse", "multivector"] },
      );

      // Check dense
      expect(result.dense).toBeDefined();
      expect(result.dense?.length).toBe(1024);

      // Check sparse
      expect(result.sparse).toBeDefined();
      expect(result.sparse?.indices).toBeInstanceOf(Int32Array);
      expect(result.sparse?.values).toBeInstanceOf(Float32Array);
      expect(result.sparse?.indices.length).toBeGreaterThan(0);
      expect(result.sparse?.indices.length).toBe(result.sparse?.values.length);

      // Check multivector
      expect(result.multivector).toBeDefined();
      expect(result.multivector?.length).toBeGreaterThan(0);
      expect(result.multivector?.[0]).toBeInstanceOf(Float32Array);
      expect(result.multivector?.[0]?.length).toBe(1024); // per-token dim
    });

    it("can request only sparse output", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputTypes: ["sparse"] },
      );

      expect(result.sparse).toBeDefined();
      expect(result.dense).toBeUndefined();
      expect(result.sparse?.indices).toBeInstanceOf(Int32Array);
      expect(result.sparse?.values).toBeInstanceOf(Float32Array);
    });

    it("can request only multivector output", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputTypes: ["multivector"] },
      );

      expect(result.multivector).toBeDefined();
      expect(result.dense).toBeUndefined();
      expect(result.multivector?.[0]).toBeInstanceOf(Float32Array);
    });
  });

  describe("encode() - embeddings quality", () => {
    it("produces similar embeddings for semantically similar texts", async () => {
      const results = await client.encode("BAAI/bge-m3", [
        { text: "Machine learning is a type of artificial intelligence" },
        { text: "ML is a form of AI" },
        { text: "The weather is nice today" },
      ]);

      // Compute cosine similarities
      const embed1 = results[0]?.dense;
      const embed2 = results[1]?.dense;
      const embed3 = results[2]?.dense;

      expect(embed1).toBeDefined();
      expect(embed2).toBeDefined();
      expect(embed3).toBeDefined();

      if (embed1 && embed2 && embed3) {
        const sim12 = cosineSimilarity(embed1, embed2);
        const sim13 = cosineSimilarity(embed1, embed3);

        // Similar texts should have higher similarity than different topics
        expect(sim12).toBeGreaterThan(sim13);
        expect(sim12).toBeGreaterThan(0.6); // Reasonably high similarity for similar texts
        expect(sim13).toBeLessThan(0.7); // Lower similarity for different topics
      }
    });
  });

  describe("encode() - quantization (outputDtype)", () => {
    it("returns Float32Array for default (float32) dtype", async () => {
      const result = await client.encode("BAAI/bge-m3", { text: "Hello world" });

      expect(result.dense).toBeInstanceOf(Float32Array);
      expect(result.dense?.length).toBe(1024);
    });

    it("returns Float32Array for float16 dtype (converted from half-precision)", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputDtype: "float16" },
      );

      // float16 is converted to Float32Array on the client side
      expect(result.dense).toBeInstanceOf(Float32Array);
      expect(result.dense?.length).toBe(1024);

      // Values should be valid floats (not NaN or Infinity)
      if (result.dense) {
        for (let i = 0; i < Math.min(10, result.dense.length); i++) {
          expect(Number.isFinite(result.dense[i])).toBe(true);
        }
      }
    });

    it("returns Int8Array for int8 dtype", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputDtype: "int8" },
      );

      expect(result.dense).toBeInstanceOf(Int8Array);
      expect(result.dense?.length).toBe(1024);

      // Int8 values should be in valid range [-128, 127]
      if (result.dense) {
        for (let i = 0; i < Math.min(10, result.dense.length); i++) {
          const val = result.dense[i] ?? 0;
          expect(val).toBeGreaterThanOrEqual(-128);
          expect(val).toBeLessThanOrEqual(127);
        }
      }
    });

    it("returns Uint8Array for uint8 dtype", async () => {
      const result = await client.encode(
        "BAAI/bge-m3",
        { text: "Hello world" },
        { outputDtype: "uint8" },
      );

      expect(result.dense).toBeInstanceOf(Uint8Array);
      expect(result.dense?.length).toBe(1024);

      // Uint8 values should be in valid range [0, 255]
      if (result.dense) {
        for (let i = 0; i < Math.min(10, result.dense.length); i++) {
          const val = result.dense[i] ?? 0;
          expect(val).toBeGreaterThanOrEqual(0);
          expect(val).toBeLessThanOrEqual(255);
        }
      }
    });

    it("quantization works with batch encoding", async () => {
      const results = await client.encode("BAAI/bge-m3", [{ text: "Hello" }, { text: "World" }], {
        outputDtype: "int8",
      });

      expect(results).toHaveLength(2);
      expect(results[0]?.dense).toBeInstanceOf(Int8Array);
      expect(results[1]?.dense).toBeInstanceOf(Int8Array);
    });
  });

  describe("Real-world workflow", () => {
    it("supports document embedding and retrieval workflow", async () => {
      // User scenario: Index documents and find most similar
      const documents = [
        { id: "doc-1", text: "Python is a popular programming language for data science" },
        { id: "doc-2", text: "Machine learning models need training data" },
        { id: "doc-3", text: "JavaScript is used for web development" },
      ];

      const query = { text: "What language is best for ML?" };

      // Encode documents
      const docEmbeddings = await client.encode("BAAI/bge-m3", documents);
      expect(docEmbeddings).toHaveLength(3);

      // Encode query
      const queryResult = await client.encode("BAAI/bge-m3", query);
      expect(queryResult.dense).toBeDefined();

      // Find most similar (simple cosine similarity)
      const queryDense = queryResult.dense;
      if (queryDense) {
        const similarities = docEmbeddings.map((doc, i) => ({
          id: documents[i]?.id,
          similarity: doc.dense ? cosineSimilarity(queryDense, doc.dense) : 0,
        }));

        similarities.sort((a, b) => b.similarity - a.similarity);

        // ML-related docs (doc-1: Python/data science, doc-2: ML training) should rank
        // higher than unrelated doc (doc-3: JavaScript/web dev)
        const topTwo = similarities.slice(0, 2).map((s) => s.id);
        expect(topTwo).toContain("doc-1"); // Python/data science is ML-related
        expect(topTwo).toContain("doc-2"); // ML training data is ML-related
        expect(similarities[2]?.id).toBe("doc-3"); // JavaScript should be least similar
      }
    });
  });
});

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
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
