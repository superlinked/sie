import { describe, expect, it } from "vitest";
import { maxsim, maxsimBatch, maxsimDocuments } from "../src/scoring.js";

describe("maxsim", () => {
  it("should return 0 for empty query", () => {
    const query: Float32Array[] = [];
    const doc = [new Float32Array([1, 0, 0])];
    expect(maxsim(query, doc)).toBe(0);
  });

  it("should return 0 for empty document", () => {
    const query = [new Float32Array([1, 0, 0])];
    const doc: Float32Array[] = [];
    expect(maxsim(query, doc)).toBe(0);
  });

  it("should compute correct score for identical vectors", () => {
    // Normalized vector: [1, 0, 0]
    const query = [new Float32Array([1, 0, 0])];
    const doc = [new Float32Array([1, 0, 0])];
    expect(maxsim(query, doc)).toBeCloseTo(1.0);
  });

  it("should compute correct score for orthogonal vectors", () => {
    const query = [new Float32Array([1, 0, 0])];
    const doc = [new Float32Array([0, 1, 0])];
    expect(maxsim(query, doc)).toBeCloseTo(0.0);
  });

  it("should sum max similarities for multi-token query", () => {
    // Query with 2 tokens, document with 2 tokens
    // Token 1: [1, 0] matches [1, 0] with score 1.0
    // Token 2: [0, 1] matches [0, 1] with score 1.0
    // Total: 2.0
    const query = [new Float32Array([1, 0]), new Float32Array([0, 1])];
    const doc = [new Float32Array([1, 0]), new Float32Array([0, 1])];
    expect(maxsim(query, doc)).toBeCloseTo(2.0);
  });

  it("should find max similarity across document tokens", () => {
    // Query: [[1, 0]]
    // Document: [[0, 1], [1, 0]] - second token should match
    const query = [new Float32Array([1, 0])];
    const doc = [new Float32Array([0, 1]), new Float32Array([1, 0])];
    expect(maxsim(query, doc)).toBeCloseTo(1.0);
  });
});

describe("maxsimDocuments", () => {
  it("should compute scores for multiple documents", () => {
    const query = [new Float32Array([1, 0])];
    const docs = [
      [new Float32Array([1, 0])],
      [new Float32Array([0, 1])],
      [new Float32Array([0.5, 0.5])],
    ];

    const scores = maxsimDocuments(query, docs);

    expect(scores).toHaveLength(3);
    expect(scores[0]).toBeCloseTo(1.0);
    expect(scores[1]).toBeCloseTo(0.0);
    expect(scores[2]).toBeCloseTo(0.5);
  });
});

describe("maxsimBatch", () => {
  it("should compute scores for query x document matrix", () => {
    const queries = [[new Float32Array([1, 0])], [new Float32Array([0, 1])]];

    const docs = [[new Float32Array([1, 0])], [new Float32Array([0, 1])]];

    const scores = maxsimBatch(queries, docs);

    // Expected: [q0d0, q0d1, q1d0, q1d1] = [1.0, 0.0, 0.0, 1.0]
    expect(scores).toHaveLength(4);
    expect(scores[0]).toBeCloseTo(1.0); // q0 vs d0
    expect(scores[1]).toBeCloseTo(0.0); // q0 vs d1
    expect(scores[2]).toBeCloseTo(0.0); // q1 vs d0
    expect(scores[3]).toBeCloseTo(1.0); // q1 vs d1
  });
});
