/**
 * Tests for SIE LlamaIndex reranker integration.
 */

import type { MessageContent, NodeWithScore } from "llamaindex";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIENodePostprocessor } from "../src/index.js";

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@superlinked/sie-sdk")>();
  const mockClient = {
    score: vi.fn(),
    close: vi.fn(),
  };

  return {
    ...actual,
    SIEClient: vi.fn().mockImplementation(() => mockClient),
  };
});

/**
 * Create a mock NodeWithScore for testing.
 */
function mockNodeWithScore(text: string, score?: number) {
  return {
    node: {
      getContent: () => text,
      toJSON: () => ({ text }),
    },
    score,
  };
}

describe("SIENodePostprocessor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const postprocessor = new SIENodePostprocessor();
    expect(postprocessor).toBeDefined();
    expect(postprocessor.modelName).toBe("jinaai/jina-reranker-v2-base-multilingual");
  });

  it("creates with custom options", () => {
    const postprocessor = new SIENodePostprocessor({
      baseUrl: "http://custom:9000",
      modelName: "custom-reranker",
      topN: 5,
      gpu: "a100-80gb",
      timeout: 60000,
    });
    expect(postprocessor.modelName).toBe("custom-reranker");
  });

  it("returns empty array for empty nodes", async () => {
    const postprocessor = new SIENodePostprocessor();
    const result = await postprocessor.postprocessNodes([], "test query");
    expect(result).toEqual([]);
  });

  it("returns nodes as-is when no query", async () => {
    const postprocessor = new SIENodePostprocessor();
    const nodes = [mockNodeWithScore("doc1", 0.5), mockNodeWithScore("doc2", 0.3)];

    const result = await postprocessor.postprocessNodes(nodes as unknown as NodeWithScore[]);
    expect(result).toBe(nodes);
  });

  it("reranks nodes with updated scores", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "1", score: 0.95, rank: 0 },
        { itemId: "0", score: 0.72, rank: 1 },
        { itemId: "2", score: 0.31, rank: 2 },
      ],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      score: mockScore,
      close: vi.fn(),
    }));

    const nodes = [
      mockNodeWithScore("First doc", 0.5),
      mockNodeWithScore("Second doc", 0.3),
      mockNodeWithScore("Third doc", 0.1),
    ];

    const postprocessor = new SIENodePostprocessor({ modelName: "test-reranker" });
    const result = await postprocessor.postprocessNodes(
      nodes as unknown as NodeWithScore[],
      "search query",
    );

    expect(mockScore).toHaveBeenCalledWith(
      "test-reranker",
      { text: "search query" },
      [{ text: "First doc" }, { text: "Second doc" }, { text: "Third doc" }],
    );

    expect(result).toHaveLength(3);
    // Sorted by score descending (server returns sorted)
    expect(result[0].node.getContent()).toBe("Second doc");
    expect(result[0].score).toBe(0.95);

    expect(result[1].node.getContent()).toBe("First doc");
    expect(result[1].score).toBe(0.72);

    expect(result[2].node.getContent()).toBe("Third doc");
    expect(result[2].score).toBe(0.31);
  });

  it("applies topN client-side", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "1", score: 0.95, rank: 0 },
        { itemId: "0", score: 0.72, rank: 1 },
      ],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      score: mockScore,
      close: vi.fn(),
    }));

    const nodes = [mockNodeWithScore("doc1"), mockNodeWithScore("doc2")];

    const postprocessor = new SIENodePostprocessor({ topN: 1 });
    const result = await postprocessor.postprocessNodes(
      nodes as unknown as NodeWithScore[],
      "query",
    );

    expect(result).toHaveLength(1);
    expect(result[0].node.getContent()).toBe("doc2");
    expect(result[0].score).toBe(0.95);
  });

  it("handles MessageContent array type", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [{ itemId: "0", score: 0.9, rank: 0 }],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      score: mockScore,
      close: vi.fn(),
    }));

    const nodes = [mockNodeWithScore("doc1")];

    const postprocessor = new SIENodePostprocessor();
    await postprocessor.postprocessNodes(
      nodes as unknown as NodeWithScore[],
      [
        { type: "text", text: "hello " },
        { type: "text", text: "world" },
      ] as unknown as MessageContent,
    );

    expect(mockScore).toHaveBeenCalledWith(
      expect.any(String),
      { text: "hello  world" },
      expect.anything(),
    );
  });
});
