/**
 * Tests for SIE LangChain reranker integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEReranker } from "../src/index.js";

function asConstructor<T extends object>(instance: T): () => T {
  return function constructorMock() {
    return instance;
  };
}

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@superlinked/sie-sdk")>();
  const mockClient = {
    score: vi.fn(),
    close: vi.fn(),
  };

  return {
    ...actual,
    SIEClient: vi.fn().mockImplementation(asConstructor(mockClient)),
  };
});

/**
 * A score mock with the SIE server's response contract: entries sorted by
 * relevance, each `itemId` echoing the item's `id` or `item-<index>` without one.
 */
function serverScore(relevance: Record<string, number>) {
  return vi.fn(async (model: string, _query: unknown, items: { id?: string; text?: string }[]) => ({
    model,
    scores: items
      .map((item, index) => ({
        itemId: item.id ?? `item-${index}`,
        score: relevance[item.text ?? ""] ?? 0,
      }))
      .sort((a, b) => b.score - a.score)
      .map((entry, rank) => ({ ...entry, rank })),
  }));
}

describe("SIEReranker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const reranker = new SIEReranker();
    expect(reranker).toBeInstanceOf(SIEReranker);
  });

  it("creates with custom options", () => {
    const reranker = new SIEReranker({
      baseUrl: "http://custom:9000",
      model: "custom-reranker",
      topK: 5,
      gpu: "a100-80gb",
      timeout: 60000,
    });
    expect(reranker).toBeInstanceOf(SIEReranker);
  });

  it("compressDocuments returns empty array for empty input", async () => {
    const reranker = new SIEReranker();
    const result = await reranker.compressDocuments([], "test query");
    expect(result).toEqual([]);
  });

  it("compressDocuments reranks documents with scores", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "1", score: 0.95, rank: 0 },
        { itemId: "0", score: 0.72, rank: 1 },
        { itemId: "2", score: 0.31, rank: 2 },
      ],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      asConstructor({
        score: mockScore,
        close: vi.fn(),
      }),
    );

    const documents = [
      { pageContent: "First document", metadata: { source: "a" } },
      { pageContent: "Second document", metadata: { source: "b" } },
      { pageContent: "Third document", metadata: { source: "c" } },
    ];

    const reranker = new SIEReranker({ model: "test-reranker" });
    const result = await reranker.compressDocuments(documents, "search query");

    expect(mockScore).toHaveBeenCalledWith("test-reranker", { text: "search query" }, [
      { id: "0", text: "First document" },
      { id: "1", text: "Second document" },
      { id: "2", text: "Third document" },
    ]);

    expect(result).toHaveLength(3);
    // Sorted by score descending (server returns sorted)
    expect(result[0].pageContent).toBe("Second document");
    expect(result[0].metadata.relevance_score).toBe(0.95);
    expect(result[0].metadata.source).toBe("b");

    expect(result[1].pageContent).toBe("First document");
    expect(result[1].metadata.relevance_score).toBe(0.72);

    expect(result[2].pageContent).toBe("Third document");
    expect(result[2].metadata.relevance_score).toBe(0.31);
  });

  it("compressDocuments maps the server's item ids back to their documents", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = serverScore({ "Doc A": 0.2, "Doc B": 0.9, "Doc C": 0.5 });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      asConstructor({
        score: mockScore,
        close: vi.fn(),
      }),
    );

    const documents = [
      { pageContent: "Doc A", metadata: {} },
      { pageContent: "Doc B", metadata: {} },
      { pageContent: "Doc C", metadata: {} },
    ];

    const reranker = new SIEReranker();
    const result = await reranker.compressDocuments(documents, "query");

    expect(result.map((doc) => doc.pageContent)).toEqual(["Doc B", "Doc C", "Doc A"]);
    expect(result.map((doc) => doc.metadata.relevance_score)).toEqual([0.9, 0.5, 0.2]);
  });

  it("compressDocuments skips score entries whose itemId was not sent", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "item-0", score: 0.99, rank: 0 },
        { itemId: "01", score: 0.98, rank: 1 },
        { itemId: "1", score: 0.5, rank: 2 },
        { itemId: "1", score: 0.4, rank: 3 },
        { itemId: "7", score: 0.3, rank: 4 },
      ],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      asConstructor({
        score: mockScore,
        close: vi.fn(),
      }),
    );

    const documents = [
      { pageContent: "Doc A", metadata: {} },
      { pageContent: "Doc B", metadata: {} },
    ];

    const reranker = new SIEReranker();
    const result = await reranker.compressDocuments(documents, "query");

    expect(result.map((doc) => [doc.pageContent, doc.metadata.relevance_score])).toEqual([
      ["Doc B", 0.5],
    ]);
  });

  it("compressDocuments applies topK client-side", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "1", score: 0.95, rank: 0 },
        { itemId: "0", score: 0.72, rank: 1 },
        { itemId: "2", score: 0.31, rank: 2 },
      ],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      asConstructor({
        score: mockScore,
        close: vi.fn(),
      }),
    );

    const documents = [
      { pageContent: "Doc A", metadata: {} },
      { pageContent: "Doc B", metadata: {} },
      { pageContent: "Doc C", metadata: {} },
    ];

    const reranker = new SIEReranker({ topK: 1 });
    const result = await reranker.compressDocuments(documents, "query");

    expect(result).toHaveLength(1);
    expect(result[0].pageContent).toBe("Doc B");
    expect(result[0].metadata.relevance_score).toBe(0.95);
  });

  it("preserves document id field", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockScore = vi.fn().mockResolvedValue({
      scores: [{ itemId: "0", score: 0.9, rank: 0 }],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      asConstructor({
        score: mockScore,
        close: vi.fn(),
      }),
    );

    const documents = [{ pageContent: "Doc", metadata: {}, id: "doc-123" }];

    const reranker = new SIEReranker();
    const result = await reranker.compressDocuments(documents, "query");

    expect(result[0].id).toBe("doc-123");
  });
});
