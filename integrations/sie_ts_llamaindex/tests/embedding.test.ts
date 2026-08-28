/**
 * Tests for SIE LlamaIndex embeddings integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEEmbedding, SIESparseEmbeddingFunction } from "../src/index.js";

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@superlinked/sie-sdk")>();
  const mockClient = {
    encode: vi.fn(),
    close: vi.fn(),
  };

  return {
    ...actual,
    SIEClient: vi.fn().mockImplementation(() => mockClient),
  };
});

// Mock llamaindex BaseEmbedding
vi.mock("llamaindex", () => {
  return {
    BaseEmbedding: class {
      embedBatchSize = 10;
    },
  };
});

describe("SIEEmbedding", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const embedding = new SIEEmbedding();
    expect(embedding).toBeInstanceOf(SIEEmbedding);
    expect(embedding.modelName).toBe("BAAI/bge-m3");
  });

  it("creates with custom options", () => {
    const embedding = new SIEEmbedding({
      baseUrl: "http://custom:9000",
      modelName: "custom-model",
      instruction: "Represent this document:",
      outputDtype: "float16",
      gpu: "a100-80gb",
      timeout: 60000,
      embedBatchSize: 50,
    });
    expect(embedding.modelName).toBe("custom-model");
    expect(embedding.embedBatchSize).toBe(50);
  });

  it("getTextEmbedding encodes with isQuery=false", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    // Use values exactly representable in Float32
    const mockEncode = vi.fn().mockResolvedValue({
      dense: new Float32Array([0.5, 0.25, 0.75]),
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embedding = new SIEEmbedding();
    const result = await embedding.getTextEmbedding("Document text");

    expect(mockEncode).toHaveBeenCalledWith(
      expect.any(String),
      { text: "Document text" },
      expect.objectContaining({
        outputTypes: ["dense"],
        isQuery: false,
      }),
    );

    expect(result).toEqual([0.5, 0.25, 0.75]);
  });

  it("getTextEmbeddings returns empty array for empty input", async () => {
    const embedding = new SIEEmbedding();
    const result = await embedding.getTextEmbeddings([]);
    expect(result).toEqual([]);
  });

  it("getTextEmbeddings encodes multiple texts", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { dense: new Float32Array([0.5, 0.25]) },
        { dense: new Float32Array([0.75, 0.125]) },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embedding = new SIEEmbedding();
    const result = await embedding.getTextEmbeddings(["Hello", "World"]);

    expect(mockEncode).toHaveBeenCalledWith(
      expect.any(String),
      [{ text: "Hello" }, { text: "World" }],
      expect.objectContaining({
        outputTypes: ["dense"],
        isQuery: false,
      }),
    );

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual([0.5, 0.25]);
    expect(result[1]).toEqual([0.75, 0.125]);
  });

  it("throws if dense is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue({});
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embedding = new SIEEmbedding();
    await expect(embedding.getTextEmbedding("test")).rejects.toThrow("missing dense embedding");
  });

  it("passes instruction and outputDtype to encode", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue({
      dense: new Float32Array([0.5, 0.25]),
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embedding = new SIEEmbedding({
      instruction: "Represent this for retrieval:",
      outputDtype: "int8",
    });
    await embedding.getTextEmbedding("test");

    expect(mockEncode).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({
        instruction: "Represent this for retrieval:",
        outputDtype: "int8",
      }),
    );
  });
});

describe("SIESparseEmbeddingFunction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const fn = new SIESparseEmbeddingFunction();
    expect(fn).toBeInstanceOf(SIESparseEmbeddingFunction);
  });

  it("encodeQueries returns empty arrays for empty input", async () => {
    const fn = new SIESparseEmbeddingFunction();
    const result = await fn.encodeQueries([]);
    expect(result).toEqual([[], []]);
  });

  it("encodeQueries encodes with isQuery=true", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { sparse: { indices: new Int32Array([1, 5]), values: new Float32Array([0.5, 0.25]) } },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const fn = new SIESparseEmbeddingFunction({ modelName: "test-model" });
    const [indices, values] = await fn.encodeQueries(["test query"]);

    expect(mockEncode).toHaveBeenCalledWith(
      "test-model",
      [{ text: "test query" }],
      expect.objectContaining({
        outputTypes: ["sparse"],
        isQuery: true,
      }),
    );

    expect(indices).toEqual([[1, 5]]);
    expect(values).toEqual([[0.5, 0.25]]);
  });

  it("encodeDocuments encodes with isQuery=false", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { sparse: { indices: new Int32Array([2, 4]), values: new Float32Array([0.5, 0.75]) } },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const fn = new SIESparseEmbeddingFunction();
    const [indices, values] = await fn.encodeDocuments(["test doc"]);

    expect(mockEncode).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({
        outputTypes: ["sparse"],
        isQuery: false,
      }),
    );

    expect(indices).toEqual([[2, 4]]);
    expect(values).toEqual([[0.5, 0.75]]);
  });

  it("returns empty arrays when sparse is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{}]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const fn = new SIESparseEmbeddingFunction();
    const [indices, values] = await fn.encodeDocuments(["test"]);

    expect(indices).toEqual([[]]);
    expect(values).toEqual([[]]);
  });

  it("handles multiple texts", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { sparse: { indices: new Int32Array([1]), values: new Float32Array([0.5]) } },
        { sparse: { indices: new Int32Array([2, 3]), values: new Float32Array([0.25, 0.75]) } },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const fn = new SIESparseEmbeddingFunction();
    const [indices, values] = await fn.encodeDocuments(["text1", "text2"]);

    expect(indices).toEqual([[1], [2, 3]]);
    expect(values).toEqual([[0.5], [0.25, 0.75]]);
  });
});
