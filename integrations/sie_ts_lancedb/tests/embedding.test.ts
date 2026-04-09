/**
 * Tests for SIE LanceDB embedding and reranker integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  SIEEmbeddingFunction,
  type SIEEmbeddingFunctionOptions,
  SIEReranker,
  type SIERerankerOptions,
} from "../src/index.js";

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", () => {
  const mockClient = {
    encode: vi.fn(),
    score: vi.fn(),
    getModel: vi.fn(),
    close: vi.fn(),
  };

  return {
    SIEClient: vi.fn().mockImplementation(() => mockClient),
    toNumberArray: (arr: Float32Array | Int32Array | number[]) => Array.from(arr),
  };
});

describe("SIEEmbeddingFunction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const func = new SIEEmbeddingFunction();
    expect(func).toBeInstanceOf(SIEEmbeddingFunction);
  });

  it("creates with custom options", () => {
    const options: SIEEmbeddingFunctionOptions = {
      baseUrl: "http://custom:9090",
      model: "custom-model",
      instruction: "Represent:",
      outputDtype: "float16",
      gpu: "a100",
      timeout: 60000,
    };
    const func = new SIEEmbeddingFunction(options);
    expect(func).toBeInstanceOf(SIEEmbeddingFunction);
  });

  it("returns empty array for empty input", async () => {
    const func = new SIEEmbeddingFunction();
    const result = await func.generateEmbeddings([]);
    expect(result).toEqual([]);
  });

  it("generates dense embeddings for texts", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { dense: new Float32Array([0.5, 0.25, 0.75]) },
        { dense: new Float32Array([1.0, 0.5, 0.25]) },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      getModel: vi.fn(),
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "BAAI/bge-m3" });
    const embeddings = await func.generateEmbeddings(["Hello world", "Goodbye world"]);

    expect(embeddings).toHaveLength(2);
    expect(embeddings[0]).toEqual([0.5, 0.25, 0.75]);
    expect(embeddings[1]).toEqual([1.0, 0.5, 0.25]);
  });

  it("calls encode with correct parameters", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ dense: new Float32Array([0.5]) }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      getModel: vi.fn(),
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({
      model: "test-model",
      instruction: "Represent this:",
      outputDtype: "float16",
    });

    await func.generateEmbeddings(["test text"]);

    expect(mockEncode).toHaveBeenCalledWith("test-model", [{ text: "test text" }], {
      outputTypes: ["dense"],
      instruction: "Represent this:",
      outputDtype: "float16",
    });
  });

  it("embedQuery passes isQuery: true", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ dense: new Float32Array([0.5, 0.25]) }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      getModel: vi.fn(),
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "test-model" });
    const result = await func.embedQuery("search text");

    expect(result).toEqual([0.5, 0.25]);
    expect(mockEncode).toHaveBeenCalledWith(
      "test-model",
      [{ text: "search text" }],
      expect.objectContaining({ isQuery: true }),
    );
  });

  it("embedDocuments does not pass isQuery", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ dense: new Float32Array([0.5]) }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      getModel: vi.fn(),
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "test-model" });
    await func.embedDocuments(["doc text"]);

    const callArgs = mockEncode.mock.calls[0]?.[2];
    expect(callArgs?.isQuery).toBeUndefined();
  });

  it("throws error when dense embedding is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ sparse: {} }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      getModel: vi.fn(),
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction();
    await expect(func.generateEmbeddings(["test"])).rejects.toThrow(
      "Encode result missing dense embedding",
    );
  });

  it("ndims queries server metadata", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockGetModel = vi.fn().mockResolvedValue(
      { name: "BAAI/bge-m3", dims: { dense: 1024 }, loaded: true, inputs: ["text"], outputs: ["dense"] },
    );
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: vi.fn(),
      getModel: mockGetModel,
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "BAAI/bge-m3" });
    const dims = await func.ndims();

    expect(dims).toBe(1024);
    expect(mockGetModel).toHaveBeenCalledOnce();
  });

  it("ndims caches after first call", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockGetModel = vi.fn().mockResolvedValue(
      { name: "test-model", dims: { dense: 384 }, loaded: true, inputs: ["text"], outputs: ["dense"] },
    );
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: vi.fn(),
      getModel: mockGetModel,
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "test-model" });
    await func.ndims();
    await func.ndims();

    expect(mockGetModel).toHaveBeenCalledOnce();
  });

  it("ndims throws for model without dense dims", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockGetModel = vi.fn().mockResolvedValue(
      { name: "multivec-only", dims: { multivector: 128 }, loaded: true, inputs: ["text"], outputs: ["multivector"] },
    );
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: vi.fn(),
      getModel: mockGetModel,
      close: vi.fn(),
    }));

    const func = new SIEEmbeddingFunction({ model: "multivec-only" });
    await expect(func.ndims()).rejects.toThrow("does not support dense");
  });
});

describe("SIEReranker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const reranker = new SIEReranker();
    expect(reranker).toBeInstanceOf(SIEReranker);
  });

  it("creates with custom options", () => {
    const options: SIERerankerOptions = {
      baseUrl: "http://custom:9090",
      model: "custom-reranker",
      column: "content",
      gpu: "a100",
      timeout: 60000,
    };
    const reranker = new SIEReranker(options);
    expect(reranker).toBeInstanceOf(SIEReranker);
  });
});
