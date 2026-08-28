/**
 * Tests for SIE ChromaDB embedding integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  SIEEmbeddingFunction,
  type SIEEmbeddingFunctionOptions,
  SIESparseEmbeddingFunction,
  type SIESparseEmbeddingFunctionOptions,
} from "../src/index.js";

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@superlinked/sie-sdk")>();
  const mockClient = {
    encode: vi.fn(),
  };

  return {
    ...actual,
    SIEClient: vi.fn().mockImplementation(() => mockClient),
  };
});

describe("SIEEmbeddingFunction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const ef = new SIEEmbeddingFunction();
    expect(ef).toBeInstanceOf(SIEEmbeddingFunction);
  });

  it("creates with custom options", () => {
    const options: SIEEmbeddingFunctionOptions = {
      baseUrl: "http://custom:9090",
      model: "custom-model",
      gpu: "a100",
      timeout: 60000,
    };
    const ef = new SIEEmbeddingFunction(options);
    expect(ef).toBeInstanceOf(SIEEmbeddingFunction);
  });

  it("returns empty array for empty input", async () => {
    const ef = new SIEEmbeddingFunction();
    const result = await ef.generate([]);
    expect(result).toEqual([]);
  });

  it("generates embeddings for texts", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { dense: new Float32Array([0.5, 0.25, 0.75]) },
        { dense: new Float32Array([1.0, 0.5, 0.25]) },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIEEmbeddingFunction({
      baseUrl: "http://localhost:8080",
      model: "BAAI/bge-m3",
    });

    const texts = ["Hello world", "Goodbye world"];
    const embeddings = await ef.generate(texts);

    expect(embeddings).toHaveLength(2);
    expect(embeddings[0]).toEqual([0.5, 0.25, 0.75]);
    expect(embeddings[1]).toEqual([1.0, 0.5, 0.25]);
  });

  it("calls encode with correct parameters", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ dense: new Float32Array([0.5, 0.25, 0.75]) }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIEEmbeddingFunction({
      baseUrl: "http://localhost:8080",
      model: "test-model",
    });

    await ef.generate(["test text"]);

    expect(mockEncode).toHaveBeenCalledWith("test-model", [{ text: "test text" }], {
      outputTypes: ["dense"],
    });
  });

  it("throws error when dense embedding is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { sparse: { indices: new Int32Array([]), values: new Float32Array([]) } },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIEEmbeddingFunction();

    await expect(ef.generate(["test"])).rejects.toThrow("Encode result missing dense embedding");
  });
});

describe("SIESparseEmbeddingFunction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const ef = new SIESparseEmbeddingFunction();
    expect(ef).toBeInstanceOf(SIESparseEmbeddingFunction);
  });

  it("creates with custom options", () => {
    const options: SIESparseEmbeddingFunctionOptions = {
      baseUrl: "http://custom:9090",
      model: "custom-model",
      gpu: "l4",
      timeout: 120000,
    };
    const ef = new SIESparseEmbeddingFunction(options);
    expect(ef).toBeInstanceOf(SIESparseEmbeddingFunction);
  });

  it("returns empty array for empty input", async () => {
    const ef = new SIESparseEmbeddingFunction();
    const result = await ef.generate([]);
    expect(result).toEqual([]);
  });

  it("generates sparse embeddings for texts", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([
      {
        sparse: {
          indices: new Int32Array([1, 5, 10]),
          values: new Float32Array([0.5, 0.25, 0.125]),
        },
      },
      {
        sparse: {
          indices: new Int32Array([2, 8]),
          values: new Float32Array([0.75, 0.5]),
        },
      },
    ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIESparseEmbeddingFunction({
      baseUrl: "http://localhost:8080",
      model: "BAAI/bge-m3",
    });

    const texts = ["Hello world", "Goodbye world"];
    const embeddings = await ef.generate(texts);

    expect(embeddings).toHaveLength(2);
    expect(embeddings[0]).toEqual({
      indices: [1, 5, 10],
      values: [0.5, 0.25, 0.125],
    });
    expect(embeddings[1]).toEqual({
      indices: [2, 8],
      values: [0.75, 0.5],
    });
  });

  it("calls encode with correct parameters", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([
      {
        sparse: {
          indices: new Int32Array([1]),
          values: new Float32Array([0.5]),
        },
      },
    ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIESparseEmbeddingFunction({
      baseUrl: "http://localhost:8080",
      model: "test-model",
    });

    await ef.generate(["test text"]);

    expect(mockEncode).toHaveBeenCalledWith("test-model", [{ text: "test text" }], {
      outputTypes: ["sparse"],
    });
  });

  it("returns empty indices/values when sparse is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{ dense: new Float32Array([0.5]) }]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIESparseEmbeddingFunction();
    const result = await ef.generate(["test"]);

    expect(result[0]).toEqual({ indices: [], values: [] });
  });

  it("generates sparse embeddings as dict format", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([
      {
        sparse: {
          indices: new Int32Array([1, 5, 10]),
          values: new Float32Array([0.5, 0.25, 0.125]),
        },
      },
    ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIESparseEmbeddingFunction({
      baseUrl: "http://localhost:8080",
      model: "BAAI/bge-m3",
    });

    const texts = ["Hello world"];
    const dictEmbeddings = await ef.generateAsDict(texts);

    expect(dictEmbeddings).toHaveLength(1);
    expect(dictEmbeddings[0]).toEqual({
      1: 0.5,
      5: 0.25,
      10: 0.125,
    });
  });

  it("handles multiple texts in dict format", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([
      {
        sparse: {
          indices: new Int32Array([1, 5, 10]),
          values: new Float32Array([0.5, 0.25, 0.125]),
        },
      },
      {
        sparse: {
          indices: new Int32Array([2, 8]),
          values: new Float32Array([0.75, 0.5]),
        },
      },
    ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
    }));

    const ef = new SIESparseEmbeddingFunction();

    const texts = ["Hello", "World"];
    const dictEmbeddings = await ef.generateAsDict(texts);

    expect(dictEmbeddings).toHaveLength(2);
    expect(dictEmbeddings[0]).toEqual({ 1: 0.5, 5: 0.25, 10: 0.125 });
    expect(dictEmbeddings[1]).toEqual({ 2: 0.75, 8: 0.5 });
  });
});
