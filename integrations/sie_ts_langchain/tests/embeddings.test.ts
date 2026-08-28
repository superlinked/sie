/**
 * Tests for SIE LangChain embeddings integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEEmbeddings, SIESparseEncoder } from "../src/index.js";

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

describe("SIEEmbeddings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const embeddings = new SIEEmbeddings();
    expect(embeddings).toBeInstanceOf(SIEEmbeddings);
  });

  it("creates with custom options", () => {
    const embeddings = new SIEEmbeddings({
      baseUrl: "http://custom:9000",
      model: "custom-model",
      instruction: "Represent this document:",
      outputDtype: "float16",
      gpu: "a100-80gb",
      timeout: 60000,
    });
    expect(embeddings).toBeInstanceOf(SIEEmbeddings);
  });

  it("embedDocuments returns empty array for empty input", async () => {
    const embeddings = new SIEEmbeddings();
    const result = await embeddings.embedDocuments([]);
    expect(result).toEqual([]);
  });

  it("embedDocuments encodes texts and returns dense vectors", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    // Use values exactly representable in Float32: 0.5, 0.25, 0.75
    const mockEncode = vi
      .fn()
      .mockResolvedValue([
        { dense: new Float32Array([0.5, 0.25, 0.75]) },
        { dense: new Float32Array([1.0, 2.0, 3.0]) },
      ]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embeddings = new SIEEmbeddings({ model: "test-model" });
    const result = await embeddings.embedDocuments(["Hello", "World"]);

    expect(mockEncode).toHaveBeenCalledWith(
      "test-model",
      [{ text: "Hello" }, { text: "World" }],
      expect.objectContaining({
        outputTypes: ["dense"],
        isQuery: false,
      }),
    );

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual([0.5, 0.25, 0.75]);
    expect(result[1]).toEqual([1.0, 2.0, 3.0]);
  });

  it("embedQuery encodes text with isQuery=true", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue({
      dense: new Float32Array([0.5, 0.25, 0.125]),
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embeddings = new SIEEmbeddings({ model: "test-model" });
    const result = await embeddings.embedQuery("What is this?");

    expect(mockEncode).toHaveBeenCalledWith(
      "test-model",
      { text: "What is this?" },
      expect.objectContaining({
        outputTypes: ["dense"],
        isQuery: true,
      }),
    );

    expect(result).toEqual([0.5, 0.25, 0.125]);
  });

  it("embedQuery throws if dense is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue({});
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const embeddings = new SIEEmbeddings();
    await expect(embeddings.embedQuery("test")).rejects.toThrow("missing dense embedding");
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

    const embeddings = new SIEEmbeddings({
      instruction: "Represent this for retrieval:",
      outputDtype: "int8",
    });
    await embeddings.embedQuery("test");

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

describe("SIESparseEncoder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const encoder = new SIESparseEncoder();
    expect(encoder).toBeInstanceOf(SIESparseEncoder);
  });

  it("encodeQueries returns empty array for empty input", async () => {
    const encoder = new SIESparseEncoder();
    const result = await encoder.encodeQueries([]);
    expect(result).toEqual([]);
  });

  it("encodeQueries encodes texts with isQuery=true", async () => {
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
      close: vi.fn(),
    }));

    const encoder = new SIESparseEncoder({ model: "test-model" });
    const result = await encoder.encodeQueries(["test query"]);

    expect(mockEncode).toHaveBeenCalledWith(
      "test-model",
      [{ text: "test query" }],
      expect.objectContaining({
        outputTypes: ["sparse"],
        isQuery: true,
      }),
    );

    expect(result).toHaveLength(1);
    expect(result[0]?.indices).toEqual([1, 5, 10]);
    expect(result[0]?.values).toEqual([0.5, 0.25, 0.125]);
  });

  it("encodeDocuments encodes texts with isQuery=false", async () => {
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

    const encoder = new SIESparseEncoder();
    const result = await encoder.encodeDocuments(["test doc"]);

    expect(mockEncode).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({
        outputTypes: ["sparse"],
        isQuery: false,
      }),
    );

    expect(result[0]).toEqual({
      indices: [2, 4],
      values: [0.5, 0.75],
    });
  });

  it("returns empty arrays when sparse is missing", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockEncode = vi.fn().mockResolvedValue([{}]);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      encode: mockEncode,
      close: vi.fn(),
    }));

    const encoder = new SIESparseEncoder();
    const result = await encoder.encodeDocuments(["test"]);

    expect(result[0]).toEqual({ indices: [], values: [] });
  });
});
