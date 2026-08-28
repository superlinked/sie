/**
 * Tests for SIE LangChain extractor integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEExtractor } from "../src/index.js";

// Default empty extract result
const emptyExtractResult = { entities: [], relations: [], classifications: [], objects: [] };

// Mock the SIEClient
vi.mock("@superlinked/sie-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@superlinked/sie-sdk")>();
  const mockClient = {
    extract: vi.fn(),
    close: vi.fn(),
  };

  return {
    ...actual,
    SIEClient: vi.fn().mockImplementation(() => mockClient),
  };
});

describe("SIEExtractor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates with default options", () => {
    const extractor = new SIEExtractor();
    expect(extractor).toBeInstanceOf(SIEExtractor);
    expect(extractor.name).toBe("sie_extract");
  });

  it("creates with custom options", () => {
    const extractor = new SIEExtractor({
      baseUrl: "http://custom:9000",
      model: "custom-ner",
      labels: ["product", "date"],
      name: "custom_extractor",
      description: "Custom description",
      gpu: "a100-80gb",
      timeout: 60000,
    });
    expect(extractor.name).toBe("custom_extractor");
    expect(extractor.description).toBe("Custom description");
  });

  it("extracts and returns JSON with all types", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue({
      entities: [
        { text: "John Smith", label: "person", score: 0.98, start: 0, end: 10 },
        { text: "Acme Corp", label: "organization", score: 0.95, start: 20, end: 29 },
      ],
      relations: [],
      classifications: [],
      objects: [],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const extractor = new SIEExtractor({ model: "test-ner" });
    const result = await extractor._call("John Smith works at Acme Corp");

    expect(mockExtract).toHaveBeenCalledWith(
      "test-ner",
      { text: "John Smith works at Acme Corp" },
      { labels: ["person", "organization", "location"] },
    );

    const parsed = JSON.parse(result);
    expect(parsed.entities).toHaveLength(2);
    expect(parsed.relations).toEqual([]);
    expect(parsed.classifications).toEqual([]);
    expect(parsed.objects).toEqual([]);
    expect(parsed.entities[0]).toEqual({
      text: "John Smith",
      label: "person",
      score: 0.98,
      start: 0,
      end: 10,
    });
  });

  it("passes custom labels to extract", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue(emptyExtractResult);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const extractor = new SIEExtractor({
      labels: ["product", "date"],
    });
    await extractor._call("test text");

    expect(mockExtract).toHaveBeenCalledWith(expect.any(String), expect.anything(), {
      labels: ["product", "date"],
    });
  });

  it("passes threshold when specified", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue(emptyExtractResult);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const extractor = new SIEExtractor({
      threshold: 0.5,
    });
    await extractor._call("test text");

    expect(mockExtract).toHaveBeenCalledWith(expect.any(String), expect.anything(), {
      labels: ["person", "organization", "location"],
      threshold: 0.5,
    });
  });

  it("returns empty result for no extractions", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue(emptyExtractResult);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const extractor = new SIEExtractor();
    const result = await extractor._call("no entities here");

    const parsed = JSON.parse(result);
    expect(parsed.entities).toEqual([]);
    expect(parsed.relations).toEqual([]);
    expect(parsed.classifications).toEqual([]);
    expect(parsed.objects).toEqual([]);
  });

  it("omits start/end when not present", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue({
      entities: [{ text: "Test", label: "thing", score: 0.9 }],
      relations: [],
      classifications: [],
      objects: [],
    });
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const extractor = new SIEExtractor();
    const result = await extractor._call("test");
    const parsed = JSON.parse(result);

    expect(parsed.entities[0]).toEqual({ text: "Test", label: "thing", score: 0.9 });
    expect(parsed.entities[0].start).toBeUndefined();
  });
});
