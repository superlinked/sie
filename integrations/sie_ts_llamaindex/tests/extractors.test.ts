/**
 * Tests for SIE LlamaIndex extractor integration.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { createSIEExtractorTool } from "../src/index.js";

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

describe("createSIEExtractorTool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates tool with default options", () => {
    const tool = createSIEExtractorTool();
    expect(tool).toBeDefined();
    expect(tool.metadata.name).toBe("sie_extract");
    expect(tool.metadata.description).toContain("Extract structured information");
  });

  it("creates tool with custom options", () => {
    const tool = createSIEExtractorTool({
      name: "custom_extractor",
      description: "Custom description",
      labels: ["product", "date"],
    });
    expect(tool.metadata.name).toBe("custom_extractor");
    expect(tool.metadata.description).toBe("Custom description");
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

    const tool = createSIEExtractorTool({ modelName: "test-ner" });
    const result = await tool.call({ text: "John Smith works at Acme Corp" });

    expect(mockExtract).toHaveBeenCalledWith(
      "test-ner",
      { text: "John Smith works at Acme Corp" },
      { labels: ["person", "organization", "location"] },
    );

    const parsed = JSON.parse(result as string);
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

    const tool = createSIEExtractorTool({
      labels: ["product", "date"],
    });
    await tool.call({ text: "test text" });

    expect(mockExtract).toHaveBeenCalledWith(expect.any(String), expect.anything(), {
      labels: ["product", "date"],
    });
  });

  it("returns empty result for no extractions", async () => {
    const { SIEClient } = await import("@superlinked/sie-sdk");
    const mockExtract = vi.fn().mockResolvedValue(emptyExtractResult);
    (SIEClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      extract: mockExtract,
      close: vi.fn(),
    }));

    const tool = createSIEExtractorTool();
    const result = await tool.call({ text: "no entities here" });

    const parsed = JSON.parse(result as string);
    expect(parsed.entities).toEqual([]);
    expect(parsed.relations).toEqual([]);
  });

  it("includes label types in default description", () => {
    const tool = createSIEExtractorTool({
      labels: ["animal", "color"],
    });
    expect(tool.metadata.description).toContain("animal");
    expect(tool.metadata.description).toContain("color");
  });
});
