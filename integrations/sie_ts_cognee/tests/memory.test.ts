import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Cognee, CogneeSearchResponse } from "@cognee/cognee-ts";
import { afterEach, expect, it, vi } from "vitest";
import { prepareStorage } from "../src/config";
import { SourceDocument } from "../src/documents";
import { SourceMemory } from "../src/memory";

const roots: string[] = [];
const datasetId = "dataset-id";
const documentId = "document-id";
function setup() {
  const storageRoot = mkdtempSync(join(tmpdir(), "sie-cognee-memory-"));
  roots.push(storageRoot);
  prepareStorage({
    llm: { baseUrl: "http://localhost:8080" },
    embedding: { baseUrl: "http://localhost:8081" },
    llmModel: "model",
    embeddingModel: "embed",
    embeddingDimensions: 4,
    storageRoot,
  });
  const add = vi.fn().mockResolvedValue({
    added: [{ id: documentId }],
    addedCount: 1,
    deduplicated: [],
    deduplicatedCount: 0,
  });
  const response: CogneeSearchResponse = {
    search_type: "CHUNKS",
    result: {
      kind: "Items",
      data: [
        {
          id: "chunk-id",
          score: 0.8,
          payload: {
            id: "chunk-id",
            type: "DocumentChunk",
            text: "Literal evidence",
            document_id: documentId,
            dataset_id: datasetId,
          },
        },
      ],
    },
    datasets: [datasetId],
    context: null,
    graphs: null,
    diagnostics: null,
    only_context: false,
    use_combined_context: false,
    verbose: false,
  };
  const search = vi.fn().mockResolvedValue(response);
  const cognee = {
    add,
    search,
    cognify: vi.fn().mockResolvedValue({ chunks: 1 }),
    users: { resetPipelineRunStatus: vi.fn().mockResolvedValue(undefined) },
    datasets: { list: vi.fn().mockResolvedValue([{ id: datasetId, name: "sources" }]) },
  } as unknown as Cognee;
  return {
    storageRoot,
    add,
    search,
    response,
    cognee,
    memory: new SourceMemory(cognee, { dataset: "sources", storageRoot }),
  };
}
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

it("uses returned document and dataset identities, preserving multiple sources for deduplicated text", async () => {
  const fixture = setup();
  const first = new SourceDocument({ text: "Literal evidence", sourceId: "A", filename: "a.txt" });
  const second = new SourceDocument({ text: "Literal evidence", sourceId: "B", filename: "b.txt" });
  await fixture.memory.add(first);
  fixture.add.mockResolvedValue({
    added: [],
    deduplicated: [{ id: documentId }],
    deduplicatedCount: 1,
  });
  expect(await fixture.memory.add(second)).toEqual({ documentId, deduplicated: true });
  const reopened = new SourceMemory(fixture.cognee, {
    dataset: "sources",
    storageRoot: fixture.storageRoot,
  });
  const chunks = await reopened.retrieve("query");
  expect(chunks[0]).toEqual({
    chunkId: "chunk-id",
    documentId,
    datasetId,
    text: "Literal evidence",
    sources: [first.metadata, second.metadata],
  });
  expect(fixture.search).toHaveBeenCalledWith("query", {
    searchType: "CHUNKS",
    datasets: ["sources"],
    topK: 5,
    saveInteraction: false,
  });
});

it("refuses source attribution for an unknown document even when its text looks identical", async () => {
  const fixture = setup();
  await fixture.memory.add(
    new SourceDocument({ text: "Literal evidence", sourceId: "A", filename: "a.txt" }),
  );
  if (fixture.response.result.kind !== "Items") throw new Error("fixture");
  const item = fixture.response.result.data[0];
  if (!item) throw new Error("fixture");
  item.payload.document_id = "other-document";
  await expect(fixture.memory.retrieve("query")).rejects.toThrow(/provenance is missing/);
});

it("rejects generated completions and cross-dataset chunks", async () => {
  const fixture = setup();
  await fixture.memory.add(
    new SourceDocument({ text: "Literal evidence", sourceId: "A", filename: "a.txt" }),
  );
  fixture.search.mockResolvedValue({
    ...fixture.response,
    result: { kind: "Text", data: "plausible answer" },
  });
  await expect(fixture.memory.retrieve("query")).rejects.toThrow(/source chunks/);
  fixture.search.mockResolvedValue({ ...fixture.response, datasets: ["another-dataset"] });
  await expect(fixture.memory.retrieve("query")).rejects.toThrow(/requested dataset/);
});

it("does not manufacture source state on a query before ingestion", async () => {
  const fixture = setup();
  await expect(fixture.memory.retrieve("query")).rejects.toThrow(/provenance is missing/);
  expect(fixture.search).not.toHaveBeenCalled();
});
