import type { ExtractResult, SIEClient } from "@superlinked/sie-sdk";
import { describe, expect, it, vi } from "vitest";
import { EvidenceOverflowError, decide, evidencePacket } from "../src/decision";
import { SourceDocument, ocrPage } from "../src/documents";
import { type SourceChunk, rerankChunks } from "../src/retrieval";

const chunks: SourceChunk[] = [
  { chunkId: "one", text: "The projector passed.", sources: [] },
  { chunkId: "two", text: "The microphone failed.", sources: [] },
];
const extraction = (data: Record<string, unknown>): ExtractResult => ({
  entities: [],
  relations: [],
  classifications: [],
  objects: [],
  data,
});
const choices = {
  confirmed: "Passed its check",
  failed: "Failed its check",
  unknown: "No result is given",
};
const options = {
  model: "decision",
  choices,
  instructions: "Did the projector pass?",
  maxChars: 1000,
};

it("keeps URL-looking source text literal and hashes the exact Unicode bytes", () => {
  const source = new SourceDocument({
    text: "https://example.org/é",
    sourceId: "DOC",
    filename: "note.txt",
    pageNumber: 2,
  });
  expect(source.toDataInput()).toEqual({ type: "text", text: "https://example.org/é" });
  expect(source.metadata).toMatchObject({ sourceId: "DOC", filename: "note.txt", pageNumber: 2 });
  expect(source.metadata.contentSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(source.metadata.contentSha256).not.toBe(
    new SourceDocument({ text: "https://example.org/e", sourceId: "DOC", filename: "note.txt" })
      .metadata.contentSha256,
  );
});

it.each(["../note.txt", "folder/note.txt", "folder\\note.txt", ".", ".."])(
  "rejects path-bearing filename %s",
  (filename) => {
    expect(() => new SourceDocument({ text: "text", sourceId: "DOC", filename })).toThrow();
  },
);

it("rejects blank sources and invalid page numbers", () => {
  expect(() => new SourceDocument({ text: " ", sourceId: "DOC", filename: "note.txt" })).toThrow();
  for (const pageNumber of [0, -1, 1.5, Number.NaN])
    expect(
      () => new SourceDocument({ text: "text", sourceId: "DOC", filename: "note.txt", pageNumber }),
    ).toThrow();
});

it("OCR preserves page provenance and rejects malformed or empty output", async () => {
  const extract = vi
    .fn()
    .mockResolvedValue({ ...extraction({}), entities: [{ text: "Step-free access confirmed" }] });
  const client = { extract } as unknown as SIEClient;
  const source = await ocrPage(client, {
    model: "ocr",
    image: new Uint8Array([1]),
    sourceId: "ACCESS",
    filename: "page.png",
    pageNumber: 3,
  });
  expect(source.text).toBe("Step-free access confirmed");
  expect(source.metadata.pageNumber).toBe(3);
  extract.mockResolvedValue({ ...extraction({}), entities: [{ text: " " }] });
  await expect(
    ocrPage(client, {
      model: "ocr",
      image: new Uint8Array([1]),
      sourceId: "ACCESS",
      filename: "page.png",
      pageNumber: 1,
    }),
  ).rejects.toThrow();
  await expect(
    ocrPage(client, {
      model: "ocr",
      image: new Uint8Array([1, 2]),
      sourceId: "ACCESS",
      filename: "page.png",
      pageNumber: 1,
      maxBytes: 1,
    }),
  ).rejects.toThrow();
});

describe("reranking", () => {
  it("orders complete finite scores while preserving the original source objects", async () => {
    const score = vi.fn().mockResolvedValue({
      scores: [
        { itemId: "two", score: 0.2 },
        { itemId: "one", score: 0.9 },
      ],
    });
    const result = await rerankChunks({ score } as unknown as SIEClient, "projector", chunks, {
      model: "rank",
      topK: 1,
    });
    expect(result).toEqual([{ chunk: chunks[0], score: 0.9 }]);
    expect(result[0]?.chunk).toBe(chunks[0]);
  });
  it.each([
    [{ itemId: "one", score: 0.4 }],
    [
      { itemId: "one", score: 0.4 },
      { itemId: "one", score: 0.5 },
    ],
    [
      { itemId: "one", score: 0.4 },
      { itemId: "unknown", score: 0.5 },
    ],
    [
      { itemId: "one", score: 0.4 },
      { itemId: "two", score: Number.NaN },
    ],
  ])("rejects invalid score identities or values", async (scores) => {
    const client = { score: vi.fn().mockResolvedValue({ scores }) } as unknown as SIEClient;
    await expect(rerankChunks(client, "projector", chunks, { model: "rank" })).rejects.toThrow();
  });
  it("returns an empty result without inference for an empty candidate set", async () => {
    const score = vi.fn();
    expect(
      await rerankChunks({ score } as unknown as SIEClient, "query", [], { model: "rank" }),
    ).toEqual([]);
    expect(score).not.toHaveBeenCalled();
  });
});

it("renders exact bounded evidence without truncation", () => {
  const packet = evidencePacket("Did it pass?", chunks, 1000);
  expect(packet).toBe(
    "Question: Did it pass?\n[one] The projector passed.\n[two] The microphone failed.",
  );
  expect(() => evidencePacket("Did it pass?", chunks, 10)).toThrow(EvidenceOverflowError);
});

it("sends the displayed evidence and preserves the server overflow guard", async () => {
  const extract = vi.fn().mockResolvedValue(
    extraction({
      action: {
        choice: "confirmed",
        probabilities: { confirmed: 0.8, failed: 0.1, unknown: 0.1 },
      },
    }),
  );
  const result = await decide({ extract } as unknown as SIEClient, "exact evidence", options);
  expect(result.choice).toBe("confirmed");
  expect(extract.mock.calls[0]?.[1]).toEqual({ id: "decision", text: "exact evidence" });
  expect(extract.mock.calls[0]?.[2]).toMatchObject({
    adapterOptions: { overflow_policy: "error" },
    waitForCapacity: true,
  });
});

it.each([
  { choice: "confirmed", probabilities: { confirmed: 0.8, failed: 0.1 } },
  { choice: "failed", probabilities: { confirmed: 0.8, failed: 0.1, unknown: 0.1 } },
  { choice: "confirmed", probabilities: { confirmed: 0.8, failed: 0.1, unknown: 0.4 } },
  { choice: "confirmed", probabilities: { confirmed: Number.NaN, failed: 0.1, unknown: 0.1 } },
  { choice: "confirmed", probabilities: { confirmed: 1.1, failed: -0.1, unknown: 0 } },
])("rejects malformed decision probabilities", async (answer) => {
  const client = {
    extract: vi.fn().mockResolvedValue(extraction({ action: answer })),
  } as unknown as SIEClient;
  await expect(decide(client, "evidence", options)).rejects.toThrow();
});

it("requires an unknown/review option and fails overflow before sending", async () => {
  const extract = vi.fn();
  const client = { extract } as unknown as SIEClient;
  await expect(
    decide(client, "evidence", { ...options, choices: { yes: "yes", no: "no" } }),
  ).rejects.toThrow();
  await expect(decide(client, "evidence", { ...options, maxChars: 3 })).rejects.toThrow(
    EvidenceOverflowError,
  );
  expect(extract).not.toHaveBeenCalled();
});
