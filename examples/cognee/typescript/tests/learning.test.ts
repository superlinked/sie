import type { CogneeData, CogneeSearchResponse } from "@cognee/cognee-ts";
import { expect, it } from "vitest";
import { type TeachingRecord, learningEvidence, nodeSets } from "../learning";

const record: TeachingRecord = {
  format: "sie-cognee-teaching-v1",
  dataset: "learning",
  datasetId: "dataset",
  storageRoot: "/tmp/state",
  teachingSession: "session",
  project: "QuillDock-123",
  expectedCode: "VIOLET-456",
  query: "What code does QuillDock-123 need?",
  stagesRun: ["persist_sessions"],
};
const response = (documentId: string, text: string): CogneeSearchResponse => ({
  search_type: "CHUNKS",
  result: {
    kind: "Items",
    data: [
      {
        id: "chunk",
        score: 0.8,
        payload: {
          id: "chunk",
          type: "DocumentChunk",
          dataset_id: "dataset",
          document_id: documentId,
          text,
        },
      },
    ],
  },
  datasets: ["dataset"],
  context: null,
  graphs: null,
  diagnostics: null,
  only_context: false,
  use_combined_context: false,
  verbose: false,
});
const document = (id: string, sets: string[]): CogneeData =>
  ({ id, node_set: JSON.stringify(sets) }) as CogneeData;

it("requires exact unique guidance, session linkage, and persisted document origin", () => {
  const docs = [document("raw", ["user_sessions_from_cache"])];
  expect(
    learningEvidence(
      record,
      docs,
      response("raw", "Session ID: session. QuillDock-123 requires VIOLET-456."),
    ).sessionTransferVerified,
  ).toBe(true);
  expect(
    learningEvidence(
      record,
      docs,
      response("raw", "Session ID: session. QuillDock-123 requires an approval code."),
    ).sessionTransferVerified,
  ).toBe(false);
  expect(
    learningEvidence(record, docs, response("raw", "QuillDock-123 VIOLET-456."))
      .sessionTransferVerified,
  ).toBe(false);
});
it("distinguishes transferred sessions from separately accepted generated lessons", () => {
  const docs = [document("lesson", ["session_learnings", "session_learnings:session"])];
  const result = response("lesson", "For QuillDock-123, always record VIOLET-456.");
  expect(learningEvidence(record, docs, result).distilledLessonVerified).toBe(false);
  const evidence = learningEvidence({ ...record, stagesRun: ["distill_sessions"] }, docs, result);
  expect(evidence.distilledLessonVerified).toBe(true);
  expect(evidence.sessionTransferVerified).toBe(false);
});
it("rejects unknown document IDs and cross-dataset evidence", () => {
  expect(() =>
    learningEvidence(record, [], response("missing", "QuillDock-123 VIOLET-456")),
  ).toThrow();
  expect(() =>
    learningEvidence(record, [], { ...response("raw", "text"), datasets: ["different"] }),
  ).toThrow();
});
it("validates node-set metadata instead of inferring an origin from text", () => {
  expect(nodeSets(null)).toEqual([]);
  expect(() => nodeSets('"session_learnings"')).toThrow();
  expect(() => nodeSets("broken")).toThrow();
});
