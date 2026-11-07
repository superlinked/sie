import type { CogneeData, CogneeSearchResponse } from "@cognee/cognee-ts";
import { expect, it } from "vitest";
import { type TeachingRecord, authoredRule, learningEvidence, nodeSets } from "../learning";

const record: TeachingRecord = {
  format: "sie-cognee-teaching-v1",
  dataset: "learning",
  datasetId: "dataset",
  storageRoot: "/tmp/state",
  teachingSession: "teaching-session-789",
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

const fullRule =
  "For every future QuillDock-123 release, the reviewer must record approval code VIOLET-456 before publishing. This is the standing project rule and continues to apply in later sessions.";

it("requires the full authored rule, session linkage, and persisted document origin", () => {
  expect(authoredRule(record.project, record.expectedCode)).toBe(fullRule);
  const docs = [document("raw", ["user_sessions_from_cache"])];
  expect(
    learningEvidence(
      record,
      docs,
      response("raw", `Session ID: teaching-session-789. Answer: ${fullRule}`),
    ).sessionTransferVerified,
  ).toBe(true);
  expect(
    learningEvidence(
      record,
      docs,
      response(
        "raw",
        `Session ID: teaching-session-789. Answer: ${fullRule.replace("VIOLET-456", "OTHER-789")}`,
      ),
    ).sessionTransferVerified,
  ).toBe(false);
  expect(learningEvidence(record, docs, response("raw", fullRule)).sessionTransferVerified).toBe(
    false,
  );
});
it.each([
  ["identifiers alone", "QuillDock-123 requires VIOLET-456."],
  ["missing publication timing", fullRule.replace(" before publishing", "")],
  ["reversed publication timing", fullRule.replace("before publishing", "after publishing")],
  [
    "missing later-session instruction",
    fullRule.replace(
      " This is the standing project rule and continues to apply in later sessions.",
      "",
    ),
  ],
  [
    "contradictory later-session instruction",
    fullRule.replace("continues to apply", "does not apply"),
  ],
  ["negated recording instruction", fullRule.replace("must record", "must not record")],
])(
  "does not verify %s even with all project, code and session identifiers",
  (_description, text) => {
    expect(text).toContain(record.project);
    expect(text).toContain(record.expectedCode);
    for (const sets of [
      ["user_sessions_from_cache"],
      ["session_learnings", "session_learnings:teaching-session-789"],
    ]) {
      const evidence = learningEvidence(
        { ...record, stagesRun: ["persist_sessions", "distill_sessions"] },
        [document("derived", sets)],
        response("derived", `Session ID: teaching-session-789. ${text}`),
      );
      expect(evidence.sessionTransferVerified).toBe(false);
      expect(evidence.distilledLessonVerified).toBe(false);
      expect(evidence.transferredSessionChunks).toEqual([]);
      expect(evidence.generatedLessonChunks).toEqual([]);
    }
  },
);
it("distinguishes complete transferred rules from separately accepted complete generated lessons", () => {
  const docs = [
    document("lesson", ["session_learnings", "session_learnings:teaching-session-789"]),
  ];
  const result = response("lesson", fullRule);
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
