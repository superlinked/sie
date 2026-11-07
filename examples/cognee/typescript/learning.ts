import type { CogneeData, CogneeSearchResponse } from "@cognee/cognee-ts";

export interface TeachingRecord {
  format: "sie-cognee-teaching-v1";
  dataset: string;
  datasetId: string;
  storageRoot: string;
  teachingSession: string;
  project: string;
  expectedCode: string;
  query: string;
  stagesRun: string[];
}

export function nodeSets(value: string | null): string[] {
  if (value === null) return [];
  const parsed: unknown = JSON.parse(value);
  if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== "string"))
    throw new Error("Invalid Cognee document node_set metadata.");
  return parsed;
}

export function learningEvidence(
  record: TeachingRecord,
  documents: CogneeData[],
  response: CogneeSearchResponse,
) {
  if (
    response.search_type !== "CHUNKS" ||
    response.result.kind !== "Items" ||
    response.datasets?.length !== 1 ||
    response.datasets[0] !== record.datasetId
  )
    throw new Error("Learning verification requires original chunks from the teaching dataset.");
  const byId = new Map(documents.map((document) => [document.id, document]));
  const transferred: { documentId: string; chunkId: string; text: string }[] = [];
  const distilled: { documentId: string; chunkId: string; text: string }[] = [];
  for (const item of response.result.data) {
    const payload = item.payload;
    if (
      !item.id ||
      payload.id !== item.id ||
      payload.type !== "DocumentChunk" ||
      payload.dataset_id !== record.datasetId ||
      typeof payload.document_id !== "string" ||
      typeof payload.text !== "string"
    )
      throw new Error("Malformed learning source chunk.");
    const document = byId.get(payload.document_id);
    if (!document) throw new Error("Retrieved learning chunk is not in the teaching dataset.");
    const sets = nodeSets(document.node_set);
    if (!payload.text.includes(record.project) || !payload.text.includes(record.expectedCode))
      continue;
    const match = { documentId: document.id, chunkId: item.id, text: payload.text };
    if (sets.includes("user_sessions_from_cache") && payload.text.includes(record.teachingSession))
      transferred.push(match);
    if (
      sets.includes("session_learnings") &&
      sets.includes(`session_learnings:${record.teachingSession}`)
    )
      distilled.push(match);
  }
  return {
    sessionTransferVerified: transferred.length > 0,
    distilledLessonVerified: record.stagesRun.includes("distill_sessions") && distilled.length > 0,
    transferredSessionChunks: transferred,
    generatedLessonChunks: distilled,
  };
}
