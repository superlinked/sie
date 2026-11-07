import type { SIEClient } from "@superlinked/sie-sdk";
import type { SourceMetadata } from "./documents";

export interface SourceChunk {
  chunkId: string;
  documentId?: string;
  datasetId?: string;
  text: string;
  sources: readonly SourceMetadata[];
}

export interface RankedChunk {
  chunk: SourceChunk;
  score: number;
}

export async function rerankChunks(
  client: SIEClient,
  query: string,
  chunks: readonly SourceChunk[],
  options: { model: string; topK?: number },
): Promise<RankedChunk[]> {
  const topK = options.topK ?? 3;
  if (!query.trim() || !Number.isSafeInteger(topK) || topK < 1)
    throw new Error("query and positive topK are required.");
  const ids = new Set(chunks.map((chunk) => chunk.chunkId));
  if (ids.size !== chunks.length || chunks.some((chunk) => !chunk.chunkId || !chunk.text.trim()))
    throw new Error("Candidates need unique nonempty identifiers and text.");
  if (!chunks.length) return [];
  const reply = await client.score(
    options.model,
    { id: "query", text: query },
    chunks.map((chunk) => ({ id: chunk.chunkId, text: chunk.text })),
    { waitForCapacity: true },
  );
  if (!Array.isArray(reply.scores)) throw new Error("Reranker returned no score list.");
  const scores = new Map<string, number>();
  for (const entry of reply.scores) {
    if (!entry || !ids.has(entry.itemId) || scores.has(entry.itemId))
      throw new Error("Reranker returned unknown or duplicate item identifiers.");
    if (typeof entry.score !== "number" || !Number.isFinite(entry.score))
      throw new Error("Reranker scores must be finite numbers.");
    scores.set(entry.itemId, entry.score);
  }
  if (scores.size !== chunks.length) throw new Error("Reranker did not score every candidate.");
  return chunks
    .map((chunk) => ({ chunk, score: scores.get(chunk.chunkId) as number }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
