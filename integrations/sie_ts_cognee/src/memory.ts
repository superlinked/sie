import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  existsSync,
  lstatSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import type { Cognee, CogneeSearchResponse } from "@cognee/cognee-ts";
import { SourceDocument, type SourceMetadata } from "./documents";
import type { SourceChunk } from "./retrieval";

interface SourceManifest {
  format: "sie-cognee-sources-v1";
  dataset: string;
  datasetId: string;
  needsBuild: boolean;
  documents: Record<string, SourceMetadata[]>;
}

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function validateSource(value: unknown): value is SourceMetadata {
  if (
    !object(value) ||
    typeof value.contentSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.contentSha256)
  )
    return false;
  try {
    new SourceDocument({
      text: "validation",
      sourceId: value.sourceId as string,
      filename: value.filename as string,
      pageNumber: value.pageNumber === null ? undefined : (value.pageNumber as number),
    });
    return value.pageNumber === null || typeof value.pageNumber === "number";
  } catch {
    return false;
  }
}

export function normalizeChunks(
  response: CogneeSearchResponse,
  manifest: SourceManifest,
): SourceChunk[] {
  if (
    response.search_type !== "CHUNKS" ||
    response.result?.kind !== "Items" ||
    !Array.isArray(response.result.data)
  ) {
    throw new Error(
      "Expected Cognee source chunks; generated graph completions and session answers are not source evidence.",
    );
  }
  if (
    !Array.isArray(response.datasets) ||
    response.datasets.length !== 1 ||
    response.datasets[0] !== manifest.datasetId
  ) {
    throw new Error("Chunk response does not match the requested dataset.");
  }
  const seen = new Set<string>();
  return response.result.data.map((entry) => {
    const payload = entry.payload;
    if (
      !entry.id ||
      seen.has(entry.id) ||
      !object(payload) ||
      payload.type !== "DocumentChunk" ||
      payload.id !== entry.id ||
      payload.dataset_id !== manifest.datasetId ||
      typeof payload.document_id !== "string" ||
      typeof payload.text !== "string" ||
      !payload.text.trim()
    ) {
      throw new Error("Malformed or cross-dataset source chunk.");
    }
    const sources = Object.hasOwn(manifest.documents, payload.document_id)
      ? manifest.documents[payload.document_id]
      : undefined;
    if (!sources?.length)
      throw new Error(
        "Source provenance is missing. Ingest this document through SourceMemory before using it as evidence.",
      );
    seen.add(entry.id);
    return {
      chunkId: entry.id,
      documentId: payload.document_id,
      datasetId: manifest.datasetId,
      text: payload.text,
      sources: sources.map((source) => ({ ...source })),
    };
  });
}

/** Dataset-scoped source memory with an explicit document-ID provenance manifest. */
export class SourceMemory {
  private readonly manifestPath: string;
  readonly dataset: string;

  constructor(
    private readonly cognee: Cognee,
    options: { dataset: string; storageRoot: string },
  ) {
    if (!options.dataset?.trim()) throw new Error("dataset must be nonempty.");
    this.dataset = options.dataset;
    const root = resolve(options.storageRoot);
    let marker: unknown;
    try {
      marker = JSON.parse(readFileSync(join(root, ".sie-cognee.json"), "utf8"));
    } catch {
      throw new Error("Initialize Cognee with createCognee before opening source memory.");
    }
    if (!object(marker) || marker.format !== "sie-cognee-typescript-v1")
      throw new Error("Source memory requires its dedicated TypeScript storage directory.");
    const digest = createHash("sha256").update(this.dataset).digest("hex");
    this.manifestPath = join(root, `sources-${digest}.json`);
  }

  private readManifest(): SourceManifest | undefined {
    if (!existsSync(this.manifestPath)) return undefined;
    let value: unknown;
    try {
      const stat = lstatSync(this.manifestPath);
      if (stat.isSymbolicLink() || stat.size > 10_000_000) throw new Error("invalid manifest file");
      value = JSON.parse(readFileSync(this.manifestPath, "utf8"));
    } catch {
      throw new Error(
        "Source provenance manifest is invalid; restore it before using this memory.",
      );
    }
    if (
      !object(value) ||
      value.format !== "sie-cognee-sources-v1" ||
      value.dataset !== this.dataset ||
      typeof value.datasetId !== "string" ||
      !value.datasetId ||
      !object(value.documents) ||
      (value.needsBuild !== undefined && typeof value.needsBuild !== "boolean")
    )
      throw new Error("Source provenance manifest does not match this dataset.");
    for (const [id, sources] of Object.entries(value.documents)) {
      if (!id || !Array.isArray(sources) || !sources.length || !sources.every(validateSource))
        throw new Error("Source provenance manifest contains invalid document metadata.");
    }
    if (value.needsBuild === undefined) value.needsBuild = true;
    return value as unknown as SourceManifest;
  }

  private writeManifest(manifest: SourceManifest): void {
    const temporary = `${this.manifestPath}.${randomUUID()}.tmp`;
    try {
      writeFileSync(temporary, `${JSON.stringify(manifest, null, 2)}\n`, {
        flag: "wx",
        mode: 0o600,
      });
      renameSync(temporary, this.manifestPath);
    } finally {
      if (existsSync(temporary)) unlinkSync(temporary);
    }
  }

  private async withWriteLock<T>(operation: () => Promise<T>): Promise<T> {
    const lockPath = `${this.manifestPath}.lock`;
    let lock: number;
    try {
      lock = openSync(lockPath, "wx", 0o600);
    } catch {
      throw new Error(
        "Another source-memory writer is active. Retry after it finishes; remove a stale .lock only after confirming its process has stopped.",
      );
    }
    try {
      return await operation();
    } finally {
      closeSync(lock);
      unlinkSync(lockPath);
    }
  }

  /** Add literal text and record the returned document ID before graph extraction. */
  async add(source: SourceDocument): Promise<{ documentId: string; deduplicated: boolean }> {
    return this.withWriteLock(async () => {
      const previous = this.readManifest();
      const result = await this.cognee.add(source.toDataInput(), this.dataset);
      const records = [...result.added, ...result.deduplicated];
      if (records.length !== 1 || !records[0]?.id)
        throw new Error("Cognee did not return exactly one source document ID.");
      const matches = (await this.cognee.datasets.list()).filter(
        (dataset) => dataset.name === this.dataset,
      );
      if (matches.length !== 1 || !matches[0]?.id)
        throw new Error("Cannot identify the source dataset.");
      const datasetId = matches[0].id;
      if (previous && previous.datasetId !== datasetId)
        throw new Error("The dataset was replaced. Use a new namespace and rebuild source memory.");
      const manifest: SourceManifest = previous ?? {
        format: "sie-cognee-sources-v1",
        dataset: this.dataset,
        datasetId,
        needsBuild: true,
        documents: Object.create(null),
      };
      const documentId = records[0].id;
      const knownDocument = Object.hasOwn(manifest.documents, documentId);
      const existing = knownDocument ? (manifest.documents[documentId] ?? []) : [];
      if (
        !existing.some((metadata) => JSON.stringify(metadata) === JSON.stringify(source.metadata))
      )
        existing.push({ ...source.metadata });
      manifest.documents[documentId] = existing;
      manifest.needsBuild ||= !knownDocument || result.addedCount > 0;
      this.writeManifest(manifest);
      return { documentId, deduplicated: result.deduplicatedCount > 0 };
    });
  }

  async build(): Promise<Awaited<ReturnType<Cognee["cognify"]>>> {
    return this.withWriteLock(async () => {
      const manifest = this.readManifest();
      if (!manifest) throw new Error("Add a source document before building memory.");
      if (manifest.needsBuild) {
        // Cognee 0.2.0 caches whole pipeline runs; new documents need the actual `cognify` run reset.
        await this.cognee.users.resetPipelineRunStatus(manifest.datasetId, "cognify");
      }
      const result = await this.cognee.cognify(this.dataset, { summarization: false });
      if (manifest.needsBuild && result.alreadyCompleted)
        throw new Error(
          "New source documents were not indexed. Retry the memory build before querying them.",
        );
      manifest.needsBuild = false;
      this.writeManifest(manifest);
      return result;
    });
  }

  async retrieve(query: string, topK = 5): Promise<SourceChunk[]> {
    if (!query.trim() || !Number.isSafeInteger(topK) || topK < 1)
      throw new Error("query and positive topK are required.");
    const manifest = this.readManifest();
    if (!manifest)
      throw new Error("Source provenance is missing. Add documents before querying this memory.");
    const response = await this.cognee.search(query, {
      searchType: "CHUNKS",
      datasets: [this.dataset],
      topK,
      saveInteraction: false,
    });
    return normalizeChunks(response, manifest);
  }
}
