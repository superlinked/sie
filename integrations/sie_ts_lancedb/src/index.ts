/**
 * SIE embedding function and reranker for LanceDB.
 *
 * Provides LanceDB-native components that use SIE for inference:
 * - SIEEmbeddingFunction: Dense embeddings registered as "sie" in LanceDB's registry
 * - SIEReranker: Cross-encoder reranking for hybrid search pipelines
 *
 * @example
 * ```typescript
 * import { connect } from "@lancedb/lancedb";
 * import { getRegistry, LanceSchema } from "@lancedb/lancedb/embedding";
 * import { SIEEmbeddingFunction } from "@superlinked/sie-lancedb";
 *
 * const func = new SIEEmbeddingFunction({ model: "BAAI/bge-m3" });
 * const schema = LanceSchema({
 *   text: func.sourceField(),
 *   vector: func.vectorField({ dims: 1024 }),
 * });
 *
 * const db = await connect("/tmp/lancedb");
 * const table = await db.createEmptyTable("docs", schema);
 * await table.add([{ text: "Hello world" }]);  // auto-embeds
 *
 * const results = await table.search("hello").limit(5).toArray();
 * ```
 */

import {
  type DType,
  type EncodeOptions,
  type EncodeResult,
  type ModelInfo,
  type ScoreResult,
  SIEClient,
  type SIEClientOptions,
  toNumberArray,
} from "@superlinked/sie-sdk";

/**
 * Configuration options for SIEEmbeddingFunction.
 */
export interface SIEEmbeddingFunctionOptions {
  /** URL of the SIE server. @default "http://localhost:8080" */
  baseUrl?: string;
  /** Model name/ID to use for encoding. @default "BAAI/bge-m3" */
  model?: string;
  /** Optional pre-configured SIEClient instance. */
  client?: SIEClient;
  /** Instruction prefix for instruction-tuned models (e.g., E5). */
  instruction?: string;
  /** Output data type: "float32" (default), "float16", "int8", "binary". */
  outputDtype?: string;
  /** Target GPU type for routing (e.g., "l4", "a100-80gb"). */
  gpu?: string;
  /** Request timeout in milliseconds. @default 180000 */
  timeout?: number;
}

/**
 * Dense text embeddings via SIE for LanceDB.
 *
 * Implements LanceDB's EmbeddingFunction interface. Embeddings are computed
 * automatically when using `sourceField()` / `vectorField()` schema helpers.
 *
 * Use `ndims()` or pass `{ dims }` to `vectorField()` for schema definition.
 * `ndims()` queries the `/v1/models` metadata API (lightweight, no model loading).
 *
 * @example
 * ```typescript
 * import { SIEEmbeddingFunction } from "@superlinked/sie-lancedb";
 * import { LanceSchema } from "@lancedb/lancedb/embedding";
 *
 * const func = new SIEEmbeddingFunction({ model: "BAAI/bge-m3" });
 * const schema = LanceSchema({
 *   text: func.sourceField(),
 *   vector: func.vectorField({ dims: 1024 }),
 * });
 * ```
 */
export class SIEEmbeddingFunction {
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly instruction: string | undefined;
  private readonly outputDtype: string | undefined;
  private readonly clientOptions: SIEClientOptions;
  private _client: SIEClient | undefined;
  private _ndims: number | undefined;

  constructor(options: SIEEmbeddingFunctionOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      model = "BAAI/bge-m3",
      client,
      instruction,
      outputDtype,
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.model = model;
    this._client = client;
    this.instruction = instruction;
    this.outputDtype = outputDtype;
    this.clientOptions = { timeout, gpu };
  }

  private get client(): SIEClient {
    if (!this._client) {
      this._client = new SIEClient(this.baseUrl, this.clientOptions);
    }
    return this._client;
  }

  /**
   * Return embedding dimensionality from /v1/models metadata.
   *
   * Queries the SIE server's model config (lightweight GET, no model
   * loading or inference). Cached after first call.
   */
  async ndims(): Promise<number> {
    if (this._ndims === undefined) {
      const info = await this.client.getModel(this.model);
      if (!info?.dims?.dense) {
        throw new Error(`Model "${this.model}" does not support dense embeddings`);
      }
      this._ndims = info.dims.dense;
    }
    return this._ndims;
  }

  /**
   * Generate dense embeddings for a list of texts.
   *
   * @param texts - Texts to embed.
   * @returns Array of embedding vectors.
   */
  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["dense"],
      instruction: this.instruction,
      outputDtype: this.outputDtype as DType | undefined,
    };

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => this.extractDense(result));
  }

  /**
   * Embed documents (no isQuery flag).
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.generateEmbeddings(texts);
  }

  /**
   * Embed a single query (passes isQuery: true for asymmetric models).
   */
  async embedQuery(text: string): Promise<number[]> {
    const items = [{ text }];
    const options: EncodeOptions = {
      outputTypes: ["dense"],
      instruction: this.instruction,
      outputDtype: this.outputDtype as DType | undefined,
      isQuery: true,
    };

    const results = await this.client.encode(this.model, items, options);
    return this.extractDense((results as EncodeResult[])[0] as EncodeResult);
  }

  private extractDense(result: EncodeResult): number[] {
    const dense = result.dense;
    if (!dense) {
      throw new Error("Encode result missing dense embedding");
    }
    return toNumberArray(dense);
  }

  async close(): Promise<void> {
    if (this._client) {
      await this._client.close();
    }
  }
}

/**
 * Configuration options for SIEReranker.
 */
export interface SIERerankerOptions {
  /** URL of the SIE server. @default "http://localhost:8080" */
  baseUrl?: string;
  /** Reranker model name/ID. @default "jinaai/jina-reranker-v2-base-multilingual" */
  model?: string;
  /** Name of the text column to score. @default "text" */
  column?: string;
  /** Optional pre-configured SIEClient instance. */
  client?: SIEClient;
  /** Target GPU type for routing. */
  gpu?: string;
  /** Request timeout in milliseconds. @default 180000 */
  timeout?: number;
}

/**
 * Cross-encoder reranker using SIE for LanceDB hybrid search.
 *
 * Implements LanceDB's Reranker interface. Plugs into hybrid search
 * pipelines via `.rerank()`.
 *
 * @example
 * ```typescript
 * import { SIEReranker } from "@superlinked/sie-lancedb";
 *
 * const reranker = new SIEReranker({
 *   model: "jinaai/jina-reranker-v2-base-multilingual",
 * });
 *
 * const results = await table
 *   .search("query", { queryType: "hybrid" })
 *   .rerank(reranker)
 *   .limit(10)
 *   .toArray();
 * ```
 */
export class SIEReranker {
  private readonly model: string;
  private readonly column: string;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;
  private _client: SIEClient | undefined;

  constructor(options: SIERerankerOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      model = "jinaai/jina-reranker-v2-base-multilingual",
      column = "text",
      client,
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.model = model;
    this.column = column;
    this._client = client;
    this.clientOptions = { timeout, gpu };
  }

  private get client(): SIEClient {
    if (!this._client) {
      this._client = new SIEClient(this.baseUrl, this.clientOptions);
    }
    return this._client;
  }

  /**
   * Rerank hybrid search results (vector + FTS).
   *
   * This is the method LanceDB calls during `.rerank()`. It scores all
   * rows against the query using SIE's cross-encoder and returns a
   * RecordBatch with `_relevance_score` added.
   */
  async rerankHybrid(
    query: string,
    vecResults: import("apache-arrow").RecordBatch,
    ftsResults: import("apache-arrow").RecordBatch,
  ): Promise<import("apache-arrow").RecordBatch> {
    const arrow = await import("apache-arrow");

    // Merge vector and FTS results, deduplicating by _rowid
    const merged = this.mergeResults(arrow, vecResults, ftsResults);

    const col = merged.getChild(this.column);
    if (!col || merged.numRows === 0) {
      return merged;
    }

    const texts: string[] = [];
    for (let i = 0; i < merged.numRows; i++) {
      texts.push(String(col.get(i)));
    }

    // Score with SIE
    const items = texts.map((text) => ({ text }));
    const scoreResult: ScoreResult = await this.client.score(this.model, { text: query }, items);

    // Build score array indexed by input position
    const scores = new Float32Array(texts.length);
    for (const entry of scoreResult.scores) {
      const idx = typeof entry.itemId === "string" ? Number.parseInt(entry.itemId) : entry.itemId;
      if (idx < scores.length) {
        scores[idx] = entry.score;
      }
    }

    // Build a new table with existing columns + _relevance_score, extract batch
    const columnArrays: Record<string, unknown[]> = {};
    for (const field of merged.schema.fields) {
      const child = merged.getChild(field.name);
      if (child) {
        const vals: unknown[] = [];
        for (let i = 0; i < child.length; i++) {
          vals.push(child.get(i));
        }
        columnArrays[field.name] = vals;
      }
    }
    columnArrays["_relevance_score"] = Array.from(scores);

    const newTable = arrow.tableFromArrays(columnArrays);
    const batch = newTable.batches[0];
    if (!batch) {
      throw new Error("Failed to create result batch");
    }
    return batch;
  }

  /**
   * Merge vector and FTS result batches, deduplicating by _rowid.
   * Rows from vecResults take priority for duplicate _rowid values.
   */
  private mergeResults(
    arrow: typeof import("apache-arrow"),
    vecResults: import("apache-arrow").RecordBatch,
    ftsResults: import("apache-arrow").RecordBatch,
  ): import("apache-arrow").RecordBatch {
    // Collect all unique column names from both batches
    const allFieldNames: string[] = [];
    const seen = new Set<string>();
    for (const field of vecResults.schema.fields) {
      allFieldNames.push(field.name);
      seen.add(field.name);
    }
    for (const field of ftsResults.schema.fields) {
      if (!seen.has(field.name)) {
        allFieldNames.push(field.name);
        seen.add(field.name);
      }
    }

    // Track seen _rowid values for deduplication
    const seenRowIds = new Set<number>();
    const rowSources: Array<{ batch: import("apache-arrow").RecordBatch; index: number }> = [];

    // Add vector results first (priority)
    const vecRowId = vecResults.getChild("_rowid");
    for (let i = 0; i < vecResults.numRows; i++) {
      const rid = vecRowId?.get(i);
      if (rid != null) {
        seenRowIds.add(typeof rid === "bigint" ? Number(rid) : (rid as number));
      }
      rowSources.push({ batch: vecResults, index: i });
    }

    // Add FTS results, skipping duplicates
    const ftsRowId = ftsResults.getChild("_rowid");
    for (let i = 0; i < ftsResults.numRows; i++) {
      const rid = ftsRowId?.get(i);
      const ridKey = rid != null ? (typeof rid === "bigint" ? Number(rid) : (rid as number)) : null;
      if (ridKey != null && seenRowIds.has(ridKey)) {
        continue;
      }
      if (ridKey != null) {
        seenRowIds.add(ridKey);
      }
      rowSources.push({ batch: ftsResults, index: i });
    }

    // Build merged column arrays
    const columnArrays: Record<string, unknown[]> = {};
    for (const fieldName of allFieldNames) {
      const vals: unknown[] = [];
      for (const { batch, index } of rowSources) {
        const child = batch.getChild(fieldName);
        vals.push(child ? child.get(index) : null);
      }
      columnArrays[fieldName] = vals;
    }

    const table = arrow.tableFromArrays(columnArrays);
    return table.batches[0]!;
  }

  async close(): Promise<void> {
    if (this._client) {
      await this._client.close();
    }
  }
}
