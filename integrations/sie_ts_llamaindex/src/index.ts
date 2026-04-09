/**
 * SIE embeddings integration for LlamaIndex.TS
 *
 * Provides embedding generation using SIE's encode endpoint:
 * - SIEEmbedding: Dense embeddings implementing BaseEmbedding
 * - SIESparseEmbeddingFunction: Sparse embeddings for hybrid search
 * - SIENodePostprocessor: Cross-encoder reranking for query pipelines
 * - createSIEExtractorTool: Entity extraction tool for agents
 *
 * @example
 * ```typescript
 * import { Settings } from "llamaindex";
 * import { SIEEmbedding } from "@superlinked/sie-llamaindex";
 *
 * // Set as default embedding model
 * Settings.embedModel = new SIEEmbedding({
 *   baseUrl: "http://localhost:8080",
 *   modelName: "BAAI/bge-m3",
 * });
 * ```
 */

import {
  type DType,
  type EncodeOptions,
  type EncodeResult,
  SIEClient,
  type SIEClientOptions,
  denseEmbedding,
  sparseEmbedding,
} from "@superlinked/sie-sdk";
import { BaseEmbedding } from "llamaindex";

/**
 * Configuration options for SIEEmbedding.
 */
export interface SIEEmbeddingOptions {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Model name/ID to use for encoding.
   * @default "BAAI/bge-m3"
   */
  modelName?: string;

  /**
   * Optional pre-configured SIEClient instance.
   */
  client?: SIEClient;

  /**
   * Optional instruction prefix for embedding (model-dependent).
   */
  instruction?: string;

  /**
   * Output dtype: "float32" (default), "float16", "int8", "binary".
   */
  outputDtype?: DType;

  /**
   * Target GPU type for routing (e.g., "l4", "a100-80gb").
   */
  gpu?: string;

  /**
   * Request timeout in milliseconds.
   * @default 180000 (3 minutes)
   */
  timeout?: number;

  /**
   * Batch size for embedding multiple texts.
   * @default 10
   */
  embedBatchSize?: number;
}

/**
 * LlamaIndex BaseEmbedding implementation using SIE.
 *
 * Wraps SIEClient.encode() to implement the LlamaIndex BaseEmbedding interface.
 *
 * @example
 * ```typescript
 * import { Settings, VectorStoreIndex, Document } from "llamaindex";
 * import { SIEEmbedding } from "@superlinked/sie-llamaindex";
 *
 * // Set as default embedding model
 * Settings.embedModel = new SIEEmbedding({
 *   baseUrl: "http://localhost:8080",
 *   modelName: "BAAI/bge-m3",
 * });
 *
 * // Create index with documents
 * const index = await VectorStoreIndex.fromDocuments([
 *   new Document({ text: "Hello world" }),
 * ]);
 *
 * // With GPU routing for multi-GPU clusters
 * const embedModel = new SIEEmbedding({
 *   baseUrl: "https://cluster.example.com",
 *   modelName: "BAAI/bge-m3",
 *   gpu: "a100-80gb",
 * });
 * ```
 */
export class SIEEmbedding extends BaseEmbedding {
  readonly modelName: string;
  private readonly instruction?: string;
  private readonly outputDtype?: DType;
  private _client: SIEClient | undefined;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  /**
   * Get embeddings for multiple text strings (documents).
   * This is a property (arrow function) to match BaseEmbedding interface.
   */
  override getTextEmbeddings = async (texts: string[]): Promise<number[][]> => {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["dense"],
      instruction: this.instruction,
      outputDtype: this.outputDtype,
      isQuery: false,
    };

    const results = await this.client.encode(this.modelName, items, options);
    return (results as EncodeResult[]).map((result) => denseEmbedding(result));
  };

  constructor(options: SIEEmbeddingOptions = {}) {
    super();

    const {
      baseUrl = "http://localhost:8080",
      modelName = "BAAI/bge-m3",
      client,
      instruction,
      outputDtype,
      gpu,
      timeout = 180_000,
      embedBatchSize = 10,
    } = options;

    this.baseUrl = baseUrl;
    this.modelName = modelName;
    this.instruction = instruction;
    this.outputDtype = outputDtype;
    this._client = client;
    this.embedBatchSize = embedBatchSize;

    this.clientOptions = {
      timeout,
      gpu,
    };
  }

  /**
   * Get or create the SIEClient.
   */
  private get client(): SIEClient {
    if (!this._client) {
      this._client = new SIEClient(this.baseUrl, this.clientOptions);
    }
    return this._client;
  }

  /**
   * Get embedding for a single text string (document).
   *
   * @param text - Text to embed.
   * @returns Embedding vector as array of numbers.
   */
  async getTextEmbedding(text: string): Promise<number[]> {
    const options: EncodeOptions = {
      outputTypes: ["dense"],
      instruction: this.instruction,
      outputDtype: this.outputDtype,
      isQuery: false,
    };

    const result = await this.client.encode(this.modelName, { text }, options);
    return denseEmbedding(result as EncodeResult);
  }

  /**
   * Close the underlying client connection.
   */
  async close(): Promise<void> {
    if (this._client) {
      await this._client.close();
    }
  }
}

/**
 * Configuration options for SIESparseEmbeddingFunction.
 */
export interface SIESparseEmbeddingFunctionOptions {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Model name/ID to use for encoding. Must support sparse output.
   * @default "BAAI/bge-m3"
   */
  modelName?: string;

  /**
   * Target GPU type for routing (e.g., "l4", "a100-80gb").
   */
  gpu?: string;

  /**
   * Request timeout in milliseconds.
   * @default 180000 (3 minutes)
   */
  timeout?: number;
}

/**
 * Sparse embedding function for LlamaIndex hybrid search.
 *
 * Compatible with LlamaIndex vector stores that support hybrid search,
 * such as QdrantVectorStore with enableHybrid: true.
 *
 * @example
 * ```typescript
 * import { QdrantVectorStore } from "llamaindex";
 * import { SIESparseEmbeddingFunction } from "@superlinked/sie-llamaindex";
 *
 * const sparseEmbedFn = new SIESparseEmbeddingFunction({
 *   baseUrl: "http://localhost:8080",
 *   modelName: "BAAI/bge-m3",
 * });
 *
 * const vectorStore = new QdrantVectorStore({
 *   client: qdrantClient,
 *   collectionName: "hybrid_docs",
 *   enableHybrid: true,
 *   sparseEmbeddingFunction: sparseEmbedFn,
 * });
 * ```
 */
export class SIESparseEmbeddingFunction {
  private readonly modelName: string;
  private _client: SIEClient | undefined;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(options: SIESparseEmbeddingFunctionOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      modelName = "BAAI/bge-m3",
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.modelName = modelName;
    this.clientOptions = {
      timeout,
      gpu,
    };
  }

  /**
   * Get or create the SIEClient.
   */
  private get client(): SIEClient {
    if (!this._client) {
      this._client = new SIEClient(this.baseUrl, this.clientOptions);
    }
    return this._client;
  }

  /**
   * Encode query texts to sparse vectors.
   *
   * @param texts - List of query texts to encode.
   * @returns Tuple of [indices_list, values_list].
   */
  async encodeQueries(texts: string[]): Promise<[number[][], number[][]]> {
    if (texts.length === 0) {
      return [[], []];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["sparse"],
      isQuery: true,
    };

    const results = await this.client.encode(this.modelName, items, options);

    const indicesList: number[][] = [];
    const valuesList: number[][] = [];

    for (const result of results as EncodeResult[]) {
      const sparse = sparseEmbedding(result);
      indicesList.push(sparse.indices);
      valuesList.push(sparse.values);
    }

    return [indicesList, valuesList];
  }

  /**
   * Encode document texts to sparse vectors.
   *
   * @param texts - List of document texts to encode.
   * @returns Tuple of [indices_list, values_list].
   */
  async encodeDocuments(texts: string[]): Promise<[number[][], number[][]]> {
    if (texts.length === 0) {
      return [[], []];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["sparse"],
      isQuery: false,
    };

    const results = await this.client.encode(this.modelName, items, options);

    const indicesList: number[][] = [];
    const valuesList: number[][] = [];

    for (const result of results as EncodeResult[]) {
      const sparse = sparseEmbedding(result);
      indicesList.push(sparse.indices);
      valuesList.push(sparse.values);
    }

    return [indicesList, valuesList];
  }

  /**
   * Close the underlying client connection.
   */
  async close(): Promise<void> {
    if (this._client) {
      await this._client.close();
    }
  }
}

export { SIENodePostprocessor, type SIENodePostprocessorOptions } from "./rerankers.js";
export { createSIEExtractorTool, type SIEExtractorToolOptions } from "./extractors.js";
