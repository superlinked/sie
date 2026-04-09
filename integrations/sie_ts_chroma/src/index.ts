/**
 * SIE embedding functions for ChromaDB.
 *
 * Provides custom embedding functions that use SIE for generating embeddings:
 * - SIEEmbeddingFunction: Dense embeddings for standard Chroma collections
 *
 * @example
 * ```typescript
 * import { ChromaClient } from "chromadb";
 * import { SIEEmbeddingFunction } from "@superlinked/sie-chroma";
 *
 * const client = new ChromaClient();
 * const embeddingFunction = new SIEEmbeddingFunction({
 *   baseUrl: "http://localhost:8080",
 *   model: "BAAI/bge-m3",
 * });
 *
 * const collection = await client.createCollection({
 *   name: "my_collection",
 *   embeddingFunction,
 * });
 *
 * await collection.add({
 *   ids: ["doc1", "doc2"],
 *   documents: ["Hello world", "Goodbye world"],
 * });
 * ```
 */

import {
  type EncodeOptions,
  type EncodeResult,
  SIEClient,
  type SIEClientOptions,
  denseEmbedding,
  sparseEmbedding,
} from "@superlinked/sie-sdk";
import type { IEmbeddingFunction } from "chromadb";

/**
 * Configuration options for SIEEmbeddingFunction.
 */
export interface SIEEmbeddingFunctionOptions {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Model name/ID to use for encoding.
   * @default "BAAI/bge-m3"
   */
  model?: string;

  /**
   * Optional pre-configured SIEClient instance.
   */
  client?: SIEClient;

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
 * Embedding function using SIE for ChromaDB collections.
 *
 * This class implements ChromaDB's IEmbeddingFunction interface,
 * allowing SIE to generate embeddings for document storage and retrieval.
 *
 * @example
 * ```typescript
 * import { ChromaClient } from "chromadb";
 * import { SIEEmbeddingFunction } from "@superlinked/sie-chroma";
 *
 * const embeddingFunction = new SIEEmbeddingFunction({
 *   baseUrl: "http://localhost:8080",
 *   model: "BAAI/bge-m3",
 * });
 *
 * const client = new ChromaClient();
 * const collection = await client.createCollection({
 *   name: "my_collection",
 *   embeddingFunction,
 * });
 *
 * // Add documents (embeddings generated automatically)
 * await collection.add({
 *   ids: ["doc1", "doc2"],
 *   documents: ["First document", "Second document"],
 * });
 *
 * // Query (query embedding generated automatically)
 * const results = await collection.query({
 *   queryTexts: ["search query"],
 *   nResults: 5,
 * });
 * ```
 */
export class SIEEmbeddingFunction implements IEmbeddingFunction {
  private readonly model: string;
  private _client: SIEClient | undefined;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(options: SIEEmbeddingFunctionOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      model = "BAAI/bge-m3",
      client,
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.model = model;
    this._client = client;
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
   * Generate embeddings for documents.
   *
   * This method is called by ChromaDB when adding documents or querying.
   *
   * @param texts - Array of document texts to embed.
   * @returns Array of embedding vectors.
   */
  async generate(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["dense"],
    };

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => denseEmbedding(result));
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
 * Sparse embedding representation for ChromaDB hybrid search.
 */
export interface SparseEmbedding {
  indices: number[];
  values: number[];
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
  model?: string;

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
 * Sparse embedding function using SIE for ChromaDB hybrid search.
 *
 * Generates sparse embeddings that can be used with ChromaDB's hybrid
 * search capabilities. Returns embeddings as {indices, values} pairs.
 *
 * @example
 * ```typescript
 * import { SIESparseEmbeddingFunction } from "@superlinked/sie-chroma";
 *
 * const sparseEf = new SIESparseEmbeddingFunction({
 *   baseUrl: "http://localhost:8080",
 *   model: "BAAI/bge-m3",
 * });
 *
 * const embeddings = await sparseEf.generate(["Hello world"]);
 * console.log(embeddings[0].indices); // [1, 5, 10, ...]
 * console.log(embeddings[0].values);  // [0.5, 0.3, 0.2, ...]
 * ```
 */
export class SIESparseEmbeddingFunction {
  private readonly model: string;
  private _client: SIEClient | undefined;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(options: SIESparseEmbeddingFunctionOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      model = "BAAI/bge-m3",
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.model = model;
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
   * Generate sparse embeddings for documents.
   *
   * @param texts - Array of document texts to embed.
   * @returns Array of sparse embeddings with indices and values.
   */
  async generate(texts: string[]): Promise<SparseEmbedding[]> {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["sparse"],
    };

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => sparseEmbedding(result));
  }

  /**
   * Generate sparse embeddings as dict format (token_id -> weight).
   *
   * This format is compatible with some ChromaDB hybrid search configurations.
   *
   * @param texts - Array of document texts to embed.
   * @returns Array of sparse embeddings as {[tokenId]: weight} dicts.
   */
  async generateAsDict(texts: string[]): Promise<Record<number, number>[]> {
    const embeddings = await this.generate(texts);
    return embeddings.map((emb) => {
      const dict: Record<number, number> = {};
      for (let i = 0; i < emb.indices.length; i++) {
        const idx = emb.indices[i];
        const val = emb.values[i];
        if (idx !== undefined && val !== undefined) {
          dict[idx] = val;
        }
      }
      return dict;
    });
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
