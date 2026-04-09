/**
 * SIE embeddings integration for LangChain.js
 *
 * Provides drop-in replacement for OpenAI embeddings using SIE's inference server:
 * - SIEEmbeddings: Dense embeddings for vector stores
 * - SIESparseEncoder: Sparse encoder for hybrid search
 * - SIEReranker: Cross-encoder reranking for retrieval pipelines
 * - SIEExtractor: Entity extraction tool for agents
 *
 * @example
 * ```typescript
 * import { SIEEmbeddings } from "@superlinked/sie-langchain";
 *
 * const embeddings = new SIEEmbeddings({
 *   baseUrl: "http://localhost:8080",
 *   model: "BAAI/bge-m3",
 * });
 *
 * const vectors = await embeddings.embedDocuments(["Hello world"]);
 * const queryVector = await embeddings.embedQuery("What is hello?");
 * ```
 */

import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
import {
  type DType,
  type EncodeOptions,
  type EncodeResult,
  SIEClient,
  type SIEClientOptions,
  denseEmbedding,
  sparseEmbedding,
} from "@superlinked/sie-sdk";

/**
 * Configuration options for SIEEmbeddings.
 */
export interface SIEEmbeddingsParams extends EmbeddingsParams {
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
   * If provided, baseUrl and other connection options are ignored.
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
}

/**
 * LangChain Embeddings implementation using SIE.
 *
 * Wraps SIEClient.encode() to implement the LangChain Embeddings interface.
 *
 * @example
 * ```typescript
 * import { SIEEmbeddings } from "@superlinked/sie-langchain";
 *
 * // Basic usage
 * const embeddings = new SIEEmbeddings({
 *   baseUrl: "http://localhost:8080",
 *   model: "BAAI/bge-m3",
 * });
 *
 * // Embed documents
 * const docVectors = await embeddings.embedDocuments([
 *   "First document",
 *   "Second document",
 * ]);
 *
 * // Embed a query (may use different encoding for asymmetric models)
 * const queryVector = await embeddings.embedQuery("What is the topic?");
 *
 * // With GPU routing
 * const gpuEmbeddings = new SIEEmbeddings({
 *   baseUrl: "https://cluster.example.com",
 *   model: "BAAI/bge-m3",
 *   gpu: "a100-80gb",
 * });
 * ```
 */
export class SIEEmbeddings extends Embeddings {
  private readonly model: string;
  private readonly instruction?: string;
  private readonly outputDtype?: DType;
  private _client: SIEClient | undefined;
  private readonly clientOptions: SIEClientOptions;

  constructor(params: SIEEmbeddingsParams = {}) {
    super(params);

    const {
      baseUrl = "http://localhost:8080",
      model = "BAAI/bge-m3",
      client,
      instruction,
      outputDtype,
      gpu,
      timeout = 180_000,
    } = params;

    this.model = model;
    this.instruction = instruction;
    this.outputDtype = outputDtype;
    this._client = client;

    this.clientOptions = {
      timeout,
      gpu,
    };

    // If no client provided, we'll create one lazily using baseUrl
    if (!client) {
      this.clientOptions.timeout = timeout;
      this.clientOptions.gpu = gpu;
      // Store baseUrl for lazy client creation
      (this as { baseUrl?: string }).baseUrl = baseUrl;
    }
  }

  /**
   * Get or create the SIEClient.
   */
  private get client(): SIEClient {
    if (!this._client) {
      const baseUrl = (this as { baseUrl?: string }).baseUrl ?? "http://localhost:8080";
      this._client = new SIEClient(baseUrl, this.clientOptions);
    }
    return this._client;
  }

  /**
   * Embed a list of documents.
   *
   * @param texts - List of document texts to embed.
   * @returns List of embedding vectors (as arrays of numbers).
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
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

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => denseEmbedding(result));
  }

  /**
   * Embed a single query text.
   *
   * For asymmetric models (like BGE-M3), this uses query-specific encoding.
   *
   * @param text - Query text to embed.
   * @returns Embedding vector as array of numbers.
   */
  async embedQuery(text: string): Promise<number[]> {
    const options: EncodeOptions = {
      outputTypes: ["dense"],
      instruction: this.instruction,
      outputDtype: this.outputDtype,
      isQuery: true,
    };

    const result = await this.client.encode(this.model, { text }, options);
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
 * Configuration options for SIESparseEncoder.
 */
export interface SIESparseEncoderOptions {
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
 * Sparse encoder for LangChain hybrid search.
 *
 * Compatible with PineconeHybridSearchRetriever's sparse_encoder interface.
 *
 * @example
 * ```typescript
 * import { SIEEmbeddings, SIESparseEncoder } from "@superlinked/sie-langchain";
 * import { PineconeHybridSearchRetriever } from "@langchain/pinecone";
 *
 * const retriever = new PineconeHybridSearchRetriever({
 *   embeddings: new SIEEmbeddings({ model: "BAAI/bge-m3" }),
 *   sparseEncoder: new SIESparseEncoder({ model: "BAAI/bge-m3" }),
 *   index: pineconeIndex,
 * });
 * ```
 */
export class SIESparseEncoder {
  private readonly model: string;
  private _client: SIEClient | undefined;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(options: SIESparseEncoderOptions = {}) {
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
   * Encode query texts to sparse vectors.
   *
   * @param texts - List of query texts to encode.
   * @returns List of objects with "indices" and "values" arrays.
   */
  async encodeQueries(texts: string[]): Promise<Array<{ indices: number[]; values: number[] }>> {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["sparse"],
      isQuery: true,
    };

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => sparseEmbedding(result));
  }

  /**
   * Encode document texts to sparse vectors.
   *
   * @param texts - List of document texts to encode.
   * @returns List of objects with "indices" and "values" arrays.
   */
  async encodeDocuments(texts: string[]): Promise<Array<{ indices: number[]; values: number[] }>> {
    if (texts.length === 0) {
      return [];
    }

    const items = texts.map((text) => ({ text }));
    const options: EncodeOptions = {
      outputTypes: ["sparse"],
      isQuery: false,
    };

    const results = await this.client.encode(this.model, items, options);
    return (results as EncodeResult[]).map((result) => sparseEmbedding(result));
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

export { SIEReranker, type SIERerankerParams } from "./rerankers.js";
export { SIEExtractor, type SIEExtractorParams } from "./extractors.js";
