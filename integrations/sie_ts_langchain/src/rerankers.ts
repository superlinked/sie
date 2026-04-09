/**
 * SIE reranker integration for LangChain.js
 *
 * Provides document reranking using SIE's score endpoint:
 * - SIEReranker: Cross-encoder reranking implementing BaseDocumentCompressor
 *
 * @example
 * ```typescript
 * import { SIEReranker } from "@superlinked/sie-langchain";
 *
 * const reranker = new SIEReranker({
 *   baseUrl: "http://localhost:8080",
 *   model: "jinaai/jina-reranker-v2-base-multilingual",
 *   topK: 3,
 * });
 *
 * const reranked = await reranker.compressDocuments(documents, "search query");
 * ```
 */

import type { DocumentInterface } from "@langchain/core/documents";
import { BaseDocumentCompressor } from "@langchain/core/retrievers/document_compressors";
import { SIEClient, type SIEClientOptions } from "@superlinked/sie-sdk";

/**
 * Configuration options for SIEReranker.
 */
export interface SIERerankerParams {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Reranker model name/ID.
   * @default "jinaai/jina-reranker-v2-base-multilingual"
   */
  model?: string;

  /**
   * Optional pre-configured SIEClient instance.
   * If provided, baseUrl and other connection options are ignored.
   */
  client?: SIEClient;

  /**
   * Number of top documents to return. If undefined, returns all documents.
   */
  topK?: number;

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
 * LangChain document compressor using SIE's reranking.
 *
 * Wraps SIEClient.score() to implement BaseDocumentCompressor.
 *
 * @example
 * ```typescript
 * import { SIEReranker } from "@superlinked/sie-langchain";
 *
 * const reranker = new SIEReranker({
 *   baseUrl: "http://localhost:8080",
 *   model: "jinaai/jina-reranker-v2-base-multilingual",
 *   topK: 3,
 * });
 *
 * // Rerank retrieved documents
 * const reranked = await reranker.compressDocuments(documents, "search query");
 *
 * // Use in a retrieval pipeline
 * import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
 *
 * const compressionRetriever = new ContextualCompressionRetriever({
 *   baseCompressor: reranker,
 *   baseRetriever: vectorStoreRetriever,
 * });
 * ```
 */
export class SIEReranker extends BaseDocumentCompressor {
  private readonly model: string;
  private readonly topK?: number;
  private _client: SIEClient | undefined;
  private readonly _ownsClient: boolean;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(params: SIERerankerParams = {}) {
    super();

    const {
      baseUrl = "http://localhost:8080",
      model = "jinaai/jina-reranker-v2-base-multilingual",
      client,
      topK,
      gpu,
      timeout = 180_000,
    } = params;

    this.baseUrl = baseUrl;
    this.model = model;
    this.topK = topK;
    this._client = client;
    this._ownsClient = !client;

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
   * Rerank documents by relevance to query.
   *
   * @param documents - Documents to rerank.
   * @param query - Query to rank documents against.
   * @returns Reranked documents with relevance_score in metadata, sorted by score descending.
   */
  async compressDocuments(
    documents: DocumentInterface[],
    query: string,
  ): Promise<DocumentInterface[]> {
    if (documents.length === 0) {
      return [];
    }

    const queryItem = { text: query };
    const docItems = documents.map((doc) => ({ text: doc.pageContent }));

    const result = await this.client.score(this.model, queryItem, docItems);

    // Map score entries back to documents with relevance_score in metadata.
    // ScoreResult.scores are already sorted by score descending.
    const reranked: DocumentInterface[] = [];
    for (const entry of result.scores) {
      const idx = Number.parseInt(entry.itemId, 10);
      const doc = documents[idx];
      if (doc) {
        reranked.push({
          pageContent: doc.pageContent,
          metadata: { ...doc.metadata, relevance_score: entry.score },
          id: doc.id,
        });
      }
    }

    if (this.topK !== undefined) {
      return reranked.slice(0, this.topK);
    }
    return reranked;
  }

  /**
   * Close the underlying client connection.
   */
  async close(): Promise<void> {
    if (this._client && this._ownsClient) {
      await this._client.close();
    }
  }
}
