/**
 * SIE reranker integration for LlamaIndex.TS
 *
 * Provides node reranking using SIE's score endpoint:
 * - SIENodePostprocessor: Cross-encoder reranking implementing BaseNodePostprocessor
 *
 * @example
 * ```typescript
 * import { SIENodePostprocessor } from "@superlinked/sie-llamaindex";
 *
 * const reranker = new SIENodePostprocessor({
 *   baseUrl: "http://localhost:8080",
 *   modelName: "jinaai/jina-reranker-v2-base-multilingual",
 *   topN: 3,
 * });
 *
 * const reranked = await reranker.postprocessNodes(nodes, "search query");
 * ```
 */

import { SIEClient, type SIEClientOptions } from "@superlinked/sie-sdk";
import { MetadataMode } from "llamaindex";
import type { BaseNodePostprocessor, MessageContent, NodeWithScore } from "llamaindex";

/**
 * Configuration options for SIENodePostprocessor.
 */
export interface SIENodePostprocessorOptions {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Reranker model name/ID.
   * @default "jinaai/jina-reranker-v2-base-multilingual"
   */
  modelName?: string;

  /**
   * Optional pre-configured SIEClient instance.
   * If provided, baseUrl and other connection options are ignored.
   */
  client?: SIEClient;

  /**
   * Number of top nodes to return. If undefined, returns all nodes.
   */
  topN?: number;

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
 * Extract a plain query string from LlamaIndex's MessageContent type.
 *
 * MessageContent is `string | MessageContentDetail[]`. For reranking
 * we need a plain string.
 */
function extractQueryString(query: MessageContent): string {
  if (typeof query === "string") {
    return query;
  }
  // MessageContentDetail[] — concatenate text parts
  return query
    .filter(
      (part): part is { type: "text"; text: string } => "type" in part && part.type === "text",
    )
    .map((part) => part.text)
    .join(" ");
}

/**
 * LlamaIndex node postprocessor using SIE's reranking.
 *
 * Wraps SIEClient.score() to implement the BaseNodePostprocessor interface.
 *
 * @example
 * ```typescript
 * import { VectorStoreIndex } from "llamaindex";
 * import { SIENodePostprocessor } from "@superlinked/sie-llamaindex";
 *
 * const reranker = new SIENodePostprocessor({
 *   modelName: "jinaai/jina-reranker-v2-base-multilingual",
 *   topN: 3,
 * });
 *
 * const queryEngine = index.asQueryEngine({
 *   nodePostprocessors: [reranker],
 * });
 *
 * const response = await queryEngine.query({ query: "What is the topic?" });
 * ```
 */
export class SIENodePostprocessor implements BaseNodePostprocessor {
  readonly modelName: string;
  private readonly topN?: number;
  private _client: SIEClient | undefined;
  private readonly _ownsClient: boolean;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(options: SIENodePostprocessorOptions = {}) {
    const {
      baseUrl = "http://localhost:8080",
      modelName = "jinaai/jina-reranker-v2-base-multilingual",
      client,
      topN,
      gpu,
      timeout = 180_000,
    } = options;

    this.baseUrl = baseUrl;
    this.modelName = modelName;
    this.topN = topN;
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
   * Rerank nodes by relevance to query.
   *
   * @param nodes - Nodes with scores to rerank.
   * @param query - Optional query string or MessageContent.
   * @returns Reranked nodes with updated scores, sorted by relevance descending.
   */
  async postprocessNodes(nodes: NodeWithScore[], query?: MessageContent): Promise<NodeWithScore[]> {
    if (nodes.length === 0 || query === undefined) {
      return nodes;
    }

    const queryText = extractQueryString(query);
    if (!queryText) {
      return nodes;
    }

    const queryItem = { text: queryText };
    const docItems = nodes.map((n) => ({ text: n.node.getContent(MetadataMode.NONE) }));

    const result = await this.client.score(this.modelName, queryItem, docItems);

    // Map score entries back to NodeWithScore with updated scores.
    // ScoreResult.scores are already sorted by score descending.
    const reranked: NodeWithScore[] = [];
    for (const entry of result.scores) {
      const idx = Number.parseInt(entry.itemId, 10);
      const original = nodes[idx];
      if (original) {
        reranked.push({
          node: original.node,
          score: entry.score,
        });
      }
    }

    if (this.topN !== undefined) {
      return reranked.slice(0, this.topN);
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
