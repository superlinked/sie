/**
 * SIE extraction tool for LangChain.js
 *
 * Provides extraction using SIE's extract endpoint:
 * - SIEExtractor: Extraction tool implementing LangChain Tool
 *
 * Returns entities, relations, classifications, and detected objects.
 *
 * @example
 * ```typescript
 * import { SIEExtractor } from "@superlinked/sie-langchain";
 *
 * const extractor = new SIEExtractor({
 *   baseUrl: "http://localhost:8080",
 *   model: "urchade/gliner_multi-v2.1",
 *   labels: ["person", "organization", "location"],
 * });
 *
 * const result = await extractor.invoke("John Smith works at Acme Corp in NYC");
 * const parsed = JSON.parse(result);
 * console.log(parsed.entities);
 * console.log(parsed.relations);
 * ```
 */

import { Tool } from "@langchain/core/tools";
import {
  type ExtractOptions,
  type ExtractResult,
  SIEClient,
  type SIEClientOptions,
} from "@superlinked/sie-sdk";

/**
 * Configuration options for SIEExtractor.
 */
export interface SIEExtractorParams {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Extraction model name/ID.
   * @default "urchade/gliner_multi-v2.1"
   */
  model?: string;

  /**
   * Optional pre-configured SIEClient instance.
   * If provided, baseUrl and other connection options are ignored.
   */
  client?: SIEClient;

  /**
   * Labels to extract (entity types, relation types, or classification labels).
   * @default ["person", "organization", "location"]
   */
  labels?: string[];

  /**
   * Minimum confidence threshold (0-1).
   */
  threshold?: number;

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
   * Tool name for use in agents.
   * @default "sie_extract"
   */
  name?: string;

  /**
   * Tool description for use in agents.
   */
  description?: string;
}

/**
 * LangChain tool for extraction using SIE.
 *
 * Wraps SIEClient.extract() to implement the LangChain Tool interface
 * for use in agents and chains. Returns JSON with entities, relations,
 * classifications, and detected objects.
 *
 * @example
 * ```typescript
 * import { SIEExtractor } from "@superlinked/sie-langchain";
 *
 * // Direct usage
 * const extractor = new SIEExtractor({
 *   model: "urchade/gliner_multi-v2.1",
 *   labels: ["person", "organization", "location"],
 * });
 * const result = await extractor.invoke("John Smith works at Acme Corp");
 * const parsed = JSON.parse(result);
 *
 * // Use in an agent
 * import { ChatOpenAI } from "@langchain/openai";
 * import { createReactAgent } from "@langchain/langgraph/prebuilt";
 *
 * const agent = createReactAgent({
 *   llm: new ChatOpenAI(),
 *   tools: [extractor],
 * });
 * ```
 */
export class SIEExtractor extends Tool {
  name: string;
  description: string;

  private readonly model: string;
  private readonly labels: string[];
  private readonly threshold?: number;
  private _client: SIEClient | undefined;
  private readonly _ownsClient: boolean;
  private readonly baseUrl: string;
  private readonly clientOptions: SIEClientOptions;

  constructor(params: SIEExtractorParams = {}) {
    const toolName = params.name ?? "sie_extract";
    const toolDescription =
      params.description ??
      "Extract structured information from text. " +
        "Input should be text to analyze. " +
        "Returns JSON with entities, relations, classifications, and detected objects.";

    super({});

    this.name = toolName;
    this.description = toolDescription;

    const {
      baseUrl = "http://localhost:8080",
      model = "urchade/gliner_multi-v2.1",
      client,
      labels = ["person", "organization", "location"],
      threshold,
      gpu,
      timeout = 180_000,
    } = params;

    this.baseUrl = baseUrl;
    this.model = model;
    this.labels = labels;
    this.threshold = threshold;
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
   * Extract structured information from text.
   *
   * @param text - Text to extract from.
   * @returns JSON string with entities, relations, classifications, and objects.
   */
  async _call(text: string): Promise<string> {
    const extractOptions: ExtractOptions = {
      labels: this.labels,
    };
    if (this.threshold !== undefined) {
      extractOptions.threshold = this.threshold;
    }

    const result: ExtractResult = await this.client.extract(this.model, { text }, extractOptions);

    return JSON.stringify({
      entities: result.entities.map((e) => ({
        text: e.text,
        label: e.label,
        score: e.score,
        ...(e.start !== undefined && { start: e.start }),
        ...(e.end !== undefined && { end: e.end }),
      })),
      relations: result.relations.map((r) => ({
        head: r.head,
        tail: r.tail,
        relation: r.relation,
        score: r.score,
      })),
      classifications: result.classifications.map((c) => ({
        label: c.label,
        score: c.score,
      })),
      objects: result.objects.map((o) => ({
        label: o.label,
        score: o.score,
        bbox: o.bbox,
      })),
    });
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
