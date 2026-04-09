/**
 * SIE extraction tool for LlamaIndex.TS
 *
 * Provides extraction using SIE's extract endpoint:
 * - createSIEExtractorTool: Factory that creates a FunctionTool for extraction
 *
 * Returns entities, relations, classifications, and detected objects.
 *
 * @example
 * ```typescript
 * import { createSIEExtractorTool } from "@superlinked/sie-llamaindex";
 *
 * const extractor = createSIEExtractorTool({
 *   modelName: "urchade/gliner_multi-v2.1",
 *   labels: ["person", "organization", "location"],
 * });
 *
 * const result = await extractor.call({ text: "John Smith works at Acme Corp" });
 * const parsed = JSON.parse(result);
 * console.log(parsed.entities);
 * ```
 */

import {
  type ExtractOptions,
  type ExtractResult,
  SIEClient,
  type SIEClientOptions,
} from "@superlinked/sie-sdk";
import { FunctionTool } from "llamaindex";

/**
 * Configuration options for createSIEExtractorTool.
 */
export interface SIEExtractorToolOptions {
  /**
   * URL of the SIE server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;

  /**
   * Extraction model name/ID.
   * @default "urchade/gliner_multi-v2.1"
   */
  modelName?: string;

  /**
   * Labels to extract.
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
   * Tool name for the agent.
   * @default "sie_extract"
   */
  name?: string;

  /**
   * Tool description for the agent.
   */
  description?: string;
}

/**
 * Internal class to hold SIE extractor state.
 */
class _SIEExtractor {
  private readonly baseUrl: string;
  private readonly modelName: string;
  private readonly labels: string[];
  private readonly threshold?: number;
  private readonly clientOptions: SIEClientOptions;
  private _client: SIEClient | undefined;

  constructor(options: SIEExtractorToolOptions) {
    this.baseUrl = options.baseUrl ?? "http://localhost:8080";
    this.modelName = options.modelName ?? "urchade/gliner_multi-v2.1";
    this.labels = options.labels ?? ["person", "organization", "location"];
    this.threshold = options.threshold;
    this.clientOptions = {
      timeout: options.timeout ?? 180_000,
      gpu: options.gpu,
    };
  }

  private get client(): SIEClient {
    if (!this._client) {
      this._client = new SIEClient(this.baseUrl, this.clientOptions);
    }
    return this._client;
  }

  async extract(text: string): Promise<string> {
    const extractOptions: ExtractOptions = {
      labels: this.labels,
    };
    if (this.threshold !== undefined) {
      extractOptions.threshold = this.threshold;
    }

    const result: ExtractResult = await this.client.extract(this.modelName, { text }, extractOptions);

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
}

/**
 * Create a LlamaIndex FunctionTool for extraction.
 *
 * Creates a tool that wraps SIE's extract endpoint for use with
 * LlamaIndex agents and workflows. Returns JSON with entities,
 * relations, classifications, and detected objects.
 *
 * @example
 * ```typescript
 * import { OpenAI, ReActAgent } from "llamaindex";
 * import { createSIEExtractorTool } from "@superlinked/sie-llamaindex";
 *
 * const extractor = createSIEExtractorTool({
 *   modelName: "urchade/gliner_multi-v2.1",
 *   labels: ["person", "organization", "location"],
 * });
 *
 * const agent = new ReActAgent({
 *   tools: [extractor],
 *   llm: new OpenAI(),
 * });
 *
 * const response = await agent.chat({
 *   message: "Extract entities from: John works at Google in NYC",
 * });
 * ```
 *
 * @param options - Configuration for the extractor tool.
 * @returns FunctionTool wrapping SIE extraction.
 */
export function createSIEExtractorTool(options: SIEExtractorToolOptions = {}) {
  const labels = options.labels ?? ["person", "organization", "location"];
  const name = options.name ?? "sie_extract";
  const description =
    options.description ??
    `Extract structured information from text. Finds entities of types: ${labels.join(", ")}. Returns entities, relations, classifications, and detected objects.`;

  const extractor = new _SIEExtractor(options);

  async function extract(input: { text: string }): Promise<string> {
    return extractor.extract(input.text);
  }

  return FunctionTool.from(extract, {
    name,
    description,
  });
}
