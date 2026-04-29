/**
 * SIE Client implementation
 *
 * @example
 * ```typescript
 * import { SIEClient } from "@superlinked/sie-sdk";
 *
 * const client = new SIEClient("http://localhost:8080");
 *
 * // Encode single item
 * const result = await client.encode("bge-m3", { text: "Hello world" });
 * console.log(result.dense); // Float32Array
 *
 * // Batch encode
 * const results = await client.encode("bge-m3", [
 *   { text: "First document" },
 *   { text: "Second document" },
 * ]);
 *
 * // With GPU routing and auto-retry for capacity
 * const resultWithGpu = await client.encode(
 *   "bge-m3",
 *   { text: "Hello" },
 *   { gpu: "l4", waitForCapacity: true },
 * );
 *
 * await client.close();
 * ```
 */

import {
  LoraLoadingError,
  ModelLoadingError,
  PoolError,
  ProvisioningError,
  RequestError,
  SIEConnectionError,
} from "./errors.js";
import {
  DEFAULT_LEASE_RENEWAL_INTERVAL,
  DEFAULT_PROVISION_TIMEOUT,
  DEFAULT_RETRY_DELAY,
  DEFAULT_TIMEOUT,
  HTTP_ACCEPTED,
  HTTP_CLIENT_ERROR_MIN,
  JSON_CONTENT_TYPE,
  LORA_LOADING_DEFAULT_DELAY,
  LORA_LOADING_ERROR_CODE,
  LORA_LOADING_MAX_RETRIES,
  MODEL_LOADING_DEFAULT_DELAY,
  MODEL_LOADING_ERROR_CODE,
  MSGPACK_CONTENT_TYPE,
  SDK_VERSION_HEADER,
  SERVER_VERSION_HEADER,
} from "./internal/constants.js";
import {
  getErrorCode,
  getRetryAfter,
  handleError,
  parseCapacityInfo,
  parseEncodeResults,
  parseExtractResults,
  parseScoreResult,
} from "./internal/parsing.js";
import { packMessage, unpackMessage } from "./msgpack.js";
import type {
  CapacityInfo,
  EncodeOptions,
  EncodeResult,
  ExtractOptions,
  ExtractResult,
  Item,
  ModelInfo,
  PoolInfo,
  SIEClientOptions,
  ScoreOptions,
  ScoreResult,
  StatusMessage,
} from "./types.js";
import { SDK_VERSION } from "./version.js";

/** Helper to sleep for a given number of milliseconds */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Sleep that can be cancelled via AbortSignal. Returns true if aborted. */
function abortableSleep(ms: number, signal: AbortSignal): Promise<boolean> {
  if (signal.aborted) return Promise.resolve(true);
  return new Promise((resolve) => {
    const onAbort = () => {
      clearTimeout(timeoutId);
      resolve(true);
    };
    const timeoutId = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve(false);
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

const _LEASE_RENEWAL_MAX_RETRIES = 5;

/**
 * SIE Client for embedding, scoring, and extraction.
 *
 * The client is async-only (no synchronous methods) and uses native fetch.
 * It handles msgpack serialization, error parsing, and retry logic.
 *
 * @example Resource pool usage
 * ```typescript
 * const client = new SIEClient("http://gateway:8080");
 *
 * // Create a dedicated pool
 * await client.createPool("eval-bench", { l4: 2 });
 *
 * // Use pool for requests
 * await client.encode("bge-m3", { text: "Hello" }, { gpu: "eval-bench/l4" });
 *
 * // Check pool status
 * const pool = await client.getPool("eval-bench");
 * console.log(`Pool state: ${pool?.status.state}`);
 *
 * // Clean up
 * await client.deletePool("eval-bench");
 * await client.close();
 * ```
 */
export class SIEClient {
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly gpu?: string;
  private readonly apiKey?: string;
  private readonly defaultWaitForCapacity: boolean;
  private readonly provisionTimeout: number;

  // Pool state: track created pools and their lease renewal scheduling
  private readonly pools: Map<
    string,
    {
      timeoutId: ReturnType<typeof setTimeout> | null;
      abortController: AbortController;
      isRenewing: boolean;
    }
  > = new Map();

  // Version negotiation state
  private versionWarningLogged = false;

  // Note: LoRA and model loading retry counters are now local to each method
  // to avoid interference between concurrent requests

  /**
   * Create a new SIE client.
   *
   * @param baseUrl - Base URL of the SIE server (e.g., "http://localhost:8080")
   * @param options - Client options
   */
  constructor(baseUrl: string, options: SIEClientOptions = {}) {
    // Remove trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.gpu = options.gpu;
    this.apiKey = options.apiKey;
    this.defaultWaitForCapacity = options.waitForCapacity ?? false;
    this.provisionTimeout = options.provisionTimeout ?? DEFAULT_PROVISION_TIMEOUT;
  }

  /**
   * Get the base URL of the SIE server.
   *
   * @returns The normalized base URL (without trailing slash)
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Encode a single item.
   *
   * @param model - Model name (e.g., "bge-m3")
   * @param item - Item to encode
   * @param options - Encode options
   * @returns Encode result with embeddings
   */
  async encode(model: string, item: Item, options?: EncodeOptions): Promise<EncodeResult>;

  /**
   * Encode multiple items.
   *
   * @param model - Model name (e.g., "bge-m3")
   * @param items - Items to encode
   * @param options - Encode options
   * @returns Array of encode results in same order as input
   */
  async encode(model: string, items: Item[], options?: EncodeOptions): Promise<EncodeResult[]>;

  /**
   * Encode one or more items.
   */
  async encode(
    model: string,
    items: Item | Item[],
    options: EncodeOptions = {},
  ): Promise<EncodeResult | EncodeResult[]> {
    const isSingleItem = !Array.isArray(items);
    const itemsArray = isSingleItem ? [items] : items;

    // Build request body - model is in URL path, not body
    // Wire format uses snake_case
    const body: Record<string, unknown> = {
      items: itemsArray,
    };

    // Add params if any are specified
    const params: Record<string, unknown> = {};
    if (options.outputTypes) {
      params.output_types = options.outputTypes;
    }
    if (options.instruction !== undefined) {
      params.instruction = options.instruction;
    }
    if (options.isQuery !== undefined) {
      params.is_query = options.isQuery;
    }
    if (options.outputDtype !== undefined) {
      params.output_dtype = options.outputDtype;
    }
    if (Object.keys(params).length > 0) {
      body.params = params;
    }

    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;
    const { pool, gpu } = this.parseGpuParam(options.gpu);

    // Model is in URL path: /v1/encode/{model}
    const response = await this.requestWithRetry(
      `/v1/encode/${encodeURIComponent(model)}`,
      body,
      pool,
      gpu,
      waitForCapacity,
      model,
    );

    // Wire format response: {"items": [...], "timing": {...}}
    interface WireResponse {
      items: unknown[];
      timing?: Record<string, unknown>;
    }

    const data = unpackMessage<WireResponse>(new Uint8Array(await response.arrayBuffer()));

    const results = parseEncodeResults(data.items);

    if (isSingleItem) {
      const first = results[0];
      if (!first) {
        throw new Error("No results returned from encode");
      }
      return first;
    }
    return results;
  }

  /**
   * List available models.
   *
   * @returns Array of model information
   */
  async listModels(): Promise<ModelInfo[]> {
    const response = await this.requestJson("/v1/models", "GET");

    // Wire format response: {"models": [...]}
    interface WireModelInfo {
      name: string;
      loaded: boolean;
      inputs: string[];
      outputs: string[];
      dims?: { dense?: number; sparse?: number; multivector?: number };
      max_sequence_length?: number;
    }

    interface WireModelsResponse {
      models: WireModelInfo[];
    }

    const data = (await response.json()) as WireModelsResponse;

    return data.models.map((m) => ({
      name: m.name,
      loaded: m.loaded,
      inputs: m.inputs,
      outputs: m.outputs,
      dims: m.dims,
      maxSequenceLength: m.max_sequence_length,
    }));
  }

  /**
   * Get details for a specific model.
   *
   * Returns model metadata including dimensions, supported inputs/outputs,
   * loaded status, and max sequence length. This is a lightweight call that
   * reads from model config — it does not load the model or trigger inference.
   *
   * @param name - Model name (e.g., "BAAI/bge-m3")
   * @returns Model information
   */
  async getModel(name: string): Promise<ModelInfo> {
    const response = await this.requestJson(`/v1/models/${encodeURIComponent(name)}`, "GET");

    interface WireModelInfo {
      name: string;
      loaded: boolean;
      inputs: string[];
      outputs: string[];
      dims?: { dense?: number; sparse?: number; multivector?: number };
      max_sequence_length?: number;
    }

    const data = (await response.json()) as WireModelInfo;

    return {
      name: data.name,
      loaded: data.loaded,
      inputs: data.inputs,
      outputs: data.outputs,
      dims: data.dims,
      maxSequenceLength: data.max_sequence_length,
    };
  }

  /**
   * Stream real-time status updates from a worker or gateway.
   *
   * @param mode - "cluster" uses gateway /ws/cluster-status, "worker" uses /ws/status.
   *               "auto" detects the endpoint via /health.
   */
  async *watch(mode: "auto" | "cluster" | "worker" = "auto"): AsyncGenerator<StatusMessage> {
    const endpoint = mode === "auto" ? await this.detectEndpointType() : mode;
    const path = endpoint === "cluster" ? "/ws/cluster-status" : "/ws/status";
    const wsUrl = this.buildWsUrl(path);
    const ws = this.createWebSocket(wsUrl);

    const queue: StatusMessage[] = [];
    let resolveNext: (() => void) | null = null;
    let rejectNext: ((error: unknown) => void) | null = null;
    let closed = false;

    const notify = () => {
      if (resolveNext) {
        resolveNext();
        resolveNext = null;
      }
    };

    const fail = (error: unknown) => {
      if (rejectNext) {
        rejectNext(error);
        rejectNext = null;
      }
    };

    const waitForMessage = () =>
      new Promise<void>((resolve, reject) => {
        resolveNext = resolve;
        rejectNext = reject;
      });

    const parseMessage = (data: unknown): StatusMessage => {
      if (typeof data === "string") {
        return JSON.parse(data) as StatusMessage;
      }
      if (data instanceof ArrayBuffer) {
        return JSON.parse(new TextDecoder().decode(new Uint8Array(data))) as StatusMessage;
      }
      if (data instanceof Uint8Array) {
        return JSON.parse(new TextDecoder().decode(data)) as StatusMessage;
      }
      throw new Error("Unsupported WebSocket message type");
    };

    const openPromise = new Promise<void>((resolve, reject) => {
      ws.addEventListener("open", () => resolve());
      ws.addEventListener("error", (event) => reject(event));
    });

    ws.addEventListener("message", (event) => {
      try {
        queue.push(parseMessage(event.data));
        notify();
      } catch (error) {
        fail(error);
      }
    });

    ws.addEventListener("close", () => {
      closed = true;
      notify();
    });

    try {
      await openPromise;
      while (!closed || queue.length > 0) {
        if (queue.length === 0) {
          await waitForMessage();
          continue;
        }
        const next = queue.shift();
        if (next) {
          yield next;
        }
      }
    } finally {
      ws.close();
    }
  }

  /**
   * Score items against a query using a reranker model.
   *
   * @param model - Model name (e.g., "bge-reranker-v2")
   * @param query - Query item
   * @param items - Items to score against the query
   * @param options - Score options
   * @returns Score result with sorted scores
   *
   * @example
   * ```typescript
   * const result = await client.score(
   *   "bge-reranker-v2",
   *   { text: "What is machine learning?" },
   *   [
   *     { id: "doc-1", text: "Machine learning is..." },
   *     { id: "doc-2", text: "Python is..." },
   *   ],
   * );
   *
   * // Scores are sorted by relevance (descending)
   * console.log(result.scores[0].itemId); // most relevant
   * ```
   */
  async score(
    model: string,
    query: Item,
    items: Item[],
    options: ScoreOptions = {},
  ): Promise<ScoreResult> {
    // Build request body
    const body: Record<string, unknown> = {
      query,
      items,
    };

    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;
    const { pool, gpu } = this.parseGpuParam(options.gpu);

    const response = await this.requestWithRetry(
      `/v1/score/${encodeURIComponent(model)}`,
      body,
      pool,
      gpu,
      waitForCapacity,
      model,
    );

    // Wire format response matches ScoreResult structure
    const data = unpackMessage<unknown>(new Uint8Array(await response.arrayBuffer()));

    return parseScoreResult(data);
  }

  /**
   * Extract entities from a single item.
   *
   * @param model - Model name (e.g., "gliner-multi-v2.1")
   * @param item - Item to extract from
   * @param options - Extract options with labels
   * @returns Extract result with entities
   */
  async extract(model: string, item: Item, options: ExtractOptions): Promise<ExtractResult>;

  /**
   * Extract entities from multiple items.
   *
   * @param model - Model name (e.g., "gliner-multi-v2.1")
   * @param items - Items to extract from
   * @param options - Extract options with labels
   * @returns Array of extract results in same order as input
   */
  async extract(model: string, items: Item[], options: ExtractOptions): Promise<ExtractResult[]>;

  /**
   * Extract entities from one or more items.
   *
   * @example
   * ```typescript
   * const result = await client.extract(
   *   "gliner-multi-v2.1",
   *   { text: "Apple was founded by Steve Jobs." },
   *   { labels: ["person", "organization"] },
   * );
   *
   * for (const entity of result.entities) {
   *   console.log(`${entity.text} (${entity.label})`);
   * }
   * // Output:
   * // Apple (organization)
   * // Steve Jobs (person)
   * ```
   */
  async extract(
    model: string,
    items: Item | Item[],
    options: ExtractOptions,
  ): Promise<ExtractResult | ExtractResult[]> {
    const isSingleItem = !Array.isArray(items);
    const itemsArray = isSingleItem ? [items] : items;

    // Build request body
    const body: Record<string, unknown> = {
      items: itemsArray,
    };

    // Add params
    const params: Record<string, unknown> = {
      labels: options.labels,
    };
    if (options.threshold !== undefined) {
      params.threshold = options.threshold;
    }
    body.params = params;

    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;
    const { pool, gpu } = this.parseGpuParam(options.gpu);

    const response = await this.requestWithRetry(
      `/v1/extract/${encodeURIComponent(model)}`,
      body,
      pool,
      gpu,
      waitForCapacity,
      model,
    );

    // Wire format response: {"items": [...]}
    interface WireResponse {
      items: unknown[];
    }

    const data = unpackMessage<WireResponse>(new Uint8Array(await response.arrayBuffer()));

    const results = parseExtractResults(data.items);

    if (isSingleItem) {
      const first = results[0];
      if (!first) {
        throw new Error("No results returned from extract");
      }
      return first;
    }
    return results;
  }

  /**
   * Close the client and cleanup resources.
   *
   * Stops pool lease renewal timers. Note that pools are not deleted
   * automatically - they are garbage collected by the gateway after inactivity.
   * This allows pool reuse if the client reconnects.
   */
  async close(): Promise<void> {
    // Stop all pool lease renewal timers and cancel in-flight renewals
    for (const [, poolState] of this.pools) {
      if (poolState.timeoutId !== null) {
        clearTimeout(poolState.timeoutId);
      }
      poolState.abortController.abort();
    }
    this.pools.clear();
  }

  /**
   * Create a resource pool for isolated capacity.
   *
   * Pools provide dedicated worker capacity, isolated from other clients.
   * Workers are assigned to pools and only serve requests from that pool.
   *
   * @param name - Pool name (used in GPU param as "poolName/machineProfile")
   * @param gpus - Machine profile requirements, e.g., { "l4": 2, "l4-spot": 1 }
   *
   * @example
   * ```typescript
   * // Create a pool with 2 L4 GPUs
   * await client.createPool("eval-bench", { l4: 2 });
   *
   * // Use the pool for requests
   * await client.encode("bge-m3", { text: "Hello" }, { gpu: "eval-bench/l4" });
   *
   * // Clean up when done
   * await client.deletePool("eval-bench");
   * ```
   */
  async createPool(name: string, gpus: Record<string, number>): Promise<void> {
    // Check if pool already exists in our tracking
    if (this.pools.has(name)) {
      return; // Pool already created
    }

    // Build pool creation request
    const requestBody = { name, gpus };

    const url = `${this.baseUrl}/v1/pools`;
    const headers: Record<string, string> = {
      "Content-Type": JSON_CONTENT_TYPE,
      Accept: JSON_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };

    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (response.status >= HTTP_CLIENT_ERROR_MIN) {
        let errorMsg = response.statusText;
        try {
          const data = (await response.json()) as { detail?: { message?: string } };
          errorMsg = data.detail?.message ?? JSON.stringify(data);
        } catch {
          // Use status text
        }
        throw new PoolError(`Failed to create pool '${name}': ${errorMsg}`, name);
      }

      // Start lease renewal loop for this pool (recursive setTimeout
      // prevents overlapping runs unlike setInterval)
      const abortController = new AbortController();
      const poolState = {
        timeoutId: null as ReturnType<typeof setTimeout> | null,
        abortController,
        isRenewing: false,
      };

      const renewLoop = async () => {
        if (abortController.signal.aborted) return;
        if (poolState.isRenewing) return;
        poolState.isRenewing = true;

        try {
          const renewUrl = `${this.baseUrl}/v1/pools/${encodeURIComponent(name)}/renew`;
          const renewHeaders: Record<string, string> = {
            Accept: JSON_CONTENT_TYPE,
          };

          if (this.apiKey) {
            renewHeaders.Authorization = `Bearer ${this.apiKey}`;
          }

          for (let attempt = 0; attempt < _LEASE_RENEWAL_MAX_RETRIES; attempt++) {
            if (abortController.signal.aborted) return;

            // Per-attempt controller: times out individual fetches and
            // forwards the pool-level abort so close()/deletePool() cancels
            // in-flight requests immediately.
            const perAttempt = new AbortController();
            const onPoolAbort = () => perAttempt.abort();
            abortController.signal.addEventListener("abort", onPoolAbort, { once: true });
            const attemptTimeout = setTimeout(() => perAttempt.abort(), this.timeout);

            try {
              const resp = await fetch(renewUrl, {
                method: "POST",
                headers: renewHeaders,
                signal: perAttempt.signal,
              });
              if (resp.ok) break;
            } catch (error) {
              // Pool-level abort → stop entirely
              if (abortController.signal.aborted) return;
              // Per-attempt timeout or network error → fall through to retry
            } finally {
              clearTimeout(attemptTimeout);
              abortController.signal.removeEventListener("abort", onPoolAbort);
            }
            if (attempt < _LEASE_RENEWAL_MAX_RETRIES - 1) {
              const aborted = await abortableSleep(
                Math.min(2 ** attempt * 1000, 10000),
                abortController.signal,
              );
              if (aborted) return;
            }
          }
        } finally {
          poolState.isRenewing = false;
        }

        // Schedule next renewal only after current run finishes
        if (!abortController.signal.aborted) {
          poolState.timeoutId = setTimeout(renewLoop, DEFAULT_LEASE_RENEWAL_INTERVAL);
        }
      };

      poolState.timeoutId = setTimeout(renewLoop, DEFAULT_LEASE_RENEWAL_INTERVAL);
      this.pools.set(name, poolState);
    } catch (error) {
      if (error instanceof PoolError) {
        throw error;
      }
      if (error instanceof Error && error.name === "AbortError") {
        throw new PoolError(`Timeout creating pool '${name}'`, name);
      }
      throw new PoolError(
        `Failed to create pool '${name}': ${error instanceof Error ? error.message : "Unknown error"}`,
        name,
      );
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get information about a pool.
   *
   * @param name - Pool name to query
   * @returns PoolInfo if pool exists, null otherwise
   *
   * @example
   * ```typescript
   * await client.createPool("eval-bench", { l4: 2 });
   * const pool = await client.getPool("eval-bench");
   * console.log(`Pool state: ${pool?.status.state}`);
   * console.log(`Workers: ${pool?.status.assignedWorkers.length}`);
   * ```
   */
  async getPool(name: string): Promise<PoolInfo | null> {
    try {
      const response = await this.requestJson(`/v1/pools/${encodeURIComponent(name)}`);
      const data = (await response.json()) as {
        name: string;
        spec: { gpus?: Record<string, number> };
        status: {
          state: string;
          assigned_workers: Array<{ name: string; url: string; gpu: string }>;
          created_at?: number;
          last_renewed?: number;
        };
      };

      return {
        name: data.name,
        spec: data.spec,
        status: {
          state: data.status.state,
          assignedWorkers: data.status.assigned_workers,
          createdAt: data.status.created_at,
          lastRenewed: data.status.last_renewed,
        },
      };
    } catch {
      // Pool might not exist
      return null;
    }
  }

  /**
   * Delete a pool.
   *
   * @param name - Pool name to delete
   * @returns true if pool was deleted, false if pool didn't exist
   *
   * @example
   * ```typescript
   * // Clean up pool when done
   * const deleted = await client.deletePool("eval-bench");
   * if (deleted) {
   *   console.log("Pool deleted successfully");
   * }
   * ```
   */
  async deletePool(name: string): Promise<boolean> {
    // Stop lease renewal first if we're tracking this pool
    const poolState = this.pools.get(name);
    if (poolState) {
      if (poolState.timeoutId !== null) {
        clearTimeout(poolState.timeoutId);
      }
      poolState.abortController.abort();
      this.pools.delete(name);
    }

    try {
      const url = `${this.baseUrl}/v1/pools/${encodeURIComponent(name)}`;
      const headers: Record<string, string> = {
        Accept: JSON_CONTENT_TYPE,
      };

      if (this.apiKey) {
        headers.Authorization = `Bearer ${this.apiKey}`;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        const response = await fetch(url, {
          method: "DELETE",
          headers,
          signal: controller.signal,
        });

        return response.ok || response.status === 404;
      } finally {
        clearTimeout(timeoutId);
      }
    } catch {
      return false;
    }
  }

  private checkServerVersion(response: Response): void {
    if (this.versionWarningLogged) return;
    const serverVersion = response.headers.get(SERVER_VERSION_HEADER);
    if (!serverVersion) return;
    try {
      const sdkParts = SDK_VERSION.split(".").map(Number);
      const serverParts = serverVersion.split(".").map(Number);
      if (sdkParts.length < 2 || serverParts.length < 2) return;
      const sdkMajor = sdkParts[0];
      const sdkMinor = sdkParts[1];
      const serverMajor = serverParts[0];
      const serverMinor = serverParts[1];
      if (
        sdkMajor === undefined ||
        sdkMinor === undefined ||
        serverMajor === undefined ||
        serverMinor === undefined
      ) {
        return;
      }
      if (sdkMajor !== serverMajor || Math.abs(sdkMinor - serverMinor) > 1) {
        console.warn(
          `[SIE SDK] Version skew detected: SDK ${SDK_VERSION}, server ${serverVersion}. Consider upgrading.`,
        );
        this.versionWarningLogged = true;
      }
    } catch {
      // Ignore parse errors
    }
  }

  /**
   * Parse GPU parameter into pool and GPU components.
   *
   * Supports "pool/gpu" format for pool routing.
   */
  private parseGpuParam(gpu?: string): { pool?: string; gpu?: string } {
    const effectiveGpu = gpu ?? this.gpu;

    if (!effectiveGpu) {
      return {};
    }

    // Parse "pool/gpu" format
    const parts = effectiveGpu.split("/");
    if (parts.length === 2 && parts[0] && parts[1]) {
      return { pool: parts[0], gpu: parts[1] };
    }

    return { gpu: effectiveGpu };
  }

  /**
   * Get current cluster capacity information.
   *
   * Queries the gateway's /health endpoint for cluster state. Useful for
   * checking if specific GPU types are available before sending requests.
   *
   * @param gpu - Optional filter to check specific GPU type availability
   * @returns CapacityInfo with worker count, GPU types, and worker details
   *
   * @example
   * ```typescript
   * // Check cluster state
   * const capacity = await client.getCapacity();
   * console.log(`Workers: ${capacity.workerCount}, GPUs: ${capacity.liveGpuTypes}`);
   *
   * // Check if L4 GPUs are available
   * const l4Capacity = await client.getCapacity("l4");
   * if (l4Capacity.workerCount > 0) {
   *   console.log("L4 workers available");
   * }
   * ```
   */
  async getCapacity(gpu?: string): Promise<CapacityInfo> {
    const response = await this.requestJson("/health");
    const data = (await response.json()) as { type?: string };

    // Check if this is a gateway (has 'type': 'gateway') or worker
    if (data.type !== "gateway") {
      throw new RequestError(
        "getCapacity() requires a gateway endpoint. This appears to be a worker.",
        "not_gateway",
        400,
      );
    }

    return parseCapacityInfo(data, gpu);
  }

  /**
   * Wait for GPU capacity to become available.
   *
   * Polls the gateway until workers with the specified GPU type are online.
   * This is useful for pre-warming the cluster before running benchmarks.
   *
   * @param gpu - GPU type to wait for (e.g., "l4", "a100-80gb")
   * @param options - Wait options
   * @returns CapacityInfo once capacity is available
   *
   * @example
   * ```typescript
   * // Wait for L4 capacity before running benchmarks
   * const capacity = await client.waitForCapacity("l4", { timeout: 300000 });
   * console.log(`Ready with ${capacity.workerCount} L4 workers`);
   *
   * // Wait and pre-load a model
   * const capacityWithModel = await client.waitForCapacity("l4", { model: "bge-m3" });
   * ```
   */
  async waitForCapacity(
    gpu: string,
    options: { model?: string; timeout?: number; pollInterval?: number } = {},
  ): Promise<CapacityInfo> {
    const timeout = options.timeout ?? this.provisionTimeout;
    const pollInterval = options.pollInterval ?? 5000;
    const startTime = Date.now();

    // If model is specified, use encode with waitForCapacity to trigger
    // both scale-up and model loading
    if (options.model) {
      await this.encode(options.model, { text: "warmup" }, { gpu, waitForCapacity: true });
      // After successful encode, get capacity info
      return this.getCapacity(gpu);
    }

    // Otherwise, poll capacity until workers are available
    while (true) {
      try {
        const capacity = await this.getCapacity(gpu);
        if (capacity.workerCount > 0) {
          return capacity;
        }
      } catch {
        // Keep trying on errors
      }

      const elapsed = Date.now() - startTime;
      if (elapsed >= timeout) {
        throw new ProvisioningError(
          `Timeout after ${elapsed}ms waiting for GPU '${gpu}' capacity`,
          gpu,
        );
      }

      // Wait before next poll
      const remaining = timeout - elapsed;
      const delay = Math.min(pollInterval, remaining);
      await sleep(delay);
    }
  }

  /**
   * Make a msgpack HTTP request with retry logic.
   *
   * Retried (only when `waitForCapacity: true`, capped by `provisionTimeout`):
   *  - 202 Accepted (provisioning)
   *  - 503 `MODEL_LOADING` / `LORA_LOADING` / no error code (scale-from-zero)
   *  - `SIEConnectionError` with `kind === "connect"` (issue #95)
   *
   * `kind === "timeout"` is NOT retried — would extend the user-visible
   * timeout from `timeout` to `provisionTimeout`.
   */
  private async requestWithRetry(
    path: string,
    body: unknown,
    pool: string | undefined,
    gpu: string | undefined,
    waitForCapacity: boolean,
    model: string,
  ): Promise<Response> {
    const startTime = Date.now();

    // Local retry counter for LoRA loading (uses retry count, not time-based)
    // Model loading uses cumulative time check, not retry counter
    let loraRetries = 0;

    while (true) {
      let response: Response;
      try {
        response = await this.request(path, body, pool, gpu);
      } catch (err) {
        // Only retry connect-time failures; see docstring for rationale.
        if (waitForCapacity && err instanceof SIEConnectionError && err.kind === "connect") {
          const elapsed = Date.now() - startTime;
          if (elapsed < this.provisionTimeout) {
            const remaining = this.provisionTimeout - elapsed;
            const delay = Math.min(DEFAULT_RETRY_DELAY, remaining);
            await sleep(delay);
            continue;
          }
        }
        throw err;
      }

      // Handle 202 (provisioning) - capacity not available
      if (response.status === HTTP_ACCEPTED) {
        const retryAfter = getRetryAfter(response);

        if (!waitForCapacity) {
          throw new ProvisioningError(
            `No capacity available for GPU '${gpu}'. Server is provisioning.`,
            gpu,
            retryAfter,
          );
        }

        // Check if we've exceeded the timeout
        const elapsed = Date.now() - startTime;
        if (elapsed >= this.provisionTimeout) {
          throw new ProvisioningError(
            `Provisioning timeout after ${elapsed}ms waiting for GPU '${gpu}'`,
            gpu,
            retryAfter,
          );
        }

        // Wait and retry
        const delay = retryAfter ?? DEFAULT_RETRY_DELAY;
        const remaining = this.provisionTimeout - elapsed;
        const actualDelay = Math.min(delay, remaining);
        await sleep(actualDelay);
        continue;
      }

      // Handle 503 with LORA_LOADING or MODEL_LOADING - auto-retry
      if (response.status === 503) {
        const clonedResponse = response.clone();
        const errorCode = await getErrorCode(clonedResponse);

        if (errorCode === LORA_LOADING_ERROR_CODE) {
          loraRetries += 1;

          if (loraRetries > LORA_LOADING_MAX_RETRIES) {
            throw new LoraLoadingError(
              `LoRA loading timeout after ${loraRetries} retries`,
              undefined, // We don't have lora name at this level
              model,
            );
          }

          // Wait and retry
          const retryAfter = getRetryAfter(response);
          const delay = retryAfter ?? LORA_LOADING_DEFAULT_DELAY;
          await sleep(delay);
          continue;
        }

        if (errorCode === MODEL_LOADING_ERROR_CODE) {
          // Check if we've exceeded the provision timeout (cumulative wall-clock time)
          const elapsed = Date.now() - startTime;
          if (elapsed >= this.provisionTimeout) {
            throw new ModelLoadingError(
              `Model loading timeout after ${(elapsed / 1000).toFixed(1)}s for '${model}'`,
              model,
            );
          }

          // Wait and retry, respecting remaining time
          const retryAfter = getRetryAfter(response);
          const delay = retryAfter ?? MODEL_LOADING_DEFAULT_DELAY;
          const remaining = this.provisionTimeout - elapsed;
          const actualDelay = Math.min(delay, remaining);
          await sleep(actualDelay);
          continue;
        }

        // Generic 503 fall-through (router/gateway scale-from-zero).
        // New code-specific 503 branches MUST go ABOVE this block.
        if (waitForCapacity) {
          const elapsed = Date.now() - startTime;
          if (elapsed < this.provisionTimeout) {
            const retryAfter = getRetryAfter(response);
            const delay = retryAfter ?? DEFAULT_RETRY_DELAY;
            const remaining = this.provisionTimeout - elapsed;
            const actualDelay = Math.min(delay, remaining);
            await sleep(actualDelay);
            continue;
          }
        }
      }

      // Handle other errors
      if (!response.ok) {
        await handleError(response, gpu);
      }

      // Success
      this.checkServerVersion(response);
      return response;
    }
  }

  /**
   * Make a single msgpack HTTP request to the SIE server (no retry logic).
   */
  private async request(
    path: string,
    body?: unknown,
    pool?: string,
    gpu?: string,
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;

    const headers: Record<string, string> = {
      Accept: MSGPACK_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };

    if (body !== undefined) {
      headers["Content-Type"] = MSGPACK_CONTENT_TYPE;
    }

    // Pool header takes precedence for routing
    if (pool) {
      headers["X-SIE-Pool"] = pool;
    }

    if (gpu) {
      headers["X-SIE-MACHINE-PROFILE"] = gpu;
    }

    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: body !== undefined ? packMessage(body) : undefined,
        signal: controller.signal,
      });

      return response;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new SIEConnectionError(`Request timeout after ${this.timeout}ms`, "timeout");
      }
      if (error instanceof TypeError) {
        throw new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Make a JSON HTTP request to the SIE server.
   * Used for endpoints that return JSON (e.g., /v1/models, /health).
   */
  private async requestJson(path: string, method: "GET" | "POST" = "GET"): Promise<Response> {
    const url = `${this.baseUrl}${path}`;

    const headers: Record<string, string> = {
      Accept: "application/json",
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };

    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        await handleError(response);
      }

      return response;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new SIEConnectionError(`Request timeout after ${this.timeout}ms`, "timeout");
      }
      if (error instanceof TypeError) {
        throw new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private buildWsUrl(path: string): string {
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `${url.pathname.replace(/\/$/, "")}${path}`;
    url.search = "";
    return url.toString();
  }

  private createWebSocket(url: string): WebSocket {
    const headers: Record<string, string> | undefined = this.apiKey
      ? { Authorization: `Bearer ${this.apiKey}` }
      : undefined;

    try {
      if (headers) {
        return new (
          WebSocket as unknown as {
            new (u: string, p?: string[], o?: { headers: Record<string, string> }): WebSocket;
          }
        )(url, [], { headers });
      }
      return new WebSocket(url);
    } catch (error) {
      if (headers) {
        throw new SIEConnectionError(
          "WebSocket auth headers are not supported in this environment",
        );
      }
      throw error;
    }
  }

  private async detectEndpointType(): Promise<"cluster" | "worker"> {
    const url = `${this.baseUrl}/health`;
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        return "worker";
      }

      const data = (await response.json()) as { type?: string };
      return data.type === "gateway" ? "cluster" : "worker";
    } catch {
      return "worker";
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
