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
  ResourceExhaustedError,
  SIEConnectionError,
  SIEStreamError,
  ServerError,
} from "./errors.js";
import { toImageWireFormat } from "./images.js";
import type { ImageInput, ImageWireFormat } from "./images.js";
import {
  DEFAULT_JOB_WAIT_POLL,
  DEFAULT_JOB_WAIT_TIMEOUT,
  DEFAULT_LEASE_RENEWAL_INTERVAL,
  DEFAULT_LONG_RUNNING_TIMEOUT,
  DEFAULT_PROVISION_TIMEOUT,
  DEFAULT_RETRY_DELAY,
  DEFAULT_TIMEOUT,
  HTTP_CLIENT_ERROR_MIN,
  HTTP_GATEWAY_TIMEOUT,
  JSON_CONTENT_TYPE,
  LORA_LOADING_DEFAULT_DELAY,
  LORA_LOADING_ERROR_CODE,
  LORA_LOADING_MAX_RETRIES,
  MODEL_LOADING_DEFAULT_DELAY,
  MODEL_LOADING_ERROR_CODE,
  MSGPACK_CONTENT_TYPE,
  PROVISIONING_ERROR_CODE,
  RESOURCE_EXHAUSTED_ERROR_CODE,
  RESOURCE_EXHAUSTED_MAX_RETRIES,
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
  parseGenerateResult,
  parseScoreResult,
  throwIfInputTooLong,
  throwIfModelLoadFailed,
} from "./internal/parsing.js";
import { nextOomRetryDelay, withProvisioningRetry } from "./internal/provisioning.js";
import { applyRetryJitter } from "./internal/retry.js";
import {
  type JobResultItem,
  type JobResults,
  type JobStatus,
  type JobSubmitResult,
  type SubmitJobOptions,
  TERMINAL_JOB_STATES,
  buildJobBody,
  decodeChunkBytes,
  jobChunks,
} from "./jobs.js";
import { packMessage, unpackMessage } from "./msgpack.js";
import { parseSseStream } from "./sse.js";
import type {
  Batch,
  CapacityInfo,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionOptions,
  ChatCompletionRequest,
  Connection,
  ConnectionCreated,
  ConnectionRevoked,
  CreatePoolOptions,
  EncodeOptions,
  EncodeResult,
  ExtractItem,
  ExtractOptions,
  ExtractResult,
  FileDeleted,
  GenerateChunk,
  GenerateGrammar,
  GenerateOptions,
  GenerateResult,
  Item,
  ModelInfo,
  PoolInfo,
  PoolSpec,
  RequestMetadata,
  SIEClientOptions,
  File as SIEFile,
  ScoreOptions,
  ScoreResult,
  StatusMessage,
  StreamGenerateOptions,
} from "./types.js";
import { SDK_VERSION } from "./version.js";

/** The `client.jobs` batch namespace. */
export interface JobsNamespace {
  /** Submit a batch job (`POST /v1/jobs`); returns the created-job envelope. */
  submit(options: SubmitJobOptions): Promise<JobSubmitResult>;
  /** Fetch a job's public status doc (`GET /v1/jobs/{id}`). */
  get(jobId: string): Promise<JobStatus>;
  /** List the org's jobs (`GET /v1/jobs`; scoped to the key's org). */
  list(): Promise<JobStatus[]>;
  /** Cancel a job (`POST /v1/jobs/{id}/cancel`); the hold's remainder releases. */
  cancel(jobId: string): Promise<JobStatus>;
  /** Retrieve a finished job's chunk refs and decode the per-item results. */
  results(jobId: string): Promise<JobResults>;
  /**
   * Poll `get` until the job reaches a terminal state, then return its status.
   * Throws a `job_wait_timeout` `RequestError` if `timeoutMs` elapses first.
   * Mirrors the Python SDK's `jobs.wait` (default 600s timeout, 2s poll).
   */
  wait(jobId: string, options?: { timeoutMs?: number; pollMs?: number }): Promise<JobStatus>;
}

/** The `client.connections` namespace (org-scoped connector auth). */
export interface ConnectionsNamespace {
  /** Create an org-scoped connection (connector auth by name). */
  add(name: string, type: string, secret: string): Promise<ConnectionCreated>;
  /** List the org's active connections (secrets redacted). */
  list(): Promise<Connection[]>;
  /** Revoke (soft-delete) a connection; frees the name for reuse. */
  revoke(name: string): Promise<ConnectionRevoked>;
}

/** Accepted upload payloads — the same shapes `fetch` sends as a body. */
export type FileUploadInput = Uint8Array | ArrayBuffer | string | Blob;

/**
 * The `client.files` OpenAI-compatible Files namespace. Method
 * names/args mirror `openai.files` so an `openai` → `sie-sdk` swap is mechanical.
 */
export interface FilesNamespace {
  /** Upload a file (`POST /v1/files`); `purpose` defaults to `"batch"`. */
  upload(
    file: FileUploadInput,
    options?: { purpose?: string; filename?: string },
  ): Promise<SIEFile>;
  /** OpenAI-exact alias for {@link upload} (`files.create({ file, purpose })`). */
  create(options: { file: FileUploadInput; purpose?: string; filename?: string }): Promise<SIEFile>;
  /** Fetch a file's metadata (`GET /v1/files/{id}`). */
  retrieve(fileId: string): Promise<SIEFile>;
  /** Download a file's raw bytes (`GET /v1/files/{id}/content`). */
  content(fileId: string): Promise<Uint8Array>;
  /** Delete a file (`DELETE /v1/files/{id}`; additive OpenAI-parity surface). */
  delete(fileId: string): Promise<FileDeleted>;
}

/**
 * The `client.batches` OpenAI-compatible Batch namespace. A
 * batch is a job over an uploaded file's JSONL lines;
 * `list` / `cancel` are the additive OpenAI-parity completion of the surface.
 * Args are OpenAI's exact snake_case body keys, so an `openai` → `sie-sdk` swap
 * is mechanical.
 */
export interface BatchesNamespace {
  /** Create a batch (`POST /v1/batches`); returns the Batch object. */
  create(options: {
    input_file_id: string;
    endpoint?: string;
    completion_window?: string;
    metadata?: Record<string, unknown>;
  }): Promise<Batch>;
  /** Fetch a batch's status (`GET /v1/batches/{id}`). */
  retrieve(batchId: string): Promise<Batch>;
  /** List the org's batches (`GET /v1/batches`; additive OpenAI-parity). */
  list(): Promise<Batch[]>;
  /** Cancel a batch (`POST /v1/batches/{id}/cancel`; additive OpenAI-parity). */
  cancel(batchId: string): Promise<Batch>;
}

/** Helper to sleep for a given number of milliseconds */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseNonnegativeMeterHeader(headers: Headers, name: string): number | undefined {
  const raw = headers.get(name);
  if (raw === null || !/^(0|[1-9]\d*)$/.test(raw)) return undefined;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

/** Parse optional request-scoped metadata from one successful terminal response. */
function parseRequestMetadata(headers: Headers): RequestMetadata | undefined {
  const metadata: RequestMetadata = {};
  const requestId = headers.get("x-sie-request-id");
  if (
    requestId !== null &&
    requestId.length > 0 &&
    requestId.length <= 256 &&
    requestId === requestId.trim() &&
    /^[\x20-\x7e]+$/.test(requestId)
  ) {
    metadata.id = requestId;
  }
  const executionIdentitySha256 = headers.get("x-sie-execution-identity-sha256");
  if (executionIdentitySha256 !== null && /^[0-9a-f]{64}$/.test(executionIdentitySha256)) {
    metadata.executionIdentitySha256 = executionIdentitySha256;
  }

  const usageHeaders = {
    inputTokens: "x-sie-units-input-tokens",
    pairs: "x-sie-units-pairs",
    images: "x-sie-units-images",
    pages: "x-sie-units-pages",
    outputTokens: "x-sie-units-output-tokens",
    audioMs: "x-sie-units-audio-ms",
  } as const;
  const usage: NonNullable<RequestMetadata["usage"]> = {};
  for (const [field, header] of Object.entries(usageHeaders) as [
    keyof typeof usageHeaders,
    string,
  ][]) {
    const value = parseNonnegativeMeterHeader(headers, header);
    if (value !== undefined) usage[field] = value;
  }
  if (Object.keys(usage).length > 0) metadata.usage = usage;

  const creditsDebited = parseNonnegativeMeterHeader(headers, "x-sie-credits-debited");
  if (creditsDebited !== undefined) metadata.creditsDebited = creditsDebited;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

function attachRequestMetadata<T extends { request?: RequestMetadata }>(
  results: T[],
  headers: Headers,
): void {
  const metadata = parseRequestMetadata(headers);
  if (metadata === undefined) return;
  for (const result of results) {
    result.request = { ...metadata };
    if (metadata.usage !== undefined) result.request.usage = { ...metadata.usage };
  }
}

/** Enforce the JavaScript safe-integer contract before signed-i64 gateway transport. */
function validateGenerationSeed(seed: number): number {
  if (!Number.isSafeInteger(seed)) {
    throw new RangeError("seed must be a JavaScript safe integer");
  }
  return seed;
}

/** Validate the discriminated native grammar before issuing a billable request. */
function validateGenerateGrammar(grammar: GenerateGrammar | Record<string, unknown>): void {
  if (typeof grammar !== "object" || grammar === null || Array.isArray(grammar)) {
    throw new TypeError("grammar must be an object");
  }
  const allowed = new Set(["json_schema", "regex", "ebnf", "label", "strict"]);
  const unknown = Object.keys(grammar).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new TypeError(`grammar contains unsupported field(s): ${unknown.sort().join(", ")}`);
  }
  const variants = ["json_schema", "regex", "ebnf"].filter((key) => Object.hasOwn(grammar, key));
  if (variants.length !== 1) {
    throw new TypeError("grammar must contain exactly one of json_schema, regex, or ebnf");
  }
  const variant = variants[0];
  const value = grammar[variant as keyof GenerateGrammar];
  if (
    variant === "json_schema"
      ? typeof value !== "object" || value === null || Array.isArray(value)
      : typeof value !== "string"
  ) {
    throw new TypeError(
      variant === "json_schema"
        ? "grammar.json_schema must be an object"
        : `grammar.${variant} must be a string`,
    );
  }
  if (grammar.label !== undefined && grammar.label !== null && typeof grammar.label !== "string") {
    throw new TypeError("grammar.label must be a string");
  }
  if (
    grammar.strict !== undefined &&
    grammar.strict !== null &&
    typeof grammar.strict !== "boolean"
  ) {
    throw new TypeError("grammar.strict must be a boolean");
  }
}

/** Serialize the controls shared by blocking and streaming native generation. */
function applyGenerateOptions(body: Record<string, unknown>, options: GenerateOptions): void {
  if (options.temperature !== undefined) body.temperature = options.temperature;
  if (options.topP !== undefined) body.top_p = options.topP;
  if (options.adapterOptions !== undefined) body.options = options.adapterOptions;
  if (options.stop !== undefined) body.stop = options.stop;
  if (options.frequencyPenalty !== undefined) body.frequency_penalty = options.frequencyPenalty;
  if (options.presencePenalty !== undefined) body.presence_penalty = options.presencePenalty;
  if (options.grammar !== undefined) {
    validateGenerateGrammar(options.grammar);
    body.grammar = options.grammar;
  }
  if (options.seed !== undefined) body.seed = validateGenerationSeed(options.seed);
  if (options.logitBias !== undefined) body.logit_bias = options.logitBias;
  if (options.routingKey !== undefined) body.routing_key = options.routingKey;
  if (options.promptCacheKey !== undefined) body.prompt_cache_key = options.promptCacheKey;
  if (options.safetyIdentifier !== undefined) body.safety_identifier = options.safetyIdentifier;
  if (options.loraAdapter !== undefined) body.lora_adapter = options.loraAdapter;
}

/** Derive an upload filename: an explicit override > a File/Blob `.name` > default. */
function resolveUploadFilename(file: FileUploadInput, filename?: string): string {
  if (filename) return filename;
  const name = (file as { name?: unknown }).name;
  if (typeof name === "string" && name.length > 0) {
    // A File's `.name` may include a path; keep just the basename.
    return name.split(/[/\\]/).pop() || "upload.jsonl";
  }
  return "upload.jsonl";
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

type ItemWithWireImages = Omit<Item, "images"> & { images?: ImageWireFormat[] };
type ItemForWire = Item | ItemWithWireImages;

interface AudioWireFormat {
  data: Uint8Array;
  format?: string;
  sample_rate?: number;
}

type ExtractItemForWire = Omit<ExtractItem, "images" | "audio"> & {
  images?: ImageWireFormat[];
  audio?: AudioWireFormat;
};

function isImageWireFormat(image: ImageInput | ImageWireFormat): image is ImageWireFormat {
  return typeof image === "object" && image !== null && "data" in image;
}

async function imageForWire(image: ImageInput | ImageWireFormat): Promise<ImageWireFormat> {
  if (isImageWireFormat(image)) {
    return image;
  }
  return toImageWireFormat(image);
}

function imageBytesToBase64(data: Uint8Array): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString("base64");
  }
  let binary = "";
  const chunkSize = 32_768;
  for (let offset = 0; offset < data.length; offset += chunkSize) {
    binary += String.fromCharCode(...data.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function generationImagesForWire(
  images: (ImageInput | ImageWireFormat)[],
): Promise<{ data: string; format: ImageWireFormat["format"] }[]> {
  return Promise.all(
    images.map(async (image) => {
      const wire = await imageForWire(image);
      return { data: imageBytesToBase64(wire.data), format: wire.format };
    }),
  );
}

async function itemImagesForWire(item: Item): Promise<ItemForWire> {
  if (!item.images || item.images.length === 0) {
    return item;
  }
  return { ...item, images: await Promise.all(item.images.map(imageForWire)) };
}

async function itemsImagesForWire(items: Item[]): Promise<ItemForWire[]> {
  return Promise.all(items.map(itemImagesForWire));
}

async function extractItemForWire(item: ExtractItem): Promise<ExtractItemForWire> {
  const { images, audio, ...wireItem } = item;
  const wireImages = images ? await Promise.all(images.map(imageForWire)) : undefined;
  if (!audio) {
    return { ...wireItem, ...(wireImages === undefined ? {} : { images: wireImages }) };
  }
  if (audio instanceof Uint8Array) {
    return {
      ...wireItem,
      ...(wireImages === undefined ? {} : { images: wireImages }),
      audio: { data: audio },
    };
  }
  const { sampleRate, ...wireAudio } = audio;
  return {
    ...wireItem,
    ...(wireImages === undefined ? {} : { images: wireImages }),
    audio: {
      ...wireAudio,
      ...(sampleRate === undefined ? {} : { sample_rate: sampleRate }),
    },
  };
}

async function itemsForExtractWire(items: ExtractItem[]): Promise<ExtractItemForWire[]> {
  return Promise.all(items.map(extractItemForWire));
}

/**
 * Pluck a mid-stream `error` block out of a `ChatCompletionChunk` and
 * convert it to `SIEStreamError`, mirroring the shape `sse.rs` emits:
 * `{ message, type, param, code }`. Returns `null` when the chunk is a
 * normal delta. Defined at module scope so it has zero coupling to
 * `SIEClient` state.
 */
function extractChatChunkError(chunk: ChatCompletionChunk): SIEStreamError | null {
  const err = (
    chunk as ChatCompletionChunk & {
      error?: { message?: string; type?: string; param?: string | null; code?: string };
    }
  ).error;
  if (!err) return null;
  return new SIEStreamError(err.message ?? "stream error", {
    code: err.code,
    errorType: err.type,
    param: err.param,
  });
}

/** SIE-native chunk variant — see `sse.rs::build_generate_chunk_event`. */
function extractGenerateChunkError(chunk: GenerateChunk): SIEStreamError | null {
  if (!chunk.error) return null;
  return new SIEStreamError(chunk.error.message, { code: chunk.error.code });
}

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
 * // Create a logical pool backed by the cluster's default worker queue
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
  private readonly controlPlaneUrl?: string;
  private readonly org?: string;

  /** Batch class — `POST/GET /v1/jobs` on the keyed gateway. */
  readonly jobs: JobsNamespace;
  /** Org-scoped connections (connector auth by name) on the control plane. */
  readonly connections: ConnectionsNamespace;
  /** OpenAI-compatible Files API — `POST/GET /v1/files`. */
  readonly files: FilesNamespace;
  /** OpenAI-compatible Batch API — `POST/GET /v1/batches`. */
  readonly batches: BatchesNamespace;

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
    // BREAKING CHANGE (0.7): default flipped from `false` to `true` to match
    // the Python SDK (`wait_for_capacity=True`). Callers that relied on
    // fail-fast 503 PROVISIONING / connect-error behaviour must now pass
    // `waitForCapacity: false` explicitly.
    this.defaultWaitForCapacity = options.waitForCapacity ?? true;
    this.provisionTimeout = options.provisionTimeout ?? DEFAULT_PROVISION_TIMEOUT;
    this.controlPlaneUrl = options.controlPlaneUrl?.replace(/\/$/, "");
    this.org = options.org;

    // First-class batch + connector surface.
    this.jobs = {
      submit: (submitOptions) => this.jobSubmit(submitOptions),
      get: (jobId) => this.jobGet(jobId),
      list: () => this.jobList(),
      cancel: (jobId) => this.jobCancel(jobId),
      results: (jobId) => this.jobResults(jobId),
      wait: (jobId, options) => this.jobWait(jobId, options),
    };
    this.connections = {
      add: (name, type, secret) => this.connectionAdd(name, type, secret),
      list: () => this.connectionList(),
      revoke: (name) => this.connectionRevoke(name),
    };
    // OpenAI-compatible Files + Batches surface — a base_url
    // swap makes an `openai` batch caller work unchanged.
    this.files = {
      upload: (file, options) => this.fileUpload(file, options),
      create: (options) => this.fileUpload(options.file, options),
      retrieve: (fileId) => this.fileRetrieve(fileId),
      content: (fileId) => this.fileContent(fileId),
      delete: (fileId) => this.fileDelete(fileId),
    };
    this.batches = {
      create: (options) => this.batchCreate(options),
      retrieve: (batchId) => this.batchRetrieve(batchId),
      list: () => this.batchList(),
      cancel: (batchId) => this.batchCancel(batchId),
    };
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
    const itemsForWire = await itemsImagesForWire(itemsArray);

    // Build request body - model is in URL path, not body
    // Wire format uses snake_case
    const body: Record<string, unknown> = {
      items: itemsForWire,
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
    attachRequestMetadata(results, response.headers);

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
  /**
   * Generate text from a prompt (walking-skeleton SDK surface).
   *
   * The SDK does not currently expose streaming chunks. The worker streams
   * to the gateway, the gateway aggregates, and the SDK returns the
   * assembled result plus SIE-native timing metadata (TTFT, TPOT,
   * attempt id).
   *
   * @example
   * ```typescript
   * const result = await client.generate(
   *   "Qwen__Qwen3-4B-Instruct-2507",
   *   "Write a haiku about the sea.",
   *   { maxNewTokens: 64, temperature: 0.7 },
   * );
   * console.log(result.text);
   * console.log(`TTFT: ${result.ttftMs}ms`);
   * ```
   */
  async generate(model: string, prompt: string, options: GenerateOptions): Promise<GenerateResult> {
    const body: Record<string, unknown> = {
      prompt,
      max_new_tokens: options.maxNewTokens,
    };
    if (options.images !== undefined) {
      body.images = await generationImagesForWire(options.images);
    }
    applyGenerateOptions(body, options);

    const { pool, gpu } = this.parseGpuParam(options.gpu);
    const headers: Record<string, string> = {
      Accept: "application/json",
      "Content-Type": JSON_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (pool) headers["X-SIE-Pool"] = pool;
    if (gpu) headers["X-SIE-MACHINE-PROFILE"] = gpu;
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;

    const safeModel = model.replaceAll("/", "__");
    const url = `${this.baseUrl}/v1/generate/${encodeURIComponent(safeModel)}`;
    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;

    const response = await withProvisioningRetry(() => this.performJsonPost(url, body, headers), {
      model,
      gpu,
      waitForCapacity,
      provisionTimeoutMs: this.provisionTimeout,
    });

    const data = (await response.json()) as Record<string, unknown>;
    if (data === null || typeof data !== "object") {
      throw new RequestError("Unexpected generate response shape");
    }
    const result = parseGenerateResult(data);
    attachRequestMetadata([result], response.headers);
    return result;
  }

  /**
   * Per-attempt JSON POST used by the non-streaming surfaces
   * ({@link generate}, {@link chatCompletions}) inside the
   * {@link withProvisioningRetry} loop.
   *
   * Translates low-level transport failures into typed errors that the
   * retry loop will surface verbatim:
   *   - `AbortError` → `SIEConnectionError` (per-attempt timeout)
   *   - `TypeError`  → `SIEConnectionError` (NOT retried — generation is
   *     non-idempotent, so a mid-flight drop must surface instead of
   *     silently re-issuing a billable generation)
   *
   * Each call uses a fresh `AbortController` so concurrent retries don't
   * share state, and the per-attempt timeout is bounded by `this.timeout`
   * (NOT the cumulative provisioning budget).
   */
  private async performJsonPost(
    url: string,
    body: unknown,
    headers: Record<string, string>,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      return await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new SIEConnectionError(`Request timeout after ${this.timeout}ms`, "timeout");
      }
      if (err instanceof TypeError) {
        // `generate()` / `chatCompletions()` are non-idempotent and carry
        // no dedup key, so a SECOND attempt issues a SECOND billable
        // generation. `fetch` throws `TypeError` for ANY network failure,
        // including a connection dropped AFTER the request body was sent
        // (mid-flight) — and it cannot reliably distinguish that from a
        // connect-time refusal. Retrying a mid-flight drop would
        // double-bill, so surface as `SIEConnectionError` and let the
        // retry loop propagate it. The SAFE pre-execution capacity
        // signals (503 PROVISIONING / MODEL_LOADING) are HTTP statuses, not
        // exceptions, so the retry loop still handles them.
        throw new SIEConnectionError(`Connection failed: ${err.message}`, "connect");
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Non-streaming chat-completion call against `/v1/chat/completions`.
   *
   * This is the OpenAI-compatible surface. The request body is forwarded
   * verbatim as JSON, so any field documented at
   * <https://platform.openai.com/docs/api-reference/chat/create> can be set;
   * the gateway will reject fields it does not yet support with
   * `400 unsupported_field`. SIE-native routing hints (`routing_key`,
   * `prompt_cache_key`) are part of the same request shape.
   *
   * Error semantics mirror `generate()`: 4xx → `RequestError`, 5xx →
   * `ServerError` (or the more specific `ModelLoadFailedError` for 502
   * `MODEL_LOAD_FAILED`), connection / timeout failures →
   * `SIEConnectionError`.
   *
   * If `req.stream === true`, this method throws `RequestError` immediately —
   * use {@link streamChatCompletions} instead. We do not auto-route because
   * the return type is fundamentally different (`Promise` vs
   * `AsyncGenerator`) and silently flipping would mis-type the call site.
   *
   * @example
   * ```typescript
   * const reply = await client.chatCompletions({
   *   model: "Qwen/Qwen3-4B-Instruct-2507",
   *   messages: [{ role: "user", content: "Write a haiku about the sea." }],
   *   max_completion_tokens: 64,
   * });
   * console.log(reply.choices[0]?.message.content);
   * ```
   */
  async chatCompletions(
    req: ChatCompletionRequest,
    options: ChatCompletionOptions = {},
  ): Promise<ChatCompletion> {
    if (req.stream === true) {
      throw new RequestError(
        "chatCompletions() cannot be used with stream:true — use streamChatCompletions() instead.",
        "invalid_request",
        400,
      );
    }
    if (req.seed !== undefined) {
      validateGenerationSeed(req.seed);
    }

    const body = { ...req, stream: false };
    const url = `${this.baseUrl}/v1/chat/completions`;
    const headers = this.buildChatHeaders("application/json");
    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;
    const provisionTimeoutMs = options.provisionTimeoutMs ?? this.provisionTimeout;

    // H1: pre-execution capacity signals (503 PROVISIONING /
    // MODEL_LOADING) MUST be handled by the shared provisioning loop.
    // The loop also surfaces `ProvisioningError` when the caller opted out
    // (`waitForCapacity: false`) or the provision budget is exhausted,
    // matching `generate()`.
    const response = await withProvisioningRetry(() => this.performJsonPost(url, body, headers), {
      model: req.model,
      gpu: undefined,
      waitForCapacity,
      provisionTimeoutMs,
    });

    this.checkServerVersion(response);

    const data = (await response.json()) as ChatCompletion;
    if (data === null || typeof data !== "object") {
      throw new RequestError("Unexpected chat.completion response shape");
    }
    attachRequestMetadata([data], response.headers);
    return data;
  }

  /**
   * Streaming chat-completion call against `/v1/chat/completions` with
   * `Accept: text/event-stream`.
   *
   * Yields `ChatCompletionChunk` events in the order the gateway emits them.
   * The terminal chunk carries `finish_reason`; if
   * `req.stream_options.include_usage === true`, a final usage-only chunk
   * (`choices: []`, populated `usage`) follows it. The generator completes
   * cleanly on the `data: [DONE]` sentinel.
   *
   * Error semantics:
   *
   *   - HTTP 4xx / 5xx **before** the stream opens → throws `RequestError` /
   *     `ServerError` (same as {@link chatCompletions}).
   *   - A chunk containing `error: { ... }` mid-stream → throws
   *     {@link SIEStreamError}. The error chunk is consumed, never yielded.
   *   - `signal.abort()` mid-stream → the generator throws
   *     `SIEConnectionError` and releases the underlying reader, which
   *     fires `StreamCancelGuard` on the gateway side.
   *
   * `req.stream` is set to `true` automatically; any existing value is
   * overwritten. We do not validate `req.stream === false` because the
   * call-site intent is unambiguous.
   *
   * @param req     The chat-completion request. See {@link ChatCompletionRequest}.
   * @param signal  Optional `AbortSignal` for cooperative cancellation.
   *
   * @example
   * ```typescript
   * const controller = new AbortController();
   * try {
   *   for await (const chunk of client.streamChatCompletions(
   *     {
   *       model: "Qwen/Qwen3-4B-Instruct-2507",
   *       messages: [{ role: "user", content: "Count to ten." }],
   *       stream_options: { include_usage: true },
   *     },
   *     controller.signal,
   *   )) {
   *     process.stdout.write(chunk.choices[0]?.delta.content ?? "");
   *   }
   * } catch (err) {
   *   if (err instanceof SIEStreamError) {
   *     console.error(`mid-stream error: ${err.code} — ${err.message}`);
   *   } else throw err;
   * }
   * ```
   */
  async *streamChatCompletions(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    if (req.seed !== undefined) {
      validateGenerationSeed(req.seed);
    }
    const body = { ...req, stream: true };
    const url = `${this.baseUrl}/v1/chat/completions`;
    yield* this.consumeSseStream<ChatCompletionChunk>(url, body, req.model, signal, (chunk) =>
      extractChatChunkError(chunk),
    );
  }

  /**
   * Streaming companion to {@link generate} — opens an SSE connection to
   * `/v1/generate/{model}` with `stream: true` and yields the SIE-native
   * chunk shape documented in
   * `packages/sie_gateway/src/handlers/sse.rs::build_generate_chunk_event`.
   *
   * Delta chunks may carry text, log probabilities, or both; callers must not
   * assume every non-terminal event has a non-empty `text_delta`. The terminal
   * chunk has `done: true`, `finish_reason`, and (typically) `usage` +
   * `ttft_ms`. The generator completes on the `data: [DONE]` sentinel.
   *
   * Error semantics match {@link streamChatCompletions}: pre-stream HTTP
   * errors throw normally, mid-stream `error` chunks throw
   * {@link SIEStreamError}.
   *
   * @example
   * ```typescript
   * for await (const chunk of client.streamGenerate(
   *   "Qwen/Qwen3-4B-Instruct-2507",
   *   "Write a haiku.",
   *   { maxNewTokens: 64, temperature: 0.7 },
   * )) {
   *   process.stdout.write(chunk.text_delta);
   *   if (chunk.done) console.log(`\nTTFT: ${chunk.ttft_ms}ms`);
   * }
   * ```
   */
  async *streamGenerate(
    model: string,
    prompt: string,
    options: StreamGenerateOptions,
    signal?: AbortSignal,
  ): AsyncGenerator<GenerateChunk, void, undefined> {
    const body: Record<string, unknown> = {
      prompt,
      max_new_tokens: options.maxNewTokens,
      stream: true,
    };
    if (options.images !== undefined) {
      body.images = await generationImagesForWire(options.images);
    }
    applyGenerateOptions(body, options);
    if (options.logprobs !== undefined) body.logprobs = options.logprobs;
    if (options.topLogprobs !== undefined) {
      body.top_logprobs = options.topLogprobs;
      body.logprobs = true;
    }

    const safeModel = model.replaceAll("/", "__");
    const url = `${this.baseUrl}/v1/generate/${encodeURIComponent(safeModel)}`;

    // Routing headers (parallel to generate()) — pool / gpu are passed
    // here even though the SSE handler also reads them from the body
    // for some endpoints, because the gateway looks at headers first.
    const { pool, gpu } = this.parseGpuParam(options.gpu);
    const waitForCapacity = options.waitForCapacity ?? this.defaultWaitForCapacity;
    yield* this.consumeSseStream<GenerateChunk>(
      url,
      body,
      model,
      signal,
      (chunk) => extractGenerateChunkError(chunk),
      { pool, gpu },
      { waitForCapacity },
    );
  }

  /**
   * Shared SSE consumption helper for the streaming methods.
   *
   * Performs a pre-stream provisioning retry loop (honoring
   * `waitForCapacity`/`provisionTimeout`), surfaces pre-stream errors via
   * {@link handleError} (so callers see the same `RequestError` /
   * `ServerError` hierarchy as the non-streaming endpoints), then iterates
   * the SSE payloads via {@link parseSseStream}. Each payload is JSON-parsed;
   * if the consumer-supplied `extractError` returns an `SIEStreamError`, the
   * generator throws it instead of yielding the chunk.
   *
   * Retry policy mirrors {@link generate}: only explicit SAFE
   * pre-execution capacity signals — `503 PROVISIONING`,
   * `503 MODEL_LOADING` and `503 RESOURCE_EXHAUSTED` (the latter only
   * under `waitForCapacity`) — are retried while the provision budget
   * remains; a `504` is post-publish and therefore terminal.
   * Once the body opens we never retry (the call is non-idempotent; a
   * mid-stream failure must not re-issue generation).
   *
   * @internal
   */
  private async *consumeSseStream<T>(
    url: string,
    body: unknown,
    model: string,
    signal: AbortSignal | undefined,
    extractError: (chunk: T) => SIEStreamError | null,
    routing?: { pool?: string; gpu?: string },
    provisioning?: { waitForCapacity?: boolean },
  ): AsyncGenerator<T, void, undefined> {
    const headers = this.buildChatHeaders("text/event-stream");
    if (routing?.pool) headers["X-SIE-Pool"] = routing.pool;
    if (routing?.gpu) headers["X-SIE-MACHINE-PROFILE"] = routing.gpu;
    const waitForCapacity = provisioning?.waitForCapacity ?? this.defaultWaitForCapacity;
    const gpu = routing?.gpu;

    // Compose the caller's signal with our internal timeout-controller so
    // both can cancel the fetch. We use a fresh controller per call so
    // multiple concurrent streams don't share state.
    const controller = new AbortController();
    const onCallerAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) {
        throw new SIEConnectionError("Stream aborted before request", "other");
      }
      signal.addEventListener("abort", onCallerAbort, { once: true });
    }

    try {
      const startTime = Date.now();
      let oomRetries = 0;
      let response: Response | undefined;

      // Pre-stream provisioning retry loop. We re-fetch on explicit SAFE
      // pre-execution capacity signals only (503 PROVISIONING / MODEL_LOADING /
      // RESOURCE_EXHAUSTED), parallel to `generate()`. The loop terminates by
      // `break`-ing on a 200 (the only status that opens a body) or by throwing.
      while (true) {
        if (signal?.aborted) {
          throw new SIEConnectionError("Stream aborted before request", "other");
        }
        // Pre-stream timeout only — once the body starts flowing we rely on
        // inter-chunk timeouts on the gateway side (`sse.rs` has its own
        // three-tier taxonomy). Setting `this.timeout` for the whole stream
        // would cap long generations at 30s. A fresh per-attempt timeout
        // covers each pre-stream fetch.
        const preStreamTimeoutId = setTimeout(() => controller.abort(), this.timeout);
        let attemptResponse: Response;
        try {
          attemptResponse = await fetch(url, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: controller.signal,
          });
        } catch (error) {
          if (signal?.aborted) {
            throw new SIEConnectionError("Stream aborted before response", "other");
          }
          if (error instanceof Error && error.name === "AbortError") {
            throw new SIEConnectionError(`Stream open timeout after ${this.timeout}ms`, "timeout");
          }
          if (error instanceof TypeError) {
            throw new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
          }
          throw error;
        } finally {
          clearTimeout(preStreamTimeoutId);
        }

        // 502 MODEL_LOAD_FAILED is terminal — surface immediately.
        await throwIfModelLoadFailed(attemptResponse, model);

        // Retry explicit SAFE pre-execution signals before the stream opens.
        // Without `waitForCapacity`, provisioning falls through to
        // `handleError` and rejects immediately.
        if (attemptResponse.status === 503) {
          const errorCode = await getErrorCode(attemptResponse.clone());
          if (errorCode === PROVISIONING_ERROR_CODE) {
            if (!waitForCapacity) {
              throw new ProvisioningError(
                "No capacity available. Server is provisioning.",
                gpu,
                getRetryAfter(attemptResponse),
              );
            }
            const elapsed = Date.now() - startTime;
            if (elapsed >= this.provisionTimeout) {
              throw new ProvisioningError(
                `Provisioning timeout after ${elapsed}ms`,
                gpu,
                getRetryAfter(attemptResponse),
              );
            }
            const retryAfter = getRetryAfter(attemptResponse);
            const delay = retryAfter ?? applyRetryJitter(DEFAULT_RETRY_DELAY);
            // Abortable: a long Retry-After sleep must yield promptly if the
            // caller aborts (`controller.signal` fires on caller-abort), not
            // wait out the full delay before the next loop's abort check.
            if (
              await abortableSleep(
                Math.min(delay, this.provisionTimeout - elapsed),
                controller.signal,
              )
            ) {
              throw new SIEConnectionError("Stream aborted while provisioning", "other");
            }
            continue;
          }
          if (errorCode === MODEL_LOADING_ERROR_CODE) {
            const elapsed = Date.now() - startTime;
            if (elapsed >= this.provisionTimeout) {
              throw new ModelLoadingError(`Model loading timeout for '${model}'`, model);
            }
            const delay = getRetryAfter(attemptResponse) ?? MODEL_LOADING_DEFAULT_DELAY;
            if (
              await abortableSleep(
                Math.min(delay, this.provisionTimeout - elapsed),
                controller.signal,
              )
            ) {
              throw new SIEConnectionError("Stream aborted while provisioning", "other");
            }
            continue;
          }
          if (errorCode === RESOURCE_EXHAUSTED_ERROR_CODE) {
            // Pre-stream capacity signal. Mirrors the Python streaming
            // surface (`next_stream_retry_delay`): retried only under
            // `waitForCapacity`, bounded by the shared OOM budget.
            if (!waitForCapacity) {
              throw new ResourceExhaustedError(
                `Server resource exhausted after ${oomRetries} retry attempt(s) for model '${model}'`,
                { model, retries: oomRetries },
              );
            }
            const delay = nextOomRetryDelay({
              retryAfter: getRetryAfter(attemptResponse),
              oomRetries,
              maxOomRetries: RESOURCE_EXHAUSTED_MAX_RETRIES,
              elapsedMs: Date.now() - startTime,
              provisionTimeoutMs: this.provisionTimeout,
              model,
            });
            oomRetries += 1;
            if (await abortableSleep(delay, controller.signal)) {
              throw new SIEConnectionError("Stream aborted while provisioning", "other");
            }
            continue;
          }
        }

        // 504 is terminal on the streaming path: post-publish, a worker may
        // already be generating, and generation is non-idempotent (Python
        // SDK parity — see `next_stream_retry_delay`).
        if (attemptResponse.status === HTTP_GATEWAY_TIMEOUT) {
          throw new ServerError(
            "Gateway timed out (504) after the request was published to the queue; " +
              "a worker may already be generating. Not retried because generation is " +
              "non-idempotent (retrying could double-bill).",
            await getErrorCode(attemptResponse.clone()),
            HTTP_GATEWAY_TIMEOUT,
          );
        }

        // Any remaining non-200 is an error.
        if (attemptResponse.status !== 200) {
          await handleError(attemptResponse);
        }

        response = attemptResponse;
        break;
      }

      if (!response) {
        throw new RequestError("Streaming request failed without producing a response");
      }
      this.checkServerVersion(response);

      const bodyStream = response.body;
      if (!bodyStream) {
        throw new RequestError("Streaming response has no body");
      }
      const reader = bodyStream.getReader();
      for await (const payload of parseSseStream(reader, signal ?? controller.signal)) {
        let chunk: T;
        try {
          chunk = JSON.parse(payload) as T;
        } catch (err) {
          throw new RequestError(
            `Failed to parse SSE chunk as JSON: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        const streamErr = extractError(chunk);
        if (streamErr) throw streamErr;
        yield chunk;
      }
    } finally {
      if (signal) signal.removeEventListener("abort", onCallerAbort);
    }
  }

  /**
   * Build the standard JSON header set for the chat-completions surface.
   * Pulled out so both the streaming and non-streaming paths agree on
   * auth / version / content-type wiring.
   */
  private buildChatHeaders(
    accept: "application/json" | "text/event-stream",
  ): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: accept,
      "Content-Type": JSON_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    return headers;
  }

  async score(
    model: string,
    query: Item,
    items: Item[],
    options: ScoreOptions = {},
  ): Promise<ScoreResult> {
    const queryForWire = await itemImagesForWire(query);
    const itemsForWire = await itemsImagesForWire(items);

    // Build request body
    const body: Record<string, unknown> = {
      query: queryForWire,
      items: itemsForWire,
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

    const result = parseScoreResult(data);
    attachRequestMetadata([result], response.headers);
    return result;
  }

  /**
   * Extract entities from a single item.
   *
   * @param model - Model name (e.g., "gliner-multi-v2.1")
   * @param item - Item to extract from
   * @param options - Extract options with labels
   * @returns Extract result with entities
   */
  async extract(model: string, item: ExtractItem, options: ExtractOptions): Promise<ExtractResult>;

  /**
   * Extract entities from multiple items.
   *
   * @param model - Model name (e.g., "gliner-multi-v2.1")
   * @param items - Items to extract from
   * @param options - Extract options with labels
   * @returns Array of extract results in same order as input
   */
  async extract(
    model: string,
    items: ExtractItem[],
    options: ExtractOptions,
  ): Promise<ExtractResult[]>;

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
    items: ExtractItem | ExtractItem[],
    options: ExtractOptions,
  ): Promise<ExtractResult | ExtractResult[]> {
    const isSingleItem = !Array.isArray(items);
    const itemsArray = isSingleItem ? [items] : items;
    const itemsForWire = await itemsForExtractWire(itemsArray);

    // Build request body
    const body: Record<string, unknown> = {
      items: itemsForWire,
    };

    // Add params
    const params: Record<string, unknown> = {
      labels: options.labels,
    };
    if (options.threshold !== undefined) {
      params.threshold = options.threshold;
    }
    if (options.adapterOptions !== undefined) {
      params.options = options.adapterOptions;
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
    attachRequestMetadata(results, response.headers);

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
   * Create or update a resource pool for isolated capacity.
   *
   * Pools provide logical capacity isolation. By default they draw from the
   * cluster's `default` Helm/NATS queue; pass `queuePool` only when the cluster
   * has a dedicated physical worker queue declared under
   * `queueRouting.staticQueuePools` for this workload.
   *
   * @param name - Pool name (used in GPU param as "poolName/machineProfile")
   * @param gpus - Optional machine profile requirements for pool readiness, e.g., { "l4": 2, "l4-spot": 1 }
   * @param gpuCaps - Optional maximum assigned workers per machine profile
   * @param queuePool - Optional Helm/NATS queue namespace backing this logical pool. Defaults to "default".
   * @param options - Optional bundle filter, warm floor, and pinned models
   *                  (Python SDK `create_pool` parity)
   *
   * @example
   * ```typescript
   * // Create or update a pool with 2 L4 GPUs
   * await client.createPool("eval-bench", { l4: 2 });
   *
   * // With a bundle filter, warm floor, and pinned models
   * await client.createPool("eval-bench", { l4: 2 }, undefined, undefined, {
   *   bundle: "default",
   *   minimumWorkerCount: 1,
   *   pinnedModels: ["bge-m3"],
   * });
   *
   * // Use the pool for requests
   * await client.encode("bge-m3", { text: "Hello" }, { gpu: "eval-bench/l4" });
   *
   * // Clean up when done
   * await client.deletePool("eval-bench");
   * ```
   */
  async createPool(
    name: string,
    gpus?: Record<string, number>,
    gpuCaps?: Record<string, number>,
    queuePool?: string,
    options: CreatePoolOptions = {},
  ): Promise<void> {
    const alreadyTracking = this.pools.has(name);

    if (options.minimumWorkerCount !== undefined && options.minimumWorkerCount < 0) {
      throw new RangeError("minimumWorkerCount must be >= 0");
    }

    // Build pool creation request
    const requestBody: {
      name: string;
      gpus?: Record<string, number>;
      gpu_caps?: Record<string, number>;
      queue_pool?: string;
      bundle?: string;
      minimum_worker_count?: number;
      pinned_models?: string[];
    } = {
      name,
    };
    if (gpus !== undefined) {
      requestBody.gpus = gpus;
    }
    if (gpuCaps) {
      requestBody.gpu_caps = gpuCaps;
    }
    if (queuePool) {
      requestBody.queue_pool = queuePool;
    }
    if (options.bundle) {
      requestBody.bundle = options.bundle;
    }
    if (options.minimumWorkerCount !== undefined) {
      requestBody.minimum_worker_count = options.minimumWorkerCount;
    }
    if (options.pinnedModels !== undefined) {
      requestBody.pinned_models = options.pinnedModels;
    }

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

      if (alreadyTracking || this.pools.has(name)) {
        return;
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
            } catch {
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
        spec: PoolSpec;
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
   * Retried (capped by `provisionTimeout`):
   *  - 503 `PROVISIONING` when `waitForCapacity: true`
   *  - 503 `MODEL_LOADING` / `LORA_LOADING`
   *  - 503 `RESOURCE_EXHAUSTED` regardless of `waitForCapacity` (bounded
   *    exponential backoff, at most `RESOURCE_EXHAUSTED_MAX_RETRIES`)
   *  - 504 gateway timeout when `waitForCapacity: true` — encode/score/
   *    extract are idempotent queue paths, so a post-publish retry is safe
   *    (unlike generate/chat, where a 504 is terminal)
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
    // Retry counter for server-side OOM (RESOURCE_EXHAUSTED). Bounded so a
    // stuck-at-OOM server cannot cause unbounded blocking.
    let oomRetries = 0;

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

      // Short-circuit terminal load failures (sie-test#85). The server
      // emits 502 MODEL_LOAD_FAILED for permanent classes (gated repos,
      // missing dependencies, unrecognised architectures); we must
      // surface the error immediately rather than burn the
      // MODEL_LOADING retry budget on a known-bad config.
      await throwIfModelLoadFailed(response, model);

      // Short-circuit token-budget overruns (#849).
      await throwIfInputTooLong(response, model);

      // Handle explicit retryable 503 signals.
      if (response.status === 503) {
        const clonedResponse = response.clone();
        const errorCode = await getErrorCode(clonedResponse);

        if (errorCode === PROVISIONING_ERROR_CODE) {
          const retryAfter = getRetryAfter(response);

          if (!waitForCapacity) {
            throw new ProvisioningError(
              `No capacity available for GPU '${gpu}'. Server is provisioning.`,
              gpu,
              retryAfter,
            );
          }

          const elapsed = Date.now() - startTime;
          if (elapsed >= this.provisionTimeout) {
            throw new ProvisioningError(
              `Provisioning timeout after ${elapsed}ms waiting for GPU '${gpu}'`,
              gpu,
              retryAfter,
            );
          }

          const delay = retryAfter ?? applyRetryJitter(DEFAULT_RETRY_DELAY);
          const remaining = this.provisionTimeout - elapsed;
          const actualDelay = Math.min(delay, remaining);
          await sleep(actualDelay);
          continue;
        }

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

        if (errorCode === RESOURCE_EXHAUSTED_ERROR_CODE) {
          // Server-side OOM. Retried regardless of `waitForCapacity`
          // (bounded budget), matching the Python SDK: the worker already
          // accepted the request and is recovering from transient capacity
          // exhaustion.
          const delay = nextOomRetryDelay({
            retryAfter: getRetryAfter(response),
            oomRetries,
            maxOomRetries: RESOURCE_EXHAUSTED_MAX_RETRIES,
            elapsedMs: Date.now() - startTime,
            provisionTimeoutMs: this.provisionTimeout,
            model,
          });
          oomRetries += 1;
          await sleep(delay);
          continue;
        }
      }

      // Handle 504 (gateway timeout): queued work was published, but the
      // gateway did not receive a worker result before its deadline.
      // Encode/score/extract are idempotent, so callers that opted into
      // waitForCapacity can retry within the provision budget (Python SDK
      // parity). On budget exhaustion this falls through to handleError.
      if (response.status === HTTP_GATEWAY_TIMEOUT && waitForCapacity) {
        const elapsed = Date.now() - startTime;
        if (elapsed < this.provisionTimeout) {
          const delay = getRetryAfter(response) ?? MODEL_LOADING_DEFAULT_DELAY;
          await sleep(Math.min(delay, this.provisionTimeout - elapsed));
          continue;
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

  // ---------------------------------------------------------------------------
  // Jobs + connections namespaces. Jobs ride the keyed gateway
  // (`/v1/jobs`); connections ride the control plane (`/internal/orgs/{org}/…`).
  // ---------------------------------------------------------------------------

  /** One JSON request over `fetch` (bearer auth reused; absolute or base-relative URL). */
  private async jsonRequest<T>(
    target: string,
    method: "GET" | "POST" | "DELETE",
    body?: unknown,
    timeoutMs: number = this.timeout,
  ): Promise<T> {
    const url = target.startsWith("http") ? target : `${this.baseUrl}${target}`;
    const headers: Record<string, string> = {
      Accept: JSON_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const init: RequestInit = { method, headers };
    if (body !== undefined) {
      headers["Content-Type"] = JSON_CONTENT_TYPE;
      init.body = JSON.stringify(body);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    init.signal = controller.signal;
    try {
      const response = await fetch(url, init);
      if (!response.ok) {
        await handleError(response);
      }
      this.checkServerVersion(response);
      const text = await response.text();
      return (text ? JSON.parse(text) : {}) as T;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new SIEConnectionError(`Request timeout after ${timeoutMs}ms`, "timeout");
      }
      if (error instanceof TypeError) {
        throw new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async jobSubmit(options: SubmitJobOptions): Promise<JobSubmitResult> {
    return this.jsonRequest<JobSubmitResult>(
      "/v1/jobs",
      "POST",
      buildJobBody(options),
      Math.max(this.timeout, DEFAULT_LONG_RUNNING_TIMEOUT),
    );
  }

  private async jobGet(jobId: string): Promise<JobStatus> {
    return this.jsonRequest<JobStatus>(`/v1/jobs/${encodeURIComponent(jobId)}`, "GET");
  }

  private async jobList(): Promise<JobStatus[]> {
    const data = await this.jsonRequest<{ object?: string; data?: JobStatus[] }>("/v1/jobs", "GET");
    return Array.isArray(data) ? data : (data.data ?? []);
  }

  private async jobCancel(jobId: string): Promise<JobStatus> {
    return this.jsonRequest<JobStatus>(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, "POST");
  }

  private async jobResults(jobId: string): Promise<JobResults> {
    const job = await this.jobGet(jobId);
    const chunks = jobChunks(job);
    const items: JobResultItem[] = [];
    for (const chunk of chunks) {
      if (chunk.state !== "succeeded" || !chunk.ref) continue;
      const raw = await this.readRef(chunk.ref);
      items.push(...decodeChunkBytes(raw));
    }
    const withDims = items.find((it) => it.dims != null);
    return {
      job_id: job.id ?? jobId,
      state: job.state,
      total_items: job.total_items,
      settled_credits: job.settled_credits,
      chunks,
      retrieved: items.length,
      dims: withDims ? withDims.dims : null,
      items,
    };
  }

  private async jobWait(
    jobId: string,
    options?: { timeoutMs?: number; pollMs?: number },
  ): Promise<JobStatus> {
    const timeoutMs = options?.timeoutMs ?? DEFAULT_JOB_WAIT_TIMEOUT;
    const pollMs = options?.pollMs ?? DEFAULT_JOB_WAIT_POLL;
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const job = await this.jobGet(jobId);
      if (job.state && TERMINAL_JOB_STATES.has(job.state)) {
        return job;
      }
      if (Date.now() >= deadline) {
        throw new RequestError(
          `job ${jobId} still ${JSON.stringify(job.state)} after ${timeoutMs}ms`,
          "job_wait_timeout",
          504,
        );
      }
      await sleep(pollMs);
    }
  }

  /** Retrieve a chunk's payload-store ref (http(s) URL). */
  private async readRef(ref: string): Promise<Uint8Array> {
    if (!ref.startsWith("http://") && !ref.startsWith("https://")) {
      throw new RequestError(
        `cannot retrieve payload-store ref ${JSON.stringify(ref)} (the TS SDK reads http(s) refs)`,
        "bad_ref",
        400,
      );
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(ref, {
        headers: { Accept: "application/octet-stream" },
        signal: controller.signal,
      });
      if (!response.ok) {
        await handleError(response);
      }
      return new Uint8Array(await response.arrayBuffer());
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

  private connectionsBase(): string {
    if (!this.controlPlaneUrl) {
      throw new RequestError(
        "connections require controlPlaneUrl on the client: new SIEClient(url, { controlPlaneUrl, org })",
        "missing_control_plane_url",
        400,
      );
    }
    if (!this.org) {
      throw new RequestError(
        "connections require org on the client: new SIEClient(url, { controlPlaneUrl, org })",
        "missing_org",
        400,
      );
    }
    return `${this.controlPlaneUrl}/internal/orgs/${encodeURIComponent(this.org)}/connections`;
  }

  private async connectionAdd(
    name: string,
    type: string,
    secret: string,
  ): Promise<ConnectionCreated> {
    return this.jsonRequest<ConnectionCreated>(this.connectionsBase(), "POST", {
      type,
      name,
      secret,
    });
  }

  private async connectionList(): Promise<Connection[]> {
    const data = await this.jsonRequest<{ connections?: Connection[] }>(
      this.connectionsBase(),
      "GET",
    );
    return Array.isArray(data) ? data : (data.connections ?? []);
  }

  private async connectionRevoke(name: string): Promise<ConnectionRevoked> {
    return this.jsonRequest<ConnectionRevoked>(
      `${this.connectionsBase()}/${encodeURIComponent(name)}`,
      "DELETE",
    );
  }

  // ---------------------------------------------------------------------------
  // Files + batches namespaces — the OpenAI-compatible file /
  // batch surface on the keyed gateway. Method names/args mirror `openai.files`
  // / `openai.batches` so switching an OpenAI-batch caller to the SDK is
  // mechanical.
  // ---------------------------------------------------------------------------

  /** POST a raw body and parse the JSON response (bearer auth reused). */
  private async rawPostJson<T>(
    path: string,
    body: FileUploadInput,
    contentType: string,
    timeoutMs: number = this.timeout,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      Accept: JSON_CONTENT_TYPE,
      "Content-Type": contentType,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        await handleError(response);
      }
      this.checkServerVersion(response);
      const text = await response.text();
      return (text ? JSON.parse(text) : {}) as T;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new SIEConnectionError(`Request timeout after ${timeoutMs}ms`, "timeout");
      }
      if (error instanceof TypeError) {
        throw new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /** GET raw bytes (bearer auth reused); used to download a file's content. */
  private async rawGetBytes(path: string): Promise<Uint8Array> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      Accept: "application/jsonl",
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(url, { method: "GET", headers, signal: controller.signal });
      if (!response.ok) {
        await handleError(response);
      }
      this.checkServerVersion(response);
      return new Uint8Array(await response.arrayBuffer());
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

  private async fileUpload(
    file: FileUploadInput,
    options?: { purpose?: string; filename?: string },
  ): Promise<SIEFile> {
    const purpose = options?.purpose ?? "batch";
    const filename = resolveUploadFilename(file, options?.filename);
    const query = new URLSearchParams({ purpose, filename }).toString();
    const body: FileUploadInput = file instanceof ArrayBuffer ? new Uint8Array(file) : file;
    return this.rawPostJson<SIEFile>(
      `/v1/files?${query}`,
      body,
      "application/jsonl",
      Math.max(this.timeout, DEFAULT_LONG_RUNNING_TIMEOUT),
    );
  }

  private async fileRetrieve(fileId: string): Promise<SIEFile> {
    return this.jsonRequest<SIEFile>(`/v1/files/${encodeURIComponent(fileId)}`, "GET");
  }

  private async fileContent(fileId: string): Promise<Uint8Array> {
    return this.rawGetBytes(`/v1/files/${encodeURIComponent(fileId)}/content`);
  }

  private async fileDelete(fileId: string): Promise<FileDeleted> {
    return this.jsonRequest<FileDeleted>(`/v1/files/${encodeURIComponent(fileId)}`, "DELETE");
  }

  private async batchCreate(options: {
    input_file_id: string;
    endpoint?: string;
    completion_window?: string;
    metadata?: Record<string, unknown>;
  }): Promise<Batch> {
    const body: Record<string, unknown> = {
      input_file_id: options.input_file_id,
      endpoint: options.endpoint ?? "/v1/embeddings",
      completion_window: options.completion_window ?? "24h",
    };
    if (options.metadata !== undefined) {
      body.metadata = options.metadata;
    }
    return this.jsonRequest<Batch>(
      "/v1/batches",
      "POST",
      body,
      Math.max(this.timeout, DEFAULT_LONG_RUNNING_TIMEOUT),
    );
  }

  private async batchRetrieve(batchId: string): Promise<Batch> {
    return this.jsonRequest<Batch>(`/v1/batches/${encodeURIComponent(batchId)}`, "GET");
  }

  private async batchList(): Promise<Batch[]> {
    const data = await this.jsonRequest<{ object?: string; data?: Batch[] }>("/v1/batches", "GET");
    return Array.isArray(data) ? data : (data.data ?? []);
  }

  private async batchCancel(batchId: string): Promise<Batch> {
    return this.jsonRequest<Batch>(`/v1/batches/${encodeURIComponent(batchId)}/cancel`, "POST");
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
      if (!headers) {
        return new WebSocket(url);
      }
      // In Node, `WebSocket` resolves to the `ws` package which accepts
      // a third `{ headers }` options argument. In browsers, the native
      // WebSocket only takes `(url, protocols)` and the third arg is
      // silently dropped. Use `Reflect.construct` with a runtime args
      // array so the call site doesn't statically appear to pass
      // superfluous trailing arguments to the lib.dom WebSocket type.
      const args: unknown[] = [url, [], { headers }];
      return Reflect.construct(WebSocket, args) as WebSocket;
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
