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
 * const result = await client.encode("BAAI/bge-m3", { text: "Hello world" });
 * console.log(result.dense); // Float32Array
 *
 * // Batch encode
 * const results = await client.encode("BAAI/bge-m3", [
 *   { text: "First document" },
 *   { text: "Second document" },
 * ]);
 *
 * // With GPU routing and auto-retry for capacity
 * const resultWithGpu = await client.encode(
 *   "BAAI/bge-m3",
 *   { text: "Hello" },
 *   { gpu: "l4", waitForCapacity: true },
 * );
 *
 * await client.close();
 * ```
 */

import {
  JobFailedError,
  LoraLoadingError,
  MalformedChunkError,
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
  ESTIMATE_PATH,
  buildEstimateEnvelope,
  getErrorCode,
  getErrorParam,
  getRetryAfter,
  handleError,
  parseCapacityInfo,
  parseEncodeResults,
  parseExtractResults,
  parseGenerateResult,
  parseScoreResult,
  readRequestId,
  settledChargeFields,
  throwIfEstimateUnroutable,
  throwIfInputTooLong,
  throwIfModelLoadFailed,
  validateBatchResultCount,
  validateRequestId,
} from "./internal/parsing.js";
import {
  admissionRetryDelay,
  nextOomRetryDelay,
  withProvisioningRetry,
} from "./internal/provisioning.js";
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
  requireConnectionName,
  requireConnectionSchemaPolicy,
  requireConnectorIdempotencyKey,
} from "./jobs.js";
import { packMessage, unpackMessage } from "./msgpack.js";
import { parseSseStream } from "./sse.js";
import type {
  AddConnectionOptions,
  Batch,
  BatchList,
  CapacityInfo,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionOptions,
  ChatCompletionRequest,
  Connection,
  ConnectionCreated,
  ConnectionRevoked,
  CostEstimate,
  CreatePoolOptions,
  EncodeOptions,
  EncodeResult,
  ExtractItem,
  ExtractOptions,
  ExtractResult,
  FileDeleted,
  FileList,
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
  WireModelInfo,
} from "./types.js";
import { SDK_VERSION } from "./version.js";

const JOB_RESULT_NOT_FOUND_ERROR_CODE = "RESULT_NOT_FOUND";
const JOB_RESULT_REF_MAX_REFRESHES = 3;

/**
 * `jobs.results()` decodes chunk refs into per-item results. A non-terminal job
 * has no stable result set: its ref list is still growing, so decoding one would
 * return a partial subset indistinguishable from a partial-FAILURE subset. The
 * SDK refuses with this code rather than return a misleading partial.
 */
const JOB_NOT_TERMINAL_ERROR_CODE = "job_not_terminal";

/**
 * Wire keys `GET /v1/models/{model}` merges in for vanilla OpenAI
 * "retrieve model" clients. They are OpenAI-envelope scaffolding, not SIE
 * model metadata (`id` duplicates `name`; `object`/`owned_by` are constants;
 * `created` is a fixed sentinel), so `ModelInfo` neither declares nor carries
 * them. Pinned as deliberately excluded in
 * `packages/wire-fixtures/model_info.json`.
 */
type OpenAiCompatModelKeys = Partial<Record<"id" | "object" | "created" | "owned_by", unknown>>;

/**
 * Map a wire `/v1/models` entry to the client-facing shape.
 *
 * Total by construction: every key except the three camelCase renames and the
 * OpenAI-compat keys flows through the rest spread, so a field added to the
 * endpoint reaches callers without a code change here — only `ModelInfo` and
 * `WireModelInfo` need to declare it. The previous hardcoded allowlist silently
 * dropped everything it did not name.
 */
function toModelInfo(wire: WireModelInfo): ModelInfo {
  const {
    max_sequence_length,
    last_error,
    pending_generation,
    aliases,
    id: _id,
    object: _object,
    created: _created,
    owned_by: _ownedBy,
    ...rest
  } = wire as WireModelInfo & OpenAiCompatModelKeys;

  return {
    ...rest,
    // The one normalized field. A gateway always sends `aliases`, but a single
    // SIE server omits it entirely, and the rest spread would then hand the
    // caller `undefined` for a field `ModelInfo` declares as always present.
    aliases: aliases ?? [],
    ...(max_sequence_length !== undefined ? { maxSequenceLength: max_sequence_length } : {}),
    ...(last_error !== undefined ? { lastError: last_error } : {}),
    ...(pending_generation !== undefined ? { pendingGeneration: pending_generation } : {}),
  };
}

/** The `client.jobs` batch namespace. */
export interface JobsNamespace {
  /** Submit a batch job; exact connector replays may return current status. */
  submit(options: SubmitJobOptions): Promise<JobSubmitResult | JobStatus>;
  /** Fetch a job's public status doc (`GET /v1/jobs/{id}`). */
  get(jobId: string): Promise<JobStatus>;
  /** List the org's jobs (`GET /v1/jobs`; scoped to the key's org). */
  list(): Promise<JobStatus[]>;
  /** Cancel a job (`POST /v1/jobs/{id}/cancel`); the hold's remainder releases. */
  cancel(jobId: string): Promise<JobStatus>;
  /** Confirm one exact connector plan revision for execution. */
  execute(jobId: string, planRevision: number, idempotencyKey: string): Promise<JobStatus>;
  /** Repair one exact recovery-required connector attempt and cutoff. */
  repair(
    jobId: string,
    planRevision: number,
    recoveryAttemptOrdinal: number,
    idempotencyKey: string,
  ): Promise<JobStatus>;
  /**
   * Retrieve a terminal job's chunk refs and decode the per-item results.
   *
   * Every chunk that published a ref is read — including a `failed` chunk,
   * whose ref carries its SUCCESSFUL (already-billed) siblings alongside the
   * per-item failures; only chunks with no ref at all are skipped. Each item's
   * `success` and `error` distinguish the two.
   *
   * Throws a `job_not_terminal` `RequestError` (409) when the job has not
   * reached a terminal state — decoding one then would return a partial subset
   * indistinguishable from a partial-failure subset. Warns (via `console.warn`)
   * when fewer items are retrieved than `total_items`, and separately when a
   * chunk ref's bytes could not be decoded.
   */
  results(jobId: string): Promise<JobResults>;
  /**
   * Poll `get` until the job reaches a terminal state, then return its status.
   * Throws a `job_wait_timeout` `RequestError` if `timeoutMs` elapses first.
   * Mirrors the Python SDK's `jobs.wait` (default 600s timeout, 2s poll).
   *
   * With `raiseOnFailure: true` a non-successful terminal
   * (`failed`/`suspended`/`cancelled`) throws {@link JobFailedError} carrying
   * the status doc's `outcome`/`error_code`, so the failure is actionable
   * without re-reading the doc. The default is unchanged and back-compatible:
   * every terminal (and the `planned` plan phase) is returned as-is.
   */
  wait(
    jobId: string,
    options?: { timeoutMs?: number; pollMs?: number; raiseOnFailure?: boolean },
  ): Promise<JobStatus>;
}

/** The `client.connections` namespace (org-scoped connector auth). */
export interface ConnectionsNamespace {
  /** Create an org-scoped connection (connector auth by name). */
  add(
    name: string,
    type: string,
    secret: string,
    options?: AddConnectionOptions,
  ): Promise<ConnectionCreated>;
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
  /** List the org's live files with OpenAI cursor/filter arguments. */
  list(options?: {
    after?: string;
    limit?: number;
    order?: "asc" | "desc";
    purpose?: string;
  }): Promise<SIEFile[]>;
  /** Return one file cursor page, including pagination metadata. */
  listPage(options?: {
    after?: string;
    limit?: number;
    order?: "asc" | "desc";
    purpose?: string;
  }): Promise<FileList>;
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
  /** List the org's batches; the historical array return is preserved. */
  list(options?: { after?: string; limit?: number }): Promise<Batch[]>;
  /** Return one batch cursor page, including pagination metadata. */
  listPage(options?: { after?: string; limit?: number }): Promise<BatchList>;
  /** Cancel a batch (`POST /v1/batches/{id}/cancel`; additive OpenAI-parity). */
  cancel(batchId: string): Promise<Batch>;
}

/** Helper to sleep for a given number of milliseconds */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const CONTENT_SAFE_MEDIA_TYPES = new Set([
  "application/json",
  "application/problem+json",
  "text/html",
]);

function terminalResponseDiagnostic(response: Response, bodyBytes: number): string {
  const rawContentType = response.headers.get("content-type");
  let contentType = "unknown";
  if (rawContentType !== null) {
    const candidate = rawContentType.split(";", 1)[0]?.trim().toLowerCase() ?? "";
    contentType = CONTENT_SAFE_MEDIA_TYPES.has(candidate) ? candidate : "other";
  }
  return `status=${response.status}, content_type=${contentType}, body_bytes=${bodyBytes}`;
}

async function parseTerminalJsonObject(
  response: Response,
  owner: string,
): Promise<Record<string, unknown>> {
  const body = new Uint8Array(await response.arrayBuffer());
  if (!response.ok) {
    throw new RequestError(
      `Unexpected ${owner} HTTP response (${terminalResponseDiagnostic(response, body.byteLength)})`,
      undefined,
      response.status,
    );
  }
  let data: unknown;
  try {
    data = JSON.parse(new TextDecoder().decode(body));
  } catch {
    throw new RequestError(
      `Malformed ${owner} JSON response (${terminalResponseDiagnostic(response, body.byteLength)})`,
      undefined,
      response.status,
    );
  }
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new RequestError(
      `Unexpected ${owner} response shape (${terminalResponseDiagnostic(response, body.byteLength)})`,
      undefined,
      response.status,
    );
  }
  return data as Record<string, unknown>;
}

function parseNonnegativeMeterHeader(headers: Headers, name: string): number | undefined {
  const raw = headers.get(name);
  if (raw === null || !/^(0|[1-9]\d*)$/.test(raw)) return undefined;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

/**
 * Parse optional request-scoped metadata from one successful terminal response.
 *
 * `body` is the decoded response envelope, when the caller has one. The settled
 * charge is read from its `usage` block FIRST and falls back to the
 * `x-sie-credits-debited` header, which is the only source on a gateway that
 * predates in-body surfacing (#2434). The two always agree when both are
 * present; the body additionally names the rate book, which no header does.
 */
function parseRequestMetadata(headers: Headers, body?: unknown): RequestMetadata | undefined {
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
  const executionBindingSha256 = headers.get("x-sie-execution-binding-sha256");
  if (executionBindingSha256 !== null && /^[0-9a-f]{64}$/.test(executionBindingSha256)) {
    metadata.executionBindingSha256 = executionBindingSha256;
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

  const settled = settledChargeFromBody(body);
  if (settled !== undefined) {
    usage.creditsCharged = settled.creditsCharged;
    usage.rateBookVersion = settled.rateBookVersion;
    metadata.creditsDebited = settled.creditsCharged;
    metadata.rateBookVersion = settled.rateBookVersion;
  } else {
    const creditsDebited = parseNonnegativeMeterHeader(headers, "x-sie-credits-debited");
    if (creditsDebited !== undefined) metadata.creditsDebited = creditsDebited;
  }
  if (Object.keys(usage).length > 0) metadata.usage = usage;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

/**
 * The settled charge from a response envelope's `usage` block (#2434).
 *
 * Shares `settledChargeFields`' validation so the metadata path and the result
 * parsers can never disagree about what counts as a publishable charge.
 * Anything malformed is treated as absent, so the header fallback still
 * applies.
 */
function settledChargeFromBody(
  body: unknown,
): { creditsCharged: number; rateBookVersion: string } | undefined {
  if (typeof body !== "object" || body === null || Array.isArray(body)) return undefined;
  const { creditsCharged, rateBookVersion } = settledChargeFields(
    (body as Record<string, unknown>).usage,
  );
  if (creditsCharged === undefined || rateBookVersion === undefined) return undefined;
  return { creditsCharged, rateBookVersion };
}

function attachRequestMetadata<T extends { request?: RequestMetadata }>(
  results: T[],
  headers: Headers,
  body?: unknown,
): void {
  const metadata = parseRequestMetadata(headers, body);
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
function validatedStreamRetryAfter(error: {
  code?: string;
  retry_after_s?: unknown;
}): number | undefined {
  const value = error.retry_after_s;
  return error.code === RESOURCE_EXHAUSTED_ERROR_CODE &&
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= 60
    ? value * 1_000
    : undefined;
}

function validatedStreamErrorParam(error: { param?: unknown }): string | null | undefined {
  return typeof error.param === "string" || error.param === null ? error.param : undefined;
}

function extractChatChunkError(chunk: ChatCompletionChunk): SIEStreamError | null {
  if (!chunk.error) return null;
  // The gateway request id rides in-band on the chat error chunk too (#3136)
  // — the `chatcmpl-*` id is not the correlation key gateway logs use, and
  // streamed responses have no terminal headers. Forward it for correlation.
  const requestId = validateRequestId(chunk.request_id);
  return new SIEStreamError(chunk.error.message ?? "stream error", {
    code: chunk.error.code,
    errorType: chunk.error.type,
    param: validatedStreamErrorParam(chunk.error),
    requestId,
    retryAfter: validatedStreamRetryAfter(chunk.error),
  });
}

/**
 * Return only the `scheme://host[:port]` origin of `url`.
 *
 * Path, query, fragment, and any `user:password@` userinfo are dropped, so a
 * baseUrl carrying embedded credentials OR a token query parameter never
 * reaches a log line. Logging uses only —
 * requests still target the real URL. Falls back to a placeholder if the URL
 * cannot be parsed, so a malformed value never leaks verbatim.
 */
function urlOriginForLogging(url: string): string {
  try {
    const parsed = new URL(url);
    // `parsed.host` is host[:port]; `parsed.origin` would work for http(s)
    // but is "null" for opaque origins, so build it explicitly.
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return "<redacted-url>";
  }
}

/**
 * Convert a `fetch()` `TypeError` into a typed connection error.
 *
 * A URL-parse failure (e.g. a scheme-less baseUrl: Node throws
 * `TypeError: Failed to parse URL …` with `cause.code === "ERR_INVALID_URL"`)
 * is a permanent configuration error, NOT a transient network failure —
 * classify it as kind `"other"` so the connect-retry loops never spin on it,
 * and point at the fix. Every other fetch `TypeError` is a genuine
 * network-level connection failure and keeps kind `"connect"`.
 */
function connectionErrorFromFetchTypeError(error: TypeError): SIEConnectionError {
  const cause = (error as { cause?: { code?: unknown } }).cause;
  if (
    cause?.code === "ERR_INVALID_URL" ||
    error.message.includes("Failed to parse URL") ||
    error.message.includes("Invalid URL")
  ) {
    return new SIEConnectionError(
      `Invalid request URL (${error.message}). baseUrl must be an absolute http(s) URL, e.g. "http://localhost:8080".`,
      "other",
    );
  }
  return new SIEConnectionError(`Connection failed: ${error.message}`, "connect");
}

const MODAL_CONTINUATION_MAX_HOPS = 20;
const MODAL_ATTEMPT_TOKEN_QUERY_KEY = "__modal_attempt_token";

/** Resolve only Modal's documented result URL on the exact configured origin. */
function modalContinuationUrl(baseUrl: string, response: Response): string | undefined {
  if (response.status !== 303) return undefined;
  const location = response.headers.get("location");
  const hasControlCharacter = [...(location ?? "")].some((character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint <= 0x1f || codePoint === 0x7f;
  });
  if (!location || location.length > 8192 || hasControlCharacter) {
    return undefined;
  }
  try {
    const base = new URL(baseUrl);
    const resolved = new URL(location, `${baseUrl.replace(/\/$/, "")}/`);
    const tokens = resolved.searchParams.getAll(MODAL_ATTEMPT_TOKEN_QUERY_KEY);
    if (
      resolved.origin !== base.origin ||
      resolved.username !== "" ||
      resolved.password !== "" ||
      resolved.hash !== "" ||
      tokens.length !== 1 ||
      tokens[0] === ""
    ) {
      return undefined;
    }
    return resolved.toString();
  } catch {
    return undefined;
  }
}

function requireVisibleManualRedirect(response: Response): void {
  if (response.type === "opaqueredirect") {
    throw new SIEConnectionError(
      "Modal result continuations require a server-side Fetch runtime that exposes manual redirect status and Location",
      "other",
    );
  }
}

/** SIE-native chunk variant — see `sse.rs::build_generate_chunk_event`. */
function extractGenerateChunkError(chunk: GenerateChunk): SIEStreamError | null {
  if (!chunk.error) return null;
  // The gateway request id rides in-band on the error chunk (streamed
  // responses have no terminal headers), so forward it for correlation (#3136).
  const requestId = validateRequestId(chunk.request_id);
  return new SIEStreamError(chunk.error.message, {
    code: chunk.error.code,
    param: validatedStreamErrorParam(chunk.error),
    requestId,
    retryAfter: validatedStreamRetryAfter(chunk.error),
  });
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
 * await client.encode("BAAI/bge-m3", { text: "Hello" }, { gpu: "eval-bench/l4" });
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
    // Validate eagerly: a scheme-less baseUrl ("localhost:8080") would
    // otherwise only surface at request time as a fetch `TypeError`.
    let parsed: URL | undefined;
    try {
      parsed = new URL(baseUrl);
    } catch {
      parsed = undefined;
    }
    // `new URL()` normalizes `http:/v1`, `http:///v1`, `https:///v1` to a
    // bogus host `"v1"` (WHATWG slash-coalescing), so those pass a `hostname`
    // check while silently targeting the wrong host. Require a real
    // `scheme://<authority>`: `https?://` immediately followed by a non-slash
    // authority character.
    const hasRealAuthority = /^https?:\/\/[^/]/i.test(baseUrl);
    if (
      !parsed ||
      (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
      !parsed.hostname ||
      !hasRealAuthority
    ) {
      throw new TypeError(
        `Invalid baseUrl "${baseUrl}": must be an absolute http(s) URL with a host, e.g. "http://localhost:8080".`,
      );
    }
    // Remove trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, "");
    // `timeoutMs` is the unit-encoded name; `timeout` is a deprecated alias
    // for the same MILLISECONDS value. `timeoutMs` wins if both are set.
    this.timeout = options.timeoutMs ?? options.timeout ?? DEFAULT_TIMEOUT;
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
      execute: (jobId, planRevision, idempotencyKey) =>
        this.jobExecute(jobId, planRevision, idempotencyKey),
      repair: (jobId, planRevision, recoveryAttemptOrdinal, idempotencyKey) =>
        this.jobRepair(jobId, planRevision, recoveryAttemptOrdinal, idempotencyKey),
      results: (jobId) => this.jobResults(jobId),
      wait: (jobId, options) => this.jobWait(jobId, options),
    };
    this.connections = {
      add: (name, type, secret, options) => this.connectionAdd(name, type, secret, options),
      list: () => this.connectionList(),
      revoke: (name) => this.connectionRevoke(name),
    };
    // OpenAI-compatible Files + Batches surface — a base_url
    // swap makes an `openai` batch caller work unchanged.
    this.files = {
      upload: (file, options) => this.fileUpload(file, options),
      create: (options) => this.fileUpload(options.file, options),
      retrieve: (fileId) => this.fileRetrieve(fileId),
      list: (options) => this.fileList(options),
      listPage: (options) => this.fileListPage(options),
      content: (fileId) => this.fileContent(fileId),
      delete: (fileId) => this.fileDelete(fileId),
    };
    this.batches = {
      create: (options) => this.batchCreate(options),
      retrieve: (batchId) => this.batchRetrieve(batchId),
      list: (options) => this.batchList(options),
      listPage: (options) => this.batchListPage(options),
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
   * @param model - Model name (e.g., "BAAI/bge-m3")
   * @param item - Item to encode
   * @param options - Encode options
   * @returns Encode result with embeddings
   */
  async encode(model: string, item: Item, options?: EncodeOptions): Promise<EncodeResult>;

  /**
   * Encode multiple items.
   *
   * @param model - Model name (e.g., "BAAI/bge-m3")
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
    // Guard the 1:1 input-to-output contract before any positional access
    // (`results[0]` below, or index-based reassembly in callers). The queue
    // path returns mixed-success batches as 200 with only the successful
    // items, so a desynced count would otherwise misalign every
    // zip-inputs-to-outputs consumer.
    validateBatchResultCount(
      results,
      itemsArray,
      model,
      "encode",
      response.headers.get("x-sie-request-id") ?? undefined,
    );
    attachRequestMetadata(results, response.headers, data);

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
   * Price a request WITHOUT running it (`POST /v1/estimate`).
   *
   * The gateway plans the request through the same reservation planner the
   * metered path runs, against the same active rate book, and returns the plan
   * instead of holding it: no dispatch, no reservation, no credits consumed.
   * The quote is the CONSERVATIVE ceiling the live path would hold —
   * settlement bills the authoritative counts against that plan and releases
   * the rest, so the real charge is at most `estimated_credits`.
   *
   * @param endpoint - The exact target path, e.g. `"/v1/encode/BAAI/bge-m3"`
   *   (native, model in the path) or `"/v1/chat/completions"`
   *   (OpenAI-compatible, model in `request.model`).
   * @param request - The verbatim body you would send to `endpoint`. Passed
   *   through untouched — a body the SDK reshaped would be a quote for a
   *   request you never send.
   * @param options.timeout - Per-call timeout override in milliseconds.
   * @returns The planned units, applied rates, credits, and the active rate
   *   book's version + digest.
   * @throws {EstimateUnroutableError} If the active book cannot price the
   *   request. Same verdict the real request would get; the message names the
   *   unpriced identity or the dimension the planner could not bound.
   * @throws {RequestError} If the target body is invalid, with the same
   *   status/code the target route itself would return. In particular a
   *   `404 MODEL_NOT_FOUND` means this data plane does not serve the model at
   *   all — the estimate checks routability, not just priceability, so it never
   *   quotes a request that would 404.
   *
   * @example
   * ```typescript
   * const quote = await client.estimate("/v1/encode/BAAI/bge-m3", {
   *   items: [{ text: "Hello" }],
   * });
   * console.log(quote.estimated_credits, quote.rate_book_version);
   * ```
   */
  async estimate(
    endpoint: string,
    request: Record<string, unknown>,
    options: { timeout?: number } = {},
  ): Promise<CostEstimate> {
    const body = buildEstimateEnvelope(endpoint, request);
    return this.jsonRequest<CostEstimate>(
      ESTIMATE_PATH,
      "POST",
      body,
      options.timeout ?? this.timeout,
      throwIfEstimateUnroutable,
    );
  }

  /**
   * List available models.
   *
   * @returns Array of model information
   */
  async listModels(): Promise<ModelInfo[]> {
    const response = await this.requestJson("/v1/models", "GET");

    // Wire format response: {"models": [...]}
    interface WireModelsResponse {
      models: WireModelInfo[];
    }

    const data = (await response.json()) as WireModelsResponse;

    return data.models.map(toModelInfo);
  }

  /**
   * Get details for a specific model.
   *
   * Returns model metadata including dimensions, supported inputs/outputs,
   * lifecycle state, profiles, and capabilities. This is a lightweight call
   * that reads from model config — it does not load the model or trigger
   * inference.
   *
   * @param name - Model name (e.g., "BAAI/bge-m3")
   * @returns Model information
   */
  async getModel(name: string): Promise<ModelInfo> {
    const response = await this.requestJson(`/v1/models/${encodeURIComponent(name)}`, "GET");

    const data = (await response.json()) as WireModelInfo;

    return toModelInfo(data);
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
   * @param model - Model name (e.g., "BAAI/bge-reranker-v2-m3")
   * @param query - Query item
   * @param items - Items to score against the query
   * @param options - Score options
   * @returns Score result with sorted scores
   *
   * @example
   * ```typescript
   * const result = await client.score(
   *   "BAAI/bge-reranker-v2-m3",
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
   * Returns the aggregated outcome: the worker streams to the gateway,
   * the gateway aggregates, and the SDK returns the assembled result
   * plus SIE-native timing metadata (TTFT, TPOT, attempt id). To
   * consume chunks as they arrive, use {@link streamGenerate} instead.
   *
   * @example
   * ```typescript
   * const result = await client.generate(
   *   "Qwen/Qwen3-4B-Instruct-2507",
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

    const data = await parseTerminalJsonObject(response, "generate");
    const result = parseGenerateResult(data);
    attachRequestMetadata([result], response.headers, data);
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
      let response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
        redirect: "manual",
      });
      for (let hop = 0; hop < MODAL_CONTINUATION_MAX_HOPS; hop += 1) {
        requireVisibleManualRedirect(response);
        const continuationUrl = modalContinuationUrl(this.baseUrl, response);
        if (!continuationUrl) return response;
        response = await fetch(continuationUrl, {
          method: "GET",
          headers,
          signal: controller.signal,
          redirect: "manual",
        });
      }
      requireVisibleManualRedirect(response);
      if (modalContinuationUrl(this.baseUrl, response)) {
        throw new ProvisioningError(
          `Provisioning result remained in flight after ${MODAL_CONTINUATION_MAX_HOPS} continuation hops`,
        );
      }
      return response;
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
        throw connectionErrorFromFetchTypeError(err);
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

    const data = (await parseTerminalJsonObject(
      response,
      "chat.completion",
    )) as unknown as ChatCompletion;
    attachRequestMetadata([data], response.headers, data);
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
            redirect: "error",
          });
        } catch (error) {
          if (signal?.aborted) {
            throw new SIEConnectionError("Stream aborted before response", "other");
          }
          if (error instanceof Error && error.name === "AbortError") {
            throw new SIEConnectionError(`Stream open timeout after ${this.timeout}ms`, "timeout");
          }
          if (error instanceof TypeError) {
            throw connectionErrorFromFetchTypeError(error);
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
                await getErrorParam(attemptResponse.clone()),
              );
            }
            const elapsed = Date.now() - startTime;
            if (elapsed >= this.provisionTimeout) {
              throw new ProvisioningError(
                `Provisioning timeout after ${elapsed}ms`,
                gpu,
                getRetryAfter(attemptResponse),
                await getErrorParam(attemptResponse.clone()),
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
              throw new ModelLoadingError(
                `Model loading timeout for '${model}'`,
                model,
                await getErrorParam(attemptResponse.clone()),
              );
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
                {
                  model,
                  retries: oomRetries,
                  param: await getErrorParam(attemptResponse.clone()),
                },
              );
            }
            const delay = nextOomRetryDelay({
              retryAfter: getRetryAfter(attemptResponse),
              oomRetries,
              maxOomRetries: RESOURCE_EXHAUSTED_MAX_RETRIES,
              elapsedMs: Date.now() - startTime,
              provisionTimeoutMs: this.provisionTimeout,
              model,
              param: await getErrorParam(attemptResponse.clone()),
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
            readRequestId(attemptResponse),
            await getErrorParam(attemptResponse.clone()),
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
    attachRequestMetadata([result], response.headers, data);
    return result;
  }

  /**
   * Extract entities from a single item.
   *
   * @param model - Model name (e.g., "urchade/gliner_multi-v2.1")
   * @param item - Item to extract from
   * @param options - Extract options with labels
   * @returns Extract result with entities
   */
  async extract(model: string, item: ExtractItem, options: ExtractOptions): Promise<ExtractResult>;

  /**
   * Extract entities from multiple items.
   *
   * @param model - Model name (e.g., "urchade/gliner_multi-v2.1")
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
   *   "urchade/gliner_multi-v2.1",
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
    // Same positional contract as encode: `results[0]` below and index-based
    // reassembly in batch callers both assume one result per input, and the
    // queue path drops failed items from a 200 body.
    validateBatchResultCount(
      results,
      itemsArray,
      model,
      "extract",
      response.headers.get("x-sie-request-id") ?? undefined,
    );
    attachRequestMetadata(results, response.headers, data);

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
   *   pinnedModels: ["BAAI/bge-m3"],
   * });
   *
   * // Use the pool for requests
   * await client.encode("BAAI/bge-m3", { text: "Hello" }, { gpu: "eval-bench/l4" });
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
        redirect: "error",
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
                redirect: "error",
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
          redirect: "error",
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
   * const capacityWithModel = await client.waitForCapacity("l4", { model: "BAAI/bge-m3" });
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
    // First connect-retry is surfaced via `console.warn` (the SDK's existing
    // logging seam, cf. the version-skew warning) so a user does not silently
    // wait out the whole provision budget against an unreachable server.
    let warnedConnectRetry = false;

    while (true) {
      let response: Response;
      try {
        response = await this.request(path, body, pool, gpu, startTime);
      } catch (err) {
        // Only retry connect-time failures; see docstring for rationale.
        if (waitForCapacity && err instanceof SIEConnectionError && err.kind === "connect") {
          const elapsed = Date.now() - startTime;
          if (elapsed < this.provisionTimeout) {
            if (!warnedConnectRetry) {
              warnedConnectRetry = true;
              console.warn(
                `[SIE SDK] Connection to ${urlOriginForLogging(this.baseUrl)} failed (${err.message}); ` +
                  `retrying for up to ${this.provisionTimeout}ms`,
              );
            }
            const remaining = this.provisionTimeout - elapsed;
            const delay = Math.min(DEFAULT_RETRY_DELAY, remaining);
            await sleep(delay);
            continue;
          }
        }
        throw err;
      }

      // Short-circuit terminal load failures. The server
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
              await getErrorParam(response.clone()),
            );
          }

          const elapsed = Date.now() - startTime;
          if (elapsed >= this.provisionTimeout) {
            throw new ProvisioningError(
              `Provisioning timeout after ${elapsed}ms waiting for GPU '${gpu}'`,
              gpu,
              retryAfter,
              await getErrorParam(response.clone()),
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
              await getErrorParam(response.clone()),
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
              await getErrorParam(response.clone()),
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
            param: await getErrorParam(response.clone()),
          });
          oomRetries += 1;
          await sleep(delay);
          continue;
        }
      }

      // Retryable pre-execution admission backpressure (pass-2 audit B1/B2/B7):
      // a 429 RATE_LIMIT, or a retryable 503 (BILLING_CAPACITY_UNAVAILABLE /
      // QUEUE_FULL) the ladder above did not match. No work was published, so
      // retry within the provision-timeout budget honoring Retry-After; a
      // give-up throws a typed RateLimitError (429) or the server's terminal
      // 503. 402/403 credit/account errors are terminal and NOT handled here.
      const admissionDelay = await admissionRetryDelay(response, {
        startTime,
        provisionTimeoutMs: this.provisionTimeout,
      });
      if (admissionDelay !== undefined) {
        await sleep(admissionDelay);
        continue;
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
    startTime = Date.now(),
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

    const fetchWithinBudget = async (
      requestUrl: string,
      init: Omit<RequestInit, "signal">,
      followingContinuation: boolean,
    ): Promise<Response> => {
      const remaining = this.provisionTimeout - (Date.now() - startTime);
      if (followingContinuation && remaining <= 0) {
        throw new ProvisioningError(
          `Provisioning timeout after ${this.provisionTimeout}ms awaiting request result`,
          gpu,
        );
      }
      const requestTimeout = followingContinuation
        ? Math.min(this.timeout, remaining)
        : this.timeout;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), requestTimeout);
      try {
        return await fetch(requestUrl, { ...init, signal: controller.signal });
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          throw new SIEConnectionError(`Request timeout after ${requestTimeout}ms`, "timeout");
        }
        if (error instanceof TypeError) {
          if (followingContinuation) {
            throw new SIEConnectionError(
              `Failed to retrieve the in-flight request result: ${error.message}`,
              "other",
            );
          }
          throw connectionErrorFromFetchTypeError(error);
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
      }
    };

    let response = await fetchWithinBudget(
      url,
      {
        method: "POST",
        headers,
        body: body !== undefined ? packMessage(body) : undefined,
        redirect: "manual",
      },
      false,
    );

    for (let hop = 0; hop < MODAL_CONTINUATION_MAX_HOPS; hop += 1) {
      requireVisibleManualRedirect(response);
      const continuationUrl = modalContinuationUrl(this.baseUrl, response);
      if (!continuationUrl) return response;
      response = await fetchWithinBudget(
        continuationUrl,
        {
          method: "GET",
          headers,
          redirect: "manual",
        },
        true,
      );
    }
    requireVisibleManualRedirect(response);
    if (modalContinuationUrl(this.baseUrl, response)) {
      throw new ProvisioningError(
        `Provisioning result remained in flight after ${MODAL_CONTINUATION_MAX_HOPS} continuation hops`,
        gpu,
      );
    }
    return response;
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
        redirect: "error",
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
        throw connectionErrorFromFetchTypeError(error);
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

  /**
   * One JSON request over `fetch` (bearer auth reused; absolute or
   * base-relative URL).
   *
   * `onError` runs BEFORE the generic {@link handleError} dispatch, so a route
   * with its own typed failure taxonomy (the #2435 cost estimate's unroutable
   * verdict) can short-circuit without a second copy of this request path.
   */
  private async jsonRequest<T>(
    target: string,
    method: "GET" | "POST" | "DELETE",
    body?: unknown,
    timeoutMs: number = this.timeout,
    onError?: (response: Response) => Promise<void>,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const url = target.startsWith("http") ? target : `${this.baseUrl}${target}`;
    const headers: Record<string, string> = {
      Accept: JSON_CONTENT_TYPE,
      [SDK_VERSION_HEADER]: SDK_VERSION,
    };
    if (extraHeaders) Object.assign(headers, extraHeaders);
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const init: RequestInit = { method, headers };
    if (body !== undefined) {
      headers["Content-Type"] = JSON_CONTENT_TYPE;
      init.body = JSON.stringify(body);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    init.signal = controller.signal;
    init.redirect = "error";
    try {
      const response = await fetch(url, init);
      if (!response.ok) {
        if (onError) await onError(response);
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
        throw connectionErrorFromFetchTypeError(error);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async jobSubmit(options: SubmitJobOptions): Promise<JobSubmitResult | JobStatus> {
    const body = buildJobBody(options);
    let extraHeaders: Record<string, string> | undefined;
    if ("src" in body) {
      extraHeaders = { "Idempotency-Key": requireConnectorIdempotencyKey(options.idempotencyKey) };
    } else if (options.idempotencyKey != null) {
      throw new RequestError(
        "idempotencyKey applies only to connector-src jobs; inline items must omit it",
        "invalid_request",
        400,
      );
    }
    return this.jsonRequest<JobSubmitResult | JobStatus>(
      "/v1/jobs",
      "POST",
      body,
      Math.max(this.timeout, DEFAULT_LONG_RUNNING_TIMEOUT),
      undefined,
      extraHeaders,
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

  private async jobExecute(
    jobId: string,
    planRevision: number,
    idempotencyKey: string,
  ): Promise<JobStatus> {
    return this.jsonRequest<JobStatus>(
      `/v1/jobs/${encodeURIComponent(jobId)}/execute`,
      "POST",
      { plan_revision: planRevision },
      this.timeout,
      undefined,
      { "Idempotency-Key": requireConnectorIdempotencyKey(idempotencyKey) },
    );
  }

  private async jobRepair(
    jobId: string,
    planRevision: number,
    recoveryAttemptOrdinal: number,
    idempotencyKey: string,
  ): Promise<JobStatus> {
    return this.jsonRequest<JobStatus>(
      `/v1/jobs/${encodeURIComponent(jobId)}/repair`,
      "POST",
      {
        plan_revision: planRevision,
        recovery_attempt_ordinal: recoveryAttemptOrdinal,
      },
      this.timeout,
      undefined,
      { "Idempotency-Key": requireConnectorIdempotencyKey(idempotencyKey) },
    );
  }

  private async jobResults(jobId: string): Promise<JobResults> {
    let refreshes = 0;
    for (;;) {
      const job = await this.jobGet(jobId);
      const state = job.state;
      if (!state || !TERMINAL_JOB_STATES.has(state)) {
        throw new RequestError(
          `job ${jobId} is ${JSON.stringify(state)}, not terminal; results are decodable only after the job reaches a terminal state (succeeded/failed/suspended/cancelled)`,
          JOB_NOT_TERMINAL_ERROR_CODE,
          409,
        );
      }
      const chunks = jobChunks(job);
      const items: JobResultItem[] = [];
      try {
        for (const chunk of chunks) {
          // A `failed` chunk still carries a ref with its SUCCESSFUL siblings
          // (which are billed) plus the per-item failures, so its ref is read
          // too — only chunks with no ref at all are skipped. Each item's
          // `success`/`error` distinguishes the two.
          if (!chunk.ref) continue;
          const raw = await this.readRef(chunk.ref);
          try {
            items.push(...decodeChunkBytes(raw));
          } catch (error) {
            // Garbage bytes are a DECODE fault, not proof of failed
            // publication/billing — confine it and flag it distinctly rather
            // than folding it into the neutral incompleteness warning below.
            if (!(error instanceof MalformedChunkError)) throw error;
            console.warn(
              `[SIE SDK] job ${jobId} chunk (seq=${chunk.seq}) ref could not be decoded (malformed bytes); its items are omitted from the results`,
            );
          }
        }
      } catch (error) {
        const refreshable =
          error instanceof RequestError &&
          error.statusCode === 404 &&
          error.code === JOB_RESULT_NOT_FOUND_ERROR_CODE;
        if (refreshable && refreshes < JOB_RESULT_REF_MAX_REFRESHES) {
          refreshes += 1;
          continue;
        }
        throw error;
      }
      const withDims = items.find((it) => it.dims != null);
      const retrieved = items.length;
      const totalItems = job.total_items;
      if (totalItems != null && retrieved < totalItems) {
        // Neutral: state only what is known (fewer items decoded than the job's
        // item count). Do NOT assert a cause — the shortfall may be an
        // unpublished chunk OR an undecodable ref, and this call cannot prove
        // billing from the status doc.
        console.warn(
          `[SIE SDK] job ${jobId} results are incomplete: retrieved ${retrieved} of ${totalItems} items`,
        );
      }
      return {
        job_id: job.id ?? jobId,
        state,
        total_items: totalItems,
        settled_credits: job.settled_credits,
        chunks,
        retrieved,
        dims: withDims ? withDims.dims : null,
        items,
      };
    }
  }

  private async jobWait(
    jobId: string,
    options?: { timeoutMs?: number; pollMs?: number; raiseOnFailure?: boolean },
  ): Promise<JobStatus> {
    const timeoutMs = options?.timeoutMs ?? DEFAULT_JOB_WAIT_TIMEOUT;
    const pollMs = options?.pollMs ?? DEFAULT_JOB_WAIT_POLL;
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const job = await this.jobGet(jobId);
      if (job.phase === "planned") return job;
      const state = job.state;
      if (state && TERMINAL_JOB_STATES.has(state)) {
        if (options?.raiseOnFailure && state !== "succeeded") {
          const outcome = job.outcome;
          const errorCode = job.error_code;
          const reason =
            outcome || errorCode
              ? ` (outcome=${JSON.stringify(outcome)}, error_code=${JSON.stringify(errorCode)})`
              : "";
          throw new JobFailedError(`job ${jobId} terminated ${JSON.stringify(state)}${reason}`, {
            jobId: job.id ?? jobId,
            state,
            outcome,
            errorCode,
          });
        }
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
        redirect: "error",
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
        throw connectionErrorFromFetchTypeError(error);
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
    options: AddConnectionOptions = {},
  ): Promise<ConnectionCreated> {
    requireConnectionName(name);
    const schemaPolicy = requireConnectionSchemaPolicy(
      type,
      options.sourceSchema,
      options.sinkSchema,
    );
    const body: Record<string, unknown> = {
      type,
      name,
      secret,
    };
    if (schemaPolicy) {
      body.source_schema = schemaPolicy.sourceSchema;
      body.sink_schema = schemaPolicy.sinkSchema;
    }
    return this.jsonRequest<ConnectionCreated>(this.connectionsBase(), "POST", body);
  }

  private async connectionList(): Promise<Connection[]> {
    const data = await this.jsonRequest<{ connections?: Connection[] }>(
      this.connectionsBase(),
      "GET",
    );
    return Array.isArray(data) ? data : (data.connections ?? []);
  }

  private async connectionRevoke(name: string): Promise<ConnectionRevoked> {
    const canonicalName = requireConnectionName(name);
    return this.jsonRequest<ConnectionRevoked>(
      `${this.connectionsBase()}/${canonicalName}`,
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
        redirect: "error",
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
        throw connectionErrorFromFetchTypeError(error);
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
      const response = await fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
        redirect: "error",
      });
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
        throw connectionErrorFromFetchTypeError(error);
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

  private async fileList(options?: {
    after?: string;
    limit?: number;
    order?: "asc" | "desc";
    purpose?: string;
  }): Promise<SIEFile[]> {
    return (await this.fileListPage(options)).data ?? [];
  }

  private async fileListPage(options?: {
    after?: string;
    limit?: number;
    order?: "asc" | "desc";
    purpose?: string;
  }): Promise<FileList> {
    const query = new URLSearchParams();
    if (options?.after !== undefined) query.set("after", options.after);
    if (options?.limit !== undefined) query.set("limit", String(options.limit));
    if (options?.order !== undefined) query.set("order", options.order);
    if (options?.purpose !== undefined) query.set("purpose", options.purpose);
    const suffix = query.size > 0 ? `?${query.toString()}` : "";
    const data = await this.jsonRequest<FileList | SIEFile[]>(`/v1/files${suffix}`, "GET");
    if (!Array.isArray(data)) return data;
    return {
      object: "list",
      data,
      first_id: data[0]?.id ?? null,
      last_id: data.at(-1)?.id ?? null,
      has_more: false,
    };
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

  private async batchList(options?: {
    after?: string;
    limit?: number;
  }): Promise<Batch[]> {
    return (await this.batchListPage(options)).data ?? [];
  }

  private async batchListPage(options?: {
    after?: string;
    limit?: number;
  }): Promise<BatchList> {
    const query = new URLSearchParams();
    if (options?.after !== undefined) query.set("after", options.after);
    if (options?.limit !== undefined) query.set("limit", String(options.limit));
    const suffix = query.size > 0 ? `?${query.toString()}` : "";
    const data = await this.jsonRequest<BatchList | Batch[]>(`/v1/batches${suffix}`, "GET");
    if (!Array.isArray(data)) return data;
    return {
      object: "list",
      data,
      first_id: data[0]?.id ?? null,
      last_id: data.at(-1)?.id ?? null,
      has_more: false,
    };
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
        redirect: "error",
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
