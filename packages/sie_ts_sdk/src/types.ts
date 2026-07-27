/**
 * Types for the SIE TypeScript SDK
 *
 * These types mirror the Python SDK (packages/sie_sdk/src/sie_sdk/types.py)
 * for full feature parity.
 */

import type { ImageInput, ImageWireFormat } from "./images.js";

/**
 * Output dtype options for quantized embeddings.
 * Matches Python DType literal.
 */
export type DType = "float32" | "float16" | "bfloat16" | "int8" | "uint8" | "binary" | "ubinary";

/**
 * Output type options for encode operation.
 */
export type OutputType = "dense" | "sparse" | "multivector";

/**
 * Document input for composite-document extractors (PDF, DOCX, HTML, ...).
 *
 * The wire format is the document bytes plus an optional format hint. The
 * hint is advisory — adapters may sniff the bytes when it is missing or
 * unrecognized.
 */
export interface DocumentInput {
  /** Document bytes (raw file content) */
  data: Uint8Array;
  /** Document format hint: "pdf", "docx", "html", etc. */
  format?: string;
}

/** Encoded audio bytes and optional decoder metadata. */
export interface AudioInput {
  /** Encoded audio bytes (WAV, MP3, FLAC, etc.) */
  data: Uint8Array;
  /** Audio container/codec format hint */
  format?: string;
  /** Source sample rate in Hz, when known */
  sampleRate?: number;
}

/**
 * A single item to encode, score, or extract from.
 *
 * For simple text encoding, just use `{ text: "your text here" }`.
 *
 * @example
 * // Simple text
 * { text: "Hello world" }
 *
 * // With ID for tracking through results
 * { id: "doc-1", text: "Document text" }
 *
 * // With images for multimodal models (ColPali, CLIP)
 * { text: "Description", images: [imageBytes] }
 *
 * // With a document for composite-document extractors (Docling, ...)
 * { document: { data: pdfBytes, format: "pdf" } }
 *
 * // Pre-encoded multivector (for use with maxsim utility)
 * { multivector: [tokenEmbedding1, tokenEmbedding2, ...] }
 */
export interface Item {
  /** Optional ID to track this item through results */
  id?: string;
  /** Text content to encode */
  text?: string;
  /** Images for multimodal models; converted to wire format by the client */
  images?: (ImageInput | ImageWireFormat)[];
  /** Document for composite-document extractors (PDF, DOCX, HTML, ...) */
  document?: DocumentInput;
  /** Pre-encoded multivector (for use with maxsim utility) */
  multivector?: Float32Array[];
  /** Arbitrary metadata (passed through to results) */
  metadata?: Record<string, unknown>;
}

/** An item accepted by extract(), including optional encoded audio. */
export interface ExtractItem extends Item {
  /** Encoded audio for speech extraction, optionally with decoder metadata. */
  audio?: AudioInput | Uint8Array;
}

/**
 * Sparse vector result with non-zero indices and values.
 * Used by SPLADE-type models.
 */
export interface SparseResult {
  /** Token indices with non-zero weights */
  indices: Int32Array;
  /** Weight values for each index */
  values: Float32Array;
}

/**
 * Server-side timing breakdown for a request.
 */
export interface TimingInfo {
  totalMs?: number;
  queueMs?: number;
  tokenizationMs?: number;
  inferenceMs?: number;
}

/** Authoritative units settled for one completed HTTP request. */
export interface RequestUsage {
  inputTokens?: number;
  pairs?: number;
  images?: number;
  pages?: number;
  outputTokens?: number;
  audioMs?: number;
}

/** Optional gateway metadata from the successful terminal response. */
export interface RequestMetadata {
  id?: string;
  /** Worker-origin immutable release/runtime identity digest. */
  executionIdentitySha256?: string;
  usage?: RequestUsage;
  creditsDebited?: number;
}

/**
 * Result of encoding a single item.
 *
 * Contains the item ID (if provided) and one or more output representations
 * depending on what was requested via outputTypes.
 */
export interface EncodeResult {
  /** Item ID (echoed from request if provided) */
  id?: string;
  /** Dense embedding vector, shape [dims] */
  dense?: Float32Array;
  /** Sparse embedding with indices and values */
  sparse?: SparseResult;
  /** Multi-vector embedding for late interaction models, shape [numTokens][tokenDims] */
  multivector?: Float32Array[];
  /** Server-side timing breakdown */
  timing?: TimingInfo;
  /** Request-scoped usage and settled debit, when supplied by the gateway. */
  request?: RequestMetadata;
}

/**
 * Model dimension information.
 */
export interface ModelDims {
  dense?: number;
  sparse?: number;
  multivector?: number;
}

/**
 * Advertised model capabilities.
 *
 * Mirrors the gateway `capabilities` object on each `/v1/models` entry
 * (`ModelCapabilitiesWire`). All fields are optional; their presence
 * depends on what the model config declares. `grammar` lists the
 * supported grammar kinds ("json_schema" | "regex" | "ebnf").
 * `code`/`sql`/`guard` are informational flags advertising validated
 * generation jobs that back the model="code"/"sql"/"guard" aliases.
 *
 * These flags mean the model *supports* a task — they are NOT a
 * precision-independent quality SLA. A flag is true at the model level even
 * when quality is profile/precision-dependent (e.g. `sql` quality regresses
 * under FP8; route SQL-critical traffic to a BF16 bundle via the `sql` alias).
 */
export interface ModelCapabilities {
  /** Supported grammar kinds: ["json_schema", "regex", "ebnf"] */
  grammar?: string[];
  /** Whether the model supports tool / function calling */
  tools?: boolean;
  /** Union of LoRA served-names across profiles (display summary) */
  lora_adapters?: string[];
  /** Per-profile LoRA served-names, keyed by profile name */
  profile_lora_adapters?: Record<string, string[]>;
  /** Validated for code generation; backs model="code" */
  code?: boolean;
  /** Supports text-to-SQL; backs model="sql". Precision-sensitive (FP8 regresses SQL) — a support flag, not a per-profile quality guarantee. */
  sql?: boolean;
  /** Generative guard model; backs model="guard" */
  guard?: boolean;
}

/**
 * Information about a model returned by listModels().
 */
export interface ModelInfo {
  /** Model name/identifier */
  name: string;
  /** Whether the model is currently loaded in memory */
  loaded: boolean;
  /** Supported input types: ["text"], ["text", "image"], ["text", "document"], etc. */
  inputs: string[];
  /** Supported output types: ["dense"], ["dense", "sparse"], etc. */
  outputs: string[];
  /** Embedding dimensions for each output type */
  dims?: ModelDims;
  /** Maximum sequence length the model supports */
  maxSequenceLength?: number;
  /** Advertised model capabilities (grammar, tools, code/sql/guard, LoRA adapters) */
  capabilities?: ModelCapabilities;
}

/**
 * A single score entry from reranking.
 */
export interface ScoreEntry {
  /** ID of the item (from request or auto-generated) */
  itemId: string;
  /** Relevance score (higher = more relevant) */
  score: number;
  /** Position in sorted order (0 = most relevant) */
  rank: number;
}

export interface ScoreUsage {
  /** Post-truncation input tokens processed */
  inputTokens: number;
  /** Images processed across query-document pairs */
  images?: number;
}

/**
 * Result of scoring items against a query.
 */
export interface ScoreResult {
  /** Model used for scoring */
  model?: string;
  /** Query ID (echoed from request if provided) */
  queryId?: string;
  /** Score entries, sorted by relevance (descending) */
  scores: ScoreEntry[];
  /** Authoritative usage when emitted by the score adapter */
  usage?: ScoreUsage;
  /** Request-scoped usage and settled debit, when supplied by the gateway. */
  request?: RequestMetadata;
}

/**
 * A single extracted entity (NER span).
 */
export interface Entity {
  /** The extracted text span */
  text: string;
  /** Entity type/label (e.g., "person", "organization") */
  label: string;
  /** Confidence score */
  score: number;
  /** Start character offset in the original text */
  start?: number;
  /** End character offset in the original text */
  end?: number;
  /** Bounding box [x, y, width, height] for image-based extraction */
  bbox?: number[];
}

/**
 * A relation triple between two entities.
 */
export interface Relation {
  /** Head entity text */
  head: string;
  /** Tail entity text */
  tail: string;
  /** Relation type label (e.g., "works_at", "founded_by") */
  relation: string;
  /** Confidence score */
  score: number;
}

/**
 * A text classification result.
 */
export interface Classification {
  /** Classification label (e.g., "positive", "negative") */
  label: string;
  /** Confidence score */
  score: number;
}

/**
 * A detected object with bounding box.
 */
export interface DetectedObject {
  /** Object class label (e.g., "person", "car") */
  label: string;
  /** Confidence score */
  score: number;
  /** Bounding box [x, y, width, height] */
  bbox: number[];
}

/** Stable per-item extraction failure. */
export interface ExtractItemError {
  /** Stable extraction error code */
  code: string;
  /** Sanitized extraction error message */
  message: string;
}

/**
 * Result of extraction for a single item.
 */
export interface ExtractResult {
  /** Item ID (echoed from request if provided) */
  id?: string;
  /** List of extracted entities */
  entities: Entity[];
  /** List of extracted relation triples */
  relations: Relation[];
  /** List of classification results */
  classifications: Classification[];
  /** List of detected objects */
  objects: DetectedObject[];
  /** Additional structured extraction data */
  data?: Record<string, unknown>;
  /** Stable per-item failure when extraction did not complete */
  error?: ExtractItemError;
  /** Request-scoped usage and settled debit, when supplied by the gateway. */
  request?: RequestMetadata;
}

/**
 * Information about a worker in the cluster.
 */
export interface WorkerInfo {
  /** Worker base URL */
  url: string;
  /** GPU type (e.g., "l4", "a100-80gb") */
  gpu: string;
  /** Number of GPUs on this worker */
  gpuCount: number;
  /** Number of worker GPU slots ready to receive work */
  readyGpuSlots: number;
  /** Whether the worker is healthy */
  healthy: boolean;
  /** Number of items in the worker's queue */
  queueDepth: number;
  /** Worker-local scheduler pending cost */
  pendingCost: number;
  /** Number of batches currently executing on the worker */
  inflightBatches: number;
  /** List of model names loaded on this worker */
  loadedModels: string[];
}

/**
 * Cluster capacity information returned by getCapacity().
 */
export interface CapacityInfo {
  /** Overall cluster status: "healthy", "degraded", "no_workers" */
  status: string;
  /** Number of healthy workers */
  workerCount: number;
  /** Number of GPUs available */
  gpuCount: number;
  /** Number of unique models loaded across all workers */
  modelsLoaded: number;
  /** Canonical machine profiles configured in the cluster */
  configuredGpuTypes: string[];
  /** Machine profiles currently running */
  liveGpuTypes: string[];
  /** List of worker details */
  workers: WorkerInfo[];
}

/**
 * Pool specification for creating resource pools.
 */
export interface PoolCreateSpec {
  /** Pool name (used in GPU param as "poolName/machineProfile") */
  name: string;
  /** Optional Helm/NATS queue namespace backing this logical pool. Defaults to "default"; non-default values must be declared under queueRouting.staticQueuePools. */
  queuePool?: string;
  /** Machine profile requirements for pool readiness, e.g., { l4: 2, "a100-40gb": 1 } */
  gpus?: Record<string, number>;
  /** Optional maximum assigned workers per machine profile */
  gpuCaps?: Record<string, number>;
}

/**
 * Optional arguments for `SIEClient.createPool`, mirroring the Python SDK's
 * `create_pool` keyword arguments (`bundle`, `minimum_worker_count`,
 * `pinned_models`).
 */
export interface CreatePoolOptions {
  /** Optional bundle filter. When set, only workers running this bundle will be assigned to the pool. */
  bundle?: string;
  /**
   * Per-pool warm floor (minimum machines kept warm). The gateway emits
   * canonical `sie.gateway.pool.warm_floor` telemetry; the collector exposes
   * `sie_gateway_pool_warm_floor` to KEDA. Defaults to 0 (scale to zero).
   */
  minimumWorkerCount?: number;
  /**
   * Model ids to keep loaded so the first request to them pays no cold
   * model-load. Each id must be a model the gateway already tracks and may be
   * profile-qualified ("model-name:profile_name"); unknown ids are rejected.
   */
  pinnedModels?: string[];
}

/**
 * Pool specification returned by the gateway.
 */
export interface PoolSpec {
  /** Pool name (used in GPU param as "poolName/machineProfile") */
  name: string;
  /** Helm/NATS queue namespace backing this logical pool */
  queue_pool?: string;
  /** Optional bundle constraint for this pool */
  bundle?: string | null;
  /** Machine profile requirements for pool readiness, e.g., { l4: 2, "a100-40gb": 1 } */
  gpus?: Record<string, number>;
  /** Optional maximum assigned workers per machine profile, as returned by the gateway */
  gpu_caps?: Record<string, number>;
  /** Optional TTL for dynamic pool leases */
  ttl_seconds?: number | null;
  /** Minimum workers kept warm for this pool */
  minimum_worker_count?: number;
  /** Models pinned for this pool */
  pinned_models?: string[];
}

/**
 * Pool status information.
 */
export interface PoolStatus {
  /** Pool state: "pending", "active", "expired" */
  state: string;
  /** Workers assigned to this pool */
  assignedWorkers: Array<{ name: string; url: string; gpu: string }>;
  /** Unix timestamp when pool was created */
  createdAt?: number;
  /** Unix timestamp of last lease renewal */
  lastRenewed?: number;
}

/**
 * Full pool information.
 */
export interface PoolInfo {
  /** Pool name */
  name: string;
  /** Pool specification */
  spec: PoolSpec;
  /** Pool status */
  status: PoolStatus;
}

// ---------------------------------------------------------------------------
// WebSocket Status Types
// ---------------------------------------------------------------------------

/**
 * Canonical model residency states. The runtime `MODEL_STATES` array is the
 * single source the `ModelState` type is derived from, so it can be checked
 * against the shared golden fixture (`packages/wire-fixtures/model_state.json`)
 * in CI. Includes `"failed"` (a terminal load-failure state) to stay in parity
 * with the Python SDK and server, which already carry it.
 */
export const MODEL_STATES = ["available", "loading", "loaded", "unloading", "failed"] as const;
export type ModelState = (typeof MODEL_STATES)[number];

export interface ClusterSummary {
  worker_count: number;
  gpu_count: number;
  models_loaded: number;
  total_qps: number;
}

export interface ClusterWorkerInfo {
  url: string;
  gpu: string;
  healthy: boolean;
  queue_depth: number;
  loaded_models: string[];
}

export interface ModelSummary {
  name: string;
  state: ModelState;
  worker_count: number;
  gpu_types: string[];
  total_queue_depth: number;
}

export interface ServerInfo {
  version: string;
  uptime_seconds: number;
  user: string;
  working_dir: string;
  pid: number;
}

export interface GPUMetrics {
  device: string;
  name: string;
  gpu_type: string;
  utilization_pct: number;
  memory_used_bytes: number;
  memory_total_bytes: number;
  memory_threshold_pct?: number;
}

export interface ModelConfig {
  hf_id: string;
  adapter: string;
  inputs: string[];
  outputs: string[];
  dims: Record<string, number | null>;
  max_sequence_length?: number;
  pooling?: string | null;
  normalize?: boolean;
  adapter_options_loadtime?: Record<string, unknown> | null;
  adapter_options_runtime?: Record<string, unknown> | null;
}

export interface ModelStatus {
  name: string;
  state: ModelState;
  device: string | null;
  memory_bytes: number;
  config: ModelConfig;
  queue_depth: number;
  queue_pending_items: number;
}

/**
 * Worker status snapshot. Request-rate and latency telemetry is exported
 * through the configured OTel destination instead of this message.
 */
export interface WorkerStatusMessage {
  timestamp: number;
  name: string;
  gpu: string;
  gpu_count: number;
  bundle: string;
  machine_profile: string;
  loaded_models: string[];
  server: ServerInfo;
  gpus: GPUMetrics[];
  models: ModelStatus[];
}

export interface ClusterStatusMessage {
  timestamp: number;
  cluster: ClusterSummary;
  workers: ClusterWorkerInfo[];
  models: ModelSummary[];
}

export type StatusMessage = WorkerStatusMessage | ClusterStatusMessage;

// ---------------------------------------------------------------------------
// Client Options
// ---------------------------------------------------------------------------

/**
 * Options for SIEClient constructor.
 */
export interface SIEClientOptions {
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Default GPU type for all requests (e.g., "l4", "a100-80gb") */
  gpu?: string;
  /** API key for authentication (sent as Bearer token) */
  apiKey?: string;
  /**
   * Whether to auto-retry retryable capacity signals (503 PROVISIONING,
   * idempotent 504 gateway timeouts, transient connect errors).
   * Default: `true`, matching the Python SDK's `wait_for_capacity=True`.
   * BREAKING (0.7): the default was previously `false` — pass `false`
   * explicitly to fail fast.
   */
  waitForCapacity?: boolean;
  /** Maximum time to wait for provisioning in milliseconds (default: 300000) */
  provisionTimeout?: number;
  /**
   * Control-plane base URL for the `connections` namespace (connector
   * auth lives on the control plane, not the keyed gateway). Required to call
   * `client.connections.*`.
   */
  controlPlaneUrl?: string;
  /** Org the `connections` namespace operates on (org-scoped by path in the POC). */
  org?: string;
}

/**
 * An org-scoped connection (connector auth by name). The secret is never
 * returned by the list endpoint (only the job-runner resolve path sees it).
 */
export interface Connection {
  id?: number;
  type?: string;
  name?: string;
  created_at?: number;
}

/** The `201` envelope from creating a connection (org + connection, no secret). */
export interface ConnectionCreated extends Connection {
  org?: string;
  account_id?: number;
}

/** The envelope from revoking (soft-deleting) a connection. */
export interface ConnectionRevoked {
  org?: string;
  account_id?: number;
  name?: string;
  state?: string;
}

/**
 * An OpenAI-shaped file object (`POST/GET /v1/files`). The
 * gateway emits `{id, object:"file", bytes, created_at, filename, purpose}`;
 * `status` / `expires_at` are the optional OpenAI fields surfaced
 * when the TTL-GC lands. Byte-for-byte OpenAI's `FileObject`.
 */
export interface File {
  id?: string;
  object?: string;
  bytes?: number;
  created_at?: number;
  filename?: string;
  purpose?: string;
  status?: string;
  expires_at?: number;
}

/** The envelope from deleting a file (OpenAI `FileDeleted`). */
export interface FileDeleted {
  id?: string;
  object?: string;
  deleted?: boolean;
}

/** Per-batch request tally (OpenAI `request_counts`). */
export interface BatchRequestCounts {
  total?: number;
  completed?: number;
  failed?: number;
}

/**
 * An OpenAI-shaped batch object (`POST/GET /v1/batches`).
 * The gateway emits `{id, object:"batch", endpoint, input_file_id,
 * completion_window, status, output_file_id, error_file_id, created_at,
 * finished_at, request_counts, errors}`. The granular OpenAI
 * timestamps (`completed_at` etc.) are optional — a strict OpenAI SDK reads them
 * when the gateway surfaces them.
 */
export interface Batch {
  id?: string;
  object?: string;
  endpoint?: string;
  input_file_id?: string;
  completion_window?: string;
  status?: string;
  output_file_id?: string | null;
  error_file_id?: string | null;
  created_at?: number;
  finished_at?: number | null;
  in_progress_at?: number | null;
  completed_at?: number | null;
  failed_at?: number | null;
  expired_at?: number | null;
  cancelled_at?: number | null;
  request_counts?: BatchRequestCounts;
  errors?: unknown;
  metadata?: Record<string, unknown> | null;
}

/** The listing envelope from `GET /v1/batches` (OpenAI cursor page). */
export interface BatchList {
  object?: string;
  data?: Batch[];
  first_id?: string;
  last_id?: string;
  has_more?: boolean;
}

/**
 * Options for encode operation.
 */
export interface EncodeOptions {
  /** Output types to request: ["dense"], ["sparse"], ["dense", "sparse", "multivector"] */
  outputTypes?: OutputType[];
  /** Instruction prefix for instruction-tuned models */
  instruction?: string;
  /** Whether this is a query (for asymmetric models) */
  isQuery?: boolean;
  /** Output dtype for quantization */
  outputDtype?: DType;
  /** GPU type for this request (overrides client default) */
  gpu?: string;
  /** Whether to wait for capacity (overrides client default) */
  waitForCapacity?: boolean;
}

/**
 * Options for score operation.
 */
export interface ScoreOptions {
  /** GPU type for this request */
  gpu?: string;
  /** Whether to wait for capacity */
  waitForCapacity?: boolean;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/** Reason the generation terminated. */
export type FinishReason = "stop" | "length" | "cancelled" | "content_filter" | "error";

/** Token usage for a single generation call. */
export interface GenerationUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

interface GrammarMetadata {
  /** Optional schema/grammar label used by structured-output backends. */
  label?: string | null;
  /** Optional strictness hint for structured-output backends. */
  strict?: boolean | null;
}

/** Constrain native generation to a JSON Schema. */
export type JsonSchemaGrammar = GrammarMetadata & {
  json_schema: Record<string, unknown>;
  regex?: never;
  ebnf?: never;
};

/** Constrain native generation to a regular expression. */
export type RegexGrammar = GrammarMetadata & {
  json_schema?: never;
  regex: string;
  ebnf?: never;
};

/** Constrain native generation to an EBNF grammar. */
export type EbnfGrammar = GrammarMetadata & {
  json_schema?: never;
  regex?: never;
  ebnf: string;
};

/** Native structured-output grammar. Exactly one grammar variant is set. */
export type GenerateGrammar = JsonSchemaGrammar | RegexGrammar | EbnfGrammar;

/** Options for the generate operation. */
export interface GenerateOptions {
  /** Hard cap on output tokens. Required. */
  maxNewTokens: number;
  /** Optional native image inputs rendered with the prompt for vision-capable models. */
  images?: (ImageInput | ImageWireFormat)[];
  /** Sampling temperature. */
  temperature?: number;
  /** Nucleus sampling cutoff. */
  topP?: number;
  /** Optional list of stop strings. */
  stop?: string[];
  /** OpenAI-compatible frequency penalty in [-2, 2]. */
  frequencyPenalty?: number;
  /** OpenAI-compatible presence penalty in [-2, 2]. */
  presencePenalty?: number;
  /**
   * Native structured-output grammar.
   *
   * The broad record arm preserves the pre-existing SDK input contract for
   * callers that keep valid grammar objects in `Record<string, unknown>`
   * variables. The SDK still validates the exact three-arm shape at runtime.
   */
  grammar?: GenerateGrammar | Record<string, unknown>;
  /**
   * Optional per-request sampling seed. Must be a JavaScript safe integer
   * (-(2^53 - 1) through 2^53 - 1) so JSON serialization preserves it exactly.
   * Reproducibility depends on the active backend and deployment configuration.
   */
  seed?: number;
  /** Token-id-to-bias map. */
  logitBias?: Record<string, number>;
  /** Stable request routing key. */
  routingKey?: string;
  /** Prompt cache affinity key. */
  promptCacheKey?: string;
  /** Opaque privacy-preserving safety identifier. */
  safetyIdentifier?: string;
  /** Served LoRA adapter name. */
  loraAdapter?: string;
  /**
   * Governed generation runtime options forwarded as `options`. Typed native
   * fields still win when both surfaces provide the same sampler control.
   */
  adapterOptions?: Record<string, unknown>;
  /** GPU type / pool spec, e.g. ``"l4"`` or ``"eval-bench/l4"``. */
  gpu?: string;
  /** Auto-retry under provisioning. */
  waitForCapacity?: boolean;
}

/** Options for streaming native generation. */
export interface StreamGenerateOptions extends GenerateOptions {
  /** Include per-token log probabilities in streamed chunks. */
  logprobs?: boolean;
  /** Number of alternate token log probabilities to return, in [0, 20]. */
  topLogprobs?: number;
}

/** Aggregated generation result. */
export interface GenerateResult {
  /** Model id the gateway dispatched to. */
  model: string;
  /** Full generated text (concatenation of all streamed deltas). */
  text: string;
  /** Termination reason. */
  finishReason: FinishReason;
  /** Prompt / completion / total token counts. */
  usage: GenerationUsage;
  /** Worker-generated attempt id. */
  attemptId?: string;
  /** Time-to-first-token in milliseconds. */
  ttftMs?: number;
  /** Average time per output token in milliseconds. */
  tpotMs?: number;
  /** Request-scoped usage and settled debit, when supplied by the gateway. */
  request?: RequestMetadata;
}

// ---------------------------------------------------------------------------
// Chat completions (OpenAI-compatible) — /v1/chat/completions
// ---------------------------------------------------------------------------

/**
 * A single message in a chat completion request.
 *
 * Accepted roles: `system`, `user`, `assistant`, `tool`, `developer`. The
 * gateway normalises `developer` → `system` before forwarding to the worker
 * (the OpenAI 2024-08 rename — most chat templates only have `system`).
 *
 * `content` may be a string OR an array of typed content parts. The gateway
 * concatenates `text` / `input_text` parts; `image_url` / `input_image` parts
 * carrying a base64 `data:` URI are accepted for vision-capable generation
 * models (non-vision/encode-only models reject with `400 unsupported_field`;
 * remote (non-`data:`) URLs reject with `400 invalid_request`). See
 * `packages/sie_gateway/src/openapi.rs` and `proxy.rs::chat_params_from_json`
 * for the canonical accepted subset.
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool" | "developer";
  content: string | ChatContentPart[] | null;
  name?: string;
  /** Required when `role === "tool"`. */
  tool_call_id?: string;
  /** Populated by the model when calling tools (assistant turns only). */
  tool_calls?: ToolCall[];
}

/**
 * One content part inside a multimodal `messages[*].content` array. Text parts
 * (`text` / `input_text`) are concatenated; image parts (`image_url` /
 * `input_image`) carry a base64 `data:` URI and are accepted for vision-capable
 * generation models.
 */
export type ChatContentPart =
  | { type: "text"; text: string }
  | { type: "input_text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "input_image"; image_url: string | { url: string } };

/** A tool call emitted by the model. */
export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** A tool the model is allowed to call. */
export interface ToolSpec {
  type: "function";
  function: {
    name: string;
    description?: string;
    /** JSON Schema describing the function arguments. */
    parameters?: Record<string, unknown>;
  };
}

/** Tool-routing directive. */
export type ToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

/** Structured-output `response_format` envelope. */
export interface ResponseFormat {
  type: "json_schema" | "json_object" | "text";
  /** JSON Schema body when `type === "json_schema"`. */
  json_schema?: unknown;
}

/** OpenAI-compatible chat-completion finish reason. */
export type ChatFinishReason = "stop" | "length" | "tool_calls" | "content_filter" | null;

/**
 * Request body for `chatCompletions` / `streamChatCompletions`.
 *
 * Field names are snake_case (the wire shape) so the SDK can hand the object
 * to `JSON.stringify` without further translation. SIE-specific routing
 * fields (`routing_key`, `prompt_cache_key`) match the gateway schema in
 * `packages/sie_gateway/src/openapi.rs`.
 *
 * The gateway honours: `model`, `messages`, `max_tokens` /
 * `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `stream`,
 * `stream_options`, `tools`, `tool_choice`, `parallel_tool_calls`,
 * `response_format`, `frequency_penalty`, `presence_penalty` (each in
 * `[-2, 2]`), `repetition_penalty`, `n`, `best_of`, `logprobs`,
 * `top_logprobs`, `logit_bias`, `seed`, `user`, `safety_identifier`,
 * `lora_adapter`, `routing_key`, and `prompt_cache_key`. Unknown fields
 * are rejected with `400 unsupported_field`.
 */
export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  /** Legacy alias; the gateway prefers `max_completion_tokens` when both set. */
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  /**
   * Non-OpenAI sampling knob (vLLM / SGLang). Integer `>= 1`; absent →
   * sampler default (top-k disabled).
   */
  top_k?: number;
  /**
   * Non-OpenAI repetition penalty (SGLang). Float in `(0.0, 2.0]`; `1.0`
   * means no penalty. Absent → sampler default.
   */
  repetition_penalty?: number;
  /** Single stop string or list of stop strings. */
  stop?: string | string[];
  /** Set to `true` to use `streamChatCompletions`. `chatCompletions` rejects this. */
  stream?: boolean;
  /** Streaming-only: ask the server to emit a final usage-only chunk before `[DONE]`. */
  stream_options?: { include_usage?: boolean };
  tools?: ToolSpec[];
  tool_choice?: ToolChoice;
  /** OpenAI parallel-tool-calls toggle (default `true`). */
  parallel_tool_calls?: boolean;
  response_format?: ResponseFormat;
  /** Accepted in the OpenAI range [-2, 2]; out-of-range values are rejected. */
  frequency_penalty?: number;
  presence_penalty?: number;
  /**
   * Multi-candidate count. Default `1`. `n > 1 && stream === true` is
   * rejected by the gateway with 400.
   */
  n?: number;
  /**
   * Generate this many candidates and return the top `n` by cumulative
   * logprob. Range `[1, 128]`; requires `best_of >= n` and `stream: false`.
   */
  best_of?: number;
  /**
   * `true` requests per-token log-probabilities on each chunk / on the
   * aggregate response. Required when `top_logprobs > 0`.
   */
  logprobs?: boolean;
  /**
   * How many alternate-token logprobs to return per position. Range
   * `[0, 20]` per the OpenAI spec; implies `logprobs: true` when `> 0`.
   */
  top_logprobs?: number;
  /**
   * `{token_id: bias_float}` map. Gateway validates per-value range
   * `[-100, 100]` and caps map size.
   */
  logit_bias?: Record<string, number>;
  /**
   * Optional per-request sampling seed. Must be a JavaScript safe integer
   * (`-(2^53 - 1)` through `2^53 - 1`) so the caller's integer is represented
   * exactly. The SDK throws `RangeError` before network I/O for invalid values.
   * Reproducibility depends on the active backend and deployment configuration.
   */
  seed?: number;
  /**
   * OpenAI's free-text end-user identifier. Accepted and logged at debug
   * level by the gateway.
   */
  user?: string;
  /**
   * OpenAI's free-text safety-tier identifier (replacement for `user` on
   * safety-sensitive accounts). Accepted but intentionally not logged.
   */
  safety_identifier?: string;
  /**
   * Multi-LoRA: served-name of the adapter to apply on the worker (SIE
   * extension). Must be a non-empty string; unknown names are rejected by
   * the gateway with 400 `unknown_lora`.
   */
  lora_adapter?: string;
  /** SIE-native routing affinity hint. */
  routing_key?: string;
  /** SIE-native prompt-cache hint. */
  prompt_cache_key?: string;
}

/** Token usage block (snake_case, matches the wire shape). */
export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** A single choice in a `ChatCompletion` (non-streaming). */
export interface ChatChoice {
  index: number;
  message: ChatMessage;
  finish_reason: ChatFinishReason;
  logprobs: null;
}

/** Non-streaming response from `chatCompletions`. */
export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  system_fingerprint: string | null;
  choices: ChatChoice[];
  usage: ChatUsage;
  /** Request-scoped usage and settled debit, when supplied by the gateway. */
  request?: RequestMetadata;
}

/** Incremental delta emitted on each streaming chunk. */
export interface ChatDelta {
  /** First chunk only, per the OpenAI streaming contract. */
  role?: "assistant";
  content?: string;
  tool_calls?: ToolCallDelta[];
}

/** Partial tool-call materialised across multiple streaming chunks. */
export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

/** A single choice in a streaming `ChatCompletionChunk`. */
export interface ChatChunkChoice {
  index: number;
  delta: ChatDelta;
  finish_reason: ChatFinishReason;
  logprobs: null;
}

/**
 * One SSE event from `streamChatCompletions`.
 *
 * The terminal-usage chunk (emitted when `stream_options.include_usage` is
 * `true`) sets `choices: []` and populates `usage`.
 */
export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  system_fingerprint: string | null;
  choices: ChatChunkChoice[];
  usage?: ChatUsage;
}

/**
 * Per-call options for `chatCompletions` controlling the pre-execution
 * provisioning / retry loop. The request body itself is the separate
 * {@link ChatCompletionRequest} argument; these knobs only govern HOW the
 * SDK talks to the gateway, not WHAT it asks for.
 *
 * All fields are optional and fall back to the client-level defaults
 * (`waitForCapacity`, `provisionTimeout`) when omitted.
 */
export interface ChatCompletionOptions {
  /**
   * When `true`, retry the SAFE pre-execution capacity signals
   * (`503 PROVISIONING`, `503 MODEL_LOADING`, `503 RESOURCE_EXHAUSTED`)
   * until `provisionTimeoutMs` elapses. When `false`, the first
   * provisioning signal throws (`ProvisioningError` / `ModelLoadingError`
   * / `ServerError`). Defaults to the client's `waitForCapacity`
   * (true unless the constructor opted out).
   */
  waitForCapacity?: boolean;
  /**
   * Total cumulative wall-clock budget (ms) for provisioning retries.
   * Independent of the per-attempt `timeout`. Defaults to the client's
   * `provisionTimeout` (typically 5 minutes).
   */
  provisionTimeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Streaming generate — /v1/generate/{model} with stream:true
// ---------------------------------------------------------------------------

/**
 * One SSE event from `streamGenerate`.
 *
 * SIE-native shape — see `packages/sie_gateway/src/handlers/sse.rs`
 * (`build_generate_chunk_event`). `usage` and `ttft_ms` only land on the
 * terminal chunk; `error` is populated when generation failed mid-stream
 * (handled by throwing `SIEStreamError`, never yielded).
 */
export interface GenerateChunk {
  request_id: string;
  seq: number;
  text_delta: string;
  /** Per-token log probabilities aligned with `text_delta`. */
  logprobs?: Array<Record<string, unknown>>;
  done: boolean;
  finish_reason?: "stop" | "length" | "cancelled" | "error";
  usage?: ChatUsage;
  /** Time-to-first-token, milliseconds. Terminal chunk only. */
  ttft_ms?: number;
  /** Populated when the worker / gateway errored mid-stream. */
  error?: { code: string; message: string };
}

/**
 * Options for extract operation.
 */
export interface ExtractOptions {
  /** Entity labels to extract (e.g., ["person", "organization"]) */
  labels: string[];
  /** Minimum confidence threshold (0-1) */
  threshold?: number;
  /** GPU type for this request */
  gpu?: string;
  /** Whether to wait for capacity */
  waitForCapacity?: boolean;
  /**
   * Adapter-specific runtime options forwarded to the server as
   * `params.options`. Used for adapter knobs that aren't part of the
   * core extract API — e.g. `{ overflow_policy: "error" }` for
   * gliclass token-budget control. Mirrors the Python SDK's `options`
   * keyword argument.
   */
  adapterOptions?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Utility Types
// ---------------------------------------------------------------------------

/**
 * Helper to convert typed arrays to regular number array.
 * Useful for JSON serialization or working with libraries that expect number[].
 */
export function toNumberArray(arr: Float32Array | Int32Array): number[] {
  return Array.from(arr);
}

/**
 * Helper to convert number array to Float32Array.
 */
export function toFloat32Array(arr: number[]): Float32Array {
  return new Float32Array(arr);
}
