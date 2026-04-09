/**
 * Types for the SIE TypeScript SDK
 *
 * These types mirror the Python SDK (packages/sie_sdk/src/sie_sdk/types.py)
 * for full feature parity.
 */

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
 * // Pre-encoded multivector (for use with maxsim utility)
 * { multivector: [tokenEmbedding1, tokenEmbedding2, ...] }
 */
export interface Item {
  /** Optional ID to track this item through results */
  id?: string;
  /** Text content to encode */
  text?: string;
  /** Images as byte arrays (JPEG/PNG) for multimodal models */
  images?: Uint8Array[];
  /** Pre-encoded multivector (for use with maxsim utility) */
  multivector?: Float32Array[];
  /** Arbitrary metadata (passed through to results) */
  metadata?: Record<string, unknown>;
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
 * Information about a model returned by listModels().
 */
export interface ModelInfo {
  /** Model name/identifier */
  name: string;
  /** Whether the model is currently loaded in memory */
  loaded: boolean;
  /** Supported input types: ["text"], ["text", "image"], etc. */
  inputs: string[];
  /** Supported output types: ["dense"], ["dense", "sparse"], etc. */
  outputs: string[];
  /** Embedding dimensions for each output type */
  dims?: ModelDims;
  /** Maximum sequence length the model supports */
  maxSequenceLength?: number;
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
}

/**
 * Information about a worker in the cluster.
 */
export interface WorkerInfo {
  /** Worker base URL */
  url: string;
  /** GPU type (e.g., "l4", "a100-80gb") */
  gpu: string;
  /** Whether the worker is healthy */
  healthy: boolean;
  /** Number of items in the worker's queue */
  queueDepth: number;
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
  /** GPU types configured in the cluster */
  configuredGpuTypes: string[];
  /** GPU types currently running */
  liveGpuTypes: string[];
  /** List of worker details */
  workers: WorkerInfo[];
}

/**
 * Pool specification for creating resource pools.
 */
export interface PoolSpec {
  /** Pool name (used in GPU param as "poolName/gpuType") */
  name: string;
  /** GPU requirements, e.g., { l4: 2, "a100-40gb": 1 } */
  gpus?: Record<string, number>;
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
  spec: { gpus?: Record<string, number> };
  /** Pool status */
  status: PoolStatus;
}

// ---------------------------------------------------------------------------
// WebSocket Status Types
// ---------------------------------------------------------------------------

export type ModelState = "available" | "loading" | "loaded" | "unloading";

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
  counters: Record<string, Record<string, number>>;
  histograms: Record<string, Record<string, Record<string, unknown>>>;
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
  /** Whether to auto-retry on 202 (provisioning) responses */
  waitForCapacity?: boolean;
  /** Maximum time to wait for provisioning in milliseconds (default: 300000) */
  provisionTimeout?: number;
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
