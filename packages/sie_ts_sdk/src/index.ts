/**
 * @superlinked/sie-sdk - Official TypeScript SDK for SIE (Search Inference Engine)
 *
 * @example
 * ```typescript
 * import { SIEClient } from "@superlinked/sie-sdk";
 *
 * const client = new SIEClient("http://localhost:8080");
 *
 * // Encode text to get embeddings
 * const result = await client.encode("bge-m3", { text: "Hello world" });
 * console.log(result.dense); // Float32Array
 *
 * // Batch encode
 * const results = await client.encode("bge-m3", [
 *   { text: "First document" },
 *   { text: "Second document" },
 * ]);
 * ```
 */

// Main client
export { SIEClient } from "./client.js";
export type {
  JobsNamespace,
  ConnectionsNamespace,
  FilesNamespace,
  BatchesNamespace,
  FileUploadInput,
} from "./client.js";

// Jobs surface (client.jobs) — pure helpers + wire types
export { buildJobBody, connectionName, TERMINAL_JOB_STATES } from "./jobs.js";
export type {
  JobState,
  JobItem,
  JobSource,
  JobFieldMap,
  SubmitJobOptions,
  JobPreflight,
  JobChunk,
  JobSubmitResult,
  JobStatus,
  JobResultItem,
  JobResults,
} from "./jobs.js";

// Version
export { SDK_VERSION } from "./version.js";

// Types
export type {
  // Core types
  AudioInput,
  ExtractItem,
  Item,
  SparseResult,
  TimingInfo,
  EncodeResult,
  ScoreEntry,
  ScoreUsage,
  ScoreResult,
  Entity,
  Relation,
  Classification,
  DetectedObject,
  ExtractItemError,
  ExtractResult,
  // Model and cluster info
  ModelCapabilities,
  ModelDims,
  ModelInfo,
  ModelState,
  WorkerInfo,
  CapacityInfo,
  ClusterSummary,
  ClusterWorkerInfo,
  ModelSummary,
  ServerInfo,
  GPUMetrics,
  ModelConfig,
  ModelStatus,
  WorkerStatusMessage,
  ClusterStatusMessage,
  StatusMessage,
  // Pool types
  CreatePoolOptions,
  PoolSpec,
  PoolStatus,
  PoolInfo,
  // Connections (client.connections)
  Connection,
  ConnectionCreated,
  ConnectionRevoked,
  // Files (client.files) — OpenAI-compatible file objects
  File as SIEFile,
  FileDeleted,
  // Batches (client.batches) — OpenAI-compatible batch objects
  Batch,
  BatchList,
  BatchRequestCounts,
  // Options
  DType,
  OutputType,
  SIEClientOptions,
  EncodeOptions,
  ScoreOptions,
  ExtractOptions,
  // Generation
  FinishReason,
  GenerationUsage,
  JsonSchemaGrammar,
  RegexGrammar,
  EbnfGrammar,
  GenerateGrammar,
  GenerateOptions,
  StreamGenerateOptions,
  GenerateResult,
  // Streaming generate (SSE)
  GenerateChunk,
  // Chat completions (OpenAI-compatible)
  ChatMessage,
  ToolCall,
  ToolSpec,
  ToolCallDelta,
  ToolChoice,
  ResponseFormat,
  ChatFinishReason,
  ChatCompletionRequest,
  ChatCompletion,
  ChatChoice,
  ChatUsage,
  ChatCompletionChunk,
  ChatChunkChoice,
  ChatDelta,
} from "./types.js";

// Utility functions
export { toNumberArray, toFloat32Array } from "./types.js";

// Encoding result helpers (for integrations)
export {
  denseEmbedding,
  sparseEmbedding,
  sparseEmbeddingMap,
  normalizeSparseVector,
  multivectorEmbedding,
  type SparseVector,
} from "./encoding.js";

// Errors
export {
  SIEError,
  SIEConnectionError,
  RequestError,
  ServerError,
  ProvisioningError,
  PoolError,
  LoraLoadingError,
  ModelLoadingError,
  ModelLoadFailedError,
  ResourceExhaustedError,
  SIEStreamError,
  InputTooLongError,
} from "./errors.js";

// Client-side scoring (MaxSim for ColBERT-style models)
export { maxsim, maxsimDocuments, maxsimBatch } from "./scoring.js";

// Low-level utilities (for advanced users)
export { packMessage, unpackMessage } from "./msgpack.js";
export {
  toImageBytes,
  toImageWireFormat,
  detectImageFormat,
  type ImageInput,
  type ImageWireFormat,
} from "./images.js";
