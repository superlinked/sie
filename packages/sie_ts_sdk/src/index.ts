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

// Version
export { SDK_VERSION } from "./version.js";

// Types
export type {
  // Core types
  Item,
  SparseResult,
  TimingInfo,
  EncodeResult,
  ScoreEntry,
  ScoreResult,
  Entity,
  Relation,
  Classification,
  DetectedObject,
  ExtractResult,
  // Model and cluster info
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
  PoolSpec,
  PoolStatus,
  PoolInfo,
  // Options
  DType,
  OutputType,
  SIEClientOptions,
  EncodeOptions,
  ScoreOptions,
  ExtractOptions,
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
