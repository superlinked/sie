/**
 * Parsing utilities for SIE responses
 */

import { ProvisioningError, RequestError, ServerError } from "../errors.js";
import { unpackMessage } from "../msgpack.js";
import type {
  CapacityInfo,
  Classification,
  DetectedObject,
  EncodeResult,
  Entity,
  ExtractResult,
  Relation,
  ScoreEntry,
  ScoreResult,
  WorkerInfo,
} from "../types.js";
import {
  HTTP_ACCEPTED,
  HTTP_CLIENT_ERROR_MAX,
  HTTP_CLIENT_ERROR_MIN,
  HTTP_SERVER_ERROR_MAX,
  HTTP_SERVER_ERROR_MIN,
  MSGPACK_CONTENT_TYPE,
} from "./constants.js";

import { getRetryAfter as getRetryAfterFromHeader } from "./retry.js";

/**
 * Parse GPU parameter from "pool/gpu" format
 */
export function parseGpuParam(param: string): { pool?: string; gpu: string } {
  const parts = param.split("/");
  if (parts.length === 2 && parts[0] !== undefined && parts[1] !== undefined) {
    return { pool: parts[0], gpu: parts[1] };
  }
  return { gpu: param };
}

/**
 * Extract Retry-After header value from Response in milliseconds
 */
export function getRetryAfter(response: Response): number | undefined {
  const header = response.headers.get("Retry-After");
  return getRetryAfterFromHeader(header);
}

/**
 * Extract error code from response body (handles both JSON and msgpack)
 */
export async function getErrorCode(response: Response): Promise<string | undefined> {
  try {
    const contentType = response.headers.get("content-type") ?? "";
    let data: Record<string, unknown>;

    if (contentType.includes(MSGPACK_CONTENT_TYPE)) {
      const buffer = await response.arrayBuffer();
      data = unpackMessage<Record<string, unknown>>(new Uint8Array(buffer));
    } else {
      data = (await response.json()) as Record<string, unknown>;
    }

    // Check error.code pattern
    if (data.error && typeof data.error === "object") {
      const error = data.error as Record<string, unknown>;
      if (typeof error.code === "string") {
        return error.code;
      }
    }

    // Check detail.code pattern
    if (data.detail && typeof data.detail === "object") {
      const detail = data.detail as Record<string, unknown>;
      if (typeof detail.code === "string") {
        return detail.code;
      }
    }

    // Check top-level code
    if (typeof data.code === "string") {
      return data.code;
    }
  } catch {
    // Ignore parsing errors
  }
  return undefined;
}

/**
 * Handle HTTP error response and throw appropriate error
 */
export async function handleError(response: Response, gpu?: string): Promise<never> {
  const { status } = response;

  let errorBody: { code?: string; detail?: string } = {};
  try {
    errorBody = (await response.json()) as { code?: string; detail?: string };
  } catch {
    // Ignore JSON parsing errors
  }

  const code = errorBody.code ?? "UNKNOWN";
  const message = errorBody.detail ?? response.statusText;

  if (status === HTTP_ACCEPTED) {
    const retryAfter = response.headers.get("Retry-After");
    throw new ProvisioningError(
      message,
      gpu,
      retryAfter ? Number.parseInt(retryAfter, 10) * 1000 : undefined,
    );
  }

  if (status >= HTTP_CLIENT_ERROR_MIN && status <= HTTP_CLIENT_ERROR_MAX) {
    throw new RequestError(message, code, status);
  }

  if (status >= HTTP_SERVER_ERROR_MIN && status <= HTTP_SERVER_ERROR_MAX) {
    throw new ServerError(message, code, status);
  }

  throw new ServerError(message, code, status);
}

// Wire format types (what server sends)
// The server wraps arrays in objects like: {"dense": {"values": Float32Array}}
interface WireDenseResult {
  values: Float32Array;
}

interface WireSparseResult {
  indices: Int32Array;
  values: Float32Array;
}

interface WireMultivectorResult {
  values: Float32Array[]; // Actually an array of Float32Arrays for each token
}

interface WireEncodeResult {
  id?: string;
  dense?: WireDenseResult; // Nested: {"values": Float32Array}
  sparse?: WireSparseResult;
  multivector?: WireMultivectorResult; // Nested: {"values": Float32Array[]}
  timing?: {
    total_ms?: number;
    queue_ms?: number;
    tokenization_ms?: number;
    inference_ms?: number;
  };
}

interface WireScoreEntry {
  item_id: string;
  score: number;
  rank: number;
}

interface WireScoreResult {
  model?: string;
  query_id?: string;
  scores: WireScoreEntry[];
}

interface WireEntity {
  text: string;
  label: string;
  score: number;
  start?: number;
  end?: number;
  bbox?: number[];
}

interface WireRelation {
  head: string;
  tail: string;
  relation: string;
  score: number;
}

interface WireClassification {
  label: string;
  score: number;
}

interface WireDetectedObject {
  label: string;
  score: number;
  bbox: number[];
}

interface WireExtractResult {
  id?: string;
  entities: WireEntity[];
  relations?: WireRelation[];
  classifications?: WireClassification[];
  objects?: WireDetectedObject[];
}

/**
 * Parse wire format to EncodeResult
 *
 * Wire format from server uses nested objects:
 * - dense: {"values": Float32Array}
 * - sparse: {"indices": Int32Array, "values": Float32Array}
 * - multivector: {"values": Float32Array[]}
 */
export function parseEncodeResult(data: WireEncodeResult): EncodeResult {
  const result: EncodeResult = {};

  if (data.id !== undefined) {
    result.id = data.id;
  }

  // Dense is nested: {"values": Float32Array}
  if (data.dense) {
    result.dense = data.dense.values;
  }

  // Sparse is already flat: {"indices": Int32Array, "values": Float32Array}
  if (data.sparse) {
    result.sparse = {
      indices: data.sparse.indices,
      values: data.sparse.values,
    };
  }

  // Multivector is nested: {"values": Float32Array[]}
  if (data.multivector) {
    result.multivector = data.multivector.values;
  }

  if (data.timing) {
    result.timing = {
      totalMs: data.timing.total_ms,
      queueMs: data.timing.queue_ms,
      tokenizationMs: data.timing.tokenization_ms,
      inferenceMs: data.timing.inference_ms,
    };
  }

  return result;
}

/**
 * Parse wire format to EncodeResult[]
 *
 * Accepts unknown[] from msgpack deserialization and casts to WireEncodeResult[].
 */
export function parseEncodeResults(data: unknown[]): EncodeResult[] {
  return (data as WireEncodeResult[]).map(parseEncodeResult);
}

/**
 * Parse wire format to ScoreEntry
 */
function parseScoreEntry(data: WireScoreEntry): ScoreEntry {
  return {
    itemId: data.item_id,
    score: data.score,
    rank: data.rank,
  };
}

/**
 * Parse wire format to ScoreResult
 *
 * Accepts unknown from msgpack deserialization and casts to WireScoreResult.
 */
export function parseScoreResult(data: unknown): ScoreResult {
  const wire = data as WireScoreResult;
  return {
    model: wire.model,
    queryId: wire.query_id,
    scores: wire.scores.map(parseScoreEntry),
  };
}

/**
 * Parse wire format to Entity
 */
function parseEntity(data: WireEntity): Entity {
  return {
    text: data.text,
    label: data.label,
    score: data.score,
    start: data.start,
    end: data.end,
    bbox: data.bbox,
  };
}

/**
 * Parse wire format to ExtractResult
 */
export function parseExtractResult(data: WireExtractResult): ExtractResult {
  return {
    id: data.id,
    entities: data.entities.map(parseEntity),
    relations: (data.relations ?? []).map(
      (r: WireRelation): Relation => ({
        head: r.head,
        tail: r.tail,
        relation: r.relation,
        score: r.score,
      }),
    ),
    classifications: (data.classifications ?? []).map(
      (c: WireClassification): Classification => ({
        label: c.label,
        score: c.score,
      }),
    ),
    objects: (data.objects ?? []).map(
      (o: WireDetectedObject): DetectedObject => ({
        label: o.label,
        score: o.score,
        bbox: o.bbox,
      }),
    ),
  };
}

/**
 * Parse wire format to ExtractResult[]
 *
 * Accepts unknown[] from msgpack deserialization and casts to WireExtractResult[].
 */
export function parseExtractResults(data: unknown[]): ExtractResult[] {
  return (data as WireExtractResult[]).map(parseExtractResult);
}

// Wire format types for capacity
interface WireWorkerInfo {
  url: string;
  gpu: string;
  healthy: boolean;
  queue_depth: number;
  loaded_models: string[];
}

interface WireCapacityResponse {
  status: string;
  type?: string;
  cluster?: {
    worker_count?: number;
    gpu_count?: number;
    models_loaded?: number;
  };
  configured_gpu_types?: string[];
  live_gpu_types?: string[];
  workers?: WireWorkerInfo[];
}

/**
 * Parse wire format to CapacityInfo
 */
export function parseCapacityInfo(data: unknown, gpuFilter?: string): CapacityInfo {
  const wire = data as WireCapacityResponse;

  // Filter workers by GPU if specified
  let workers = wire.workers ?? [];
  if (gpuFilter) {
    const gpuLower = gpuFilter.toLowerCase();
    workers = workers.filter((w) => w.gpu.toLowerCase() === gpuLower);
  }

  const parsedWorkers: WorkerInfo[] = workers.map((w) => ({
    url: w.url,
    gpu: w.gpu,
    healthy: w.healthy,
    queueDepth: w.queue_depth,
    loadedModels: w.loaded_models,
  }));

  return {
    status: wire.status,
    workerCount: gpuFilter ? parsedWorkers.length : (wire.cluster?.worker_count ?? 0),
    gpuCount: wire.cluster?.gpu_count ?? 0,
    modelsLoaded: wire.cluster?.models_loaded ?? 0,
    configuredGpuTypes: wire.configured_gpu_types ?? [],
    liveGpuTypes: wire.live_gpu_types ?? [],
    workers: parsedWorkers,
  };
}
