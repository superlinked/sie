/**
 * Parsing utilities for SIE responses
 */

import {
  InputTooLongError,
  ModelLoadFailedError,
  ProvisioningError,
  RequestError,
  ServerError,
} from "../errors.js";
import { unpackMessage } from "../msgpack.js";
import type {
  CapacityInfo,
  Classification,
  DetectedObject,
  EncodeResult,
  Entity,
  ExtractItemError,
  ExtractResult,
  FinishReason,
  GenerateResult,
  Relation,
  ScoreEntry,
  ScoreResult,
  WorkerInfo,
} from "../types.js";
import {
  HTTP_CLIENT_ERROR_MAX,
  HTTP_CLIENT_ERROR_MIN,
  HTTP_SERVER_ERROR_MAX,
  HTTP_SERVER_ERROR_MIN,
  MSGPACK_CONTENT_TYPE,
  PROVISIONING_ERROR_CODE,
} from "./constants.js";

import { getRetryAfter as getRetryAfterFromHeader } from "./retry.js";

const SIE_ERROR_CODE_HEADER = "X-SIE-Error-Code";

function normalizeErrorCode(code: string | undefined): string | undefined {
  if (code === "provisioning") return PROVISIONING_ERROR_CODE;
  return code;
}

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
 * Extract the error-detail object from a response body (JSON or msgpack).
 *
 * Returns the nested `error` / `detail` object so callers can read
 * auxiliary fields like `error_class`, `permanent`, `attempts` without
 * re-parsing. Used by the {@link throwIfModelLoadFailed} short-circuit.
 */
export async function getErrorDetail(
  response: Response,
): Promise<Record<string, unknown> | undefined> {
  try {
    const contentType = response.headers.get("content-type") ?? "";
    let data: Record<string, unknown>;

    if (contentType.includes(MSGPACK_CONTENT_TYPE)) {
      const buffer = await response.arrayBuffer();
      data = unpackMessage<Record<string, unknown>>(new Uint8Array(buffer));
    } else {
      data = (await response.json()) as Record<string, unknown>;
    }

    if (data.error && typeof data.error === "object") {
      return data.error as Record<string, unknown>;
    }
    if (data.detail && typeof data.detail === "object") {
      return data.detail as Record<string, unknown>;
    }
    if (typeof data.code === "string") {
      return data;
    }
  } catch {
    // Ignore parsing errors
  }
  return undefined;
}

/**
 * Extract error code from response body (handles both JSON and msgpack)
 */
export async function getErrorCode(response: Response): Promise<string | undefined> {
  const headerCode = response.headers.get(SIE_ERROR_CODE_HEADER);
  if (headerCode) return headerCode;

  const detail = await getErrorDetail(response);
  if (!detail) return undefined;
  const code = detail.code;
  return typeof code === "string" ? normalizeErrorCode(code) : undefined;
}

/**
 * Throw {@link ModelLoadFailedError} if the response is a 502 carrying
 * the `MODEL_LOAD_FAILED` error code.
 *
 * Used by the retry loop to short-circuit *before* engaging the
 * `MODEL_LOADING` budget. The server emits 502 + this code for
 * permanent-class failures (gated repos, missing dependencies); the SDK
 * must surface the error immediately rather than retrying for the full
 * provision timeout.
 *
 * No-op for any other status / error code.
 */
export async function throwIfModelLoadFailed(response: Response, model?: string): Promise<void> {
  if (response.status !== 502) return;
  const detail = await getErrorDetail(response.clone());
  if (!detail) return;
  if (detail.code !== "MODEL_LOAD_FAILED") return;
  const errorClass = typeof detail.error_class === "string" ? detail.error_class : undefined;
  const permanent = typeof detail.permanent === "boolean" ? detail.permanent : true;
  // Defensive: server should always send an int >= 1, but a malformed
  // payload must not crash the retry loop. Use ``Number.isFinite`` so
  // ``NaN`` (from a non-numeric string) and infinities both fall back
  // to 1, and a legitimate 0 (if the server semantics ever change) is
  // preserved instead of being clobbered by ``|| 1``.
  const attemptsRaw = detail.attempts;
  const parsedAttempts =
    typeof attemptsRaw === "number"
      ? attemptsRaw
      : typeof attemptsRaw === "string"
        ? Number.parseInt(attemptsRaw, 10)
        : Number.NaN;
  const attempts = Number.isFinite(parsedAttempts) ? parsedAttempts : 1;
  const message =
    typeof detail.message === "string" ? detail.message : `Model '${model ?? "?"}' failed to load`;
  throw new ModelLoadFailedError(message, {
    model,
    errorClass,
    permanent,
    attempts,
  });
}

/**
 * Throw {@link InputTooLongError} if the response is a 400 carrying the
 * `INPUT_TOO_LONG` error code.
 *
 * Used by the extract path to surface token-budget overruns as a typed
 * exception (so callers can catch {@link InputTooLongError} specifically)
 * instead of relying on a generic {@link RequestError} + string-matching
 * the `code`.
 *
 * No-op for any other status / error code.
 */
export async function throwIfInputTooLong(response: Response, model?: string): Promise<void> {
  if (response.status !== 400) return;
  const detail = await getErrorDetail(response.clone());
  if (!detail) return;
  if (detail.code !== "INPUT_TOO_LONG") return;
  const message =
    typeof detail.message === "string"
      ? detail.message
      : "Input exceeds the model's maximum token capacity";
  throw new InputTooLongError(message, { model });
}

/**
 * Handle HTTP error response and throw appropriate error
 */
export async function handleError(response: Response, gpu?: string): Promise<never> {
  const { status } = response;

  // Prefer nested ``error`` / ``detail`` objects (gateway + FastAPI dict detail),
  // same as Python ``handle_error``. Legacy: string ``detail``, or top-level
  // ``message``.
  const detail = await getErrorDetail(response.clone());

  let code: string | undefined;
  let message: string;

  if (detail) {
    const c = detail.code;
    code = typeof c === "string" ? c : undefined;
    const m = detail.message;
    message = typeof m === "string" ? m : JSON.stringify(detail);
  } else {
    try {
      const data = (await response.json()) as Record<string, unknown>;
      if (typeof data.detail === "string") {
        code = typeof data.code === "string" ? data.code : undefined;
        message = data.detail;
      } else if (typeof data.message === "string") {
        code = typeof data.code === "string" ? data.code : undefined;
        message = data.message;
      } else {
        code = typeof data.code === "string" ? data.code : undefined;
        message = response.statusText;
      }
    } catch {
      code = undefined;
      message = response.statusText;
    }
  }
  code = response.headers.get(SIE_ERROR_CODE_HEADER) ?? normalizeErrorCode(code);

  if (status === 503 && code === PROVISIONING_ERROR_CODE) {
    const retryAfter = getRetryAfter(response);
    throw new ProvisioningError(message, gpu, retryAfter);
  }

  if (status >= HTTP_CLIENT_ERROR_MIN && status <= HTTP_CLIENT_ERROR_MAX) {
    if (status === 400 && code === "INPUT_TOO_LONG") {
      // Fallback dispatch — ``model`` is only attached by the helper-style
      // short-circuit (``throwIfInputTooLong``) on the extract path.
      throw new InputTooLongError(message);
    }
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

interface WireScoreUsage {
  input_tokens: number;
  images?: number;
}

interface WireScoreResult {
  model?: string;
  query_id?: string;
  scores: WireScoreEntry[];
  usage?: WireScoreUsage;
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
  data?: Record<string, unknown>;
  error?: { code: string; message: string };
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
    usage: wire.usage
      ? {
          inputTokens: wire.usage.input_tokens,
          images: wire.usage.images,
        }
      : undefined,
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

function parseExtractItemError(data: unknown): ExtractItemError | undefined {
  if (data === undefined || data === null) return undefined;
  if (typeof data === "object" && !Array.isArray(data)) {
    const error = data as Record<string, unknown>;
    if (
      typeof error.code === "string" &&
      error.code.trim().length > 0 &&
      typeof error.message === "string" &&
      error.message.trim().length > 0
    ) {
      return { code: error.code, message: error.message };
    }
  }
  return {
    code: "INTERNAL_ERROR",
    message: "Malformed extraction item error",
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
    data: data.data,
    error: parseExtractItemError(data.error),
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

interface WireUsageBlock {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

interface WireGenerateResult {
  model?: string;
  text?: string;
  finish_reason?: string;
  usage?: WireUsageBlock;
  attempt_id?: string;
  ttft_ms?: number;
  tpot_ms?: number;
}

/**
 * Describe an unknown value's runtime type for error messages, mirroring
 * the granularity of Python's ``type(x).__name__`` (``typeof null`` is
 * ``"object"`` in JS, so disambiguate ``null`` explicitly).
 */
function describeType(value: unknown): string {
  if (value === null) return "null";
  return typeof value;
}

/**
 * Parse the gateway's streaming generate response envelope into a
 * :class:`GenerateResult`. Tolerant of missing *optional* fields for
 * forward compat with future surface extensions.
 *
 * ``model`` and ``text`` are required strings: a missing or non-string
 * value is surfaced as a {@link RequestError} rather than being silently
 * coerced to an empty string. A truncated / malformed envelope must not
 * look like a legitimate empty completion (silent data loss). This mirrors
 * the Python SDK's ``_parse_generate_result`` contract.
 */
/**
 * Coerce a wire-format token count into a safe non-negative integer.
 *
 * The wire `usage` envelope is untyped JSON, so a malformed payload can carry
 * a string (`"5"`), a float (`3.9`), `null`, or a missing field. The previous
 * `?? 0` only guarded null/undefined, letting strings/floats leak verbatim
 * into the SDK's `number`-typed fields. We mirror the Python SDK's int
 * coercion: keep only finite numbers and truncate toward zero; everything
 * else (string, NaN, Infinity, null) becomes `0`.
 */
function coerceTokenCount(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.trunc(v) : 0;
}

export function parseGenerateResult(data: Record<string, unknown>): GenerateResult {
  const wire = data as WireGenerateResult;
  if (typeof wire.model !== "string") {
    throw new RequestError(
      `Generate response missing string 'model' field: got ${describeType(wire.model)}`,
    );
  }
  if (typeof wire.text !== "string") {
    throw new RequestError(
      `Generate response missing string 'text' field: got ${describeType(wire.text)}`,
    );
  }
  const usage = wire.usage ?? {};
  const finish = (wire.finish_reason ?? "stop") as FinishReason;
  return {
    model: wire.model,
    text: wire.text,
    finishReason: finish,
    usage: {
      promptTokens: coerceTokenCount(usage.prompt_tokens),
      completionTokens: coerceTokenCount(usage.completion_tokens),
      totalTokens: coerceTokenCount(usage.total_tokens),
    },
    attemptId: wire.attempt_id,
    ttftMs: wire.ttft_ms,
    tpotMs: wire.tpot_ms,
  };
}

// Wire format types for capacity
interface WireWorkerInfo {
  url: string;
  gpu: string;
  gpu_count?: number;
  ready_gpu_slots?: number;
  healthy: boolean;
  queue_depth: number;
  pending_cost?: number;
  inflight_batches?: number;
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
    gpuCount: w.gpu_count ?? 0,
    readyGpuSlots: w.ready_gpu_slots ?? w.gpu_count ?? (w.healthy ? 1 : 0),
    healthy: w.healthy,
    queueDepth: w.queue_depth,
    pendingCost: w.pending_cost ?? 0,
    inflightBatches: w.inflight_batches ?? 0,
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
