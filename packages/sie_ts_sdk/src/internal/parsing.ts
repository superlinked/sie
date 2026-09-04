/**
 * Parsing utilities for SIE responses
 */

import {
  AccountInactiveError,
  AccountStateUnavailableError,
  EstimateUnroutableError,
  IncompleteBatchError,
  InputTooLongError,
  InsufficientCreditsError,
  ModelLoadFailedError,
  ProvisioningError,
  RateLimitError,
  RequestError,
  ServerError,
  SpendLimitError,
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
  ACCOUNT_PENDING_REVIEW_ERROR_CODE,
  ACCOUNT_STATE_UNAVAILABLE_ERROR_CODE,
  ACCOUNT_SUSPENDED_ERROR_CODE,
  HTTP_CLIENT_ERROR_MAX,
  HTTP_CLIENT_ERROR_MIN,
  HTTP_FORBIDDEN,
  HTTP_PAYMENT_REQUIRED,
  HTTP_SERVER_ERROR_MAX,
  HTTP_SERVER_ERROR_MIN,
  HTTP_TOO_MANY_REQUESTS,
  INSUFFICIENT_CREDITS_ERROR_CODE,
  KEY_SPEND_LIMIT_EXCEEDED_ERROR_CODE,
  MSGPACK_CONTENT_TYPE,
  PROVISIONING_ERROR_CODE,
} from "./constants.js";

import { getRetryAfter as getRetryAfterFromHeader } from "./retry.js";

const SIE_ERROR_CODE_HEADER = "X-SIE-Error-Code";
const REQUEST_ID_HEADER = "x-sie-request-id";
const INVALID_ERROR_MESSAGE = "Request failed";

function normalizeErrorCode(code: string | undefined): string | undefined {
  if (code === "provisioning") return PROVISIONING_ERROR_CODE;
  return code;
}

/**
 * Read the gateway request id from a terminal response, applying the same
 * validation as `parseRequestMetadata` (non-empty visible ASCII, no
 * surrounding whitespace, bounded length). Returns `undefined` when absent
 * or malformed so errors never carry an attacker-shaped id (#3136).
 */
export function readRequestId(response: Response): string | undefined {
  return validateRequestId(response.headers.get(REQUEST_ID_HEADER));
}

/**
 * Validate one candidate request id (non-empty visible ASCII, no surrounding
 * whitespace, bounded length). Shared by the header path above and the
 * in-band stream error path so streamed ids obey the same rule and errors
 * never carry an attacker-shaped id (#3136).
 */
export function validateRequestId(value: unknown): string | undefined {
  if (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= 256 &&
    value === value.trim() &&
    /^[\x20-\x7e]+$/.test(value)
  ) {
    return value;
  }
  return undefined;
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

/** Preserve the OpenAI-compatible nullable/string `error.param` contract. */
function errorParamFromDetail(
  detail: Record<string, unknown> | undefined,
): string | null | undefined {
  const param = detail?.param;
  return typeof param === "string" || param === null ? param : undefined;
}

/** Extract a validated nullable/string error parameter from JSON or msgpack. */
export async function getErrorParam(response: Response): Promise<string | null | undefined> {
  return errorParamFromDetail(await getErrorDetail(response));
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
    param: errorParamFromDetail(detail),
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
  throw new InputTooLongError(message, { model, param: errorParamFromDetail(detail) });
}

/** The one exact path of the cost-estimate dry run (#2435). */
export const ESTIMATE_PATH = "/v1/estimate";

/**
 * Error codes the gateway answers when it cannot PRICE a request.
 *
 * Both are the live billing path's own "this request is not sellable" verdict.
 * A slab-capacity 503 (`BILLING_CAPACITY_UNAVAILABLE`) is a retryable gateway
 * condition and deliberately stays a plain {@link ServerError}.
 */
const ESTIMATE_UNROUTABLE_ERROR_CODES = new Set(["QUEUE_UNAVAILABLE"]);

/**
 * The `POST /v1/estimate` envelope: the target path plus its verbatim body.
 *
 * The gateway prices whatever `request` would have been sent to `endpoint`, so
 * the SDK does NOT reshape it. Anything normalized on the way in would be
 * something the quote priced and the request never sent.
 *
 * @throws {TypeError} If the envelope arguments are malformed — a caller
 *   mistake about WHAT is being priced, caught here rather than as a 400.
 */
export function buildEstimateEnvelope(
  endpoint: string,
  request: Record<string, unknown>,
): { endpoint: string; request: Record<string, unknown> } {
  if (typeof endpoint !== "string" || !endpoint.startsWith("/")) {
    throw new TypeError(`estimate endpoint must be the exact target path (got ${endpoint})`);
  }
  if (request === null || typeof request !== "object" || Array.isArray(request)) {
    throw new TypeError("estimate request must be the target request body object");
  }
  return { endpoint, request: { ...request } };
}

/**
 * Throw {@link EstimateUnroutableError} for an estimate the active rate book
 * cannot price. No-op for any other status / error code.
 *
 * Short-circuits {@link handleError}'s generic 5xx dispatch so callers can
 * catch "this request is not sellable" specifically, without string-matching a
 * code. The message is the planner's own reason.
 */
export async function throwIfEstimateUnroutable(response: Response): Promise<void> {
  if (response.status !== 503) return;
  // The SHARED code reader, so this follows the same `X-SIE-Error-Code`
  // header-over-body precedence (and the same normalization) as `handleError`
  // and the Python twin's `get_error_code`. Reading `detail.code` directly
  // would classify an edge-normalized code differently in one SDK than the
  // other. `detail` is kept only for the fallback message.
  const code = await getErrorCode(response.clone());
  if (code === undefined || !ESTIMATE_UNROUTABLE_ERROR_CODES.has(code)) return;
  const detail = await getErrorDetail(response.clone());
  const message =
    detail && typeof detail.message === "string"
      ? detail.message
      : "the active rate book cannot price this request";
  throw new EstimateUnroutableError(message, code, errorParamFromDetail(detail));
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
  let param: string | null | undefined;

  if (detail) {
    const c = detail.code;
    code = typeof c === "string" ? c : undefined;
    const m = detail.message;
    message = typeof m === "string" ? m : INVALID_ERROR_MESSAGE;
    param = errorParamFromDetail(detail);
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
      param = errorParamFromDetail(data);
    } catch {
      code = undefined;
      message = response.statusText;
      param = undefined;
    }
  }
  code = response.headers.get(SIE_ERROR_CODE_HEADER) ?? normalizeErrorCode(code);
  const requestId = readRequestId(response);

  if (status === 503 && code === PROVISIONING_ERROR_CODE) {
    const retryAfter = getRetryAfter(response);
    throw new ProvisioningError(message, gpu, retryAfter, param);
  }

  // Rate limit (pass-2 audit B1). Retried on the admission ladder in the
  // buffered loops; a give-up there throws RateLimitError directly. This arm
  // covers the terminal paths (streaming, listModels, estimate, …) so a 429 is
  // always a typed RateLimitError rather than a generic RequestError.
  if (status === HTTP_TOO_MANY_REQUESTS) {
    throw new RateLimitError(message, {
      retryAfter: getRetryAfter(response),
      code,
      requestId,
      param,
    });
  }

  // Terminal credit / account failures (pass-2 audit B3). 402/403 are NEVER
  // retried — they have no arm on any retry ladder, so they surface here on the
  // first response. Unrecognised 402/403 codes stay generic RequestError.
  if (status === HTTP_PAYMENT_REQUIRED) {
    if (code === INSUFFICIENT_CREDITS_ERROR_CODE)
      throw new InsufficientCreditsError(message, { requestId, param });
    if (code === KEY_SPEND_LIMIT_EXCEEDED_ERROR_CODE)
      throw new SpendLimitError(message, { requestId, param });
  }
  if (
    status === HTTP_FORBIDDEN &&
    (code === ACCOUNT_SUSPENDED_ERROR_CODE || code === ACCOUNT_PENDING_REVIEW_ERROR_CODE)
  ) {
    throw new AccountInactiveError(message, code, requestId, param);
  }
  if (status === 503 && code === ACCOUNT_STATE_UNAVAILABLE_ERROR_CODE) {
    throw new AccountStateUnavailableError(message, requestId, param);
  }

  if (status >= HTTP_CLIENT_ERROR_MIN && status <= HTTP_CLIENT_ERROR_MAX) {
    if (status === 400 && code === "INPUT_TOO_LONG") {
      // Fallback dispatch — ``model`` is only attached by the helper-style
      // short-circuit (``throwIfInputTooLong``) on the extract path.
      throw new InputTooLongError(message, { param });
    }
    throw new RequestError(message, code, status, requestId, param);
  }

  if (status >= HTTP_SERVER_ERROR_MIN && status <= HTTP_SERVER_ERROR_MAX) {
    throw new ServerError(message, code, status, requestId, param);
  }

  throw new ServerError(message, code, status, requestId, param);
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
 * Best-effort ids of submitted items absent from a shortened response.
 *
 * Only computed when ids identify every position on both sides: every
 * submitted item carries a string `id` and every returned item echoes one.
 * Otherwise the set difference could mislabel a present-but-unnamed item as
 * missing, so the diagnostic degrades to `undefined` (positional counts only).
 *
 * Runs only while building an error, so it degrades rather than throwing.
 */
function missingResultIds(
  results: readonly unknown[],
  submitted: readonly unknown[],
): string[] | undefined {
  const idOf = (value: unknown): string | undefined => {
    if (typeof value !== "object" || value === null) {
      return undefined;
    }
    const id = (value as { id?: unknown }).id;
    return typeof id === "string" ? id : undefined;
  };

  const submittedIds = submitted.map(idOf);
  if (submittedIds.some((id) => id === undefined)) {
    return undefined;
  }
  const returnedIds = new Set<string>();
  for (const result of results) {
    const id = idOf(result);
    if (id === undefined) {
      return undefined;
    }
    returnedIds.add(id);
  }
  const missing = submittedIds.filter(
    (id): id is string => id !== undefined && !returnedIds.has(id),
  );
  return missing.length > 0 ? missing : undefined;
}

/**
 * Guard the positional batch contract: exactly one result per input item.
 *
 * Encode and extract are positional — both return `results[0]` for a
 * single-item request, and batch callers reassemble results by index. The
 * contract breaks on an HTTP 200 whose `items` list is *shorter* than the
 * request: the gateway returns mixed-success batches as `200` carrying only
 * the successful items (a per-item server-side failure — an input exceeding
 * the model's `max_sequence_length`, say — is dropped from the body, not
 * surfaced as an error envelope). Without this check the short list flows into
 * positional access and silently misaligns every result after the drop.
 *
 * Mirrors the Python SDK's `validate_batch_result_count`, including its error
 * codes, so both SDKs fail identically.
 *
 * @throws {IncompleteBatchError} If the counts differ.
 */
export function validateBatchResultCount(
  results: readonly unknown[],
  submitted: readonly unknown[],
  model: string,
  operation: "encode" | "extract",
  requestId?: string,
): void {
  if (results.length === submitted.length) {
    return;
  }
  const [label, noun, code] =
    operation === "encode"
      ? ["Encode", "embedding(s)", "ENCODE_RESULT_COUNT_MISMATCH"]
      : ["Extract", "extraction result(s)", "EXTRACT_RESULT_COUNT_MISMATCH"];
  const missingIds = missingResultIds(results, submitted);
  let message = `${label} response desync for model ${JSON.stringify(model)}: server returned ${results.length} ${noun} for ${submitted.length} input item(s); expected exactly one per input. An input may have failed server-side (e.g. exceeding the model's max_sequence_length) and been dropped from the batch.`;
  if (missingIds !== undefined) {
    message += ` Missing item id(s): ${missingIds.join(", ")}.`;
  }
  throw new IncompleteBatchError(message, {
    expected: submitted.length,
    received: results.length,
    code,
    model,
    missingIds,
    requestId,
  });
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
/**
 * The settled charge (#2434) carried by one wire `usage` block, as camelCase
 * fields ready to spread into a parsed block.
 *
 * Both halves are required: a charge with no book version cannot be reconciled,
 * and a version with no charge describes nothing. Anything malformed yields an
 * empty object, so absence stays absence.
 */
export function settledChargeFields(usage: unknown): {
  creditsCharged?: number;
  rateBookVersion?: string;
} {
  if (typeof usage !== "object" || usage === null || Array.isArray(usage)) return {};
  const credits = (usage as Record<string, unknown>).credits_charged;
  const version = (usage as Record<string, unknown>).rate_book_version;
  if (typeof credits !== "number" || !Number.isSafeInteger(credits) || credits < 0) return {};
  if (typeof version !== "string" || version.length === 0) return {};
  return { creditsCharged: credits, rateBookVersion: version };
}

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
      // #2434: the gateway merges the settled charge into this same block, so
      // rebuilding it field-by-field must carry the charge across. Absence
      // stays absence — a request that committed no debit gets neither key.
      ...settledChargeFields(usage),
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
