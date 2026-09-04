/**
 * Error classes for the SIE TypeScript SDK.
 *
 * These errors mirror the Python SDK (packages/sie_sdk/src/sie_sdk/client/errors.py)
 * for consistent error handling across languages.
 *
 * @example
 * // Catching specific error types
 * try {
 *   await client.encode("model", { text: "hello" });
 * } catch (error) {
 *   if (error instanceof RequestError) {
 *     console.error(`Bad request (${error.code}): ${error.message}`);
 *   } else if (error instanceof ProvisioningError) {
 *     console.log(`GPU ${error.gpu} is provisioning, retry after ${error.retryAfter}ms`);
 *   } else if (error instanceof SIEConnectionError) {
 *     console.error("Cannot reach server:", error.message);
 *   }
 * }
 */

/**
 * Base error for all SIE SDK errors.
 *
 * All SIE errors extend this class, so you can catch all SDK errors with:
 * `catch (error) { if (error instanceof SIEError) { ... } }`
 */
export class SIEError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SIEError";
    // Maintain proper prototype chain for instanceof checks
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * `SIEConnectionError` failure category. Only `"connect"` is auto-retried
 * under `waitForCapacity: true`; `"timeout"` and `"other"` fail fast.
 */
export type SIEConnectionErrorKind = "connect" | "timeout" | "other";

/**
 * Error connecting to the SIE server.
 *
 * Raised when:
 * - Network is unreachable
 * - DNS resolution fails
 * - Connection times out
 * - Server refuses connection
 */
export class SIEConnectionError extends SIEError {
  readonly kind: SIEConnectionErrorKind;

  constructor(message: string, kind: SIEConnectionErrorKind = "other") {
    super(message);
    this.name = "SIEConnectionError";
    this.kind = kind;
  }
}

/**
 * Terminal request or response-contract error.
 *
 * Raised for 4xx responses, malformed successful response bodies, and other
 * terminal response-shape violations. Common 4xx cases include:
 * - 400: Bad request (invalid parameters, malformed body)
 * - 401: Unauthorized (missing or invalid API key)
 * - 403: Forbidden (insufficient permissions)
 * - 404: Not found (invalid endpoint or model)
 * - 422: Validation error (invalid input format)
 */
export class RequestError extends SIEError {
  /** Error code from the server (e.g., "INVALID_MODEL", "VALIDATION_ERROR") */
  readonly code: string | undefined;
  /** HTTP status code, when the failure came from a terminal response. */
  readonly statusCode: number | undefined;
  /**
   * Gateway request id (`x-sie-request-id` header) when the terminal
   * response carried one. Mirrors the Python SDK's `RequestError.request.id`
   * so failures stay correlatable with gateway logs (#3136).
   */
  readonly requestId: string | undefined;
  /** Offending request field from the server error envelope, when known. */
  readonly param: string | null | undefined;

  constructor(
    message: string,
    code?: string,
    statusCode?: number,
    requestId?: string,
    param?: string | null,
  ) {
    super(message);
    this.name = "RequestError";
    this.code = code;
    this.statusCode = statusCode;
    this.requestId = requestId;
    this.param = param;
  }
}

/**
 * Error from the server (5xx responses).
 *
 * Raised when the server encounters an internal error:
 * - 500: Internal server error
 * - 502: Bad gateway
 * - 503: Service unavailable
 * - 504: Gateway timeout
 */
export class ServerError extends SIEError {
  /** Error code from the server (e.g., "INTERNAL_ERROR", "LORA_LOADING") */
  readonly code: string | undefined;
  /** HTTP status code (500-599) */
  readonly statusCode: number | undefined;
  /**
   * Gateway request id (`x-sie-request-id` header) when the terminal
   * response carried one. Mirrors the Python SDK's `ServerError.request.id`
   * so failures stay correlatable with gateway logs (#3136).
   */
  readonly requestId: string | undefined;
  /** Offending request field from the server error envelope, when known. */
  readonly param: string | null | undefined;

  constructor(
    message: string,
    code?: string,
    statusCode?: number,
    requestId?: string,
    param?: string | null,
  ) {
    super(message);
    this.name = "ServerError";
    this.code = code;
    this.statusCode = statusCode;
    this.requestId = requestId;
    this.param = param;
  }
}

/**
 * Error when capacity is not available and provisioning timed out.
 *
 * Raised when:
 * - Server returns 503 with PROVISIONING code
 * - waitForCapacity is false (caller doesn't want to wait)
 * - Or provisioning timeout exceeded
 *
 * The caller can use `retryAfter` to know when to retry.
 */
export class ProvisioningError extends SIEError {
  /** The GPU type that was requested */
  readonly gpu: string | undefined;
  /** Suggested retry delay in milliseconds (from server Retry-After header) */
  readonly retryAfter: number | undefined;
  /** Offending request field from the server error envelope, when known. */
  readonly param: string | null | undefined;

  constructor(message: string, gpu?: string, retryAfter?: number, param?: string | null) {
    super(message);
    this.name = "ProvisioningError";
    this.gpu = gpu;
    this.retryAfter = retryAfter;
    this.param = param;
  }
}

/**
 * Error related to resource pool operations.
 *
 * Raised when:
 * - Pool creation fails (e.g., insufficient capacity)
 * - Pool not found
 * - Pool in invalid state (e.g., expired)
 * - Pool lease renewal fails
 */
export class PoolError extends SIEError {
  /** Name of the pool */
  readonly poolName: string | undefined;
  /** Current pool state (if known): "pending", "active", "expired" */
  readonly state: string | undefined;

  constructor(message: string, poolName?: string, state?: string) {
    super(message);
    this.name = "PoolError";
    this.poolName = poolName;
    this.state = state;
  }
}

/**
 * Error when LoRA adapter is loading and retry limit exceeded.
 *
 * Raised when:
 * - Server returns 503 with LORA_LOADING code
 * - Retry limit is exceeded
 *
 * This usually means the adapter is being loaded from disk/network
 * and the caller should wait longer or reduce request rate.
 */
export class LoraLoadingError extends SIEError {
  /** The LoRA adapter that was requested */
  readonly lora: string | undefined;
  /** The model the LoRA was requested for */
  readonly model: string | undefined;
  /** Offending request field from the server error envelope, when known. */
  readonly param: string | null | undefined;

  constructor(message: string, lora?: string, model?: string, param?: string | null) {
    super(message);
    this.name = "LoraLoadingError";
    this.lora = lora;
    this.model = model;
    this.param = param;
  }
}

/**
 * Error when model is loading and retry limit exceeded.
 *
 * Raised when:
 * - Server returns 503 with MODEL_LOADING code
 * - Retry limit is exceeded
 *
 * This usually means the model is being loaded from disk/HuggingFace
 * and the caller should wait longer.
 */
export class ModelLoadingError extends SIEError {
  /** The model that was requested */
  readonly model: string | undefined;
  /** Offending request field from the server error envelope, when known. */
  readonly param: string | null | undefined;

  constructor(message: string, model?: string, param?: string | null) {
    super(message);
    this.name = "ModelLoadingError";
    this.model = model;
    this.param = param;
  }
}

/**
 * Error when the server has exhausted its OOM-recovery strategies.
 *
 * Raised when:
 * - Server returns 503 with RESOURCE_EXHAUSTED code
 * - SDK retry limit is exceeded (or the next backoff would exhaust the
 *   provision-timeout budget)
 *
 * Subclass of {@link ServerError} so callers that already catch
 * `ServerError` continue to behave correctly; new code can catch
 * `ResourceExhaustedError` specifically to react to sustained GPU
 * pressure (e.g., back off, route elsewhere, scale up). Mirrors the
 * Python SDK's `ResourceExhaustedError`.
 */
export class ResourceExhaustedError extends ServerError {
  /** The model that was requested */
  readonly model: string | undefined;
  /** Number of retry attempts made before giving up */
  readonly retries: number;

  constructor(
    message: string,
    options?: { model?: string; retries?: number; param?: string | null },
  ) {
    super(message, "RESOURCE_EXHAUSTED", 503, undefined, options?.param);
    this.name = "ResourceExhaustedError";
    this.model = options?.model;
    this.retries = options?.retries ?? 0;
  }
}

/**
 * Error surfaced mid-stream from `streamChatCompletions` / `streamGenerate`.
 *
 * The SSE wire shape includes optional
 * `error: {message, type, param, code, retry_after_s?}` on the terminal chunk.
 * When the SDK sees such a chunk it does NOT yield the chunk; instead
 * it throws `SIEStreamError`, mirroring the non-streaming `handleError` path
 * so callers can catch the same way they would for HTTP-level failures.
 *
 * Compare with `RequestError` / `ServerError`: those fire before the SSE
 * stream opens (HTTP 4xx / 5xx). `SIEStreamError` fires after at least one
 * byte has gone out — the connection itself was healthy, but the worker /
 * gateway emitted an error envelope partway through generation.
 */
export class SIEStreamError extends SIEError {
  /** SIE-native error code (e.g. `context_exceeded`, `empty_model_output`, `cancelled`). */
  readonly code: string | undefined;
  /** OpenAI-style error type (e.g. `context_length_exceeded`, `server_error`). */
  readonly errorType: string | undefined;
  /** Offending field name when known. */
  readonly param: string | null | undefined;
  /** Validated RESOURCE_EXHAUSTED retry hint, in milliseconds. */
  readonly retryAfter: number | undefined;
  /**
   * Gateway request id carried in-band by the error chunk (SIE-native
   * generate shape only — streamed responses have no terminal headers).
   * Lets callers correlate typed terminal errors like `empty_model_output`
   * with gateway logs (#3136).
   */
  readonly requestId: string | undefined;

  constructor(
    message: string,
    options?: {
      code?: string;
      errorType?: string;
      param?: string | null;
      requestId?: string;
      retryAfter?: number;
    },
  ) {
    super(message);
    this.name = "SIEStreamError";
    this.code = options?.code;
    this.errorType = options?.errorType;
    this.param = options?.param;
    this.requestId = options?.requestId;
    this.retryAfter = options?.retryAfter;
  }
}

/**
 * Error when the server reports a *terminal* model-load failure.
 *
 * Distinct from {@link ModelLoadingError} — this is thrown on the first
 * response (no retry budget consumed) when the server returns HTTP
 * `502 MODEL_LOAD_FAILED`. The server uses this code for permanent-class
 * failures (gated repos, missing dependencies, unrecognised model
 * architectures) where retrying would waste time.
 *
 * Permanent failures will not auto-clear; an operator must fix the
 * underlying cause (e.g. set `HF_TOKEN`, accept the model license on
 * HuggingFace, upgrade `transformers`).
 */
export class ModelLoadFailedError extends ServerError {
  /** The model that was requested */
  readonly model: string | undefined;
  /**
   * Server-side classification: one of `GATED`, `OOM`, `DEPENDENCY`,
   * `NOT_FOUND`, `NETWORK`, `UNKNOWN`. Use this to route to specific
   * remediation paths (e.g. surface a "set HF_TOKEN" hint for `GATED`).
   */
  readonly errorClass: string | undefined;
  /** Whether the failure is non-retryable per server policy. */
  readonly permanent: boolean;
  /** How many load attempts the server has logged. */
  readonly attempts: number;

  constructor(
    message: string,
    options?: {
      model?: string;
      errorClass?: string;
      permanent?: boolean;
      attempts?: number;
      param?: string | null;
    },
  ) {
    super(message, "MODEL_LOAD_FAILED", 502, undefined, options?.param);
    this.name = "ModelLoadFailedError";
    this.model = options?.model;
    this.errorClass = options?.errorClass;
    this.permanent = options?.permanent ?? true;
    this.attempts = options?.attempts ?? 1;
  }
}

/**
 * Error when the request input exceeds the model's maximum token capacity.
 *
 * Thrown when the server returns HTTP `400 INPUT_TOO_LONG` for an
 * extraction request. Distinct from generic {@link RequestError} so
 * callers can branch on token-budget failures specifically (truncate
 * the input client-side, switch to a longer-context model, or surface
 * a targeted error to the end user) without parsing the error code.
 *
 * Subclass of {@link RequestError} so existing 4xx handlers continue
 * to work; new code can catch {@link InputTooLongError} for tailored
 * handling.
 */
export class InputTooLongError extends RequestError {
  /** The model that was requested */
  readonly model: string | undefined;

  constructor(message: string, options?: { model?: string; param?: string | null }) {
    super(message, "INPUT_TOO_LONG", 400, undefined, options?.param);
    this.name = "InputTooLongError";
    this.model = options?.model;
  }
}

/**
 * A successful (HTTP 200) batch response dropped or added items.
 *
 * The gateway's queue path returns mixed-success batches as `200` carrying
 * only the successful items — a per-item failure is dropped from the body, not
 * surfaced as an error envelope. Batch responses are positional (item `id` is
 * optional), so a shortened body silently shifts every item after the dropped
 * one: a zip-inputs-to-outputs consumer would store results against the wrong
 * inputs. The SDK guards the 1:1 input-to-output contract on every batch
 * response and throws this instead of returning a desynced array.
 *
 * Subclass of {@link ServerError} — the server violated the response-shape
 * contract even though the HTTP status was 200 — so existing `ServerError`
 * handlers keep working. `statusCode` is 200 for the same reason: the response
 * was not an HTTP error. Catch this specifically to retry item-wise
 * (single-item batches get per-item error visibility), using {@link missingIds}
 * when the submitted items carried ids. Mirrors the Python SDK's
 * `IncompleteBatchError`.
 */
export class IncompleteBatchError extends ServerError {
  /** Number of items submitted in this HTTP request. */
  readonly expected: number;

  /** Number of items the response body carried. */
  readonly received: number;

  /** The model that was requested. */
  readonly model: string | undefined;

  /**
   * Ids of submitted items absent from the response.
   *
   * Only populated when ids identify every position on both sides (every
   * submitted item carried an `id` and every returned item echoed one);
   * `undefined` otherwise, since the set difference could otherwise mislabel a
   * present-but-unnamed item as missing.
   */
  readonly missingIds: string[] | undefined;

  constructor(
    message: string,
    options: {
      expected: number;
      received: number;
      code?: string;
      model?: string;
      missingIds?: string[];
      requestId?: string;
      param?: string | null;
    },
  ) {
    super(message, options.code, 200, options.requestId, options.param);
    this.name = "IncompleteBatchError";
    this.expected = options.expected;
    this.received = options.received;
    this.model = options.model;
    this.missingIds = options.missingIds;
  }
}

/**
 * A job reached a non-successful terminal state (`failed`/`suspended`/`cancelled`).
 *
 * Thrown by {@link SIEClient.jobs}`.wait` only when called with
 * `raiseOnFailure: true`; the default remains back-compatible and returns the
 * terminal status doc unchanged. The gateway's terminal reason rides `outcome`
 * / `error_code` on the status doc, so this surfaces them and a caller can
 * branch on the failure without re-reading the doc. Mirrors the Python SDK's
 * `JobFailedError`.
 */
export class JobFailedError extends SIEError {
  /** The job that failed. */
  readonly jobId: string | undefined;

  /** The terminal state (`failed`, `suspended`, or `cancelled`). */
  readonly state: string | undefined;

  /**
   * The gateway's terminal outcome (e.g. `reexecution_required`), or
   * `null`/`undefined` when the status doc carried none.
   */
  readonly outcome: string | null | undefined;

  /**
   * The gateway's terminal error code (e.g. `RESULT_HANDLE_EXPIRED`), or
   * `null`/`undefined` when absent.
   */
  readonly errorCode: string | null | undefined;

  constructor(
    message: string,
    options: {
      jobId?: string;
      state?: string;
      outcome?: string | null;
      errorCode?: string | null;
    } = {},
  ) {
    super(message);
    this.name = "JobFailedError";
    this.jobId = options.jobId;
    this.state = options.state;
    this.outcome = options.outcome;
    this.errorCode = options.errorCode;
  }
}

/**
 * A chunk ref's bytes could not be decoded as a msgpack `WorkResult` array.
 *
 * Distinct from a chunk that published no ref at all: the bytes exist but are
 * garbage (not msgpack, or not a list). That is a DECODE fault, not evidence of
 * failed publication or billing, so `jobs.results` confines it (one bad chunk
 * cannot sink the whole call) and flags it separately, never conflating it with
 * a genuinely-unpublished, already-billed chunk. Mirrors the Python SDK's
 * `MalformedChunkError`.
 */
export class MalformedChunkError extends SIEError {
  constructor(message: string) {
    super(message);
    this.name = "MalformedChunkError";
  }
}

/**
 * The gateway cannot PRICE the request, so it will not run it either.
 *
 * Thrown by {@link SIEClient.estimate} when the dry run answers `503`: the
 * active rate book declares no rate for the request's
 * (model, profile, operation, region) identity, or the planner cannot bound one
 * of the dimensions that book DOES price (an unbounded generation input, say,
 * or a sealed lane whose GPU class is unresolved).
 *
 * This is deliberately the same verdict the real request would get — the
 * estimate runs the live planner — so treat it as "this request is not sellable
 * right now", not as an estimator limitation. `message` is the planner's own
 * reason, naming the unpriced identity or the dimension it could not bound.
 *
 * Subclass of {@link ServerError} so existing 5xx handlers keep working.
 */
export class EstimateUnroutableError extends ServerError {
  constructor(message: string, code?: string, param?: string | null) {
    super(message, code, 503, undefined, param);
    this.name = "EstimateUnroutableError";
  }
}

/**
 * Error when the gateway rate-limits the caller and retries are exhausted.
 *
 * Thrown when the gateway returns HTTP `429 TOO_MANY_REQUESTS` (code
 * `RATE_LIMIT`, per-key or per-account, default-on) and the SDK's bounded,
 * `Retry-After`-honoring retry budget (capped by the provision timeout) is
 * spent. The SDK honors the server's `Retry-After` on each attempt before
 * giving up.
 *
 * Subclass of {@link RequestError} so existing 4xx handlers keep working; new
 * code can catch {@link RateLimitError} specifically to back off at a higher
 * level, shed load, or route elsewhere. Mirrors the Python SDK's
 * `RateLimitError`.
 */
export class RateLimitError extends RequestError {
  /** Last `Retry-After` hint the server supplied, in milliseconds. */
  readonly retryAfter: number | undefined;

  constructor(
    message: string,
    options?: { retryAfter?: number; code?: string; requestId?: string; param?: string | null },
  ) {
    super(message, options?.code ?? "RATE_LIMIT", 429, options?.requestId, options?.param);
    this.name = "RateLimitError";
    this.retryAfter = options?.retryAfter;
  }
}

/**
 * Error when the account has insufficient credits to run the request.
 *
 * Thrown when the gateway returns HTTP `402 PAYMENT_REQUIRED` with code
 * `INSUFFICIENT_CREDITS`. This is a TERMINAL billing failure — the SDK never
 * retries it, because retrying a credit failure would be wrong.
 *
 * Subclass of {@link RequestError} so existing 4xx handlers keep working.
 * Mirrors the Python SDK's `InsufficientCreditsError`.
 */
export class InsufficientCreditsError extends RequestError {
  constructor(message: string, options?: { requestId?: string; param?: string | null }) {
    super(message, "INSUFFICIENT_CREDITS", 402, options?.requestId, options?.param);
    this.name = "InsufficientCreditsError";
  }
}

/**
 * Error when the API key's configured spend limit is exceeded.
 *
 * Thrown when the gateway returns HTTP `402 PAYMENT_REQUIRED` with code
 * `KEY_SPEND_LIMIT_EXCEEDED`. This is a TERMINAL policy failure — the SDK
 * never retries it. Distinct from {@link InsufficientCreditsError}: the
 * account may have credits, but this key has hit its own spend cap.
 *
 * Subclass of {@link RequestError} so existing 4xx handlers keep working.
 * Mirrors the Python SDK's `SpendLimitError`.
 */
export class SpendLimitError extends RequestError {
  constructor(message: string, options?: { requestId?: string; param?: string | null }) {
    super(message, "KEY_SPEND_LIMIT_EXCEEDED", 402, options?.requestId, options?.param);
    this.name = "SpendLimitError";
  }
}

/**
 * Error when the account is not permitted to submit work.
 *
 * Thrown when the gateway returns HTTP `403 FORBIDDEN` with code
 * `ACCOUNT_SUSPENDED` or `ACCOUNT_PENDING_REVIEW`. This is a TERMINAL
 * account-state failure — the SDK never retries it, because the account must
 * be activated/reinstated out of band before work is accepted. Branch on
 * `code` to distinguish suspended vs pending review.
 *
 * Subclass of {@link RequestError} so existing 4xx handlers keep working.
 * Mirrors the Python SDK's `AccountInactiveError`.
 */
export class AccountInactiveError extends RequestError {
  constructor(message: string, code?: string, requestId?: string, param?: string | null) {
    super(message, code, 403, requestId, param);
    this.name = "AccountInactiveError";
  }
}

/**
 * Error when the gateway cannot resolve the account's admission state.
 *
 * Thrown when the gateway returns HTTP `503` with code
 * `ACCOUNT_STATE_UNAVAILABLE` — a fail-closed infrastructure signal (the
 * control plane could not resolve account state), distinct from a customer
 * suspension. Surfaced as a typed TERMINAL error rather than being retried on
 * the SDK's admission ladder: a caller may re-issue the whole request, but the
 * SDK does not silently loop on an unresolved account state.
 *
 * Subclass of {@link ServerError} so existing 5xx handlers keep working.
 * Mirrors the Python SDK's `AccountStateUnavailableError`.
 */
export class AccountStateUnavailableError extends ServerError {
  constructor(message: string, requestId?: string, param?: string | null) {
    super(message, "ACCOUNT_STATE_UNAVAILABLE", 503, requestId, param);
    this.name = "AccountStateUnavailableError";
  }
}
