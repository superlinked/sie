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
 * Error in the request (4xx responses).
 *
 * Raised when the client sends an invalid request:
 * - 400: Bad request (invalid parameters, malformed body)
 * - 401: Unauthorized (missing or invalid API key)
 * - 403: Forbidden (insufficient permissions)
 * - 404: Not found (invalid endpoint or model)
 * - 422: Validation error (invalid input format)
 */
export class RequestError extends SIEError {
  /** Error code from the server (e.g., "INVALID_MODEL", "VALIDATION_ERROR") */
  readonly code: string | undefined;
  /** HTTP status code (400-499) */
  readonly statusCode: number | undefined;

  constructor(message: string, code?: string, statusCode?: number) {
    super(message);
    this.name = "RequestError";
    this.code = code;
    this.statusCode = statusCode;
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

  constructor(message: string, code?: string, statusCode?: number) {
    super(message);
    this.name = "ServerError";
    this.code = code;
    this.statusCode = statusCode;
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

  constructor(message: string, gpu?: string, retryAfter?: number) {
    super(message);
    this.name = "ProvisioningError";
    this.gpu = gpu;
    this.retryAfter = retryAfter;
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

  constructor(message: string, lora?: string, model?: string) {
    super(message);
    this.name = "LoraLoadingError";
    this.lora = lora;
    this.model = model;
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

  constructor(message: string, model?: string) {
    super(message);
    this.name = "ModelLoadingError";
    this.model = model;
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

  constructor(message: string, options?: { model?: string; retries?: number }) {
    super(message, "RESOURCE_EXHAUSTED", 503);
    this.name = "ResourceExhaustedError";
    this.model = options?.model;
    this.retries = options?.retries ?? 0;
  }
}

/**
 * Error surfaced mid-stream from `streamChatCompletions` / `streamGenerate`.
 *
 * The SSE wire shape includes optional `error: {message, type, param, code}`
 * (chat) or `error: {code, message}` (SIE-native generate) on the terminal
 * chunk. When the SDK sees such a chunk it does NOT yield the chunk; instead
 * it throws `SIEStreamError`, mirroring the non-streaming `handleError` path
 * so callers can catch the same way they would for HTTP-level failures.
 *
 * Compare with `RequestError` / `ServerError`: those fire before the SSE
 * stream opens (HTTP 4xx / 5xx). `SIEStreamError` fires after at least one
 * byte has gone out — the connection itself was healthy, but the worker /
 * gateway emitted an error envelope partway through generation.
 */
export class SIEStreamError extends SIEError {
  /** SIE-native error code (e.g. `context_exceeded`, `cancelled`). */
  readonly code: string | undefined;
  /** OpenAI-style error type (e.g. `context_length_exceeded`, `server_error`). */
  readonly errorType: string | undefined;
  /** Offending field name when known (chat shape only). */
  readonly param: string | null | undefined;

  constructor(
    message: string,
    options?: { code?: string; errorType?: string; param?: string | null },
  ) {
    super(message);
    this.name = "SIEStreamError";
    this.code = options?.code;
    this.errorType = options?.errorType;
    this.param = options?.param;
  }
}

/**
 * Error when the server reports a *terminal* model-load failure.
 *
 * Distinct from {@link ModelLoadingError} — this is thrown on the first
 * response (no retry budget consumed) when the server returns HTTP
 * `502 MODEL_LOAD_FAILED`. The server uses this code for permanent-class
 * failures (gated repos, missing dependencies, unrecognised model
 * architectures) where retrying would waste time. See sie-test#85.
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
    },
  ) {
    super(message, "MODEL_LOAD_FAILED", 502);
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

  constructor(message: string, options?: { model?: string }) {
    super(message, "INPUT_TOO_LONG", 400);
    this.name = "InputTooLongError";
    this.model = options?.model;
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
  constructor(message: string, code?: string) {
    super(message, code, 503);
    this.name = "EstimateUnroutableError";
  }
}
