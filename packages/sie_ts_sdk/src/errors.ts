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
 * - Server returns 202 (no capacity, provisioning)
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
