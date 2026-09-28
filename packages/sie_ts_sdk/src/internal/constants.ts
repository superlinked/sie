/**
 * Internal constants for the SIE TypeScript SDK
 */

export const MSGPACK_CONTENT_TYPE = "application/msgpack";
export const JSON_CONTENT_TYPE = "application/json";

export const HTTP_CLIENT_ERROR_MIN = 400;
export const HTTP_CLIENT_ERROR_MAX = 499;
export const HTTP_SERVER_ERROR_MIN = 500;
export const HTTP_SERVER_ERROR_MAX = 599;
export const HTTP_GATEWAY_TIMEOUT = 504;

// Default timeouts and delays. The per-attempt timeout must outlast the
// gateway's default 120 s request deadline so its typed 504 reaches the caller
// before the client gives up on a request the server is still running
// (matches the Python SDK's DEFAULT_READ_TIMEOUT_S).
export const DEFAULT_TIMEOUT = 150_000; // 150 seconds
// Floor for long-running POSTs (jobs.submit / files.upload / batches.create) so
// a large upload / connector preflight / batch creation does not abort while the
// server keeps working. Matches the Python SDK's 120s floor.
export const DEFAULT_LONG_RUNNING_TIMEOUT = 120_000; // 2 minutes
export const DEFAULT_PROVISION_TIMEOUT = 900_000; // 15 minutes (900s matches the Python SDK's DEFAULT_PROVISION_TIMEOUT_S)
export const DEFAULT_RETRY_DELAY = 5_000; // 5 seconds (matches Python SDK)
export const DEFAULT_MAX_RETRY_DELAY = 30_000; // 30 seconds
export const DEFAULT_LEASE_RENEWAL_INTERVAL = 60_000; // 1 minute

// jobs.wait() polling — mirrors the Python SDK's jobs.wait defaults.
export const DEFAULT_JOB_WAIT_TIMEOUT = 600_000; // 10 minutes
export const DEFAULT_JOB_WAIT_POLL = 2_000; // 2 seconds

// LoRA loading retry settings
export const LORA_LOADING_MAX_RETRIES = 10; // Max retries for LoRA loading
export const LORA_LOADING_DEFAULT_DELAY = 1_000; // 1 second default retry delay
export const LORA_LOADING_ERROR_CODE = "LORA_LOADING"; // Error code from server

// Model loading retry settings
export const MODEL_LOADING_MAX_RETRIES = 60; // Max retries (60 * 5s = 5 min)
export const MODEL_LOADING_DEFAULT_DELAY = 5_000; // 5 seconds default retry delay
export const MODEL_LOADING_ERROR_CODE = "MODEL_LOADING"; // Error code from server
export const PROVISIONING_ERROR_CODE = "PROVISIONING"; // Error code from gateway provisioning

// RESOURCE_EXHAUSTED (server-side OOM) retry settings — mirror the Python SDK
// (packages/sie_sdk/src/sie_sdk/client/_shared.py).
export const RESOURCE_EXHAUSTED_MAX_RETRIES = 3; // Max bounded retries
export const RESOURCE_EXHAUSTED_DEFAULT_DELAY = 5_000; // Base backoff (ms)
export const RESOURCE_EXHAUSTED_MAX_DELAY = 30_000; // Backoff ceiling (ms)
export const RESOURCE_EXHAUSTED_ERROR_CODE = "RESOURCE_EXHAUSTED"; // Error code from server

// Pre-execution admission backpressure / billing signals (pass-2 audit).
// Emitted BEFORE any work is published to the queue, so retrying is idempotent
// even on the non-idempotent generate paths. Retried on the admission ladder
// (`admissionRetryDelay`), bounded by the provision-timeout budget and the
// server's `Retry-After`. Mirrors the Python SDK
// (packages/sie_sdk/src/sie_sdk/client/_shared.py).
//
// B1 — 429 rate limit (rate_limit.rs): per-key/per-account, default-on.
export const HTTP_TOO_MANY_REQUESTS = 429;
export const RATE_LIMIT_ERROR_CODE = "RATE_LIMIT";
export const RATE_LIMIT_DEFAULT_DELAY = 1_000; // Fallback (ms) when the server omits Retry-After
// B2 — 503 BILLING_CAPACITY_UNAVAILABLE (slab_ledger.rs): a gateway-local
// billing-family cap, NOT customer credit exhaustion. B7 — 503 QUEUE_FULL
// (self-hosted server, #3180): transient queue backpressure. Both send
// Retry-After: 1 and were previously unmatched by the retry ladder.
export const BILLING_CAPACITY_UNAVAILABLE_ERROR_CODE = "BILLING_CAPACITY_UNAVAILABLE";
export const QUEUE_FULL_ERROR_CODE = "QUEUE_FULL";
export const BACKPRESSURE_503_ERROR_CODES: ReadonlySet<string> = new Set([
  BILLING_CAPACITY_UNAVAILABLE_ERROR_CODE,
  QUEUE_FULL_ERROR_CODE,
]);
export const BACKPRESSURE_503_DEFAULT_DELAY = 1_000; // Fallback (ms) when the server omits Retry-After
// The gateway answers a publish rejected by queue backpressure with 503
// QUEUE_UNAVAILABLE (`transport_failure` in the OpenAI envelope used by the
// generation routes) plus Retry-After. The same codes without Retry-After
// report failures that may follow publication, so only the hinted form is
// retryable.
export const QUEUE_UNAVAILABLE_ERROR_CODE = "QUEUE_UNAVAILABLE";
export const OPENAI_TRANSPORT_FAILURE_ERROR_CODE = "transport_failure";
export const RETRY_AFTER_GATED_503_ERROR_CODES: ReadonlySet<string> = new Set([
  QUEUE_UNAVAILABLE_ERROR_CODE,
  OPENAI_TRANSPORT_FAILURE_ERROR_CODE,
]);

// Terminal credit / account errors (pass-2 audit B3) — NEVER retried. 402/403
// credit/account failures are mapped to typed exceptions in `handleError` and,
// having no arm on any retry ladder, surface on the first response.
export const HTTP_PAYMENT_REQUIRED = 402;
export const HTTP_FORBIDDEN = 403;
export const INSUFFICIENT_CREDITS_ERROR_CODE = "INSUFFICIENT_CREDITS";
export const KEY_SPEND_LIMIT_EXCEEDED_ERROR_CODE = "KEY_SPEND_LIMIT_EXCEEDED";
export const ACCOUNT_SUSPENDED_ERROR_CODE = "ACCOUNT_SUSPENDED";
export const ACCOUNT_PENDING_REVIEW_ERROR_CODE = "ACCOUNT_PENDING_REVIEW";
export const ACCOUNT_STATE_UNAVAILABLE_ERROR_CODE = "ACCOUNT_STATE_UNAVAILABLE";

// Version negotiation headers
export const SDK_VERSION_HEADER = "X-SIE-SDK-Version";
export const SERVER_VERSION_HEADER = "X-SIE-Server-Version";
