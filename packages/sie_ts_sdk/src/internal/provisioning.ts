/**
 * Shared provisioning / retry loop for non-streaming POST endpoints.
 *
 * Both {@link SIEClient.generate} and {@link SIEClient.chatCompletions}
 * receive the same pre-execution capacity signals from the gateway —
 * `503` with a known error code (`PROVISIONING`, `MODEL_LOADING` or
 * `RESOURCE_EXHAUSTED`) — and retry them under a `provisionTimeout`
 * budget. Only `503 PROVISIONING` is gated by the caller's
 * `waitForCapacity` flag; `503 MODEL_LOADING` and `503
 * RESOURCE_EXHAUSTED` are retried regardless (the worker already
 * accepted the request). A `504` is post-publish, and a rejected
 * `performFetch` may fire after the body was sent, so both are terminal
 * here — this loop serves non-idempotent generation and a retry could
 * double-bill.
 *
 * This helper centralises that loop. Callers supply a `performFetch`
 * callback that issues a fresh `fetch` per attempt (the request must be
 * re-buildable, which the JSON chat path satisfies trivially since the
 * body is a plain object). The loop returns the first successful
 * response or throws a typed error.
 *
 * The streaming path keeps its own inline copy because it needs
 * abortable sleeps composed with the caller's `AbortSignal` (see
 * `consumeSseStream` in `client.ts`).
 */

import {
  ModelLoadingError,
  ProvisioningError,
  RateLimitError,
  RequestError,
  ResourceExhaustedError,
  ServerError,
} from "../errors.js";
import {
  BACKPRESSURE_503_DEFAULT_DELAY,
  BACKPRESSURE_503_ERROR_CODES,
  DEFAULT_RETRY_DELAY,
  HTTP_GATEWAY_TIMEOUT,
  HTTP_TOO_MANY_REQUESTS,
  MODEL_LOADING_DEFAULT_DELAY,
  MODEL_LOADING_ERROR_CODE,
  PROVISIONING_ERROR_CODE,
  RATE_LIMIT_DEFAULT_DELAY,
  RESOURCE_EXHAUSTED_ERROR_CODE,
  RESOURCE_EXHAUSTED_MAX_RETRIES,
} from "./constants.js";
import {
  getErrorCode,
  getErrorParam,
  getRetryAfter,
  handleError,
  readRequestId,
  throwIfModelLoadFailed,
} from "./parsing.js";
import { applyRetryJitter, computeOomBackoff } from "./retry.js";

/** Options controlling the provisioning retry loop. */
export interface ProvisioningOptions {
  /** Model name (used to populate `ModelLoadingError.model`). */
  model: string;
  /** GPU label passed through to `ProvisioningError`. May be `undefined`. */
  gpu: string | undefined;
  /**
   * Controls `503 PROVISIONING` ONLY. When `true`, the loop retries a
   * `PROVISIONING` signal until `provisionTimeoutMs` is exhausted; when
   * `false`, the first `PROVISIONING` signal throws `ProvisioningError`
   * (the call-site opted out of capacity waiting). It does NOT gate any
   * other outcome:
   *   - `503 MODEL_LOADING` and `503 RESOURCE_EXHAUSTED` are retried
   *     regardless of this flag (each retry bounded by
   *     `provisionTimeoutMs`; `RESOURCE_EXHAUSTED` additionally bounded by
   *     `RESOURCE_EXHAUSTED_MAX_RETRIES`) because the worker already
   *     accepted the request — matching the Python SDK.
   *   - A `504` gateway timeout and any rejected `performFetch`
   *     (fetch-level connection failure) are ALWAYS terminal here, never
   *     retried under any flag value: this loop serves non-idempotent
   *     generation, so a retry could double-bill.
   */
  waitForCapacity: boolean;
  /**
   * Total cumulative wall-clock budget (ms) for retries. Defaults to
   * `DEFAULT_PROVISION_TIMEOUT` if omitted.
   */
  provisionTimeoutMs: number;
}

/** Sleep for `ms` milliseconds. Non-abortable; the non-streaming surface
 * does not expose an AbortSignal to the caller. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Budget check + backoff for one `503 RESOURCE_EXHAUSTED` retry.
 *
 * Mirrors the Python SDK's `_handle_oom_retry`: throws
 * {@link ResourceExhaustedError} when the retry budget is exhausted
 * (`oomRetries >= maxOomRetries`), the provision budget has elapsed, or
 * the next backoff would consume the remaining budget without leaving
 * room for the retried request to run (surfacing the *root cause* now
 * instead of a later "provisioning timeout"). Otherwise returns the
 * delay (ms) to sleep before the next attempt.
 *
 * Distinct from MODEL_LOADING: the model is already resident, the
 * request just lost the race for compute resources. This is a SAFE
 * pre-execution signal (the worker rejected the request before running
 * it), so on the buffered paths it is retried regardless of
 * `waitForCapacity`, matching the Python SDK.
 *
 * @internal
 */
export function nextOomRetryDelay(opts: {
  retryAfter: number | undefined;
  oomRetries: number;
  maxOomRetries: number;
  elapsedMs: number;
  provisionTimeoutMs: number;
  model: string;
  param?: string | null;
}): number {
  const { retryAfter, oomRetries, maxOomRetries, elapsedMs, provisionTimeoutMs, model, param } =
    opts;
  const message = `Server resource exhausted after ${oomRetries} retry attempt(s) for model '${model}'`;
  if (oomRetries >= maxOomRetries || elapsedMs >= provisionTimeoutMs) {
    throw new ResourceExhaustedError(message, { model, retries: oomRetries, param });
  }
  const delay = computeOomBackoff(retryAfter, oomRetries);
  if (delay >= provisionTimeoutMs - elapsedMs) {
    throw new ResourceExhaustedError(message, { model, retries: oomRetries, param });
  }
  return delay;
}

/**
 * Delay (ms) before retrying a pre-execution admission rejection, or
 * `undefined` when the response is not a retryable admission signal (the
 * caller then falls through to its existing terminal handling).
 *
 * Handles the pass-2 audit backpressure/billing signals that are safe to retry
 * because NO work has been published to the queue yet — they are admission
 * decisions the gateway/self-hosted server makes *before* dispatch, so
 * retrying is idempotent even on the non-idempotent generate paths:
 *
 * - `429 TOO_MANY_REQUESTS` (code `RATE_LIMIT`, B1) — per-key/per-account rate
 *   limiting. On give-up (provision-timeout budget spent) throws a typed
 *   {@link RateLimitError} carrying the last `Retry-After`.
 * - `503 BILLING_CAPACITY_UNAVAILABLE` (B2 — a gateway-local billing-family
 *   cap, NOT customer credit exhaustion) and `503 QUEUE_FULL` (B7 — self-hosted
 *   queue backpressure, #3180). On give-up throws the server's terminal 503
 *   verbatim via {@link handleError} (a {@link ServerError} preserving the code).
 *
 * Retry timing mirrors the PROVISIONING arm: the server-supplied `Retry-After`
 * is honored verbatim, and only the SDK's own fallback default is jittered.
 * Give-up mirrors `nextOomRetryDelay`: when the budget is spent OR the next
 * wait would consume the rest of it (leaving no room for the retried request to
 * run), the typed root cause is surfaced NOW instead of sleeping the budget
 * away and letting the outer loop mask it.
 *
 * IMPORTANT: 402/403 credit/account failures are deliberately NOT handled here.
 * They are terminal and must never be retried; {@link handleError} maps them to
 * typed exceptions on the first response. Mirrors the Python SDK's
 * `admission_retry_delay`.
 *
 * @internal
 */
export async function admissionRetryDelay(
  response: Response,
  opts: { startTime: number; provisionTimeoutMs: number },
): Promise<number | undefined> {
  const { startTime, provisionTimeoutMs } = opts;
  const { status } = response;

  if (status === HTTP_TOO_MANY_REQUESTS) {
    const retryAfter = getRetryAfter(response);
    const elapsed = Date.now() - startTime;
    const remaining = provisionTimeoutMs - elapsed;
    const delay = retryAfter ?? applyRetryJitter(Math.min(RATE_LIMIT_DEFAULT_DELAY, remaining));
    if (remaining <= 0 || delay >= remaining) {
      // Preserve the gateway request id from the final 429 so the typed
      // give-up stays correlatable with gateway logs, matching the direct
      // terminal 429 mapped by `handleError` (#3136).
      throw new RateLimitError(
        `Rate limited (429); retry budget (${provisionTimeoutMs}ms) exhausted after ${elapsed}ms`,
        {
          retryAfter,
          requestId: readRequestId(response),
          param: await getErrorParam(response.clone()),
        },
      );
    }
    return delay;
  }

  if (status === 503) {
    const code = await getErrorCode(response.clone());
    if (code !== undefined && BACKPRESSURE_503_ERROR_CODES.has(code)) {
      const retryAfter = getRetryAfter(response);
      const elapsed = Date.now() - startTime;
      const remaining = provisionTimeoutMs - elapsed;
      const delay =
        retryAfter ?? applyRetryJitter(Math.min(BACKPRESSURE_503_DEFAULT_DELAY, remaining));
      if (remaining <= 0 || delay >= remaining) {
        // Budget spent (or the next wait would consume it): surface the
        // server's terminal 503 (ServerError, code preserved). `handleError`
        // always throws.
        await handleError(response);
      }
      return delay;
    }
  }

  return undefined;
}

/**
 * Wrap a non-streaming POST attempt in the shared provisioning retry loop.
 *
 * The `performFetch` callback MUST re-issue the request from scratch on
 * each invocation — never reuse a consumed `Response`. It is responsible
 * for its own per-attempt timeout and for translating low-level
 * `TypeError` / `AbortError` into `SIEConnectionError`.
 *
 * The loop returns the first non-retryable success (`status === 200`).
 * Any other terminal status is handed to {@link handleError}, which
 * always throws.
 *
 * @internal
 */
export async function withProvisioningRetry(
  performFetch: () => Promise<Response>,
  opts: ProvisioningOptions,
): Promise<Response> {
  const startTime = Date.now();
  let oomRetries = 0;

  while (true) {
    // INTENTIONAL divergence from the Python SDK: Python's generate() retries
    // `httpx.ConnectError` because httpx guarantees it fires *before* the
    // request body is sent. `fetch` offers no such distinction — a rejected
    // fetch promise (`TypeError`) can mean a pre-connect DNS/refused failure
    // OR a connection reset *after* the (non-idempotent, non-deduped) request
    // body was transmitted, when a worker may already be generating. Retrying
    // here could issue a second billable generation, so fetch-level failures
    // from `performFetch` are surfaced as terminal `SIEConnectionError`
    // instead of being retried.
    const response = await performFetch();

    // 502 MODEL_LOAD_FAILED is terminal — surface immediately.
    await throwIfModelLoadFailed(response, opts.model);

    if (response.status === 503) {
      const errorCode = await getErrorCode(response.clone());
      if (errorCode === PROVISIONING_ERROR_CODE) {
        if (!opts.waitForCapacity) {
          throw new ProvisioningError(
            "No capacity available. Server is provisioning.",
            opts.gpu,
            getRetryAfter(response),
            await getErrorParam(response.clone()),
          );
        }
        const elapsed = Date.now() - startTime;
        if (elapsed >= opts.provisionTimeoutMs) {
          throw new ProvisioningError(
            `Provisioning timeout after ${elapsed}ms`,
            opts.gpu,
            getRetryAfter(response),
            await getErrorParam(response.clone()),
          );
        }
        const retryAfter = getRetryAfter(response);
        const delay = retryAfter ?? applyRetryJitter(DEFAULT_RETRY_DELAY);
        await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
        continue;
      }
      if (errorCode === MODEL_LOADING_ERROR_CODE) {
        const elapsed = Date.now() - startTime;
        if (elapsed >= opts.provisionTimeoutMs) {
          throw new ModelLoadingError(
            `Model loading timeout for '${opts.model}'`,
            opts.model,
            await getErrorParam(response.clone()),
          );
        }
        const delay = getRetryAfter(response) ?? MODEL_LOADING_DEFAULT_DELAY;
        await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
        continue;
      }
      if (errorCode === RESOURCE_EXHAUSTED_ERROR_CODE) {
        // Retried regardless of `waitForCapacity` (bounded budget), matching
        // the Python SDK: the signal fires before any generation starts.
        const delay = nextOomRetryDelay({
          retryAfter: getRetryAfter(response),
          oomRetries,
          maxOomRetries: RESOURCE_EXHAUSTED_MAX_RETRIES,
          elapsedMs: Date.now() - startTime,
          provisionTimeoutMs: opts.provisionTimeoutMs,
          model: opts.model,
          param: await getErrorParam(response.clone()),
        });
        oomRetries += 1;
        await sleep(delay);
        continue;
      }
    }

    // Retryable pre-execution admission backpressure (pass-2 audit B1/B2/B7):
    // a 429 RATE_LIMIT, or a retryable 503 (BILLING_CAPACITY_UNAVAILABLE /
    // QUEUE_FULL). These are admission rejections BEFORE the work is published,
    // so retrying is safe (unlike the post-publish 504 below). Honor
    // Retry-After within the provision-timeout budget; a give-up throws a typed
    // RateLimitError (429) or the server's terminal 503. 402/403 credit/account
    // errors are terminal and NOT handled here.
    const admissionDelay = await admissionRetryDelay(response, {
      startTime,
      provisionTimeoutMs: opts.provisionTimeoutMs,
    });
    if (admissionDelay !== undefined) {
      await sleep(admissionDelay);
      continue;
    }

    // Do NOT retry 504. A 504 GATEWAY_TIMEOUT is a *post-publish* timeout:
    // the work item is already on the queue and a worker may be — or have
    // finished — generating. Generation is non-idempotent and carries no
    // dedup key, so retrying could issue a SECOND billable generation.
    // The pre-execution 503 signals above remain retryable because those
    // fire before any generation can have started (Python SDK parity).
    if (response.status === HTTP_GATEWAY_TIMEOUT) {
      throw new ServerError(
        "Gateway timed out (504) after the request was published to the queue; " +
          "a worker may already be generating. Not retried because generation is " +
          "non-idempotent (retrying could double-bill). Re-issue manually if needed.",
        await getErrorCode(response.clone()),
        HTTP_GATEWAY_TIMEOUT,
        readRequestId(response),
        await getErrorParam(response.clone()),
      );
    }

    if (!response.ok) {
      await handleError(response);
    }

    // Defensive: handleError always throws on !ok, but if a future caller
    // adds a non-200 success status we still want to surface it cleanly.
    if (response.status !== 200) {
      throw new RequestError(`Unexpected response status ${response.status}`);
    }
    return response;
  }
}
