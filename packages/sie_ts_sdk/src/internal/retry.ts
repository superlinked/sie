/**
 * Retry logic with exponential backoff and jitter
 */

import { DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY } from "./constants.js";

/**
 * Compute backoff with decorrelated jitter
 * @param attempt - The current attempt number (0-indexed)
 * @param baseDelay - Base delay in milliseconds
 * @param maxDelay - Maximum delay in milliseconds
 */
export function computeBackoffWithJitter(
  attempt: number,
  baseDelay: number = DEFAULT_RETRY_DELAY,
  maxDelay: number = DEFAULT_MAX_RETRY_DELAY,
): number {
  const exponentialDelay = baseDelay * 2 ** attempt;
  const cappedDelay = Math.min(exponentialDelay, maxDelay);
  // Decorrelated jitter: random value between 0 and cappedDelay
  return Math.random() * cappedDelay;
}

/**
 * Parse Retry-After header value
 * @param header - The Retry-After header value
 * @returns Delay in milliseconds, or undefined if invalid
 */
export function getRetryAfter(header: string | null): number | undefined {
  if (!header) return undefined;

  // Try parsing as seconds (integer)
  const seconds = Number.parseInt(header, 10);
  if (!Number.isNaN(seconds) && seconds > 0) {
    return seconds * 1000;
  }

  // Try parsing as HTTP-date
  const date = new Date(header);
  if (!Number.isNaN(date.getTime())) {
    const delay = date.getTime() - Date.now();
    return delay > 0 ? delay : undefined;
  }

  return undefined;
}
