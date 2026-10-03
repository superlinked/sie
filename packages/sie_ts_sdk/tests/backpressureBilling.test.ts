/**
 * Tests for the pass-2 audit backpressure / billing signals the gateway emits
 * that the SDK previously discarded:
 *
 *   B1 — 429 RATE_LIMIT                    -> retried (Retry-After), typed RateLimitError on give-up
 *   B2 — 503 BILLING_CAPACITY_UNAVAILABLE  -> retried (Retry-After), terminal ServerError on give-up
 *   B7 — 503 QUEUE_FULL                    -> retried (Retry-After)
 *   B3 — 402/403 credit & account failures -> TERMINAL, typed, NEVER retried (single attempt)
 *
 * Mirrors the harness in `client.test.ts`: mock `fetch` with a canned response
 * sequence and drive the fake timer to release the retry sleeps. Covers the
 * inline buffered loop (encode), the shared `withProvisioningRetry` loop
 * (generate), and the terminal `handleError` mapping.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  AccountInactiveError,
  AccountStateUnavailableError,
  InsufficientCreditsError,
  RateLimitError,
  RequestError,
  ServerError,
  SpendLimitError,
} from "../src/errors.js";
import { handleError } from "../src/internal/parsing.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function detailResponse(
  status: number,
  code: string,
  headers: Record<string, string> = {},
  param?: string | null,
): Response {
  return new Response(JSON.stringify({ detail: { code, message: code.toLowerCase(), param } }), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

function encodeSuccess(): Response {
  return new Response(packMessage({ items: [{ dense: { values: new Float32Array([0.1]) } }] }), {
    status: 200,
    headers: { "Content-Type": "application/msgpack" },
  });
}

describe("backpressure & billing signals (pass-2 audit B1/B2/B7/B3)", () => {
  beforeEach(() => {
    // mockReset (not mockClear) drains any leftover `mockResolvedValueOnce`
    // queue so an unconsumed success from a "does not retry" test cannot leak
    // into the next test.
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ── B1: 429 rate limit ──────────────────────────────────────────────

  it("retries 429 RATE_LIMIT honoring Retry-After, then succeeds", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });
    mockFetch
      .mockResolvedValueOnce(
        detailResponse(429, "RATE_LIMIT", { "Retry-After": "2", "X-SIE-Error-Code": "RATE_LIMIT" }),
      )
      .mockResolvedValueOnce(encodeSuccess());

    const promise = client.encode("bge-m3", { text: "test" });

    // Retry-After (2s) honored verbatim — no retry before the full 2s.
    await vi.advanceTimersByTimeAsync(1_999);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    const result = await promise;
    expect(result.dense).toBeInstanceOf(Float32Array);
    expect(mockFetch).toHaveBeenCalledTimes(2);
    await client.close();
  });

  it("throws RateLimitError after the retry budget is exhausted", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 5_000,
      provisionTimeout: 1_000,
    });
    mockFetch.mockResolvedValue(detailResponse(429, "RATE_LIMIT", { "Retry-After": "1" }));

    const promise = client.encode("bge-m3", { text: "test" });
    const expectation = expect(promise).rejects.toThrow(RateLimitError);
    await vi.advanceTimersByTimeAsync(1_000);
    await expectation;
    await client.close();
  });

  it("exposes the server Retry-After on the thrown RateLimitError", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 5_000,
      provisionTimeout: 1_000,
    });
    mockFetch.mockResolvedValue(detailResponse(429, "RATE_LIMIT", { "Retry-After": "1" }));

    const promise = client.encode("bge-m3", { text: "test" }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_000);
    const err = await promise;
    expect(err).toBeInstanceOf(RateLimitError);
    expect((err as RateLimitError).retryAfter).toBe(1_000); // ms
    expect((err as RateLimitError).statusCode).toBe(429);
    await client.close();
  });

  it("preserves the gateway request id on the exhausted-429 give-up", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 5_000,
      provisionTimeout: 1_000,
    });
    // The give-up must carry x-sie-request-id from the final 429, matching the
    // direct terminal 429 mapped by handleError (#3136).
    mockFetch.mockResolvedValue(
      detailResponse(429, "RATE_LIMIT", { "Retry-After": "1", "x-sie-request-id": "req-abc123" }),
    );

    const promise = client.encode("bge-m3", { text: "test" }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_000);
    const err = await promise;
    expect(err).toBeInstanceOf(RateLimitError);
    expect((err as RateLimitError).requestId).toBe("req-abc123");
    await client.close();
  });

  // ── B2 / B7: retryable 503 backpressure ─────────────────────────────

  it.each(["BILLING_CAPACITY_UNAVAILABLE", "QUEUE_FULL"])(
    "retries 503 %s honoring Retry-After, then succeeds",
    async (code) => {
      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      mockFetch
        .mockResolvedValueOnce(detailResponse(503, code, { "Retry-After": "1" }))
        .mockResolvedValueOnce(encodeSuccess());

      const promise = client.encode("bge-m3", { text: "test" });
      await vi.advanceTimersByTimeAsync(1_000);
      const result = await promise;

      expect(result.dense).toBeInstanceOf(Float32Array);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      await client.close();
    },
  );

  it("throws ServerError (code preserved) when the backpressure 503 budget is spent", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 5_000,
      provisionTimeout: 1_000,
    });
    mockFetch.mockResolvedValue(
      detailResponse(503, "BILLING_CAPACITY_UNAVAILABLE", { "Retry-After": "1" }),
    );

    const promise = client.encode("bge-m3", { text: "test" }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_000);
    const err = await promise;

    expect(err).toBeInstanceOf(ServerError);
    expect(err).not.toBeInstanceOf(RateLimitError);
    expect((err as ServerError).code).toBe("BILLING_CAPACITY_UNAVAILABLE");
    expect((err as ServerError).statusCode).toBe(503);
    await client.close();
  });

  // ── B3: terminal credit / account failures (NEVER retried) ──────────

  it.each([
    [402, "INSUFFICIENT_CREDITS", InsufficientCreditsError],
    [402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError],
    [403, "ACCOUNT_SUSPENDED", AccountInactiveError],
    [403, "ACCOUNT_PENDING_REVIEW", AccountInactiveError],
    [503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError],
  ] as const)(
    "maps %s %s to a terminal typed error and does not retry",
    async (status, code, ctor) => {
      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      // A success is queued AFTER the failure to prove the SDK never reaches it.
      mockFetch
        .mockResolvedValueOnce(detailResponse(status, code))
        .mockResolvedValueOnce(encodeSuccess());

      await expect(client.encode("bge-m3", { text: "test" })).rejects.toBeInstanceOf(ctor);
      // Single attempt — no retry.
      expect(mockFetch).toHaveBeenCalledTimes(1);
      await client.close();
    },
  );

  // ── withProvisioningRetry parity (generate path) ────────────────────

  it("retries 429 on the generate path (withProvisioningRetry), then succeeds", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });
    mockFetch
      .mockResolvedValueOnce(detailResponse(429, "RATE_LIMIT", { "Retry-After": "1" }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            model: "m",
            text: "hi",
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            attempt_id: "a",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );

    const promise = client.generate("m", "Hi", { maxNewTokens: 8 });
    await vi.advanceTimersByTimeAsync(1_000);
    const result = await promise;

    expect(result.text).toBe("hi");
    expect(mockFetch).toHaveBeenCalledTimes(2);
    await client.close();
  });

  it("does not retry 402 on the generate path", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });
    mockFetch.mockResolvedValue(detailResponse(402, "INSUFFICIENT_CREDITS"));

    await expect(client.generate("m", "Hi", { maxNewTokens: 8 })).rejects.toBeInstanceOf(
      InsufficientCreditsError,
    );
    expect(mockFetch).toHaveBeenCalledTimes(1);
    await client.close();
  });
});

// handleError runs under real timers — it never sleeps.
describe("handleError billing/backpressure mapping", () => {
  it.each([
    [429, "RATE_LIMIT", RateLimitError],
    [402, "INSUFFICIENT_CREDITS", InsufficientCreditsError],
    [402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError],
    [403, "ACCOUNT_SUSPENDED", AccountInactiveError],
    [403, "ACCOUNT_PENDING_REVIEW", AccountInactiveError],
    [503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError],
  ] as const)("maps %s %s", async (status, code, ctor) => {
    await expect(handleError(detailResponse(status, code, {}, "account"))).rejects.toMatchObject({
      name: ctor.name,
      param: "account",
    });
  });

  it("leaves an unrecognized 403 code as a generic RequestError", async () => {
    const err = await handleError(detailResponse(403, "INVALID_KEY")).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RequestError);
    expect(err).not.toBeInstanceOf(AccountInactiveError);
  });
});
