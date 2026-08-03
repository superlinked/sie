/**
 * The settled charge surfaced in a response's `usage` block (#2434).
 *
 * The gateway publishes `usage.credits_charged` + `usage.rate_book_version` on
 * every settled response and keeps the `x-sie-credits-debited` header
 * unchanged for compatibility. The SDK reads the body first — it is the only
 * source that also names the rate book — and falls back to the header for a
 * gateway that predates in-body surfacing. All tests use a mocked `fetch`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import type { ChatCompletion } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const BOOK = "2026-07-22-production-bootstrap-v1";

function completion(usage: Record<string, unknown>): ChatCompletion {
  return {
    id: "chatcmpl-settled",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "m",
    system_fingerprint: null,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "hi" },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: usage as ChatCompletion["usage"],
  };
}

function jsonResponse(body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

async function chat(body: unknown, headers: Record<string, string> = {}) {
  mockFetch.mockResolvedValueOnce(jsonResponse(body, headers));
  const client = new SIEClient("http://localhost:8080");
  return client.chatCompletions({ model: "m", messages: [{ role: "user", content: "hi" }] });
}

describe("settled charge", () => {
  beforeEach(() => mockFetch.mockClear());
  afterEach(() => vi.clearAllMocks());

  it("reads the charge from the body, which also names the rate book", async () => {
    const result = await chat(
      completion({
        prompt_tokens: 9,
        completion_tokens: 3,
        total_tokens: 12,
        credits_charged: 7,
        rate_book_version: BOOK,
      }),
      { "x-sie-credits-debited": "7" },
    );

    expect(result.usage.credits_charged).toBe(7);
    expect(result.usage.rate_book_version).toBe(BOOK);
    expect(result.request?.creditsDebited).toBe(7);
    expect(result.request?.rateBookVersion).toBe(BOOK);
    expect(result.request?.usage?.creditsCharged).toBe(7);
  });

  it("falls back to the header when the body carries no charge", async () => {
    const result = await chat(
      completion({ prompt_tokens: 9, completion_tokens: 3, total_tokens: 12 }),
      { "x-sie-credits-debited": "7" },
    );

    expect(result.request?.creditsDebited).toBe(7);
    expect(result.request?.rateBookVersion).toBeUndefined();
  });

  it("surfaces no charge at all when settlement did not commit", async () => {
    const result = await chat(
      completion({ prompt_tokens: 9, completion_tokens: 3, total_tokens: 12 }),
      { "x-sie-request-id": "req-fault" },
    );

    expect(result.request?.creditsDebited).toBeUndefined();
    expect(result.request?.rateBookVersion).toBeUndefined();
  });

  it("treats a settled zero as a charge", async () => {
    const result = await chat(
      completion({
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        credits_charged: 0,
        rate_book_version: BOOK,
      }),
    );

    expect(result.request?.creditsDebited).toBe(0);
    expect(result.request?.usage?.creditsCharged).toBe(0);
  });

  it("carries the charge through the generate result's own usage block", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        {
          model: "m",
          text: "hi",
          finish_reason: "stop",
          usage: {
            prompt_tokens: 9,
            completion_tokens: 3,
            total_tokens: 12,
            credits_charged: 7,
            rate_book_version: BOOK,
          },
        },
        { "x-sie-credits-debited": "7" },
      ),
    );
    const client = new SIEClient("http://localhost:8080");
    const result = await client.generate("m", "hi", { maxNewTokens: 16 });

    // The parser rebuilds `usage` field-by-field; the settled charge must
    // survive that rebuild rather than being reachable only via `request`.
    expect(result.usage.creditsCharged).toBe(7);
    expect(result.usage.rateBookVersion).toBe(BOOK);
    expect(result.usage.totalTokens).toBe(12);
  });

  it("leaves the generate usage block charge-free when nothing settled", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "hi",
        finish_reason: "stop",
        usage: { prompt_tokens: 9, completion_tokens: 3, total_tokens: 12 },
      }),
    );
    const client = new SIEClient("http://localhost:8080");
    const result = await client.generate("m", "hi", { maxNewTokens: 16 });

    expect(result.usage.creditsCharged).toBeUndefined();
    expect(result.usage.rateBookVersion).toBeUndefined();
  });

  it("ignores a half or malformed charge and falls back to the header", async () => {
    for (const usage of [
      { credits_charged: 7 },
      { rate_book_version: BOOK },
      { credits_charged: -1, rate_book_version: BOOK },
      { credits_charged: "7", rate_book_version: BOOK },
      { credits_charged: 1.5, rate_book_version: BOOK },
      { credits_charged: 7, rate_book_version: "" },
    ]) {
      const result = await chat(
        completion({ prompt_tokens: 1, completion_tokens: 1, total_tokens: 2, ...usage }),
        { "x-sie-credits-debited": "3" },
      );
      expect(result.request?.creditsDebited).toBe(3);
      expect(result.request?.rateBookVersion).toBeUndefined();
    }
  });
});
