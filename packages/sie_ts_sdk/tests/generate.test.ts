/**
 * SIEClient.generate() tests.
 *
 * Verify that:
 * - The request body is JSON (not msgpack).
 * - The aggregated response envelope parses into a ``GenerateResult``.
 * - The SDK surfaces SIE-native timing metadata (ttftMs, tpotMs, attemptId).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  RequestError,
  ResourceExhaustedError,
  SIEConnectionError,
  ServerError,
} from "../src/errors.js";
import { MINIMAL_JPEG_BASE64, MINIMAL_JPEG_BYTES } from "./fixtures.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("SIEClient.generate", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("parses the streaming envelope into a GenerateResult", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "Qwen__Qwen3-4B-Instruct-2507",
        text: "Hello world!",
        finish_reason: "stop",
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        attempt_id: "att-abc",
        ttft_ms: 120.5,
        tpot_ms: 45.2,
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const result = await client.generate("Qwen__Qwen3-4B-Instruct-2507", "Hi", {
      maxNewTokens: 32,
    });

    expect(result.model).toBe("Qwen__Qwen3-4B-Instruct-2507");
    expect(result.text).toBe("Hello world!");
    expect(result.finishReason).toBe("stop");
    expect(result.usage.promptTokens).toBe(5);
    expect(result.usage.completionTokens).toBe(3);
    expect(result.usage.totalTokens).toBe(8);
    expect(result.attemptId).toBe("att-abc");
    expect(result.ttftMs).toBe(120.5);
    expect(result.tpotMs).toBe(45.2);
    const omittedSamplerBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(omittedSamplerBody).not.toHaveProperty("temperature");
    expect(omittedSamplerBody).not.toHaveProperty("top_p");
  });

  it("sends a JSON body with snake_case field names", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("m", "Hi", {
      maxNewTokens: 8,
      temperature: 1.0,
      topP: 1.0,
      stop: ["</s>"],
      frequencyPenalty: 0.25,
      presencePenalty: -0.5,
      grammar: { regex: "[a-z]+", label: null, strict: null },
      seed: -1,
      logitBias: { "123": 1.5 },
      routingKey: "tenant-7",
      promptCacheKey: "prompt-9",
      safetyIdentifier: "safety-3",
      loraAdapter: "sql-adapter",
      adapterOptions: { overall_timeout_s: 30 },
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/generate/m");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.headers.Accept).toBe("application/json");
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      prompt: "Hi",
      max_new_tokens: 8,
      temperature: 1.0,
      top_p: 1.0,
      stop: ["</s>"],
      frequency_penalty: 0.25,
      presence_penalty: -0.5,
      grammar: { regex: "[a-z]+", label: null, strict: null },
      seed: -1,
      logit_bias: { "123": 1.5 },
      routing_key: "tenant-7",
      prompt_cache_key: "prompt-9",
      safety_identifier: "safety-3",
      lora_adapter: "sql-adapter",
      options: { overall_timeout_s: 30 },
    });
  });

  it("serializes native images as canonical base64 JSON envelopes", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "vision-model",
        text: "caption",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("vision-model", "Describe this image", {
      maxNewTokens: 8,
      images: [MINIMAL_JPEG_BYTES],
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.images).toEqual([{ data: MINIMAL_JPEG_BASE64, format: "jpeg" }]);
    expect(body.images[0].data).not.toBeInstanceOf(Uint8Array);
  });

  it("accepts a valid grammar held in the legacy broad record type", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "123",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );
    const grammar: Record<string, unknown> = { regex: "\\d+" };

    const client = new SIEClient("http://localhost:8080");
    await client.generate("m", "Return digits", { maxNewTokens: 8, grammar });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.grammar).toEqual(grammar);
  });

  it.each([
    {},
    { json_schema: {}, regex: "x" },
    { regex: 123 },
    { ebnf: "root", unknown: true },
    { regex: "x", label: 123 },
    { ebnf: "root", strict: "yes" },
  ])("rejects an invalid grammar before request: %j", async (grammar) => {
    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.generate("m", "Hi", {
        maxNewTokens: 8,
        grammar: grammar as never,
      }),
    ).rejects.toThrow();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it.each([Number.MAX_SAFE_INTEGER + 1, Number.MIN_SAFE_INTEGER - 1, 1.5, Number.NaN])(
    "rejects a seed that JSON cannot preserve exactly: %s",
    async (seed) => {
      const client = new SIEClient("http://localhost:8080");
      await expect(client.generate("m", "Hi", { maxNewTokens: 8, seed })).rejects.toThrow(
        RangeError,
      );
      expect(mockFetch).not.toHaveBeenCalled();
    },
  );

  it("normalizes HF-style model ids to SIE-safe route ids", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "Qwen__Qwen3-4B-Instruct-2507",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("Qwen/Qwen3-4B-Instruct-2507", "Hi", { maxNewTokens: 8 });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/generate/Qwen__Qwen3-4B-Instruct-2507");
  });

  it("throws RequestError on non-object response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify("not an object"), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toBeInstanceOf(
      RequestError,
    );
  });

  // H4 regression: a truncated / malformed envelope must NOT silently
  // produce an empty completion. Missing or non-string `model` / `text`
  // raises (matches the Python SDK contract).
  it("throws RequestError when the envelope is missing model", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        text: "hello",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toThrow(
      /missing string 'model'/,
    );
  });

  it("throws RequestError when the envelope is missing text", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toThrow(
      /missing string 'text'/,
    );
  });

  it("throws RequestError when model/text are present but not strings", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: 123,
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toBeInstanceOf(
      RequestError,
    );
  });

  it("forwards gpu and pool routing headers", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("m", "hi", { maxNewTokens: 8, gpu: "eval-bench/l4" });

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers["X-SIE-Pool"]).toBe("eval-bench");
    expect(init.headers["X-SIE-MACHINE-PROFILE"]).toBe("l4");
  });
});

/**
 * B1c regression: generate() is non-idempotent (no dedup key), so a
 * `fetch` `TypeError` — which can be raised for a connection dropped
 * AFTER the request body was sent (mid-flight) — must NOT be retried.
 * Retrying would issue a SECOND billable generation. The safe
 * pre-execution capacity signals (503 PROVISIONING, 503 MODEL_LOADING)
 * are detected from the HTTP status and ARE still retried.
 */
describe("SIEClient.generate retry semantics (B1c)", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("does NOT retry a mid-flight TypeError even when waitForCapacity is true", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    // `fetch` throws TypeError for any network failure; a second attempt
    // could double-bill, so generate() must surface it without retrying.
    mockFetch.mockRejectedValue(new TypeError("fetch failed"));

    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true }),
    ).rejects.toBeInstanceOf(SIEConnectionError);

    // Crucially: exactly ONE call, no retry.
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("does NOT retry a mid-flight TypeError when waitForCapacity is false", async () => {
    const client = new SIEClient("http://localhost:8080", { timeout: 1000 });

    mockFetch.mockRejectedValue(new TypeError("fetch failed"));

    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: false }),
    ).rejects.toBeInstanceOf(SIEConnectionError);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("still retries the safe 503 PROVISIONING status path under waitForCapacity", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    mockFetch
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          model: "m",
          text: "ok",
          finish_reason: "stop",
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          attempt_id: "a",
        }),
      );

    const promise = client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true });
    // DEFAULT_RETRY_DELAY = 5_000ms.
    await vi.advanceTimersByTimeAsync(5_000);
    const result = await promise;

    expect(result.text).toBe("ok");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("still retries the safe 503 MODEL_LOADING status path under waitForCapacity", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    const modelLoading = new Response(
      JSON.stringify({ error: { code: "MODEL_LOADING", message: "loading" } }),
      {
        status: 503,
        headers: { "Content-Type": "application/json" },
      },
    );

    mockFetch.mockResolvedValueOnce(modelLoading).mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "loaded",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const promise = client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true });
    // MODEL_LOADING_DEFAULT_DELAY = 5_000ms.
    await vi.advanceTimersByTimeAsync(5_000);
    const result = await promise;

    expect(result.text).toBe("loaded");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("retries the safe 503 RESOURCE_EXHAUSTED pre-execution signal", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    mockFetch
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ error: { code: "RESOURCE_EXHAUSTED", message: "out of memory" } }),
          { status: 503, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          model: "m",
          text: "recovered",
          finish_reason: "stop",
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          attempt_id: "a",
        }),
      );

    const promise = client.generate("m", "hi", { maxNewTokens: 8 });
    // First OOM backoff is jittered but never exceeds the 5s base.
    await vi.advanceTimersByTimeAsync(5_000);
    const result = await promise;

    expect(result.text).toBe("recovered");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("surfaces sustained RESOURCE_EXHAUSTED as ResourceExhaustedError", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    mockFetch.mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({ error: { code: "RESOURCE_EXHAUSTED", message: "out of memory" } }),
          { status: 503, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    const promise = client.generate("m", "hi", { maxNewTokens: 8 });
    const expectation = expect(promise).rejects.toThrow(ResourceExhaustedError);

    // Max backoff schedule (no Retry-After): ≤5s, ≤10s, ≤20s → ≤35s total.
    await vi.advanceTimersByTimeAsync(35_000);
    await expectation;

    // RESOURCE_EXHAUSTED_MAX_RETRIES = 3 → 4 requests (initial + 3 retries).
    expect(mockFetch).toHaveBeenCalledTimes(4);
  });

  it("does NOT retry a 504 gateway timeout even when waitForCapacity is true", async () => {
    // 504 is post-publish: a worker may already be generating. Retrying
    // could double-bill, so generate() must surface it terminally.
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    mockFetch.mockResolvedValue(
      new Response(
        JSON.stringify({ error: { code: "GATEWAY_TIMEOUT", message: "result deadline" } }),
        { status: 504, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true }),
    ).rejects.toThrow(/non-idempotent/);
    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true }),
    ).rejects.toBeInstanceOf(ServerError);

    // Crucially: one call per generate() invocation, no retry.
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
