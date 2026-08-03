/**
 * SIEClient.chatCompletions() tests (non-streaming surface).
 *
 * Mirrors `tests/generate.test.ts` for shape conventions. All tests use a
 * mocked `fetch` — no live gateway required.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { ProvisioningError, RequestError, ServerError } from "../src/errors.js";
import type { ChatCompletion } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

const SAMPLE_COMPLETION: ChatCompletion = {
  id: "chatcmpl-abc",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "Qwen__Qwen3-4B-Instruct-2507",
  system_fingerprint: null,
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hi there!" },
      finish_reason: "stop",
      logprobs: null,
    },
  ],
  usage: { prompt_tokens: 9, completion_tokens: 3, total_tokens: 12 },
};

describe("SIEClient.chatCompletions", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns parsed ChatCompletion on success", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(SAMPLE_COMPLETION, 200, {
        "x-sie-request-id": "req-chat",
        "x-sie-units-output-tokens": "3",
        "x-sie-credits-debited": "9",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const result = await client.chatCompletions({
      model: "Qwen/Qwen3-4B-Instruct-2507",
      messages: [{ role: "user", content: "Hi" }],
    });

    expect(result.id).toBe("chatcmpl-abc");
    expect(result.choices[0]?.message.content).toBe("Hi there!");
    expect(result.usage.total_tokens).toBe(12);
    expect(result.request).toEqual({
      id: "req-chat",
      usage: { outputTokens: 3 },
      creditsDebited: 9,
    });
  });

  it("POSTs JSON to /v1/chat/completions with stream:false in the body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));

    const client = new SIEClient("http://localhost:8080");
    await client.chatCompletions({
      model: "m",
      messages: [{ role: "user", content: "Hi" }],
      max_completion_tokens: 32,
      temperature: 0.7,
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/chat/completions");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.headers.Accept).toBe("application/json");
    const body = JSON.parse(init.body);
    expect(body.stream).toBe(false);
    expect(body.model).toBe("m");
    expect(body.max_completion_tokens).toBe(32);
    expect(body.temperature).toBe(0.7);
  });

  it.each([Number.MAX_SAFE_INTEGER + 1, Number.MIN_SAFE_INTEGER - 1, 1.5, Number.NaN])(
    "rejects a seed outside the JavaScript safe-integer contract: %s",
    async (seed) => {
      const client = new SIEClient("http://localhost:8080");
      await expect(
        client.chatCompletions({
          model: "m",
          messages: [{ role: "user", content: "hi" }],
          seed,
        }),
      ).rejects.toThrow(RangeError);
      expect(mockFetch).not.toHaveBeenCalled();
    },
  );

  it.each([Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER])(
    "forwards a safe-integer boundary seed unchanged: %s",
    async (seed) => {
      mockFetch.mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));
      const client = new SIEClient("http://localhost:8080");
      await client.chatCompletions({
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        seed,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.seed).toBe(seed);
    },
  );

  it("throws RequestError on 400", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        {
          error: {
            message: "messages cannot be empty",
            type: "invalid_request_error",
            code: "invalid_request",
            param: "messages",
          },
        },
        400,
      ),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.chatCompletions({ model: "m", messages: [] })).rejects.toBeInstanceOf(
      RequestError,
    );
  });

  it("throws ServerError on 500", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ error: { message: "boom", type: "server_error", code: "internal" } }, 500),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.chatCompletions({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    ).rejects.toBeInstanceOf(ServerError);
  });

  it("throws RequestError when stream:true is passed", async () => {
    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.chatCompletions({
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    ).rejects.toBeInstanceOf(RequestError);
    // Must not have hit the network.
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("includes Authorization header when apiKey is set", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));
    const client = new SIEClient("http://localhost:8080", { apiKey: "sk-test" });
    await client.chatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    });
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer sk-test");
  });

  // H1: provisioning must be a non-2xx typed capacity signal, not a successful
  // response that can be cast to `ChatCompletion`.
  it("throws ProvisioningError on 503 PROVISIONING when waitForCapacity is false (H1)", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }), {
        status: 503,
        headers: { "Content-Type": "application/json", "Retry-After": "7" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.chatCompletions(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { waitForCapacity: false },
      ),
    ).rejects.toBeInstanceOf(ProvisioningError);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("retries 503 PROVISIONING then returns the parsed completion when waitForCapacity is true (H1)", async () => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }),
            {
              status: 503,
              headers: { "Content-Type": "application/json" },
            },
          ),
        )
        .mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const promise = client.chatCompletions(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { waitForCapacity: true },
      );
      // DEFAULT_RETRY_DELAY = 5_000ms.
      await vi.advanceTimersByTimeAsync(5_000);
      const result = await promise;

      expect(result.id).toBe("chatcmpl-abc");
      expect(result.choices[0]?.message.content).toBe("Hi there!");
      expect(mockFetch).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries a 503 MODEL_LOADING then returns the completion when waitForCapacity is true (H1)", async () => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ error: { code: "MODEL_LOADING", message: "loading" } }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          }),
        )
        .mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const promise = client.chatCompletions(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { waitForCapacity: true },
      );
      // MODEL_LOADING_DEFAULT_DELAY = 5_000ms.
      await vi.advanceTimersByTimeAsync(5_000);
      const result = await promise;

      expect(result.id).toBe("chatcmpl-abc");
      expect(mockFetch).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not retry a generic 503 when waitForCapacity is true (H1)", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: "service unavailable" }, 503));

    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.chatCompletions(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { waitForCapacity: true },
      ),
    ).rejects.toBeInstanceOf(ServerError);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("throws ProvisioningError when provisioning exceeds provisionTimeoutMs (H1)", async () => {
    vi.useFakeTimers();
    try {
      // Always answer 503 PROVISIONING so the loop keeps looping until the
      // budget is exhausted.
      mockFetch.mockImplementation(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }),
            {
              status: 503,
              headers: { "Content-Type": "application/json", "Retry-After": "5" },
            },
          ),
        ),
      );

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 600_000, // ignored — per-call override below
      });
      const promise = client.chatCompletions(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { waitForCapacity: true, provisionTimeoutMs: 10_000 },
      );
      // Attach the assertion BEFORE advancing fake timers — the promise
      // rejects synchronously once the next attempt sees elapsed >= 10s,
      // and advancing 30s drives the loop past the budget. Attaching late
      // would surface as an unhandled rejection in the test harness.
      const assertion = expect(promise).rejects.toBeInstanceOf(ProvisioningError);
      await vi.advanceTimersByTimeAsync(30_000);
      await assertion;
      // We expect more than one attempt (multiple retries) before timeout.
      expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(2);
    } finally {
      vi.useRealTimers();
    }
  });

  // M6: every newly typed field in `ChatCompletionRequest` must land on the
  // wire under its snake_case name. A regression here means the gateway
  // either drops the field silently (worst case) or 400s on the canonical
  // shape (visible case) — both are user-visible breakage.
  it("forwards every newly typed M6 field verbatim on the wire", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SAMPLE_COMPLETION));

    const client = new SIEClient("http://localhost:8080");
    await client.chatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
      n: 2,
      best_of: 4,
      logprobs: true,
      top_logprobs: 5,
      lora_adapter: "my-lora",
      top_k: 40,
      repetition_penalty: 1.1,
      logit_bias: { "1234": 5, "9999": -7.5 },
      user: "end-user-42",
      safety_identifier: "safety-tier-A",
      parallel_tool_calls: false,
      // also assert existing-typed fields still survive the rewrite.
      seed: 42,
      frequency_penalty: 0.2,
      presence_penalty: -0.3,
      routing_key: "k1",
      prompt_cache_key: "cache-1",
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [, init] = mockFetch.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.n).toBe(2);
    expect(body.best_of).toBe(4);
    expect(body.logprobs).toBe(true);
    expect(body.top_logprobs).toBe(5);
    expect(body.lora_adapter).toBe("my-lora");
    expect(body.top_k).toBe(40);
    expect(body.repetition_penalty).toBe(1.1);
    expect(body.logit_bias).toEqual({ "1234": 5, "9999": -7.5 });
    expect(body.user).toBe("end-user-42");
    expect(body.safety_identifier).toBe("safety-tier-A");
    expect(body.parallel_tool_calls).toBe(false);
    expect(body.seed).toBe(42);
    expect(body.frequency_penalty).toBe(0.2);
    expect(body.presence_penalty).toBe(-0.3);
    expect(body.routing_key).toBe("k1");
    expect(body.prompt_cache_key).toBe("cache-1");
    // stream must be forced false on the non-streaming method.
    expect(body.stream).toBe(false);
  });
});
