/**
 * SIEClient streaming surface tests (`streamChatCompletions`, `streamGenerate`).
 *
 * SSE responses are built inline via `ReadableStream` from string fixtures
 * — no live gateway required. The fixtures mirror the wire shape emitted
 * by `packages/sie_gateway/src/handlers/sse.rs`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  ProvisioningError,
  RequestError,
  SIEConnectionError,
  SIEStreamError,
} from "../src/errors.js";
import type { ChatCompletionChunk, GenerateChunk } from "../src/types.js";
import { MINIMAL_JPEG_BASE64, MINIMAL_JPEG_BYTES } from "./fixtures.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

/**
 * Build an SSE response from a list of payload strings. Each payload is
 * framed as `data: <json>\n\n`; the final `[DONE]` terminator is added
 * automatically unless `includeDone: false`.
 */
function sseResponse(payloads: string[], opts: { includeDone?: boolean } = {}): Response {
  const includeDone = opts.includeDone ?? true;
  const encoder = new TextEncoder();
  const frames = payloads.map((p) => `data: ${p}\n\n`);
  if (includeDone) frames.push("data: [DONE]\n\n");
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

/** Same as `sseResponse` but enqueues frames lazily so tests can race the abort. */
function slowSseResponse(payloads: string[]): {
  response: Response;
  releaseNext: () => void;
  closed: () => boolean;
} {
  const encoder = new TextEncoder();
  let nextIdx = 0;
  let isClosed = false;
  let resolveNext: (() => void) | null = null;
  const gate = () =>
    new Promise<void>((res) => {
      resolveNext = res;
    });

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (nextIdx >= payloads.length) {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        isClosed = true;
        return;
      }
      await gate();
      const frame = `data: ${payloads[nextIdx]}\n\n`;
      nextIdx += 1;
      controller.enqueue(encoder.encode(frame));
    },
    cancel() {
      isClosed = true;
      // Unblock any in-flight pull() so reader.cancel() can resolve.
      const r = resolveNext;
      resolveNext = null;
      r?.();
    },
  });
  return {
    response: new Response(stream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }),
    releaseNext: () => {
      const r = resolveNext;
      resolveNext = null;
      r?.();
    },
    closed: () => isClosed,
  };
}

/**
 * Like `sseResponse` but the raw `data:` frames are passed verbatim (so a
 * test can emit a deliberately malformed JSON payload) and the underlying
 * stream's `cancel()` flips a flag. Used to assert BUG 6: any non-clean
 * teardown of the stream (caller break, mid-stream throw, JSON parse error)
 * must `cancel()` the body so the worker stops generating.
 */
function trackedSseResponse(rawFrames: string[]): {
  response: Response;
  cancelled: () => boolean;
} {
  const encoder = new TextEncoder();
  let wasCancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of rawFrames) controller.enqueue(encoder.encode(f));
      controller.close();
    },
    cancel() {
      wasCancelled = true;
    },
  });
  return {
    response: new Response(stream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }),
    cancelled: () => wasCancelled,
  };
}

function chatChunk(content: string, opts: Partial<ChatCompletionChunk> = {}): string {
  const chunk: ChatCompletionChunk = {
    id: "chatcmpl-x",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "m",
    system_fingerprint: null,
    choices: [
      {
        index: 0,
        delta: { content },
        finish_reason: null,
        logprobs: null,
      },
    ],
    ...opts,
  };
  return JSON.stringify(chunk);
}

function generateChunk(seq: number, text_delta: string, opts: Partial<GenerateChunk> = {}): string {
  const chunk: GenerateChunk = {
    request_id: "req-1",
    seq,
    text_delta,
    done: false,
    ...opts,
  };
  return JSON.stringify(chunk);
}

describe("SIEClient.streamChatCompletions", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("yields chunks in order and stops cleanly on [DONE]", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        chatChunk("Hello", {
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello" },
              finish_reason: null,
              logprobs: null,
            },
          ],
        }),
        chatChunk(" "),
        chatChunk("world"),
        chatChunk("!"),
        chatChunk("", {
          choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
        }),
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(5);
    expect(chunks[0]?.choices[0]?.delta.role).toBe("assistant");
    expect(chunks[0]?.choices[0]?.delta.content).toBe("Hello");
    expect(chunks.map((c) => c.choices[0]?.delta.content ?? "").join("")).toBe("Hello world!");
    expect(chunks[4]?.choices[0]?.finish_reason).toBe("stop");
  });

  it("yields a trailing usage-only chunk when include_usage is true", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        chatChunk("Hi"),
        chatChunk("", {
          choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
        }),
        JSON.stringify({
          id: "chatcmpl-x",
          object: "chat.completion.chunk",
          created: 1_700_000_000,
          model: "m",
          system_fingerprint: null,
          choices: [],
          usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
        } satisfies ChatCompletionChunk),
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
      stream_options: { include_usage: true },
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(3);
    const last = chunks[2];
    expect(last?.choices).toEqual([]);
    expect(last?.usage?.total_tokens).toBe(6);
  });

  it("throws SIEStreamError on mid-stream error chunk and stops iteration", async () => {
    const errorChunkBody = {
      id: "chatcmpl-x",
      object: "chat.completion.chunk" as const,
      created: 1_700_000_000,
      model: "m",
      system_fingerprint: null,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" as const, logprobs: null }],
      error: {
        message: "prompt too long",
        type: "context_length_exceeded",
        param: null,
        code: "context_exceeded",
      },
    };
    mockFetch.mockResolvedValueOnce(
      sseResponse([chatChunk("Hello"), JSON.stringify(errorChunkBody)]),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    });

    const first = await gen.next();
    expect(first.done).toBe(false);
    expect(first.value?.choices[0]?.delta.content).toBe("Hello");

    try {
      await gen.next();
      throw new Error("expected SIEStreamError");
    } catch (err) {
      expect(err).toBeInstanceOf(SIEStreamError);
      const streamErr = err as SIEStreamError;
      expect(streamErr.code).toBe("context_exceeded");
      expect(streamErr.errorType).toBe("context_length_exceeded");
      expect(streamErr.message).toBe("prompt too long");
    }
  });

  it("yields tool_calls delta unchanged", async () => {
    const toolChunk = JSON.stringify({
      id: "chatcmpl-x",
      object: "chat.completion.chunk",
      created: 1_700_000_000,
      model: "m",
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_abc",
                type: "function",
                function: { name: "get_weather", arguments: '{"city":' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    } satisfies ChatCompletionChunk);

    mockFetch.mockResolvedValueOnce(sseResponse([toolChunk]));

    const client = new SIEClient("http://localhost:8080");
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "weather?" }],
    })) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    const calls = chunks[0]?.choices[0]?.delta.tool_calls;
    expect(calls?.[0]?.function?.name).toBe("get_weather");
    expect(calls?.[0]?.function?.arguments).toBe('{"city":');
  });

  it("sets stream:true automatically and uses Accept: text/event-stream", async () => {
    mockFetch.mockResolvedValueOnce(sseResponse([chatChunk("ok")]));

    const client = new SIEClient("http://localhost:8080");
    // Drain the generator.
    for await (const _ of client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
      stream: false, // should be overridden
      seed: -1,
    })) {
      // noop
    }

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/chat/completions");
    expect(init.headers.Accept).toBe("text/event-stream");
    const body = JSON.parse(init.body);
    expect(body.stream).toBe(true);
    expect(body.seed).toBe(-1);
  });

  it.each([Number.MAX_SAFE_INTEGER + 1, Number.MIN_SAFE_INTEGER - 1, 1.5, Number.NaN])(
    "rejects a chat seed outside the JavaScript safe-integer contract: %s",
    async (seed) => {
      const client = new SIEClient("http://localhost:8080");
      const stream = client.streamChatCompletions({
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        seed,
      });
      await expect(stream.next()).rejects.toThrow(RangeError);
      expect(mockFetch).not.toHaveBeenCalled();
    },
  );

  it.each([Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER])(
    "forwards a safe-integer boundary chat seed unchanged: %s",
    async (seed) => {
      mockFetch.mockResolvedValueOnce(sseResponse([chatChunk("ok")]));
      const client = new SIEClient("http://localhost:8080");
      for await (const _ of client.streamChatCompletions({
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        seed,
      })) {
        // Drain the generator.
      }

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.seed).toBe(seed);
    },
  );

  it("throws RequestError on 400 before the stream opens", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: { message: "bad", type: "invalid_request_error", code: "invalid_request" },
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      ),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamChatCompletions({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    });
    await expect(gen.next()).rejects.toBeInstanceOf(RequestError);
  });

  // 3s test-level timeout: we expect either resolution or rejection
  // within a few hundred ms; anything longer indicates a hang.
  it("respects AbortController mid-stream and ends the generator", { timeout: 3_000 }, async () => {
    const slow = slowSseResponse([chatChunk("Hello"), chatChunk(" world"), chatChunk("!")]);
    mockFetch.mockResolvedValueOnce(slow.response);

    const client = new SIEClient("http://localhost:8080");
    const controller = new AbortController();
    const gen = client.streamChatCompletions(
      {
        model: "m",
        messages: [{ role: "user", content: "hi" }],
      },
      controller.signal,
    );

    // Kick off the first `gen.next()` BEFORE releasing the gate: the
    // underlying source's `pull()` must already be awaiting `gate()`
    // by the time we resolve it, otherwise `resolveNext` is still
    // null and the release no-ops.
    const firstP = gen.next();
    setTimeout(() => slow.releaseNext(), 25);
    const first = await firstP;
    expect(first.done).toBe(false);
    expect(first.value?.choices[0]?.delta.content).toBe("Hello");

    // Abort and pull again. We expect `SIEConnectionError` because
    // the parser checks `signal.aborted` synchronously at the top of
    // each read loop and races `read()` against the signal.
    controller.abort();
    try {
      await gen.next();
      throw new Error("expected SIEConnectionError");
    } catch (err) {
      expect(err).toBeInstanceOf(SIEConnectionError);
    }
  });
});

describe("SIEClient.streamGenerate", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("yields SIE-native chunks in order with a terminal usage+ttft chunk", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        generateChunk(0, "Hello", {
          logprobs: [{ token: "Hello", logprob: -0.25, top_logprobs: [] }],
        }),
        generateChunk(1, " world"),
        generateChunk(2, "", {
          done: true,
          finish_reason: "stop",
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
          ttft_ms: 123.4,
        }),
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const chunks: GenerateChunk[] = [];
    for await (const chunk of client.streamGenerate("m", "hi", { maxNewTokens: 8 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(3);
    expect(chunks.map((c) => c.text_delta).join("")).toBe("Hello world");
    expect(chunks[0]?.logprobs?.[0]?.token).toBe("Hello");
    const last = chunks[2];
    expect(last?.done).toBe(true);
    expect(last?.finish_reason).toBe("stop");
    expect(last?.usage?.total_tokens).toBe(6);
    expect(last?.ttft_ms).toBe(123.4);
  });

  it("throws SIEStreamError when chunk.error is present", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        generateChunk(0, "Hello"),
        generateChunk(0, "", {
          done: true,
          finish_reason: "error",
          error: { code: "cancelled", message: "client closed" },
        }),
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamGenerate("m", "hi", { maxNewTokens: 8 });
    await gen.next(); // first delta
    try {
      await gen.next();
      throw new Error("expected SIEStreamError");
    } catch (err) {
      expect(err).toBeInstanceOf(SIEStreamError);
      expect((err as SIEStreamError).code).toBe("cancelled");
      expect((err as SIEStreamError).message).toBe("client closed");
    }
  });

  it("POSTs JSON to /v1/generate/<safeModel> with stream:true in the body", async () => {
    mockFetch.mockResolvedValueOnce(sseResponse([generateChunk(0, "x")]));
    const client = new SIEClient("http://localhost:8080");
    for await (const _ of client.streamGenerate("Qwen/Qwen3-4B-Instruct-2507", "hi", {
      maxNewTokens: 4,
      images: [MINIMAL_JPEG_BYTES],
      temperature: 0.5,
      topP: 0.9,
      stop: ["</s>"],
      frequencyPenalty: 0.25,
      presencePenalty: -0.5,
      grammar: { regex: "[a-z]+", label: null, strict: null },
      seed: -2,
      logitBias: { "123": 1.5 },
      topLogprobs: 3,
      routingKey: "tenant-7",
      promptCacheKey: "prompt-9",
      safetyIdentifier: "safety-3",
      loraAdapter: "sql-adapter",
    })) {
      // noop
    }
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/generate/Qwen__Qwen3-4B-Instruct-2507");
    expect(init.headers.Accept).toBe("text/event-stream");
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      prompt: "hi",
      max_new_tokens: 4,
      images: [{ data: MINIMAL_JPEG_BASE64, format: "jpeg" }],
      temperature: 0.5,
      top_p: 0.9,
      stop: ["</s>"],
      frequency_penalty: 0.25,
      presence_penalty: -0.5,
      grammar: { regex: "[a-z]+", label: null, strict: null },
      seed: -2,
      logit_bias: { "123": 1.5 },
      logprobs: true,
      top_logprobs: 3,
      routing_key: "tenant-7",
      prompt_cache_key: "prompt-9",
      safety_identifier: "safety-3",
      lora_adapter: "sql-adapter",
      stream: true,
    });
  });

  it("rejects an unsafe seed before opening the generation stream", async () => {
    const client = new SIEClient("http://localhost:8080");
    const stream = client.streamGenerate("m", "hi", {
      maxNewTokens: 4,
      seed: Number.MAX_SAFE_INTEGER + 1,
    });
    await expect(stream.next()).rejects.toThrow(RangeError);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("forwards gpu/pool routing headers", async () => {
    mockFetch.mockResolvedValueOnce(sseResponse([generateChunk(0, "x")]));
    const client = new SIEClient("http://localhost:8080");
    for await (const _ of client.streamGenerate("m", "hi", {
      maxNewTokens: 4,
      gpu: "eval-bench/l4",
    })) {
      // noop
    }
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers["X-SIE-Pool"]).toBe("eval-bench");
    expect(init.headers["X-SIE-MACHINE-PROFILE"]).toBe("l4");
  });

  // BUG 6 (MEDIUM-HIGH): a JSON parse error mid-stream raises a RequestError
  // inside consumeSseStream's for-await — that early throw must `cancel()` the
  // underlying body so the worker stops generating, not just release the lock.
  it("cancels the stream when a chunk fails to parse as JSON (BUG 6)", async () => {
    const { response, cancelled } = trackedSseResponse([
      `data: ${generateChunk(0, "Hello")}\n\n`,
      "data: {not valid json\n\n",
      `data: ${generateChunk(1, "never reached")}\n\n`,
    ]);
    mockFetch.mockResolvedValueOnce(response);

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamGenerate("m", "hi", { maxNewTokens: 8 });
    const first = await gen.next();
    expect(first.value?.text_delta).toBe("Hello");
    await expect(gen.next()).rejects.toBeInstanceOf(RequestError);
    expect(cancelled()).toBe(true);
  });

  // BUG 6 (MEDIUM-HIGH): a caller that `break`s out of the for-await must
  // cancel the underlying body so the GPU work behind it is torn down.
  it("cancels the stream when the caller breaks early (BUG 6)", async () => {
    const { response, cancelled } = trackedSseResponse([
      `data: ${generateChunk(0, "Hello")}\n\n`,
      `data: ${generateChunk(1, " world")}\n\n`,
      `data: ${generateChunk(2, "!")}\n\n`,
    ]);
    mockFetch.mockResolvedValueOnce(response);

    const client = new SIEClient("http://localhost:8080");
    for await (const chunk of client.streamGenerate("m", "hi", { maxNewTokens: 8 })) {
      expect(chunk.text_delta).toBe("Hello");
      break;
    }
    expect(cancelled()).toBe(true);
  });

  // BUG 13a (MEDIUM): provisioning on the streaming path with
  // waitForCapacity:false must reject with ProvisioningError.
  it("rejects 503 PROVISIONING with ProvisioningError on the streaming path (BUG 13a)", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }), {
        status: 503,
        headers: { "Content-Type": "application/json", "Retry-After": "7" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamGenerate(
      "m",
      "hi",
      { maxNewTokens: 8, waitForCapacity: false },
      undefined,
    );
    try {
      await gen.next();
      throw new Error("expected ProvisioningError");
    } catch (err) {
      expect(err).toBeInstanceOf(ProvisioningError);
      expect((err as ProvisioningError).retryAfter).toBe(7_000);
    }
  });

  // BUG 13b (MEDIUM): the streaming path must retry SAFE pre-execution
  // model-loading signals before opening the stream — parallel to generate().
  it("retries 503 MODEL_LOADING then streams the 200 when waitForCapacity:true (BUG 13b)", async () => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ error: { code: "MODEL_LOADING", message: "loading" } }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          }),
        )
        .mockResolvedValueOnce(
          sseResponse([
            generateChunk(0, "Hello"),
            generateChunk(1, "", { done: true, finish_reason: "stop" }),
          ]),
        );

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const chunks: GenerateChunk[] = [];
      const consume = (async () => {
        for await (const chunk of client.streamGenerate("m", "hi", {
          maxNewTokens: 8,
          waitForCapacity: true,
        })) {
          chunks.push(chunk);
        }
      })();
      // MODEL_LOADING_DEFAULT_DELAY = 5_000ms between attempts.
      await vi.advanceTimersByTimeAsync(5_000);
      await consume;

      expect(mockFetch.mock.calls.length).toBe(2);
      expect(chunks.map((c) => c.text_delta).join("")).toBe("Hello");
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries 503 MODEL_LOADING even when waitForCapacity:false (BUG 13b)", async () => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ error: { code: "MODEL_LOADING", message: "loading" } }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          }),
        )
        .mockResolvedValueOnce(
          sseResponse([
            generateChunk(0, "Loaded"),
            generateChunk(1, "", { done: true, finish_reason: "stop" }),
          ]),
        );

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const chunks: GenerateChunk[] = [];
      const consume = (async () => {
        for await (const chunk of client.streamGenerate(
          "m",
          "hi",
          { maxNewTokens: 8, waitForCapacity: false },
          undefined,
        )) {
          chunks.push(chunk);
        }
      })();
      await vi.advanceTimersByTimeAsync(5_000);
      await consume;

      expect(mockFetch.mock.calls.length).toBe(2);
      expect(chunks.map((c) => c.text_delta).join("")).toBe("Loaded");
    } finally {
      vi.useRealTimers();
    }
  });

  // A generic 503 is not a provisioning contract. It must stay terminal even
  // when the caller opts into capacity waiting.
  it("does NOT retry a generic 503 when waitForCapacity:true", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "service unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamGenerate(
      "m",
      "hi",
      { maxNewTokens: 8, waitForCapacity: true },
      undefined,
    );
    await expect(gen.next()).rejects.toBeTruthy();
    expect(mockFetch.mock.calls.length).toBe(1);
  });

  it("does NOT retry a generic 503 when waitForCapacity:false", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "service unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const gen = client.streamGenerate(
      "m",
      "hi",
      { maxNewTokens: 8, waitForCapacity: false },
      undefined,
    );
    await expect(gen.next()).rejects.toBeTruthy();
    expect(mockFetch.mock.calls.length).toBe(1);
  });

  // The provisioning Retry-After sleep must be abortable: an abort fired
  // DURING the sleep must surface SIEConnectionError promptly rather than
  // waiting out the full delay (which the next-iteration abort check would
  // only catch after the sleep resolved).
  it("aborts promptly during a provisioning Retry-After sleep", { timeout: 3_000 }, async () => {
    vi.useFakeTimers();
    try {
      // Always answer 503 PROVISIONING so the loop stays in the provisioning sleep.
      mockFetch.mockImplementation(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({ error: { code: "PROVISIONING", message: "provisioning" } }),
            {
              status: 503,
              headers: { "Content-Type": "application/json", "Retry-After": "300" },
            },
          ),
        ),
      );

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 600_000,
      });
      const controller = new AbortController();
      const gen = client.streamGenerate(
        "m",
        "hi",
        { maxNewTokens: 8, waitForCapacity: true },
        controller.signal,
      );

      const nextP = gen.next();
      // Let the first fetch resolve and the loop enter the long sleep.
      await vi.advanceTimersByTimeAsync(10);
      controller.abort();
      await expect(nextP).rejects.toBeInstanceOf(SIEConnectionError);
      // Only the single pre-sleep fetch was issued — we never woke from the
      // 300s sleep to fetch again.
      expect(mockFetch.mock.calls.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
