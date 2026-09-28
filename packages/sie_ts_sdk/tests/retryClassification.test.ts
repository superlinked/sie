/**
 * Retry classification, timeouts and partial batches through the real client.
 *
 * `fetch` is the stubbed transport: each case replays the shared
 * `packages/wire-fixtures/retry_classification.json` response followed by a
 * success, so every decision runs through the production retry loops. The
 * Python SDK asserts the same table in `test_retry_classification.py`.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { IncompleteBatchError, SIEConnectionError, SIEError } from "../src/errors.js";
import { getRetryAfter } from "../src/internal/retry.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

interface ResponseCase {
  name: string;
  status: number;
  headers: Record<string, string>;
  body: unknown;
  idempotent: "retry" | "terminal";
  generation: "retry" | "terminal";
}

interface RetryAfterCase {
  header: string;
  seconds: number | null | "future";
}

const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../wire-fixtures/retry_classification.json", import.meta.url)),
    "utf8",
  ),
) as { responses: ResponseCase[]; retry_after: RetryAfterCase[] };

type Operation = "encode" | "generate" | "chatCompletions";
const OPERATIONS: Operation[] = ["encode", "generate", "chatCompletions"];

function successFor(operation: Operation): Response {
  if (operation === "encode") {
    return new Response(
      packMessage({ model: "m", items: [{ id: "a", dense: { values: new Float32Array([0.5]) } }] }),
      { status: 200, headers: { "Content-Type": "application/msgpack" } },
    );
  }
  const body =
    operation === "generate"
      ? { model: "m", text: "ok", finish_reason: "stop" }
      : { id: "chat-1", object: "chat.completion", model: "m", choices: [] };
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function fixtureResponse(testCase: ResponseCase): Response {
  return new Response(JSON.stringify(testCase.body), {
    status: testCase.status,
    headers: { "Content-Type": "application/json", ...testCase.headers },
  });
}

function call(
  client: SIEClient,
  operation: Operation,
  options: { timeoutMs?: number } = {},
): Promise<unknown> {
  if (operation === "encode") {
    return client.encode("m", { id: "a", text: "hi" });
  }
  if (operation === "generate") {
    return client.generate("m", "hi", { maxNewTokens: 4, ...options });
  }
  return client.chatCompletions(
    { model: "m", messages: [{ role: "user", content: "hi" }] },
    options,
  );
}

async function settle(promise: Promise<unknown>, advanceMs: number) {
  const outcome = promise.then(
    (value) => ({ ok: true as const, value }),
    (error: unknown) => ({ ok: false as const, error }),
  );
  await vi.advanceTimersByTimeAsync(advanceMs);
  return outcome;
}

function fetchFailure(cause: unknown): TypeError {
  return new TypeError("fetch failed", { cause });
}

function codedError(code: string): Error {
  return Object.assign(new Error(code), { code });
}

describe("shared retry classification table", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  for (const operation of OPERATIONS) {
    const operationClass = operation === "encode" ? "idempotent" : "generation";
    for (const testCase of fixture.responses) {
      it(`${operation}: ${testCase.name} is ${testCase[operationClass]}`, async () => {
        mockFetch
          .mockResolvedValueOnce(fixtureResponse(testCase))
          .mockResolvedValueOnce(successFor(operation));
        const client = new SIEClient("http://localhost:8080");

        const outcome = await settle(call(client, operation), 10_000);

        if (testCase[operationClass] === "retry") {
          expect(outcome.ok).toBe(true);
          expect(mockFetch).toHaveBeenCalledTimes(2);
        } else {
          expect(outcome.ok).toBe(false);
          expect(outcome.ok ? undefined : outcome.error).toBeInstanceOf(SIEError);
          expect(mockFetch).toHaveBeenCalledTimes(1);
        }
      });
    }
  }

  for (const testCase of fixture.retry_after) {
    it(`parses Retry-After ${JSON.stringify(testCase.header)}`, () => {
      vi.useRealTimers();
      const parsed = getRetryAfter(testCase.header);
      if (testCase.seconds === "future") {
        expect(parsed).toBeGreaterThan(0);
      } else if (testCase.seconds === null) {
        expect(parsed).toBeUndefined();
      } else {
        expect(parsed).toBe(testCase.seconds * 1000);
      }
    });
  }

  it("retries queue backpressure before a stream opens", async () => {
    const backpressure = fixture.responses.find(
      (testCase) => testCase.name === "openai_transport_failure_backpressure",
    );
    if (!backpressure) throw new Error("fixture case missing");
    mockFetch.mockResolvedValueOnce(fixtureResponse(backpressure)).mockResolvedValueOnce(
      new Response('data: {"request_id":"r","seq":0,"text_delta":"ok","done":true}\n\n', {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    const chunks: unknown[] = [];
    const consumed = (async () => {
      for await (const chunk of client.streamGenerate("m", "hi", { maxNewTokens: 4 })) {
        chunks.push(chunk);
      }
    })();
    await vi.advanceTimersByTimeAsync(10_000);
    await consumed;

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(chunks).toHaveLength(1);
  });
});

describe("fetch failure classification", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each([
    ["DNS name not found", codedError("ENOTFOUND")],
    ["expired certificate", codedError("CERT_HAS_EXPIRED")],
    ["untrusted certificate", codedError("UNABLE_TO_VERIFY_LEAF_SIGNATURE")],
    ["TLS protocol mismatch", codedError("ERR_SSL_WRONG_VERSION_NUMBER")],
  ])("does not retry a permanent failure: %s", async (_label, cause) => {
    mockFetch
      .mockRejectedValueOnce(fetchFailure(cause))
      .mockResolvedValueOnce(successFor("encode"));
    const client = new SIEClient("https://gateway.example.test");

    const outcome = await settle(call(client, "encode"), 10_000);

    expect(outcome.ok).toBe(false);
    const error = outcome.ok ? undefined : outcome.error;
    expect(error).toBeInstanceOf(SIEConnectionError);
    expect((error as SIEConnectionError).kind).toBe("other");
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["connection refused", codedError("ECONNREFUSED")],
    ["temporary DNS failure", codedError("EAI_AGAIN")],
    ["connect timeout", codedError("UND_ERR_CONNECT_TIMEOUT")],
    [
      "every address refused",
      Object.assign(new AggregateError([codedError("ECONNREFUSED"), codedError("ENETUNREACH")]), {
        code: "ECONNREFUSED",
      }),
    ],
    ["no cause", undefined],
  ])("retries a transient connection failure: %s", async (_label, cause) => {
    mockFetch
      .mockRejectedValueOnce(fetchFailure(cause))
      .mockResolvedValueOnce(successFor("encode"));
    const client = new SIEClient("http://localhost:8080");

    const outcome = await settle(call(client, "encode"), 10_000);

    expect(outcome.ok).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("does not retry a response timeout reported by the runtime", async () => {
    mockFetch
      .mockRejectedValueOnce(fetchFailure(codedError("UND_ERR_HEADERS_TIMEOUT")))
      .mockResolvedValueOnce(successFor("encode"));
    const client = new SIEClient("http://localhost:8080");

    const outcome = await settle(call(client, "encode"), 10_000);

    expect(outcome.ok).toBe(false);
    expect((outcome.ok ? undefined : outcome.error) as SIEConnectionError).toMatchObject({
      kind: "timeout",
    });
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe("per-call generation timeout", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
    mockFetch.mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(["generate", "chatCompletions"] as const)(
    "%s waits for the per-call timeoutMs instead of the client default",
    async (operation) => {
      const client = new SIEClient("http://localhost:8080", { timeoutMs: 1_000 });

      const outcome = settle(call(client, operation, { timeoutMs: 5_000 }), 0);
      await vi.advanceTimersByTimeAsync(1_000);
      expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(4_000);
      const settled = await outcome;

      expect(settled.ok).toBe(false);
      expect((settled.ok ? undefined : settled.error) as SIEConnectionError).toMatchObject({
        kind: "timeout",
      });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    },
  );

  it("defaults to a per-attempt timeout longer than the gateway's 120 s deadline", async () => {
    const client = new SIEClient("http://localhost:8080");

    const outcome = settle(call(client, "generate"), 0);
    await vi.advanceTimersByTimeAsync(121_000);
    expect(mockFetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(30_000);
    const settled = await outcome;

    expect(settled.ok).toBe(false);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe("partial batches", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("keeps the returned results on IncompleteBatchError", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(
        packMessage({
          model: "m",
          items: [
            {
              id: "b",
              entities: [],
              error: { code: "INPUT_TOO_LONG", message: "too long" },
            },
          ],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/msgpack", "X-SIE-Request-ID": "req-1" },
        },
      ),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await client
      .extract(
        "m",
        [
          { id: "a", text: "x" },
          { id: "b", text: "y" },
        ],
        { labels: ["person"] },
      )
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(IncompleteBatchError);
    const incomplete = error as IncompleteBatchError;
    expect(incomplete.missingIds).toEqual(["a"]);
    expect(incomplete.results).toHaveLength(1);
    expect(incomplete.results[0]).toMatchObject({
      id: "b",
      error: { code: "INPUT_TOO_LONG", message: "too long" },
      request: { id: "req-1" },
    });
  });
});
