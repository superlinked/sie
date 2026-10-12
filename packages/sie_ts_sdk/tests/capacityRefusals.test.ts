/**
 * Capacity refusals: the final attempt at the end of the provisioning budget,
 * and the request metadata a refusal leaves on the thrown error.
 *
 * A `503` capacity refusal can be the local refusal that answers a failed
 * remote attempt, and its `X-SIE-*` headers then say which side answered. The
 * thrown error carries them as `request`; when the budget runs out, those of
 * the last refusal. Mirrors the Python SDK's `test_provisioning.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  LoraLoadingError,
  ModelLoadingError,
  ProvisioningError,
  ResourceExhaustedError,
} from "../src/errors.js";
import {
  DEFAULT_PROVISION_TIMEOUT,
  LORA_LOADING_MAX_RETRIES,
  RESOURCE_EXHAUSTED_MAX_RETRIES,
} from "../src/internal/constants.js";
import { handleError } from "../src/internal/parsing.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const REFUSAL_HEADERS = {
  "X-SIE-Request-ID": "req-refused",
  "X-SIE-Served-By": "local",
  "X-SIE-Fallback-Reason": "provisioning",
  "X-SIE-Fallback-Error": "QUEUE_FULL",
};
const REFUSAL_METADATA = {
  id: "req-refused",
  servedBy: "local",
  fallbackReason: "provisioning",
  fallbackError: "QUEUE_FULL",
};
const CAPACITY_READY_AT_MS = 700_000;

type Operation = "encode" | "generate" | "streamGenerate";
const OPERATIONS: Operation[] = ["encode", "generate", "streamGenerate"];

function refusal(code: string, retryAfter = "900"): Response {
  return new Response(
    JSON.stringify({ error: { code, message: "Capacity for model 'm' is starting." } }),
    {
      status: 503,
      headers: {
        "Content-Type": "application/json",
        "Retry-After": retryAfter,
        ...REFUSAL_HEADERS,
      },
    },
  );
}

function success(operation: Operation): Response {
  if (operation === "encode") {
    return new Response(packMessage({ items: [{ dense: { values: new Float32Array([0.1]) } }] }), {
      status: 200,
      headers: { "Content-Type": "application/msgpack" },
    });
  }
  if (operation === "generate") {
    return new Response(JSON.stringify({ model: "m", text: "ok", finish_reason: "stop" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  const chunk = { request_id: "r", seq: 0, text_delta: "ok", done: true };
  return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function call(
  client: SIEClient,
  operation: Operation,
  waitForCapacity?: boolean,
): Promise<unknown> {
  if (operation === "encode") return client.encode("m", { text: "hi" }, { waitForCapacity });
  if (operation === "generate") {
    return client.generate("m", "hi", { maxNewTokens: 4, waitForCapacity });
  }
  const chunks: unknown[] = [];
  for await (const chunk of client.streamGenerate("m", "hi", {
    maxNewTokens: 4,
    waitForCapacity,
  })) {
    chunks.push(chunk);
  }
  return chunks;
}

function settle(promise: Promise<unknown>): Promise<{ error?: unknown }> {
  return promise.then(
    () => ({}),
    (error: unknown) => ({ error }),
  );
}

describe("the final attempt at the end of the provisioning budget", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(OPERATIONS)(
    "%s uses capacity that became ready during the last wait",
    async (operation) => {
      const start = Date.now();
      mockFetch.mockImplementation(async () =>
        Date.now() - start >= CAPACITY_READY_AT_MS ? success(operation) : refusal("PROVISIONING"),
      );
      const client = new SIEClient("http://localhost:8080");

      const settled = settle(call(client, operation));
      await vi.advanceTimersByTimeAsync(DEFAULT_PROVISION_TIMEOUT);

      expect(await settled).toEqual({});
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(Date.now() - start).toBe(DEFAULT_PROVISION_TIMEOUT);
    },
  );

  it.each(
    OPERATIONS.flatMap((operation) => [
      { operation, code: "PROVISIONING", type: ProvisioningError },
      { operation, code: "MODEL_LOADING", type: ModelLoadingError },
    ]),
  )(
    "$operation throws the $code refusal of the final attempt with its metadata",
    async ({ operation, code, type }) => {
      mockFetch.mockImplementation(async () => refusal(code));
      const client = new SIEClient("http://localhost:8080");

      const settled = settle(call(client, operation));
      await vi.advanceTimersByTimeAsync(DEFAULT_PROVISION_TIMEOUT);
      const { error } = await settled;

      expect(error).toBeInstanceOf(type);
      expect((error as ProvisioningError | ModelLoadingError).request).toEqual(REFUSAL_METADATA);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    },
  );
});

describe("request metadata on capacity refusals", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(OPERATIONS)("%s without waiting throws ProvisioningError with it", async (operation) => {
    mockFetch.mockImplementation(async () => refusal("PROVISIONING", "30"));
    const client = new SIEClient("http://localhost:8080");

    const { error } = await settle(call(client, operation, false));

    expect(error).toBeInstanceOf(ProvisioningError);
    expect(error).toMatchObject({ retryAfter: 30_000, request: REFUSAL_METADATA });
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("handleError keeps it on a ProvisioningError", async () => {
    const { error } = await settle(handleError(refusal("PROVISIONING", "60")));

    expect(error).toBeInstanceOf(ProvisioningError);
    expect(error).toMatchObject({ retryAfter: 60_000, request: REFUSAL_METADATA });
  });

  it("encode throws LoraLoadingError with the last refusal's", async () => {
    mockFetch.mockImplementation(async () => refusal("LORA_LOADING", "1"));
    const client = new SIEClient("http://localhost:8080");

    const settled = settle(client.encode("m", { text: "hi" }));
    await vi.advanceTimersByTimeAsync(LORA_LOADING_MAX_RETRIES * 1_000);
    const { error } = await settled;

    expect(error).toBeInstanceOf(LoraLoadingError);
    expect((error as LoraLoadingError).request).toEqual(REFUSAL_METADATA);
    expect(mockFetch).toHaveBeenCalledTimes(LORA_LOADING_MAX_RETRIES + 1);
  });

  it("encode throws ResourceExhaustedError with the last refusal's", async () => {
    mockFetch.mockImplementation(async () => refusal("RESOURCE_EXHAUSTED", "1"));
    const client = new SIEClient("http://localhost:8080");

    const settled = settle(client.encode("m", { text: "hi" }));
    await vi.advanceTimersByTimeAsync(60_000);
    const { error } = await settled;

    expect(error).toBeInstanceOf(ResourceExhaustedError);
    expect((error as ResourceExhaustedError).request).toEqual(REFUSAL_METADATA);
    expect(mockFetch).toHaveBeenCalledTimes(RESOURCE_EXHAUSTED_MAX_RETRIES + 1);
  });
});

describe("capacity error constructors", () => {
  it("keep their positional arguments and default to no request metadata", () => {
    expect(new ProvisioningError("no capacity", "l4", 30_000, "model")).toMatchObject({
      message: "no capacity",
      gpu: "l4",
      retryAfter: 30_000,
      param: "model",
      request: undefined,
    });
    expect(new ModelLoadingError("loading", "m", "model")).toMatchObject({
      message: "loading",
      model: "m",
      param: "model",
      request: undefined,
    });
    expect(new LoraLoadingError("lora loading", "adapter", "m", "model")).toMatchObject({
      message: "lora loading",
      lora: "adapter",
      model: "m",
      param: "model",
      request: undefined,
    });
  });
});
