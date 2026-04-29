/**
 * Tests for the ``ModelLoadFailedError`` short-circuit (sie-test#85).
 *
 * A 502 ``MODEL_LOAD_FAILED`` response must:
 * - throw {@link ModelLoadFailedError} immediately on the first response
 * - never engage the ``MODEL_LOADING`` retry budget (no sleep, no retry)
 * - expose the structured ``errorClass`` / ``permanent`` / ``attempts``
 *   fields from the server payload
 *
 * The 503 ``MODEL_LOADING`` retry behavior must be unchanged.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { ModelLoadFailedError, ModelLoadingError, SIEError } from "../src/errors.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function loadFailedResponse(
  options: {
    errorClass?: string;
    permanent?: boolean;
    attempts?: number;
    message?: string;
  } = {},
): Response {
  const body = JSON.stringify({
    detail: {
      code: "MODEL_LOAD_FAILED",
      message: options.message ?? "Model 'org/test' failed to load",
      error_class: options.errorClass ?? "GATED",
      permanent: options.permanent ?? true,
      attempts: options.attempts ?? 1,
    },
  });
  return new Response(body, {
    status: 502,
    headers: { "Content-Type": "application/json" },
  });
}

function modelLoadingResponse(): Response {
  // ``Retry-After: 1`` is the smallest positive integer the parser accepts;
  // values <=0 fall through to ``MODEL_LOADING_DEFAULT_DELAY`` (5s) which
  // would blow past the test timeout.
  return new Response(JSON.stringify({ detail: { code: "MODEL_LOADING", message: "loading" } }), {
    status: 503,
    headers: {
      "Content-Type": "application/json",
      "Retry-After": "1",
    },
  });
}

function encodeSuccessResponse(): Response {
  return new Response(
    packMessage({ items: [{ dense: { values: new Float32Array([0.1, 0.2]) } }] }),
    {
      status: 200,
      headers: { "Content-Type": "application/msgpack" },
    },
  );
}

describe("ModelLoadFailedError class", () => {
  it("is a SIEError but not a ModelLoadingError", () => {
    const err = new ModelLoadFailedError("test", { model: "x" });
    expect(err).toBeInstanceOf(SIEError);
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(ModelLoadingError);
    expect(err.name).toBe("ModelLoadFailedError");
  });

  it("defaults permanent=true and attempts=1", () => {
    const err = new ModelLoadFailedError("test");
    expect(err.permanent).toBe(true);
    expect(err.attempts).toBe(1);
  });

  it("preserves all structured fields", () => {
    const err = new ModelLoadFailedError("gated", {
      model: "org/m",
      errorClass: "GATED",
      permanent: true,
      attempts: 3,
    });
    expect(err.model).toBe("org/m");
    expect(err.errorClass).toBe("GATED");
    expect(err.permanent).toBe(true);
    expect(err.attempts).toBe(3);
  });
});

describe("502 MODEL_LOAD_FAILED short-circuit", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    mockFetch.mockClear();
  });

  it("throws ModelLoadFailedError on the first response", async () => {
    mockFetch.mockResolvedValueOnce(loadFailedResponse());
    const client = new SIEClient("http://localhost:8080");

    await expect(client.encode("org/test", { text: "hi" })).rejects.toThrow(ModelLoadFailedError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("populates errorClass / permanent / attempts from payload", async () => {
    mockFetch.mockResolvedValueOnce(
      loadFailedResponse({
        errorClass: "DEPENDENCY",
        permanent: true,
        attempts: 4,
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    try {
      await client.encode("org/test", { text: "hi" });
      throw new Error("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ModelLoadFailedError);
      const e = err as ModelLoadFailedError;
      expect(e.errorClass).toBe("DEPENDENCY");
      expect(e.permanent).toBe(true);
      expect(e.attempts).toBe(4);
      expect(e.model).toBe("org/test");
    }
  });

  it("does not consume the MODEL_LOADING retry budget", async () => {
    // Even with a long provision timeout, a 502 must surface immediately.
    mockFetch.mockResolvedValueOnce(loadFailedResponse());
    const client = new SIEClient("http://localhost:8080", {
      provisionTimeout: 600_000, // 10 min — would burn 5 min on 503 loop
    });

    const startTime = Date.now();
    await expect(client.encode("org/test", { text: "hi" })).rejects.toThrow(ModelLoadFailedError);
    const elapsed = Date.now() - startTime;

    // Must complete in well under a second; no retry sleeps consumed.
    expect(elapsed).toBeLessThan(1000);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("does not affect 503 MODEL_LOADING retry behavior", async () => {
    // Fake timers so the 1s Retry-After sleep doesn't blow the test
    // budget. Vitest's ``vi.useFakeTimers`` advances ``setTimeout``
    // synchronously when paired with ``vi.runAllTimersAsync`` below.
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(modelLoadingResponse())
        .mockResolvedValueOnce(modelLoadingResponse())
        .mockResolvedValueOnce(encodeSuccessResponse());

      const client = new SIEClient("http://localhost:8080");
      const promise = client.encode("bge-m3", { text: "hi" });
      // Drain pending timers (the sleep() between retries) without
      // wall-clock waits.
      await vi.runAllTimersAsync();
      const result = await promise;
      expect(result.dense).toBeDefined();
      expect(mockFetch).toHaveBeenCalledTimes(3);
    } finally {
      vi.useRealTimers();
    }
  });
});
