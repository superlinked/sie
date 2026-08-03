/**
 * Tests for the cost-estimate dry run (`POST /v1/estimate`, #2435).
 *
 * The SDK's job on this route is narrow and load-bearing:
 * - send the target request VERBATIM inside `{endpoint, request}` — a body the
 *   SDK reshaped would be a quote for a request the caller never sends;
 * - return the gateway's projection unchanged, so the customer reads the same
 *   ceiling the metered path would hold;
 * - map "the book cannot price this" onto a typed {@link EstimateUnroutableError}
 *   rather than a generic 5xx, because that verdict is the request's, not the
 *   estimator's.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { EstimateUnroutableError, RequestError, SIEError, ServerError } from "../src/errors.js";
import { buildEstimateEnvelope, throwIfEstimateUnroutable } from "../src/internal/parsing.js";
import type { CostEstimate } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const QUOTE: CostEstimate = {
  endpoint: "/v1/encode/BAAI/bge-m3",
  identity: {
    model: "BAAI/bge-m3",
    profile: "default",
    operation: "encode",
    region: "us",
  },
  estimated_credits: 261,
  unit_ceilings: { input_tokens: 261 },
  applied_rates: [{ unit: "input_tokens", rate_numerator: 1, rate_denominator: 1 }],
  rate_book_version: "2026-07-26-production-bootstrap-v2",
  rate_book_sha256: "a".repeat(64),
  rounding_rule: "ceil-once-per-terminal-event",
  estimate_basis: "conservative pre-dispatch reservation ceiling",
  minimum_billed_units: null,
};

const UNROUTABLE = {
  detail: {
    code: "QUEUE_UNAVAILABLE",
    message:
      'missing rate for model="acme/unpriced", profile="default", operation="encode", region="us"',
  },
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * The error a rejecting call MUST produce.
 *
 * A bare `.catch(e => e as X)` binds the RESOLVED quote when the call
 * unexpectedly succeeds, so the assertions that follow read as type confusion
 * instead of "no error was thrown". This fails on the spot instead.
 */
async function rejection<T>(call: Promise<unknown>): Promise<T> {
  let resolved: unknown;
  let caught: unknown;
  let threw = false;
  try {
    resolved = await call;
  } catch (error) {
    threw = true;
    caught = error;
  }
  // Asserted OUTSIDE the try, so the failure cannot be caught and returned as
  // if it were the error under test.
  if (!threw) expect.fail(`expected a rejection, got ${JSON.stringify(resolved)}`);
  return caught as T;
}

function sentBody(): { endpoint: string; request: Record<string, unknown> } {
  const init = mockFetch.mock.calls[0][1] as RequestInit;
  return JSON.parse(init.body as string);
}

describe("estimate envelope", () => {
  it("carries the target body verbatim", () => {
    const request = { items: [{ text: "Hello" }], params: { output_types: ["dense"] } };
    expect(buildEstimateEnvelope("/v1/encode/BAAI/bge-m3", request)).toEqual({
      endpoint: "/v1/encode/BAAI/bge-m3",
      request,
    });
  });

  it("detaches the caller object so a later mutation cannot change what was priced", () => {
    const request: Record<string, unknown> = { items: [{ text: "Hello" }] };
    const envelope = buildEstimateEnvelope("/v1/encode/m", request);
    request.items = [];
    expect(envelope.request.items).toEqual([{ text: "Hello" }]);
  });

  it.each([
    ["v1/encode/m", { items: [] }],
    ["", { items: [] }],
    ["/v1/encode/m", ["not", "an", "object"]],
    ["/v1/encode/m", null],
  ])("rejects a malformed envelope client-side (%s)", (endpoint, request) => {
    expect(() =>
      buildEstimateEnvelope(endpoint as string, request as Record<string, unknown>),
    ).toThrow(TypeError);
  });
});

describe("EstimateUnroutableError class", () => {
  it("is a ServerError and SIEError", () => {
    const err = new EstimateUnroutableError("nope", "QUEUE_UNAVAILABLE");
    expect(err).toBeInstanceOf(ServerError);
    expect(err).toBeInstanceOf(SIEError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("EstimateUnroutableError");
    expect(err.code).toBe("QUEUE_UNAVAILABLE");
    expect(err.statusCode).toBe(503);
  });
});

describe("throwIfEstimateUnroutable", () => {
  it("throws on an unpriced identity and preserves the planner's reason", async () => {
    await expect(throwIfEstimateUnroutable(jsonResponse(503, UNROUTABLE))).rejects.toThrow(
      /acme\/unpriced/,
    );
  });

  it("is a no-op for a retryable billing-capacity 503", async () => {
    const capacity = {
      detail: {
        code: "BILLING_CAPACITY_UNAVAILABLE",
        message: "audio billing preflight is at capacity",
      },
    };
    await expect(throwIfEstimateUnroutable(jsonResponse(503, capacity))).resolves.toBeUndefined();
  });

  it("is a no-op for retryable provisioning capacity", async () => {
    const capacity = {
      detail: {
        code: "PROVISIONING",
        message: "a matching worker is scaling from zero",
      },
    };
    await expect(throwIfEstimateUnroutable(jsonResponse(503, capacity))).resolves.toBeUndefined();
  });

  it("is a no-op for non-503 responses", async () => {
    await expect(
      throwIfEstimateUnroutable(jsonResponse(400, { detail: { code: "QUEUE_UNAVAILABLE" } })),
    ).resolves.toBeUndefined();
  });
});

describe("client.estimate", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("posts the envelope to /v1/estimate and returns the quote", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(200, QUOTE));
    const client = new SIEClient("http://localhost:8080");

    const quote = await client.estimate("/v1/encode/BAAI/bge-m3", {
      items: [{ text: "Hello" }],
    });

    expect(quote).toEqual(QUOTE);
    expect(quote.estimated_credits).toBe(261);
    expect(quote.unit_ceilings).toEqual({ input_tokens: 261 });
    expect(quote.applied_rates[0].rate_denominator).toBe(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://localhost:8080/v1/estimate");
    expect((mockFetch.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect(sentBody()).toEqual({
      endpoint: "/v1/encode/BAAI/bge-m3",
      request: { items: [{ text: "Hello" }] },
    });
  });

  it("passes an OpenAI-compatible body through untouched (model stays in the body)", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(200, QUOTE));
    const client = new SIEClient("http://localhost:8080");
    const body = {
      model: "Qwen/Qwen3.6-27B",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 64,
      n: 3,
    };

    await client.estimate("/v1/chat/completions", body);

    expect(sentBody()).toEqual({ endpoint: "/v1/chat/completions", request: body });
  });

  it("throws EstimateUnroutableError for an unpriced identity", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(503, UNROUTABLE));
    const client = new SIEClient("http://localhost:8080");

    await expect(
      client.estimate("/v1/encode/acme/unpriced", { items: [{ text: "Hello" }] }),
    ).rejects.toThrowError(EstimateUnroutableError);

    mockFetch.mockResolvedValueOnce(jsonResponse(503, UNROUTABLE));
    const error = await rejection<EstimateUnroutableError>(
      client.estimate("/v1/encode/acme/unpriced", { items: [{ text: "Hello" }] }),
    );
    expect(error.code).toBe("QUEUE_UNAVAILABLE");
    expect(error.statusCode).toBe(503);
    expect(error.message).toContain("acme/unpriced");
    expect(error).toBeInstanceOf(ServerError);
  });

  it("leaves a retryable billing-capacity 503 as a plain ServerError", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(503, {
        detail: {
          code: "BILLING_CAPACITY_UNAVAILABLE",
          message: "audio billing preflight is at capacity",
        },
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await rejection<ServerError>(
      client.estimate("/v1/extract/audio/asr", { items: [{ text: "x" }] }),
    );
    expect(error).toBeInstanceOf(ServerError);
    expect(error).not.toBeInstanceOf(EstimateUnroutableError);
    expect(error.code).toBe("BILLING_CAPACITY_UNAVAILABLE");
  });

  it("surfaces an unroutable model as a 404 RequestError, not a quote", async () => {
    // The gateway checks routability, not just priceability: a model this data
    // plane does not serve answers the live 404 even when the rate book prices
    // it. A regression that turned that into a 200 quote would otherwise pass.
    mockFetch.mockResolvedValueOnce(
      jsonResponse(404, {
        detail: { code: "MODEL_NOT_FOUND", message: 'Model "acme/absent" not found.' },
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await rejection<RequestError>(
      client.estimate("/v1/encode/acme/absent", { items: [{ text: "Hello" }] }),
    );
    expect(error).toBeInstanceOf(RequestError);
    expect(error).not.toBeInstanceOf(EstimateUnroutableError);
    expect(error.statusCode).toBe(404);
    expect(error.code).toBe("MODEL_NOT_FOUND");
  });

  it("surfaces a malformed target body as a RequestError", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(400, {
        detail: { code: "INVALID_REQUEST", message: "'items' must be an array" },
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await rejection<RequestError>(
      client.estimate("/v1/encode/BAAI/bge-m3", { items: "nope" }),
    );
    expect(error).toBeInstanceOf(RequestError);
    expect(error.code).toBe("INVALID_REQUEST");
    expect(error.statusCode).toBe(400);
  });

  it("returns a sealed rate card with its minimum-billed floor", async () => {
    const sealed: CostEstimate = {
      ...QUOTE,
      endpoint: "/v1/generate/org-42/support-gen",
      identity: {
        model: "custom.l4",
        profile: "default",
        operation: "sealed",
        region: "us",
      },
      unit_ceilings: { gpu_second: 130 },
      applied_rates: [{ unit: "gpu_second", rate_numerator: 7, rate_denominator: 1 }],
      estimated_credits: 910,
      minimum_billed_units: { gpu_second: 1 },
      estimate_basis: "sealed custom-lane rate card; duration is execution-dependent",
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(200, sealed));
    const client = new SIEClient("http://localhost:8080");

    const quote = await client.estimate("/v1/generate/org-42/support-gen", {
      prompt: "hi",
      max_new_tokens: 32,
    });

    expect(quote.minimum_billed_units).toEqual({ gpu_second: 1 });
    expect(quote.applied_rates[0].unit).toBe("gpu_second");
    expect(quote.estimate_basis).toContain("duration");
  });

  it("rejects a malformed envelope before touching the network", async () => {
    const client = new SIEClient("http://localhost:8080");
    await expect(client.estimate("v1/encode/m", { items: [] })).rejects.toThrow(TypeError);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("sends the bearer token like every other authenticated route", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(200, QUOTE));
    const client = new SIEClient("http://localhost:8080", { apiKey: "sk-sie-test" });

    await client.estimate("/v1/encode/BAAI/bge-m3", { items: [{ text: "Hello" }] });

    const headers = (mockFetch.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer sk-sie-test");
    expect(headers["Content-Type"]).toBe("application/json");
  });
});
