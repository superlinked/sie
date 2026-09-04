/**
 * Tests for the ``InputTooLongError`` short-circuit (#849).
 *
 * A 400 ``INPUT_TOO_LONG`` response on the extract path must:
 * - throw {@link InputTooLongError} immediately on the first response
 * - carry ``code === "INPUT_TOO_LONG"`` and ``statusCode === 400``
 * - expose ``model`` from caller context
 * - not be confused with generic {@link RequestError} so callers can
 *   branch on token-budget failures specifically
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { InputTooLongError, RequestError, SIEError } from "../src/errors.js";
import { handleError } from "../src/internal/parsing.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function inputTooLongResponse(message = "Input exceeds capacity (4096 tokens)"): Response {
  return new Response(
    JSON.stringify({ detail: { code: "INPUT_TOO_LONG", message, param: "input" } }),
    {
      status: 400,
      headers: { "Content-Type": "application/json" },
    },
  );
}

function validationErrorResponse(): Response {
  return new Response(
    JSON.stringify({ detail: { code: "VALIDATION_ERROR", message: "bad input" } }),
    { status: 400, headers: { "Content-Type": "application/json" } },
  );
}

describe("InputTooLongError class", () => {
  it("is a RequestError and SIEError", () => {
    const err = new InputTooLongError("test", { model: "x" });
    expect(err).toBeInstanceOf(RequestError);
    expect(err).toBeInstanceOf(SIEError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("InputTooLongError");
  });

  it("populates code, statusCode, and model", () => {
    const err = new InputTooLongError("too long", { model: "gliclass-large" });
    expect(err.code).toBe("INPUT_TOO_LONG");
    expect(err.statusCode).toBe(400);
    expect(err.model).toBe("gliclass-large");
  });

  it("defaults model to undefined", () => {
    const err = new InputTooLongError("too long");
    expect(err.model).toBeUndefined();
  });
});

describe("400 INPUT_TOO_LONG short-circuit", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    mockFetch.mockClear();
  });

  it("throws InputTooLongError on the first response", async () => {
    mockFetch.mockResolvedValueOnce(inputTooLongResponse());
    const client = new SIEClient("http://localhost:8080");

    await expect(
      client.extract("gliclass-large", { text: "hi" }, { labels: ["a", "b"] }),
    ).rejects.toThrow(InputTooLongError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("populates model and message", async () => {
    mockFetch.mockResolvedValueOnce(inputTooLongResponse("Too many tokens"));
    const client = new SIEClient("http://localhost:8080");

    try {
      await client.extract("gliclass-large", { text: "hi" }, { labels: ["a"] });
      throw new Error("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(InputTooLongError);
      const e = err as InputTooLongError;
      expect(e.model).toBe("gliclass-large");
      expect(e.code).toBe("INPUT_TOO_LONG");
      expect(e.statusCode).toBe(400);
      expect(e.message).toBe("Too many tokens");
      expect(e.param).toBe("input");
    }
  });

  it("does not consume any retry budget", async () => {
    mockFetch.mockResolvedValueOnce(inputTooLongResponse());
    const client = new SIEClient("http://localhost:8080", {
      provisionTimeout: 600_000,
    });

    const startTime = Date.now();
    await expect(
      client.extract("gliclass-large", { text: "hi" }, { labels: ["a"] }),
    ).rejects.toThrow(InputTooLongError);
    const elapsed = Date.now() - startTime;

    expect(elapsed).toBeLessThan(1000);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("does not classify other 400s as InputTooLongError", async () => {
    mockFetch.mockResolvedValueOnce(validationErrorResponse());
    const client = new SIEClient("http://localhost:8080");

    try {
      await client.extract("gliclass-large", { text: "hi" }, { labels: ["a"] });
      throw new Error("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(RequestError);
      expect(err).not.toBeInstanceOf(InputTooLongError);
      const e = err as RequestError;
      expect(e.code).toBe("VALIDATION_ERROR");
      expect(e.statusCode).toBe(400);
    }
  });
});

describe("handleError dispatch (direct)", () => {
  // Locks in the secondary fallthrough so reordering the conditions in
  // ``parsing.handleError`` cannot silently regress the typed dispatch.

  it("raises InputTooLongError on 400 + INPUT_TOO_LONG", async () => {
    await expect(handleError(inputTooLongResponse("Too many tokens"))).rejects.toMatchObject({
      name: "InputTooLongError",
      code: "INPUT_TOO_LONG",
      statusCode: 400,
      message: "Too many tokens",
      param: "input",
    });
  });

  it("does not classify other 400s as InputTooLongError", async () => {
    await expect(handleError(validationErrorResponse())).rejects.toSatisfy((err: unknown) => {
      return err instanceof RequestError && !(err instanceof InputTooLongError);
    });
  });
});
