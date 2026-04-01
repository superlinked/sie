/**
 * Error handling tests focused on real user scenarios.
 *
 * These tests verify that users can:
 * 1. Catch specific error types in try/catch blocks
 * 2. Access error properties for debugging and logging
 * 3. Distinguish between different failure modes
 * 4. Build proper error handling in their applications
 */

import { describe, expect, it } from "vitest";
import {
  LoraLoadingError,
  PoolError,
  ProvisioningError,
  RequestError,
  SIEConnectionError,
  SIEError,
  ServerError,
} from "../src/errors.js";

describe("Error hierarchy for try/catch patterns", () => {
  it("allows catching all SIE errors with base class", () => {
    // User scenario: "I want to catch any SDK error and log it"
    const errors = [
      new SIEConnectionError("network failed"),
      new RequestError("bad request", "INVALID", 400),
      new ServerError("server error", "INTERNAL", 500),
      new ProvisioningError("provisioning", "l4"),
      new PoolError("pool expired", "my-pool", "expired"),
      new LoraLoadingError("lora loading", "adapter1", "bge-m3"),
    ];

    for (const error of errors) {
      expect(error).toBeInstanceOf(SIEError);
      expect(error).toBeInstanceOf(Error);
      // User can access .message on any error
      expect(typeof error.message).toBe("string");
      expect(error.message.length).toBeGreaterThan(0);
    }
  });

  it("allows catching specific error types", () => {
    // User scenario: "I want to handle network errors differently from API errors"
    const connectionError = new SIEConnectionError("DNS lookup failed");
    const requestError = new RequestError("Invalid model", "INVALID_MODEL", 404);

    expect(connectionError).toBeInstanceOf(SIEConnectionError);
    expect(connectionError).not.toBeInstanceOf(RequestError);

    expect(requestError).toBeInstanceOf(RequestError);
    expect(requestError).not.toBeInstanceOf(SIEConnectionError);
  });
});

describe("SIEConnectionError - network failures", () => {
  it("provides clear message for debugging", () => {
    // User scenario: "My request failed, what went wrong?"
    const error = new SIEConnectionError("Connection refused: ECONNREFUSED 127.0.0.1:8080");

    expect(error.message).toContain("ECONNREFUSED");
    expect(error.name).toBe("SIEConnectionError");
  });

  it("works with standard Error patterns", () => {
    const error = new SIEConnectionError("timeout");

    // Users expect standard Error properties
    expect(error.stack).toBeDefined();
    expect(error.toString()).toContain("SIEConnectionError");
  });
});

describe("RequestError - client errors (4xx)", () => {
  it("provides error code for programmatic handling", () => {
    // User scenario: "I want to show different messages for different error types"
    const invalidModel = new RequestError("Model 'foo' not found", "INVALID_MODEL", 404);
    const validationError = new RequestError("text field required", "VALIDATION_ERROR", 422);

    expect(invalidModel.code).toBe("INVALID_MODEL");
    expect(invalidModel.statusCode).toBe(404);

    expect(validationError.code).toBe("VALIDATION_ERROR");
    expect(validationError.statusCode).toBe(422);
  });

  it("allows conditional handling based on status code", () => {
    // User scenario: "If 401, redirect to login; if 400, show validation error"
    const unauthorized = new RequestError("Invalid API key", "UNAUTHORIZED", 401);
    const badRequest = new RequestError("Missing required field", "BAD_REQUEST", 400);

    // User code pattern:
    function handleError(error: RequestError): string {
      if (error.statusCode === 401) {
        return "Please check your API key";
      }
      if (error.statusCode === 400) {
        return `Invalid request: ${error.message}`;
      }
      return "Unknown error";
    }

    expect(handleError(unauthorized)).toBe("Please check your API key");
    expect(handleError(badRequest)).toContain("Missing required field");
  });

  it("works without optional parameters", () => {
    // Some errors might not have code/status from server
    const error = new RequestError("Something went wrong");

    expect(error.code).toBeUndefined();
    expect(error.statusCode).toBeUndefined();
    expect(error.message).toBe("Something went wrong");
  });
});

describe("ServerError - server errors (5xx)", () => {
  it("indicates server-side issues for retry logic", () => {
    // User scenario: "If server error, I should retry; if client error, I shouldn't"
    const serverError = new ServerError("Internal error", "INTERNAL_ERROR", 500);
    const requestError = new RequestError("Bad input", "BAD_REQUEST", 400);

    function shouldRetry(error: SIEError): boolean {
      return error instanceof ServerError;
    }

    expect(shouldRetry(serverError)).toBe(true);
    expect(shouldRetry(requestError)).toBe(false);
  });

  it("provides status code for logging", () => {
    const errors = [
      new ServerError("Internal error", "INTERNAL", 500),
      new ServerError("Bad gateway", "BAD_GATEWAY", 502),
      new ServerError("Service unavailable", "UNAVAILABLE", 503),
    ];

    for (const error of errors) {
      expect(error.statusCode).toBeGreaterThanOrEqual(500);
      expect(error.statusCode).toBeLessThan(600);
    }
  });
});

describe("ProvisioningError - capacity scaling", () => {
  it("provides retry-after hint for backoff", async () => {
    // User scenario: "GPU is scaling up, when should I retry?"
    const error = new ProvisioningError(
      "GPU l4 is provisioning, please retry",
      "l4",
      5000, // retry after 5 seconds
    );

    expect(error.gpu).toBe("l4");
    expect(error.retryAfter).toBe(5000);

    // User can implement backoff:
    async function waitAndRetry(err: ProvisioningError): Promise<number> {
      const delay = err.retryAfter ?? 10000; // default 10s
      // await sleep(delay);
      return delay;
    }

    await expect(waitAndRetry(error)).resolves.toBe(5000);
  });

  it("handles missing retry-after gracefully", () => {
    // Server might not always provide retry-after
    const error = new ProvisioningError("No capacity available", "a100-80gb");

    expect(error.gpu).toBe("a100-80gb");
    expect(error.retryAfter).toBeUndefined();
  });

  it("identifies which GPU was requested", () => {
    // User scenario: "I requested multiple GPU types, which one failed?"
    const error = new ProvisioningError("Provisioning in progress", "l4");

    expect(error.gpu).toBe("l4");
  });
});

describe("PoolError - resource pool issues", () => {
  it("identifies the problematic pool", () => {
    // User scenario: "Which pool failed?"
    const error = new PoolError("Pool has expired", "eval-bench", "expired");

    expect(error.poolName).toBe("eval-bench");
    expect(error.state).toBe("expired");
  });

  it("helps diagnose pool lifecycle issues", () => {
    // User scenario: "Why did my pool fail?"
    const expired = new PoolError("Pool expired after 1 hour", "prod-pool", "expired");
    const pending = new PoolError("Pool still provisioning", "new-pool", "pending");
    const notFound = new PoolError("Pool not found", "missing-pool");

    expect(expired.state).toBe("expired");
    expect(pending.state).toBe("pending");
    expect(notFound.state).toBeUndefined(); // pool might not exist
  });
});

describe("LoraLoadingError - adapter loading", () => {
  it("identifies which LoRA and model failed", () => {
    // User scenario: "I'm using fine-tuned models, which adapter failed?"
    const error = new LoraLoadingError(
      "LoRA adapter still loading after 30s",
      "custom-ner-adapter",
      "bge-m3",
    );

    expect(error.lora).toBe("custom-ner-adapter");
    expect(error.model).toBe("bge-m3");
  });

  it("allows conditional handling for LoRA vs other errors", () => {
    // User scenario: "LoRA loading is expected, I'll just wait longer"
    const loraError = new LoraLoadingError("Loading", "adapter", "model");
    const serverError = new ServerError("Failed", "INTERNAL", 500);

    function getRetryDelay(error: SIEError): number {
      if (error instanceof LoraLoadingError) {
        return 30000; // Wait 30s for LoRA loading
      }
      if (error instanceof ServerError) {
        return 5000; // Quick retry for server errors
      }
      return 0; // Don't retry client errors
    }

    expect(getRetryDelay(loraError)).toBe(30000);
    expect(getRetryDelay(serverError)).toBe(5000);
  });
});

describe("Error message and stack traces", () => {
  it("includes helpful context in messages", () => {
    // User scenario: "I need to log errors with context for debugging"
    const error = new RequestError('Model "nonexistent-model" not found', "MODEL_NOT_FOUND", 404);

    expect(error.message).toContain("nonexistent-model");
    expect(error.toString()).toContain("RequestError");
  });

  it("preserves stack traces for debugging", () => {
    function innerFunction(): never {
      throw new SIEConnectionError("Failed in inner function");
    }

    function outerFunction(): void {
      innerFunction();
    }

    try {
      outerFunction();
    } catch (error) {
      expect(error).toBeInstanceOf(SIEConnectionError);
      expect((error as Error).stack).toContain("innerFunction");
    }
  });
});

describe("Real-world error handling patterns", () => {
  it("supports comprehensive try/catch pattern", () => {
    // This is how a real user would handle errors
    function handleSIEError(error: unknown): { type: string; retry: boolean; message: string } {
      if (error instanceof ProvisioningError) {
        return {
          type: "provisioning",
          retry: true,
          message: `GPU ${error.gpu} is provisioning. Retry after ${error.retryAfter ?? 10000}ms`,
        };
      }
      if (error instanceof LoraLoadingError) {
        return {
          type: "lora",
          retry: true,
          message: `LoRA ${error.lora} for ${error.model} is loading`,
        };
      }
      if (error instanceof PoolError) {
        return {
          type: "pool",
          retry: false,
          message: `Pool ${error.poolName} is ${error.state ?? "unavailable"}`,
        };
      }
      if (error instanceof RequestError) {
        return {
          type: "request",
          retry: false,
          message: `Bad request (${error.code ?? "UNKNOWN"}): ${error.message}`,
        };
      }
      if (error instanceof ServerError) {
        return {
          type: "server",
          retry: true,
          message: `Server error (${error.statusCode ?? 500}): ${error.message}`,
        };
      }
      if (error instanceof SIEConnectionError) {
        return {
          type: "connection",
          retry: true,
          message: `Connection failed: ${error.message}`,
        };
      }
      if (error instanceof SIEError) {
        return {
          type: "unknown",
          retry: false,
          message: error.message,
        };
      }
      throw error; // Re-throw non-SIE errors
    }

    // Test various error scenarios
    expect(handleSIEError(new ProvisioningError("Scaling", "l4", 5000))).toEqual({
      type: "provisioning",
      retry: true,
      message: "GPU l4 is provisioning. Retry after 5000ms",
    });

    expect(handleSIEError(new RequestError("Invalid", "BAD", 400))).toEqual({
      type: "request",
      retry: false,
      message: "Bad request (BAD): Invalid",
    });

    // Non-SIE errors should be re-thrown
    expect(() => handleSIEError(new TypeError("wrong type"))).toThrow(TypeError);
  });
});
