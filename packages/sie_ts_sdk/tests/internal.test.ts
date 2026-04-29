/**
 * Internal utilities tests focused on real user scenarios.
 *
 * These tests verify that:
 * 1. Retry logic with backoff works correctly for resilient requests
 * 2. GPU parameter parsing handles pool/gpu format correctly
 * 3. Response parsing correctly transforms wire format to SDK types
 * 4. Capacity info parsing correctly handles gateway health responses
 */

import { describe, expect, it } from "vitest";
import { parseCapacityInfo, parseGpuParam } from "../src/internal/parsing.js";
import { computeBackoffWithJitter, getRetryAfter } from "../src/internal/retry.js";

describe("Retry logic - exponential backoff with jitter", () => {
  it("should return delay within expected range for first attempt", () => {
    // First attempt (attempt=0) should have delay between 0 and baseDelay
    const delays: number[] = [];
    for (let i = 0; i < 100; i++) {
      delays.push(computeBackoffWithJitter(0, 1000, 30000));
    }

    // All delays should be between 0 and 1000 (baseDelay)
    for (const delay of delays) {
      expect(delay).toBeGreaterThanOrEqual(0);
      expect(delay).toBeLessThanOrEqual(1000);
    }
  });

  it("should increase delay range with each attempt (exponential)", () => {
    // Collect max delays for each attempt level
    const maxDelays: number[] = [];
    for (let attempt = 0; attempt < 5; attempt++) {
      let maxDelay = 0;
      for (let i = 0; i < 100; i++) {
        const delay = computeBackoffWithJitter(attempt, 1000, 30000);
        if (delay > maxDelay) maxDelay = delay;
      }
      maxDelays.push(maxDelay);
    }

    // Max delays should generally increase with attempt number
    // (due to randomness, we check the trend is upward)
    const delay0 = maxDelays[0] ?? 0;
    const delay1 = maxDelays[1] ?? 0;
    const delay3 = maxDelays[3] ?? 0;
    expect(delay1).toBeGreaterThan(delay0 * 0.5);
    expect(delay3).toBeGreaterThan(delay1 * 0.5);
  });

  it("should respect max delay cap", () => {
    // Even with many attempts, should never exceed maxDelay
    const maxDelay = 5000;
    for (let attempt = 0; attempt < 20; attempt++) {
      for (let i = 0; i < 50; i++) {
        const delay = computeBackoffWithJitter(attempt, 1000, maxDelay);
        expect(delay).toBeLessThanOrEqual(maxDelay);
      }
    }
  });

  it("should add jitter to prevent thundering herd", () => {
    // Same attempt number should produce different delays (jitter)
    const delays = new Set<number>();
    for (let i = 0; i < 20; i++) {
      delays.add(computeBackoffWithJitter(3, 1000, 30000));
    }

    // With jitter, we should get many different values
    expect(delays.size).toBeGreaterThan(10);
  });

  it("should use defaults when not specified", () => {
    // Should work with defaults (5s base, 30s max - matches Python SDK)
    const delay = computeBackoffWithJitter(0);
    expect(delay).toBeGreaterThanOrEqual(0);
    expect(delay).toBeLessThanOrEqual(5000);
  });
});

describe("Retry-After header parsing", () => {
  it("should parse integer seconds", () => {
    const delay = getRetryAfter("5");
    expect(delay).toBe(5000); // 5 seconds in ms
  });

  it("should parse large integer seconds", () => {
    const delay = getRetryAfter("300");
    expect(delay).toBe(300000); // 5 minutes in ms
  });

  it("should return undefined for null header", () => {
    const delay = getRetryAfter(null);
    expect(delay).toBeUndefined();
  });

  it("should return undefined for empty header", () => {
    const delay = getRetryAfter("");
    expect(delay).toBeUndefined();
  });

  it("should return undefined for zero", () => {
    const delay = getRetryAfter("0");
    expect(delay).toBeUndefined();
  });

  it("should return undefined for negative value", () => {
    const delay = getRetryAfter("-5");
    expect(delay).toBeUndefined();
  });

  it("should parse HTTP date format", () => {
    // Set a date 10 seconds in the future
    const futureDate = new Date(Date.now() + 10000);
    const httpDate = futureDate.toUTCString();

    const delay = getRetryAfter(httpDate);

    // Should be approximately 10 seconds (within 1s tolerance)
    expect(delay).toBeDefined();
    expect(delay).toBeGreaterThan(8000);
    expect(delay).toBeLessThan(12000);
  });

  it("should return undefined for past date", () => {
    const pastDate = new Date(Date.now() - 10000);
    const httpDate = pastDate.toUTCString();

    const delay = getRetryAfter(httpDate);
    expect(delay).toBeUndefined();
  });

  it("should return undefined for invalid format", () => {
    const delay = getRetryAfter("not-a-valid-value");
    expect(delay).toBeUndefined();
  });
});

describe("GPU parameter parsing", () => {
  it("should parse simple GPU type", () => {
    // User scenario: "I just want to use any L4 GPU"
    const result = parseGpuParam("l4");

    expect(result.gpu).toBe("l4");
    expect(result.pool).toBeUndefined();
  });

  it("should parse pool/gpu format", () => {
    // User scenario: "I want to use L4 from my dedicated pool"
    const result = parseGpuParam("eval-bench/l4");

    expect(result.pool).toBe("eval-bench");
    expect(result.gpu).toBe("l4");
  });

  it("should handle various GPU types", () => {
    const gpuTypes = ["l4", "a100-40gb", "a100-80gb", "h100", "t4"];

    for (const gpu of gpuTypes) {
      const result = parseGpuParam(gpu);
      expect(result.gpu).toBe(gpu);
    }
  });

  it("should handle various pool names", () => {
    const pools = ["prod", "staging", "eval-bench", "my_custom_pool", "pool123"];

    for (const pool of pools) {
      const result = parseGpuParam(`${pool}/l4`);
      expect(result.pool).toBe(pool);
      expect(result.gpu).toBe("l4");
    }
  });

  it("should handle pool names with special characters", () => {
    // Pool names might have hyphens, underscores
    const result = parseGpuParam("my-eval_pool/a100-80gb");

    expect(result.pool).toBe("my-eval_pool");
    expect(result.gpu).toBe("a100-80gb");
  });

  it("should treat multiple slashes as no pool (invalid format)", () => {
    // Edge case: what if there are multiple slashes?
    // This is an invalid format - we only expect "pool/gpu" with exactly one slash
    const result = parseGpuParam("pool/with/slash/l4");

    // Current implementation requires exactly 2 parts, otherwise no pool
    expect(result.pool).toBeUndefined();
    expect(result.gpu).toBe("pool/with/slash/l4");
  });
});

describe("Real-world retry scenarios", () => {
  it("should provide reasonable delays for 202 provisioning retries", () => {
    // User scenario: GPU is provisioning, need to retry with backoff
    // Typical provisioning takes 30-60 seconds, we should have reasonable delays

    const delays: number[] = [];
    for (let attempt = 0; attempt < 10; attempt++) {
      delays.push(computeBackoffWithJitter(attempt, 5000, 30000));
    }

    // First few attempts should be quick (under 10s each)
    expect(delays[0]).toBeLessThan(5000);
    expect(delays[1]).toBeLessThan(10000);

    // Later attempts should back off more
    expect(delays[5]).toBeLessThan(30000);

    // Total wait over 10 retries should be reasonable for provisioning
    const totalWait = delays.reduce((a, b) => a + b, 0);
    expect(totalWait).toBeLessThan(300000); // Under 5 minutes total
  });

  it("should respect server Retry-After hint when provided", () => {
    // Server says retry after 10 seconds
    const serverHint = getRetryAfter("10");

    expect(serverHint).toBe(10000);

    // This should take precedence over computed backoff
    const computedDelay = computeBackoffWithJitter(0, 1000, 30000);
    expect(serverHint).toBeGreaterThan(computedDelay);
  });
});

describe("Capacity info parsing", () => {
  it("should parse gateway health response", () => {
    const wireData = {
      status: "healthy",
      type: "gateway",
      cluster: {
        worker_count: 3,
        gpu_count: 3,
        models_loaded: 5,
      },
      configured_gpu_types: ["l4", "a100-80gb"],
      live_gpu_types: ["l4"],
      workers: [
        {
          url: "http://worker1:8080",
          gpu: "l4",
          healthy: true,
          queue_depth: 5,
          loaded_models: ["bge-m3", "e5-large"],
        },
        {
          url: "http://worker2:8080",
          gpu: "l4",
          healthy: true,
          queue_depth: 3,
          loaded_models: ["bge-m3"],
        },
      ],
    };

    const result = parseCapacityInfo(wireData);

    expect(result.status).toBe("healthy");
    expect(result.workerCount).toBe(3);
    expect(result.gpuCount).toBe(3);
    expect(result.modelsLoaded).toBe(5);
    expect(result.configuredGpuTypes).toEqual(["l4", "a100-80gb"]);
    expect(result.liveGpuTypes).toEqual(["l4"]);
    expect(result.workers).toHaveLength(2);
    expect(result.workers[0]?.url).toBe("http://worker1:8080");
    expect(result.workers[0]?.gpu).toBe("l4");
    expect(result.workers[0]?.healthy).toBe(true);
    expect(result.workers[0]?.queueDepth).toBe(5);
    expect(result.workers[0]?.loadedModels).toEqual(["bge-m3", "e5-large"]);
  });

  it("should filter workers by GPU when specified", () => {
    const wireData = {
      status: "healthy",
      type: "gateway",
      cluster: {
        worker_count: 4,
        gpu_count: 4,
        models_loaded: 5,
      },
      workers: [
        { url: "http://worker1:8080", gpu: "l4", healthy: true, queue_depth: 0, loaded_models: [] },
        { url: "http://worker2:8080", gpu: "l4", healthy: true, queue_depth: 0, loaded_models: [] },
        {
          url: "http://worker3:8080",
          gpu: "a100-80gb",
          healthy: true,
          queue_depth: 0,
          loaded_models: [],
        },
        {
          url: "http://worker4:8080",
          gpu: "a100-80gb",
          healthy: true,
          queue_depth: 0,
          loaded_models: [],
        },
      ],
    };

    // Filter for L4 only
    const l4Result = parseCapacityInfo(wireData, "l4");
    expect(l4Result.workers).toHaveLength(2);
    expect(l4Result.workerCount).toBe(2); // Uses filtered count when GPU specified

    // Filter for A100
    const a100Result = parseCapacityInfo(wireData, "a100-80gb");
    expect(a100Result.workers).toHaveLength(2);
    expect(a100Result.workerCount).toBe(2);
  });

  it("should handle missing optional fields", () => {
    const minimalData = {
      status: "no_workers",
    };

    const result = parseCapacityInfo(minimalData);

    expect(result.status).toBe("no_workers");
    expect(result.workerCount).toBe(0);
    expect(result.gpuCount).toBe(0);
    expect(result.modelsLoaded).toBe(0);
    expect(result.configuredGpuTypes).toEqual([]);
    expect(result.liveGpuTypes).toEqual([]);
    expect(result.workers).toEqual([]);
  });

  it("should handle case-insensitive GPU filtering", () => {
    const wireData = {
      status: "healthy",
      workers: [
        { url: "http://worker1:8080", gpu: "L4", healthy: true, queue_depth: 0, loaded_models: [] },
        { url: "http://worker2:8080", gpu: "l4", healthy: true, queue_depth: 0, loaded_models: [] },
      ],
    };

    // Filter with lowercase should match both
    const result = parseCapacityInfo(wireData, "l4");
    expect(result.workers).toHaveLength(2);
  });
});
