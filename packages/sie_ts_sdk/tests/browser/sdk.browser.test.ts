import { expect, test } from "@playwright/test";

/**
 * Browser compatibility tests for @superlinked/sie-sdk
 *
 * These tests verify that the SDK works correctly in browser environments
 * without relying on Node.js-specific APIs.
 */

test.describe("SDK Browser Compatibility", () => {
  test.beforeEach(async ({ page }) => {
    // Capture console messages for debugging
    const consoleMessages: string[] = [];
    page.on("console", (msg) => {
      consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
    });
    page.on("pageerror", (err) => {
      consoleMessages.push(`[pageerror] ${err.message}`);
    });

    // Navigate to the test harness
    await page.goto("/tests/browser/index.html");

    // Wait for SDK to load (with better error handling)
    try {
      await page.waitForFunction(() => window.__SDK_READY__ === true || window.__SDK_ERROR__, {
        timeout: 10000,
      });
    } catch (e) {
      console.error("Console messages:", consoleMessages.join("\n"));
      throw e;
    }

    // Check for errors
    const error = await page.evaluate(() => window.__SDK_ERROR__);
    if (error) {
      console.error("Console messages:", consoleMessages.join("\n"));
      throw new Error(`SDK failed to load: ${error}`);
    }
  });

  test("SDK loads without errors", async ({ page }) => {
    const ready = await page.evaluate(() => window.__SDK_READY__);
    expect(ready).toBe(true);
  });

  test("SIEClient class is available", async ({ page }) => {
    const hasClient = await page.evaluate(() => typeof window.SIEClient === "function");
    expect(hasClient).toBe(true);
  });

  test("SIEClient can be instantiated", async ({ page }) => {
    const result = await page.evaluate(() => {
      const client = new window.SIEClient("http://localhost:8080");
      return {
        isObject: typeof client === "object",
        hasEncode: typeof client.encode === "function",
        hasScore: typeof client.score === "function",
        hasExtract: typeof client.extract === "function",
        hasListModels: typeof client.listModels === "function",
        hasClose: typeof client.close === "function",
      };
    });

    expect(result.isObject).toBe(true);
    expect(result.hasEncode).toBe(true);
    expect(result.hasScore).toBe(true);
    expect(result.hasExtract).toBe(true);
    expect(result.hasListModels).toBe(true);
    expect(result.hasClose).toBe(true);
  });

  test("SIEClient accepts options", async ({ page }) => {
    const result = await page.evaluate(() => {
      try {
        const _client = new window.SIEClient("http://localhost:8080", {
          timeout: 60000,
          gpu: "a100",
        });
        return { success: true };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    });

    expect(result.success).toBe(true);
  });

  test("maxsim function is available", async ({ page }) => {
    const hasMaxsim = await page.evaluate(() => typeof window.maxsim === "function");
    expect(hasMaxsim).toBe(true);
  });

  test("maxsimBatch function is available", async ({ page }) => {
    const hasMaxsimBatch = await page.evaluate(() => typeof window.maxsimBatch === "function");
    expect(hasMaxsimBatch).toBe(true);
  });

  test("maxsim computes correct scores", async ({ page }) => {
    const result = await page.evaluate(() => {
      // Create test vectors (multivector format: array of token vectors)
      // Query: single token with vector [1, 0, 0, 0]
      const queryTokens = [new Float32Array([1, 0, 0, 0])];
      // Doc: single token with same vector (perfect match)
      const docTokens = [new Float32Array([1, 0, 0, 0])];

      // Compute maxsim (query multivector vs doc multivector)
      const score = window.maxsim(queryTokens, docTokens);
      return score;
    });

    // Perfect match should give score of 1
    expect(result).toBeCloseTo(1.0, 5);
  });

  test("Float32Array works correctly", async ({ page }) => {
    const result = await page.evaluate(() => {
      const arr = new Float32Array([1.5, 2.5, 3.5]);
      return {
        length: arr.length,
        first: arr[0],
        isTypedArray: arr instanceof Float32Array,
      };
    });

    expect(result.length).toBe(3);
    expect(result.first).toBe(1.5);
    expect(result.isTypedArray).toBe(true);
  });

  test("Int32Array works correctly", async ({ page }) => {
    const result = await page.evaluate(() => {
      const arr = new Int32Array([1, 2, 3]);
      return {
        length: arr.length,
        first: arr[0],
        isTypedArray: arr instanceof Int32Array,
      };
    });

    expect(result.length).toBe(3);
    expect(result.first).toBe(1);
    expect(result.isTypedArray).toBe(true);
  });

  test("No Node.js globals are required", async ({ page }) => {
    // Verify that SDK doesn't depend on Node.js-specific globals
    const result = await page.evaluate(() => {
      return {
        // These should NOT exist in browser
        hasProcess: typeof process !== "undefined" && process?.versions?.node !== undefined,
        hasRequire: typeof require === "function",
        hasBuffer: typeof Buffer !== "undefined",
        hasFs:
          typeof require === "function" &&
          (() => {
            try {
              require("node:fs");
              return true;
            } catch {
              return false;
            }
          })(),
        // SDK should still work
        sdkReady: window.__SDK_READY__,
      };
    });

    expect(result.sdkReady).toBe(true);
    expect(result.hasRequire).toBe(false);
    // Note: hasProcess may be true if test runner injects it, but Node version should be undefined
    expect(result.hasFs).toBe(false);
  });
});

// Extend window type for TypeScript
declare global {
  interface Window {
    __SDK_READY__?: boolean;
    __SDK_ERROR__?: Error;
    SIEClient: new (
      baseUrl: string,
      options?: Record<string, unknown>,
    ) => {
      encode: (...args: unknown[]) => Promise<unknown>;
      score: (...args: unknown[]) => Promise<unknown>;
      extract: (...args: unknown[]) => Promise<unknown>;
      listModels: () => Promise<unknown>;
      close: () => Promise<void>;
    };
    maxsim: (query: Float32Array[], doc: Float32Array[]) => number;
    maxsimBatch: (queries: Float32Array[][], docs: Float32Array[][]) => Float32Array;
  }
}
