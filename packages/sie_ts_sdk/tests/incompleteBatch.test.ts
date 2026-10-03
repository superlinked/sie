/**
 * Positional batch-contract guard (architecture-review finding U1).
 *
 * The gateway's queue path answers a mixed-success batch with `200` carrying
 * only the *successful* items, and batch results are positional (item `id` is
 * optional). A shortened body therefore shifts every result after the dropped
 * item, so a zip-inputs-to-outputs consumer stores results against the wrong
 * inputs — silently. Both `encode` and `extract` guard the 1:1 contract and
 * throw {@link IncompleteBatchError} instead of returning a desynced array.
 *
 * Mirrors the Python SDK's `test_incomplete_batch.py`, including error codes.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { IncompleteBatchError, SIEError, ServerError } from "../src/errors.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function msgpackResponse(body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(packMessage(body), {
    status: 200,
    headers: { "Content-Type": "application/msgpack", ...headers },
  });
}

function dense(value: number): { dense: { values: Float32Array } } {
  return { dense: { values: new Float32Array([value, value]) } };
}

describe("IncompleteBatchError class", () => {
  it("is a ServerError and SIEError", () => {
    const error = new IncompleteBatchError("desync", { expected: 2, received: 1 });
    expect(error).toBeInstanceOf(ServerError);
    expect(error).toBeInstanceOf(SIEError);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("IncompleteBatchError");
  });

  it("carries the counts and a 200 status (the response was not an HTTP error)", () => {
    const error = new IncompleteBatchError("desync", {
      expected: 3,
      received: 2,
      model: "bge-m3",
      code: "ENCODE_RESULT_COUNT_MISMATCH",
    });
    expect(error.expected).toBe(3);
    expect(error.received).toBe(2);
    expect(error.model).toBe("bge-m3");
    expect(error.statusCode).toBe(200);
    expect(error.missingIds).toBeUndefined();
  });
});

describe("encode batch guard", () => {
  let client: SIEClient;

  beforeEach(() => {
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(() => {
    mockFetch.mockReset();
  });

  it("throws when the response drops an item", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({ items: [dense(0.1)] }, { "x-sie-request-id": "req-drop" }),
    );

    await expect(
      client.encode("bge-m3", [{ text: "ok" }, { text: "over-length" }]),
    ).rejects.toThrow(IncompleteBatchError);
  });

  it("reports counts, model, code, and request id", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({ items: [dense(0.1)] }, { "x-sie-request-id": "req-drop" }),
    );

    try {
      await client.encode("bge-m3", [{ text: "ok" }, { text: "over-length" }]);
      expect.unreachable("expected IncompleteBatchError");
    } catch (error) {
      expect(error).toBeInstanceOf(IncompleteBatchError);
      const incomplete = error as IncompleteBatchError;
      expect(incomplete.expected).toBe(2);
      expect(incomplete.received).toBe(1);
      expect(incomplete.model).toBe("bge-m3");
      expect(incomplete.code).toBe("ENCODE_RESULT_COUNT_MISMATCH");
      expect(incomplete.requestId).toBe("req-drop");
      expect(incomplete.message).toContain("1 embedding(s) for 2 input item(s)");
    }
  });

  it("names the dropped ids when every item carries one", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({
        items: [
          { id: "doc-a", ...dense(0.1) },
          { id: "doc-c", ...dense(0.3) },
        ],
      }),
    );

    try {
      await client.encode("bge-m3", [
        { id: "doc-a", text: "a" },
        { id: "doc-b", text: "b" },
        { id: "doc-c", text: "c" },
      ]);
      expect.unreachable("expected IncompleteBatchError");
    } catch (error) {
      const incomplete = error as IncompleteBatchError;
      expect(incomplete.missingIds).toEqual(["doc-b"]);
      expect(incomplete.message).toContain("doc-b");
    }
  });

  it("degrades to counts only when ids are absent", async () => {
    mockFetch.mockResolvedValueOnce(msgpackResponse({ items: [dense(0.1)] }));

    try {
      await client.encode("bge-m3", [{ text: "a" }, { text: "b" }]);
      expect.unreachable("expected IncompleteBatchError");
    } catch (error) {
      expect((error as IncompleteBatchError).missingIds).toBeUndefined();
    }
  });

  it("passes through when the counts match", async () => {
    mockFetch.mockResolvedValueOnce(msgpackResponse({ items: [dense(0.1), dense(0.2)] }));

    const results = await client.encode("bge-m3", [{ text: "a" }, { text: "b" }]);
    expect(results).toHaveLength(2);
  });
});

describe("extract batch guard", () => {
  let client: SIEClient;

  beforeEach(() => {
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(() => {
    mockFetch.mockReset();
  });

  it("throws with the extract-specific code and wording", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({
        items: [{ entities: [{ text: "Apple", label: "org", score: 0.9, start: 0, end: 5 }] }],
      }),
    );

    try {
      await client.extract("gliner", [{ text: "Apple info" }, { text: "Tesla info" }], {
        labels: ["org"],
      });
      expect.unreachable("expected IncompleteBatchError");
    } catch (error) {
      expect(error).toBeInstanceOf(IncompleteBatchError);
      const incomplete = error as IncompleteBatchError;
      expect(incomplete.code).toBe("EXTRACT_RESULT_COUNT_MISMATCH");
      expect(incomplete.message).toContain("extraction result(s)");
    }
  });

  it("passes through when the counts match", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({
        items: [
          { entities: [{ text: "Apple", label: "org", score: 0.9, start: 0, end: 5 }] },
          { entities: [{ text: "Tesla", label: "org", score: 0.95, start: 0, end: 5 }] },
        ],
      }),
    );

    const results = await client.extract(
      "gliner",
      [{ text: "Apple info" }, { text: "Tesla info" }],
      { labels: ["org"] },
    );
    expect(results).toHaveLength(2);
  });
});
