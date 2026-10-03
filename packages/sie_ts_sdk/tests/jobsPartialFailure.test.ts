/**
 * Job result surfacing and terminal-state guard (architecture-review finding U2).
 *
 * A `failed` chunk still publishes a result ref carrying its SUCCESSFUL —
 * already billed — siblings alongside the per-item failures, so dropping the
 * whole chunk discards work the caller paid for. And a still-running job's ref
 * list is still growing, so decoding it returns a partial subset
 * indistinguishable from a partial-FAILURE subset.
 *
 * Mirrors the Python SDK's `test_jobs.py` (PR #3219), including error codes.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { JobFailedError, MalformedChunkError, SIEError } from "../src/errors.js";
import { decodeChunkBytes, decodeResultItem } from "../src/jobs.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** One chunk ref carrying a successful item AND a failed sibling (both billed). */
function mixedChunk(): Uint8Array {
  return packMessage([
    {
      success: true,
      id: "ok-1",
      units: { input_tokens: 5 },
      result_msgpack: packMessage({ dense: { dims: 2, values: [0.1, 0.2] } }),
    },
    {
      success: false,
      id: "bad-1",
      error: "tokenization failed",
      error_code: "INFERENCE_ERROR",
    },
  ]);
}

/** A terminal `failed` job: chunk 0 published a ref; chunk 1 lost its results. */
function partialFailureJob() {
  return {
    id: "job-1",
    state: "failed",
    total_items: 3,
    settled_credits: 5,
    output: {
      kind: "refs",
      chunks: [
        {
          seq: 0,
          items: 2,
          state: "failed",
          ref: "http://refs.local/chunk-0",
          error: "1 of 2 items failed",
        },
        { seq: 1, items: 1, state: "failed", ref: null, error: "result publication failed" },
      ],
    },
  };
}

describe("JobFailedError class", () => {
  it("is an SIEError carrying the terminal reason", () => {
    const error = new JobFailedError("job job-1 terminated", {
      jobId: "job-1",
      state: "failed",
      outcome: "reexecution_required",
      errorCode: "RESULT_HANDLE_EXPIRED",
    });
    expect(error).toBeInstanceOf(SIEError);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("JobFailedError");
    expect(error.jobId).toBe("job-1");
    expect(error.state).toBe("failed");
    expect(error.outcome).toBe("reexecution_required");
    expect(error.errorCode).toBe("RESULT_HANDLE_EXPIRED");
  });
});

describe("decodeResultItem", () => {
  it("populates error and surfaces the wire work_item_id", () => {
    const ok = decodeResultItem({
      success: true,
      id: "a",
      result_msgpack: packMessage({ dense: { dims: 2, values: [0.1, 0.2] } }),
    });
    expect(ok.success).toBe(true);
    expect(ok.error).toBeUndefined(); // a success carries no failure detail

    const bad = decodeResultItem({
      success: false,
      work_item_id: "b",
      error: "boom",
      error_code: "INFERENCE_ERROR",
    });
    expect(bad.success).toBe(false);
    // id surfaced from the wire `work_item_id`, never fabricated
    expect(bad.id).toBe("b");
    expect(bad.error).toEqual({ code: "INFERENCE_ERROR", message: "boom" });
  });

  it("preserves a falsy-but-valid id", () => {
    // Item ids can be numbers: id 0 (and an empty-string id) are VALID and must
    // not be replaced by work_item_id via a truthiness fallback.
    expect(decodeResultItem({ success: true, id: 0, work_item_id: "wi-9" }).id).toBe(0);
    expect(decodeResultItem({ success: true, id: "", work_item_id: "wi-9" }).id).toBe("");
    // The work_item_id fallback only applies when id is genuinely absent.
    expect(decodeResultItem({ success: false, work_item_id: "wi-9" }).id).toBe("wi-9");
  });

  it("tolerates malformed payloads", () => {
    // A non-object element degrades to a null result rather than throwing out of
    // the per-chunk decode loop.
    const item = decodeResultItem(42);
    expect(item.id).toBeNull();
    expect(item.success).toBeNull();
    expect(item.error).toBeUndefined();
    // A record whose result_msgpack is garbage still decodes (vector is null).
    const garbage = decodeResultItem({
      success: true,
      id: "x",
      result_msgpack: new Uint8Array([0xc1, 0xc1, 0xc1]),
    });
    expect(garbage.id).toBe("x");
    expect(garbage.dense).toBeNull();
  });
});

describe("decodeChunkBytes", () => {
  it("signals malformed refs distinctly", () => {
    // Garbage bytes → a distinct MalformedChunkError, not a silent empty list, so
    // the caller can tell a decode fault from an unpublished chunk.
    expect(() => decodeChunkBytes(new Uint8Array([0xc1, 0xc1, 0xc1]))).toThrow(MalformedChunkError);
    // Valid msgpack that is not a list of results is also malformed.
    expect(() => decodeChunkBytes(packMessage({ not: "a list" }))).toThrow(MalformedChunkError);
    // A well-formed list still decodes normally.
    expect(decodeChunkBytes(mixedChunk()).map((it) => it.id)).toEqual(["ok-1", "bad-1"]);
  });
});

describe("jobs.results partial-failure surfacing", () => {
  let client: SIEClient;
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    client = new SIEClient("http://gw:8080");
    warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    mockFetch.mockReset();
    warn.mockRestore();
  });

  it("surfaces a failed chunk's billed successful items and per-item errors", async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResponse(partialFailureJob()))
      .mockResolvedValueOnce(new Response(mixedChunk(), { status: 200 }));

    const results = await client.jobs.results("job-1");

    // The `failed` chunk's ref is READ (not dropped); the ref-less chunk is skipped.
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch.mock.calls[1][0]).toBe("http://refs.local/chunk-0");
    expect(results.state).toBe("failed");
    expect(results.retrieved).toBe(2);

    const [ok, bad] = results.items;
    expect([ok.id, ok.success]).toEqual(["ok-1", true]);
    expect(Array.from(ok.dense as number[])).toEqual([0.1, 0.2]);
    expect(ok.error).toBeUndefined();
    expect([bad.id, bad.success]).toEqual(["bad-1", false]);
    expect(bad.error).toEqual({ code: "INFERENCE_ERROR", message: "tokenization failed" });

    // The incompleteness warning is NEUTRAL — it claims no cause.
    const messages = warn.mock.calls.map((call) => String(call[0]));
    expect(messages.some((m) => m.includes("incomplete: retrieved 2 of 3"))).toBe(true);
    expect(messages.some((m) => m.includes("billed") || m.includes("publishing"))).toBe(false);
  });

  it("throws job_not_terminal on a running job without reading any ref", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        id: "job-1",
        state: "running",
        total_items: 3,
        output: { kind: "refs", chunks: [{ seq: 0, items: 1, ref: "http://refs.local/c0" }] },
      }),
    );

    await expect(client.jobs.results("job-1")).rejects.toMatchObject({
      code: "job_not_terminal",
      statusCode: 409,
    });
    // Only the status GET happened — no ref was read.
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("confines a garbage chunk ref without sinking the whole call", async () => {
    mockFetch
      .mockResolvedValueOnce(
        jsonResponse({
          id: "job-1",
          state: "failed",
          total_items: 3,
          settled_credits: 5,
          output: {
            kind: "refs",
            chunks: [
              { seq: 0, items: 2, state: "failed", ref: "http://refs.local/good" },
              { seq: 1, items: 1, state: "succeeded", ref: "http://refs.local/garbage" },
            ],
          },
        }),
      )
      .mockResolvedValueOnce(new Response(mixedChunk(), { status: 200 }))
      .mockResolvedValueOnce(new Response(new Uint8Array([0xc1, 0xc1, 0xc1]), { status: 200 }));

    const results = await client.jobs.results("job-1");

    // The good chunk's 2 items survive; the garbage ref contributes none.
    expect(results.retrieved).toBe(2);
    expect(results.items.map((it) => it.id)).toEqual(["ok-1", "bad-1"]);
    const messages = warn.mock.calls.map((call) => String(call[0]));
    // The malformed ref is flagged distinctly (a decode fault)...
    expect(messages.some((m) => m.includes("could not be decoded"))).toBe(true);
    // ...and the neutral incompleteness warning still fires.
    expect(messages.some((m) => m.includes("incomplete: retrieved 2 of 3"))).toBe(true);
  });
});

describe("jobs.wait raiseOnFailure", () => {
  let client: SIEClient;

  beforeEach(() => {
    client = new SIEClient("http://gw:8080");
  });

  afterEach(() => {
    mockFetch.mockReset();
  });

  it("throws JobFailedError carrying the terminal outcome", async () => {
    const failed = {
      id: "job-1",
      state: "failed",
      outcome: "reexecution_required",
      error_code: "RESULT_HANDLE_EXPIRED",
    };
    // A fresh Response per call — a body can only be read once.
    mockFetch.mockImplementation(() => Promise.resolve(jsonResponse(failed)));

    try {
      await client.jobs.wait("job-1", { pollMs: 0, raiseOnFailure: true });
      expect.unreachable("expected JobFailedError");
    } catch (error) {
      expect(error).toBeInstanceOf(JobFailedError);
      const jobFailed = error as JobFailedError;
      expect(jobFailed.jobId).toBe("job-1");
      expect(jobFailed.state).toBe("failed");
      expect(jobFailed.outcome).toBe("reexecution_required");
      expect(jobFailed.errorCode).toBe("RESULT_HANDLE_EXPIRED");
      expect(jobFailed.message).toContain("reexecution_required");
    }

    // Default stays back-compatible: the terminal doc is returned, not thrown.
    expect((await client.jobs.wait("job-1", { pollMs: 0 })).state).toBe("failed");
  });

  it("returns a succeeded terminal unchanged", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "job-1", state: "succeeded", total_items: 2 }));
    const job = await client.jobs.wait("job-1", { pollMs: 0, raiseOnFailure: true });
    expect(job.state).toBe("succeeded");
  });

  it("returns a stable planned connector phase without raising", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({ id: "job-plan-1", state: "queued", execution: "plan", phase: "planned" }),
    );
    const job = await client.jobs.wait("job-plan-1", { pollMs: 0, raiseOnFailure: true });
    expect(job.phase).toBe("planned");
  });
});

describe("decodeResultItem scalar validation", () => {
  // These values arrive off the wire unvalidated, and `results()` now reads
  // FAILED chunks too, so off-contract scalars must degrade rather than be
  // asserted straight into the public shape.
  it("normalizes a non-scalar id to null", () => {
    expect(decodeResultItem({ id: {}, success: true }).id).toBeNull();
    expect(decodeResultItem({ work_item_id: [1], success: true }).id).toBeNull();
  });

  it("preserves falsy-but-valid ids", () => {
    expect(decodeResultItem({ id: 0, success: true }).id).toBe(0);
    expect(decodeResultItem({ id: "", success: true }).id).toBe("");
  });

  it("normalizes a non-boolean success to null", () => {
    expect(decodeResultItem({ id: "a", success: "false" }).success).toBeNull();
    expect(decodeResultItem({ id: "a", success: 0 }).success).toBeNull();
    expect(decodeResultItem({ id: "a", success: false }).success).toBe(false);
  });

  it("omits non-string error fields but keeps the failure signal", () => {
    const item = decodeResultItem({ id: "a", success: false, error_code: 1, error: {} });
    expect(item.error).toBeDefined();
    expect(item.error?.code).toBeUndefined();
    expect(item.error?.message).toBeUndefined();
  });

  it("keeps well-formed error fields", () => {
    const item = decodeResultItem({
      id: "a",
      success: false,
      error_code: "OOM",
      error: "out of memory",
    });
    expect(item.error).toEqual({ code: "OOM", message: "out of memory" });
  });
});
