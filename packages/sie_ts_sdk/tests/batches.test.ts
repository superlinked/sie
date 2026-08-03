import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const GW = "http://gw:8080";
const BATCH = {
  id: "batch-1",
  object: "batch",
  endpoint: "/v1/embeddings",
  input_file_id: "file-in",
  completion_window: "24h",
  status: "completed",
  output_file_id: "file-out",
  request_counts: { total: 2, completed: 2, failed: 0 },
};

describe("client.batches", () => {
  beforeEach(() => mockFetch.mockClear());

  it("create posts the OpenAI-shaped body to /v1/batches", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(BATCH, 200));
    const client = new SIEClient(GW, { apiKey: "sk-sie-x" });
    const out = await client.batches.create({
      input_file_id: "file-in",
      endpoint: "/v1/embeddings",
    });
    expect(out.id).toBe("batch-1");
    expect(out.request_counts?.total).toBe(2);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://gw:8080/v1/batches");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      input_file_id: "file-in",
      endpoint: "/v1/embeddings",
      completion_window: "24h",
    });
    expect(init.headers.Authorization).toBe("Bearer sk-sie-x");
  });

  it("create includes metadata and defaults endpoint/completion_window", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(BATCH, 200));
    const client = new SIEClient(GW);
    await client.batches.create({ input_file_id: "file-in", metadata: { run: "eval-7" } });
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({
      input_file_id: "file-in",
      endpoint: "/v1/embeddings",
      completion_window: "24h",
      metadata: { run: "eval-7" },
    });
  });

  it("retrieve and cancel hit the expected URLs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(BATCH));
    const client = new SIEClient(GW);
    await client.batches.retrieve("batch-1");
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/batches/batch-1");

    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "batch-1", status: "cancelling" }));
    const out = await client.batches.cancel("batch-1");
    expect(out.status).toBe("cancelling");
    expect(mockFetch.mock.calls[1][0]).toBe("http://gw:8080/v1/batches/batch-1/cancel");
    expect(mockFetch.mock.calls[1][1].method).toBe("POST");
  });

  it("list returns the data array", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ object: "list", data: [{ id: "batch-1" }, { id: "batch-2" }] }),
    );
    const client = new SIEClient(GW);
    const batches = await client.batches.list();
    expect(batches.map((b) => b.id)).toEqual(["batch-1", "batch-2"]);
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/batches");
  });

  it("list encodes cursor arguments and preserves the array return", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "list",
        data: [{ id: "batch-2" }],
        first_id: "batch-2",
        last_id: "batch-2",
        has_more: false,
      }),
    );
    const client = new SIEClient(GW);
    const batches = await client.batches.list({
      after: 'batch/"a b"&limit=1',
      limit: 9,
    });
    expect(batches.map((batch) => batch.id)).toEqual(["batch-2"]);
    expect(mockFetch.mock.calls[0][0]).toBe(
      "http://gw:8080/v1/batches?after=batch%2F%22a+b%22%26limit%3D1&limit=9",
    );
  });

  it("listPage exposes cursor metadata", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "list",
        data: [{ id: "batch-1" }, { id: "batch-2" }],
        first_id: "batch-1",
        last_id: "batch-2",
        has_more: true,
      }),
    );
    const client = new SIEClient(GW);
    const page = await client.batches.listPage();
    expect(page.data?.map((batch) => batch.id)).toEqual(["batch-1", "batch-2"]);
    expect(page.first_id).toBe("batch-1");
    expect(page.last_id).toBe("batch-2");
    expect(page.has_more).toBe(true);
  });

  it("listPage normalizes an empty legacy bare array", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse([]));
    const client = new SIEClient(GW);
    const page = await client.batches.listPage();
    expect(page.data).toEqual([]);
    expect(page.first_id).toBeNull();
    expect(page.last_id).toBeNull();
    expect(page.has_more).toBe(false);
  });
});
