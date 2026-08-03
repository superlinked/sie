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
const FILE = {
  id: "file-abc",
  object: "file",
  bytes: 42,
  created_at: 1,
  filename: "in.jsonl",
  purpose: "batch",
};

describe("client.files", () => {
  beforeEach(() => mockFetch.mockClear());

  it("upload posts a raw body to /v1/files with purpose + filename query", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(FILE, 200));
    const client = new SIEClient(GW, { apiKey: "sk-sie-x" });
    const bytes = new TextEncoder().encode('{"custom_id":"a"}\n');
    const out = await client.files.upload(bytes, { purpose: "batch", filename: "in.jsonl" });
    expect(out.id).toBe("file-abc");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/v1/files?");
    expect(url).toContain("purpose=batch");
    expect(url).toContain("filename=in.jsonl");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/jsonl");
    expect(init.headers.Authorization).toBe("Bearer sk-sie-x");
    expect(init.body).toBe(bytes);
  });

  it("create is an OpenAI-exact alias ({ file, purpose })", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(FILE, 200));
    const client = new SIEClient(GW);
    const out = await client.files.create({ file: "line1\nline2\n", purpose: "batch" });
    expect(out.id).toBe("file-abc");
    expect(mockFetch.mock.calls[0][0]).toContain("purpose=batch");
    expect(mockFetch.mock.calls[0][1].body).toBe("line1\nline2\n");
  });

  it("derives the filename from a File's .name", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(FILE, 200));
    const client = new SIEClient(GW);
    const blob = new File([new TextEncoder().encode("x")], "batch_input.jsonl");
    await client.files.upload(blob);
    expect(mockFetch.mock.calls[0][0]).toContain("filename=batch_input.jsonl");
  });

  it("retrieve and delete hit the expected URLs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(FILE));
    const client = new SIEClient(GW);
    await client.files.retrieve("file-abc");
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/files/file-abc");

    mockFetch.mockResolvedValueOnce(
      jsonResponse({ id: "file-abc", object: "file", deleted: true }),
    );
    const del = await client.files.delete("file-abc");
    expect(del.deleted).toBe(true);
    expect(mockFetch.mock.calls[1][0]).toBe("http://gw:8080/v1/files/file-abc");
    expect(mockFetch.mock.calls[1][1].method).toBe("DELETE");
  });

  it("list encodes cursor and filter arguments and preserves the array return", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "list",
        data: [FILE],
        first_id: "file-abc",
        last_id: "file-abc",
        has_more: true,
      }),
    );
    const client = new SIEClient(GW);
    const files = await client.files.list({
      after: 'file/"a b"&purpose=x',
      limit: 7,
      order: "asc",
      purpose: "batch",
    });
    expect(files.map((file) => file.id)).toEqual(["file-abc"]);
    expect(mockFetch.mock.calls[0][0]).toBe(
      "http://gw:8080/v1/files?after=file%2F%22a+b%22%26purpose%3Dx&limit=7&order=asc&purpose=batch",
    );
  });

  it("listPage exposes cursor metadata", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "list",
        data: [FILE],
        first_id: "file-abc",
        last_id: "file-abc",
        has_more: true,
      }),
    );
    const client = new SIEClient(GW);
    const page = await client.files.listPage();
    expect(page.data?.map((file) => file.id)).toEqual(["file-abc"]);
    expect(page.first_id).toBe("file-abc");
    expect(page.last_id).toBe("file-abc");
    expect(page.has_more).toBe(true);
  });

  it("listPage normalizes a legacy bare array", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse([FILE, { ...FILE, id: "file-def", filename: "next.jsonl" }]),
    );
    const client = new SIEClient(GW);
    const page = await client.files.listPage();
    expect(page.data?.map((file) => file.id)).toEqual(["file-abc", "file-def"]);
    expect(page.first_id).toBe("file-abc");
    expect(page.last_id).toBe("file-def");
    expect(page.has_more).toBe(false);
  });

  it("content returns the raw file bytes", async () => {
    const payload = new TextEncoder().encode('{"custom_id":"a","response":{"status_code":200}}\n');
    mockFetch.mockResolvedValueOnce(new Response(payload, { status: 200 }));
    const client = new SIEClient(GW);
    const out = await client.files.content("file-out");
    expect(new TextDecoder().decode(out)).toBe(
      '{"custom_id":"a","response":{"status_code":200}}\n',
    );
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/files/file-out/content");
  });
});
