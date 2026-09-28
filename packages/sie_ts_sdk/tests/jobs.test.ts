import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { buildJobBody, connectionName } from "../src/jobs.js";
import type { JobStatus } from "../src/jobs.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SUBMIT_RESP = {
  id: "job-1",
  object: "job",
  operation: "encode",
  model: "BAAI/bge-m3",
  state: "queued",
  total_items: 2,
  chunks: 1,
  preflight: { estimated_credits: 64 },
};

const CONNECTOR_STATUS_WITH_NUMERIC_REVISIONS: JobStatus = {
  id: "job-1",
  object: "job",
  operation: "encode",
  model: "BAAI/bge-m3",
  state: "succeeded",
  checkpoint: { published_revision: 3 },
  publication: { revision: 4 },
  attempt: {
    ordinal: 2,
    action: "repair",
    state: "published",
    recovery_attempt_ordinal: 1,
  },
};

describe("buildJobBody (pure slot mapping)", () => {
  it("types checkpoint and publication revisions as wire-format numbers", () => {
    expect(CONNECTOR_STATUS_WITH_NUMERIC_REVISIONS.checkpoint?.published_revision).toBe(3);
    expect(CONNECTOR_STATUS_WITH_NUMERIC_REVISIONS.publication?.revision).toBe(4);
    expect(CONNECTOR_STATUS_WITH_NUMERIC_REVISIONS.attempt?.recovery_attempt_ordinal).toBe(1);
  });

  it("maps an inline list to items", () => {
    expect(buildJobBody({ source: ["a", "b"], model: "m" })).toEqual({
      operation: "encode",
      model: "m",
      items: [{ text: "a" }, { text: "b" }],
    });
  });

  it("routes a connector URI to src + derived connection", () => {
    const body = buildJobBody({
      source: "postgres://warehouse?query=x",
      model: "m",
      sink: "postgres://warehouse?table=t",
      execution: "plan",
    });
    expect(body.src).toBe("postgres://warehouse?query=x");
    expect(body.connection).toBe("warehouse");
    expect(body.sink).toBe("postgres://warehouse?table=t");
    expect(body.execution).toBe("plan");
    expect(body.sink_connection).toBeUndefined();
  });

  it("threads a distinct sink connection", () => {
    const body = buildJobBody({
      source: "postgres://wh?query=x",
      model: "m",
      sink: "s3://out/vecs",
      execution: "run",
    });
    expect(body.sink_connection).toBe("out");
  });

  it("requires explicit connector execution and rejects it for inline jobs", () => {
    expect(() =>
      buildJobBody({
        source: "postgres://warehouse?query=x",
        model: "m",
        sink: "postgres://warehouse?table=t",
      }),
    ).toThrow(/require execution/);
    expect(() => buildJobBody({ source: ["a"], model: "m", execution: "plan" })).toThrow(
      /only to connector/,
    );
  });

  it("rejects unavailable schedule / watch triggers", () => {
    expect(() => buildJobBody({ source: ["a"], model: "m", when: "schedule:*/5 * * * *" })).toThrow(
      /not available/,
    );
    expect(() => buildJobBody({ source: ["a"], model: "m", when: "watch:s3://in" })).toThrow(
      /not available/,
    );
  });

  it("derives connection names from URIs", () => {
    expect(connectionName("postgres://warehouse?query=x")).toBe("warehouse");
    expect(connectionName("s3://customer-bucket/in/")).toBe("customer-bucket");
  });

  it.each([
    "../other",
    "warehouse/name",
    "warehouse%2fname",
    "warehouse\n",
    "warehouse\r",
    "warehouse\u2028",
    "café",
    "a".repeat(129),
  ])("rejects non-canonical connector name %s", (name) => {
    if (name !== "warehouse/name") {
      expect(() => connectionName(`postgres://${name}?query=x`)).toThrow(/connection name/);
    }
    expect(() =>
      buildJobBody({
        source: "postgres://warehouse?query=x",
        model: "m",
        sink: "postgres://warehouse?table=out",
        connection: name,
      }),
    ).toThrow(/connection name/);
    expect(() =>
      buildJobBody({
        source: "postgres://warehouse?query=x",
        model: "m",
        sink: "postgres://warehouse?table=out",
        sinkConnection: name,
      }),
    ).toThrow(/connection name/);
  });

  // field_map / output_field + the internal upload:// scheme

  it("rides field_map + output_field on connector jobs", () => {
    const body = buildJobBody({
      source: "postgres://wh?query=select id, body, source_url from docs",
      model: "BAAI/bge-m3",
      sink: "postgres://wh?table=doc_vectors",
      fieldMap: {
        id_field: "id",
        input_field: "body",
        carry: ["source_url"],
        input_type: "text",
      },
      outputField: "embedding",
      execution: "plan",
    });
    expect(body.field_map).toEqual({
      id_field: "id",
      input_field: "body",
      input_type: "text",
      carry: ["source_url"],
    });
    expect(body.output_field).toBe("embedding");
  });

  it("rejects field_map on inline items and bad slots", () => {
    expect(() =>
      buildJobBody({ source: ["a"], model: "m", fieldMap: { id_field: "id" } }),
    ).toThrowError(/connector-src/);
    expect(() =>
      buildJobBody({
        source: "postgres://wh?query=x",
        model: "m",
        sink: "postgres://wh?table=t",
        execution: "plan",
        fieldMap: { id_column: "id" } as never,
      }),
    ).toThrowError(/unknown field_map key/);
    expect(() =>
      buildJobBody({
        source: "postgres://wh?query=x",
        model: "m",
        sink: "postgres://wh?table=t",
        execution: "plan",
        fieldMap: { input_type: "rows" } as never,
      }),
    ).toThrowError(/input_type/);
  });

  it.each([
    { sink: "inplace" },
    { sink: "postgres://wh?table=vecs" },
    { sinkConnection: "wh" },
    { connection: "wh" },
  ])("rejects connector fields on inline items: %j", (connectorFields) => {
    expect(() => buildJobBody({ source: ["a"], model: "m", ...connectorFields })).toThrowError(
      /connector-src/,
    );
  });

  it("derives no connection for the internal upload:// scheme", () => {
    const body = buildJobBody({
      source: "upload://file-abc?format=csv",
      model: "m",
      sink: "upload://file-out",
      fieldMap: { id_field: "doc_id", input_field: "text" },
      execution: "run",
    });
    expect(body.src).toBe("upload://file-abc?format=csv");
    expect(body.sink).toBe("upload://file-out");
    expect(body.connection).toBeUndefined();
    expect(body.sink_connection).toBeUndefined();
    // upload source → external sink still threads the sink's connection.
    const cross = buildJobBody({
      source: "upload://file-abc",
      model: "m",
      sink: "postgres://wh?table=doc_vectors",
      sinkConnection: "wh",
      execution: "run",
    });
    expect(cross.connection).toBeUndefined();
    expect(cross.sink_connection).toBe("wh");
    expect(() =>
      buildJobBody({
        source: "upload://file-abc",
        model: "m",
        sink: "upload://file-out",
        execution: "plan",
      }),
    ).toThrow(/run-only/);
  });

  it("forwards op inputs via options as-is (op matrix)", () => {
    // score: options.query rides untouched (connector form).
    const score = buildJobBody({
      source: "postgres://wh?query=x",
      model: "m",
      operation: "score",
      sink: "postgres://wh?table=scores",
      options: { query: "rank these documents" },
      execution: "run",
    });
    expect(score.options).toEqual({ query: "rank these documents" });

    // extract: labels + output_schema forwarded (inline form).
    const extract = buildJobBody({
      source: ["some text"],
      model: "m",
      operation: "extract",
      options: { labels: ["PERSON", "ORG"], output_schema: { type: "object" } },
    });
    expect(extract.options).toEqual({
      labels: ["PERSON", "ORG"],
      output_schema: { type: "object" },
    });

    // Absent / empty options stay off the wire (additive-only).
    expect(buildJobBody({ source: ["a"], model: "m" }).options).toBeUndefined();
    expect(buildJobBody({ source: ["a"], model: "m", options: {} }).options).toBeUndefined();
  });
});

describe("client.jobs", () => {
  beforeEach(() => mockFetch.mockClear());

  it("submit posts the inline body to /v1/jobs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080", { apiKey: "sk-sie-x" });
    const result = await client.jobs.submit({ source: ["a", "b"], model: "BAAI/bge-m3" });
    expect(result.id).toBe("job-1");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://gw:8080/v1/jobs");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      operation: "encode",
      model: "BAAI/bge-m3",
      items: [{ text: "a" }, { text: "b" }],
    });
    expect(init.headers.Authorization).toBe("Bearer sk-sie-x");
  });

  it("submit maps a connector job body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.submit({
      source: "postgres://warehouse?query=x",
      model: "m",
      sink: "s3://out/vecs",
      execution: "plan",
      idempotencyKey: "plan-warehouse-1",
    });
    const init = mockFetch.mock.calls[0][1];
    const body = JSON.parse(init.body);
    expect(body.src).toBe("postgres://warehouse?query=x");
    expect(body.connection).toBe("warehouse");
    expect(body.sink_connection).toBe("out");
    expect(body.execution).toBe("plan");
    expect(init.headers["Idempotency-Key"]).toBe("plan-warehouse-1");
  });

  it("submit forwards score query and extract labels via options", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.submit({
      source: "postgres://wh?query=x",
      model: "BAAI/bge-m3",
      operation: "score",
      sink: "postgres://wh?table=scores",
      options: { query: "rank these documents" },
      execution: "run",
      idempotencyKey: "run-score-1",
    });
    const scoreBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(scoreBody.operation).toBe("score");
    expect(scoreBody.options).toEqual({ query: "rank these documents" });

    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    await client.jobs.submit({
      source: ["some text"],
      model: "urchade/gliner_small-v2.1",
      operation: "extract",
      options: { labels: ["PERSON", "ORG"] },
    });
    const extractBody = JSON.parse(mockFetch.mock.calls[1][1].body);
    expect(extractBody.operation).toBe("extract");
    expect(extractBody.options).toEqual({ labels: ["PERSON", "ORG"] });
  });

  it("requires a retry-stable idempotency key for connectors before fetch", async () => {
    const client = new SIEClient("http://gw:8080");
    await expect(
      client.jobs.submit({
        source: "postgres://warehouse?query=x",
        model: "m",
        sink: "postgres://warehouse?table=out",
        execution: "plan",
      }),
    ).rejects.toThrow(/idempotencyKey/);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("keeps inline submissions free of connector idempotency headers", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.submit({ source: ["a"], model: "m" });
    expect(mockFetch.mock.calls[0][1].headers["Idempotency-Key"]).toBeUndefined();
  });

  it("get and cancel hit the expected URLs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "running" }));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.get("job-1");
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/jobs/job-1");

    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "cancelled" }));
    const out = await client.jobs.cancel("job-1");
    expect(out.state).toBe("cancelled");
    expect(mockFetch.mock.calls[1][0]).toBe("http://gw:8080/v1/jobs/job-1/cancel");
    expect(mockFetch.mock.calls[1][1].method).toBe("POST");
  });

  it("execute posts the exact plan revision with one idempotency key", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "queued" }));
    const client = new SIEClient("http://gw:8080");

    await client.jobs.execute("job-1", 3, "execute-plan-3");

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://gw:8080/v1/jobs/job-1/execute");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ plan_revision: 3 });
    expect(init.headers["Idempotency-Key"]).toBe("execute-plan-3");
  });

  it("execute rejects an invalid idempotency key before fetch", async () => {
    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.execute("job-1", 3, "")).rejects.toThrow(/idempotencyKey/);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("repair posts the exact plan revision and recovery predecessor", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "running" }, 202));
    const client = new SIEClient("http://gw:8080");

    await client.jobs.repair("job-1", 3, 2, "repair-plan-3-attempt-2");

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://gw:8080/v1/jobs/job-1/repair");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      plan_revision: 3,
      recovery_attempt_ordinal: 2,
    });
    expect(init.headers["Idempotency-Key"]).toBe("repair-plan-3-attempt-2");
  });

  it("repair rejects an invalid idempotency key before fetch", async () => {
    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.repair("job-1", 3, 2, "")).rejects.toThrow(/idempotencyKey/);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("list returns the data array", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ object: "list", data: [{ id: "job-1" }, { id: "job-2" }] }),
    );
    const client = new SIEClient("http://gw:8080");
    const jobs = await client.jobs.list();
    expect(jobs.map((j) => j.id)).toEqual(["job-1", "job-2"]);
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/jobs");
  });

  it("results reads http refs and decodes per-item embeddings", async () => {
    const chunkBytes = packMessage([
      {
        success: true,
        id: "0",
        units: { input_tokens: 5 },
        result_msgpack: packMessage({ dense: { dims: 4, values: [0.1, 0.2, 0.3, 0.4] } }),
      },
    ]);
    const job = {
      id: "job-1",
      state: "succeeded",
      total_items: 1,
      settled_credits: 5,
      output: {
        kind: "refs",
        chunks: [{ seq: 0, items: 1, state: "succeeded", ref: "http://refs.local/c0" }],
      },
    };
    mockFetch
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(new Response(chunkBytes, { status: 200 }));

    const client = new SIEClient("http://gw:8080");
    const results = await client.jobs.results("job-1");
    // snake_case throughout (matches the wire + the Python SDK + JobStatus).
    expect(results.job_id).toBe("job-1");
    expect(results.total_items).toBe(1);
    expect(results.settled_credits).toBe(5);
    expect(results.retrieved).toBe(1);
    expect(results.dims).toBe(4);
    expect(Array.from(results.items[0].dense as number[])).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it("results refreshes the job after a replica miss and discards partial items", async () => {
    const chunk = (id: string) =>
      packMessage([
        {
          success: true,
          id,
          units: { input_tokens: 5 },
          result_msgpack: packMessage({ dense: { dims: 2, values: [0.1, 0.2] } }),
        },
      ]);
    const job = (prefix: string) => ({
      id: "job-1",
      state: "succeeded",
      total_items: 2,
      output: {
        kind: "refs",
        chunks: [
          { seq: 0, items: 1, state: "succeeded", ref: `http://refs.local/${prefix}-0` },
          { seq: 1, items: 1, state: "succeeded", ref: `http://refs.local/${prefix}-1` },
        ],
      },
    });
    mockFetch
      .mockResolvedValueOnce(jsonResponse(job("stale")))
      .mockResolvedValueOnce(new Response(chunk("discarded"), { status: 200 }))
      .mockResolvedValueOnce(
        jsonResponse({ detail: { code: "RESULT_NOT_FOUND", message: "not on this replica" } }, 404),
      )
      .mockResolvedValueOnce(jsonResponse(job("fresh")))
      .mockResolvedValueOnce(new Response(chunk("0"), { status: 200 }))
      .mockResolvedValueOnce(new Response(chunk("1"), { status: 200 }));

    const client = new SIEClient("http://gw:8080");
    const results = await client.jobs.results("job-1");

    expect(mockFetch).toHaveBeenCalledTimes(6);
    expect(results.retrieved).toBe(2);
    expect(results.items.map((item) => item.id)).toEqual(["0", "1"]);
    expect(results.chunks.map((chunk) => chunk.ref)).toEqual([
      "http://refs.local/fresh-0",
      "http://refs.local/fresh-1",
    ]);
  });

  it("results bounds replica-miss refreshes", async () => {
    const job = {
      id: "job-1",
      state: "succeeded",
      output: {
        kind: "refs",
        chunks: [{ seq: 0, items: 1, state: "succeeded", ref: "http://refs.local/stale" }],
      },
    };
    const expectedAttempts = 4;
    for (let attempt = 0; attempt < expectedAttempts; attempt += 1) {
      mockFetch
        .mockResolvedValueOnce(jsonResponse(job))
        .mockResolvedValueOnce(
          jsonResponse(
            { detail: { code: "RESULT_NOT_FOUND", message: "not on this replica" } },
            404,
          ),
        );
    }

    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.results("job-1")).rejects.toMatchObject({
      code: "RESULT_NOT_FOUND",
      statusCode: 404,
    });
    expect(mockFetch).toHaveBeenCalledTimes(expectedAttempts * 2);
  });

  it.each([
    [404, "NOT_FOUND"],
    [503, "STORAGE_UNAVAILABLE"],
  ])("results does not refresh unrelated ref failure %i %s", async (status, code) => {
    const job = {
      id: "job-1",
      state: "succeeded",
      output: {
        kind: "refs",
        chunks: [{ seq: 0, items: 1, state: "succeeded", ref: "http://refs.local/ref" }],
      },
    };
    mockFetch
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(
        jsonResponse({ detail: { code, message: "terminal ref failure" } }, status),
      );

    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.results("job-1")).rejects.toMatchObject({ code, statusCode: status });
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("wait polls until the job reaches a terminal state", async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "running" }))
      .mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "succeeded" }));
    const client = new SIEClient("http://gw:8080");
    const job = await client.jobs.wait("job-1", { pollMs: 0 });
    expect(job.state).toBe("succeeded");
    expect(mockFetch.mock.calls.length).toBe(2);
  });

  it("wait returns a stable planned connector phase without polling again", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ id: "job-plan-1", state: "queued", execution: "plan", phase: "planned" }),
    );
    const client = new SIEClient("http://gw:8080");
    const job = await client.jobs.wait("job-plan-1", { pollMs: 0 });
    expect(job.phase).toBe("planned");
    expect(mockFetch.mock.calls.length).toBe(1);
  });

  it("wait throws a job_wait_timeout RequestError when the deadline passes", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "job-1", state: "running" }));
    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.wait("job-1", { timeoutMs: 0, pollMs: 0 })).rejects.toMatchObject({
      code: "job_wait_timeout",
    });
  });

  it("submit floors the abort timeout to 120s (survives a 30s client timeout)", () => {
    vi.useFakeTimers();
    try {
      let capturedInit: RequestInit | undefined;
      mockFetch.mockImplementationOnce((_url: string, init: RequestInit) => {
        capturedInit = init;
        return new Promise<Response>(() => {}); // never settles; we only inspect the signal
      });
      const client = new SIEClient("http://gw:8080", { timeoutMs: 30_000 });
      client.jobs.submit({ source: ["a"], model: "m" }).catch(() => {}); // swallow the eventual abort
      expect(capturedInit?.signal?.aborted).toBe(false);
      vi.advanceTimersByTime(30_000);
      expect(capturedInit?.signal?.aborted).toBe(false); // past the 30s client timeout, still alive
      vi.advanceTimersByTime(90_001);
      expect(capturedInit?.signal?.aborted).toBe(true); // aborts at the 120s floor
    } finally {
      vi.useRealTimers();
    }
  });
});
