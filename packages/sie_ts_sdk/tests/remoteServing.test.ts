/**
 * Forbidding remote serving, and reading which side served.
 *
 * The header contract is shared with the server and the Python SDK through
 * packages/wire-fixtures/serving_disclosure.json. Requests go through the real
 * client with a stubbed `fetch`, so the option and the parsing run on the
 * production request and response paths.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { ModelLoadFailedError, RequestError, ServerError } from "../src/errors.js";
import { handleError } from "../src/internal/parsing.js";
import { packMessage } from "../src/msgpack.js";
import {
  FALLBACK_ERROR_PATTERN,
  FALLBACK_REASONS,
  type RequestMetadata,
  SERVED_BY_VALUES,
  type SIEClientOptions,
  UPSTREAM_NAME_PATTERN,
} from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

interface DisclosureHeader {
  name: string;
  values?: string[];
  pattern?: string;
}

const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../wire-fixtures/serving_disclosure.json", import.meta.url)),
    "utf8",
  ),
) as {
  request_header: { name: string; values: string[] };
  response_headers: Record<
    "served_by" | "upstream" | "fallback_reason" | "fallback_error",
    DisclosureHeader
  >;
};
const header = fixture.response_headers;

const SERVED_REMOTELY: Record<string, string> = {
  [header.served_by.name]: "remote",
  [header.upstream.name]: "team-sie",
  [header.fallback_reason.name]: "model_loading",
};

type Operation = "encode" | "generate" | "chatCompletions";
const OPERATIONS: Operation[] = ["encode", "generate", "chatCompletions"];

function reply(operation: Operation, headers: Record<string, string>): Response {
  if (operation === "encode") {
    return new Response(
      packMessage({ model: "m", items: [{ dense: { values: new Float32Array([0.5]) } }] }),
      { status: 200, headers: { "Content-Type": "application/msgpack", ...headers } },
    );
  }
  const body =
    operation === "generate"
      ? { model: "m", text: "ok", finish_reason: "stop" }
      : { id: "c", object: "chat.completion", created: 1, model: "m", choices: [] };
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

async function call(client: SIEClient, operation: Operation): Promise<RequestMetadata | undefined> {
  if (operation === "encode") return (await client.encode("m", { text: "hi" })).request;
  if (operation === "generate")
    return (await client.generate("m", "hi", { maxNewTokens: 4 })).request;
  return (await client.chatCompletions({ model: "m", messages: [{ role: "user", content: "hi" }] }))
    .request;
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    () => {
      throw new Error("expected a rejection");
    },
    (error: unknown) => error,
  );
}

function sentHeaders(): Record<string, string> {
  return mockFetch.mock.calls[0]?.[1]?.headers as Record<string, string>;
}

describe("remote serving", () => {
  beforeEach(() => mockFetch.mockReset());
  afterEach(() => vi.clearAllMocks());

  it("declares the same headers and values as the shared fixture", () => {
    expect(fixture.request_header).toEqual({ name: "X-SIE-Remote", values: ["forbid"] });
    expect([...SERVED_BY_VALUES]).toEqual(header.served_by.values);
    expect([...FALLBACK_REASONS]).toEqual(header.fallback_reason.values);
    expect(UPSTREAM_NAME_PATTERN.source).toBe(header.upstream.pattern);
    expect(FALLBACK_ERROR_PATTERN.source).toBe(header.fallback_error.pattern);
  });

  it.each(OPERATIONS)("a forbidding client sends X-SIE-Remote on %s", async (operation) => {
    mockFetch.mockResolvedValueOnce(reply(operation, { [header.served_by.name]: "local" }));
    const client = new SIEClient("http://localhost:8080", { remote: "forbid" });

    const request = await call(client, operation);

    expect(sentHeaders()[fixture.request_header.name]).toBe("forbid");
    expect(request?.servedBy).toBe("local");
  });

  it("a default client leaves the choice to the server", async () => {
    mockFetch.mockResolvedValueOnce(reply("encode", {}));
    const client = new SIEClient("http://localhost:8080");

    await call(client, "encode");

    expect(sentHeaders()[fixture.request_header.name]).toBeUndefined();
  });

  it.each(["a".repeat(64), "A".repeat(64), "a".repeat(65)])(
    "validates serving worker instance provenance %s",
    async (value) => {
      mockFetch.mockResolvedValueOnce(reply("encode", { "X-SIE-Runtime-Instance": value }));
      const client = new SIEClient("http://localhost:8080");
      const request = await call(client, "encode");
      expect(request?.runtimeInstanceId).toBe(value === "a".repeat(64) ? value : undefined);
    },
  );

  it.each(OPERATIONS)(
    "a remotely served %s result names the upstream and the reason",
    async (operation) => {
      mockFetch.mockResolvedValueOnce(reply(operation, SERVED_REMOTELY));
      const client = new SIEClient("http://localhost:8080");

      const request = await call(client, operation);

      expect(request).toMatchObject({
        servedBy: "remote",
        upstream: "team-sie",
        fallbackReason: "model_loading",
      });
    },
  );

  it("refuses any other remote option", () => {
    const options = { remote: "allow" } as unknown as SIEClientOptions;

    expect(() => new SIEClient("http://localhost:8080", options)).toThrow(TypeError);
  });

  it("a local refusal after a failed remote attempt carries both outcomes on the error", async () => {
    const response = new Response(
      JSON.stringify({
        detail: { code: "MODEL_LOADING", message: "Model 'm' is loading, please retry" },
      }),
      {
        status: 503,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": "5",
          [header.fallback_reason.name]: "model_loading",
          [header.fallback_error.name]: "INFERENCE_ERROR",
        },
      },
    );

    const error = await rejection(handleError(response));

    expect(error).toBeInstanceOf(ServerError);
    expect((error as ServerError).request).toEqual({
      fallbackReason: "model_loading",
      fallbackError: "INFERENCE_ERROR",
    });
  });

  it.each(OPERATIONS)(
    "a restored refusal to %s exposes which side answered on the thrown error",
    async (operation) => {
      mockFetch.mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: { code: "MODEL_LOAD_FAILED", message: "load failed", permanent: true },
          }),
          {
            status: 502,
            headers: {
              "Content-Type": "application/json",
              "X-SIE-Request-ID": "req-restored",
              [header.served_by.name]: "local",
              [header.fallback_reason.name]: "unhealthy",
              [header.fallback_error.name]: "INFERENCE_ERROR",
            },
          },
        ),
      );
      const client = new SIEClient("http://localhost:8080");

      const error = await rejection(call(client, operation));

      expect(error).toBeInstanceOf(ModelLoadFailedError);
      expect((error as ModelLoadFailedError).requestId).toBe("req-restored");
      expect((error as ModelLoadFailedError).request).toEqual({
        id: "req-restored",
        servedBy: "local",
        fallbackReason: "unhealthy",
        fallbackError: "INFERENCE_ERROR",
      });
    },
  );

  it("a terminal streaming answer exposes which side answered on the thrown error", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ error: { code: "GATEWAY_TIMEOUT", message: "result deadline" } }),
        {
          status: 504,
          headers: {
            "Content-Type": "application/json",
            [header.served_by.name]: "remote",
            [header.upstream.name]: "team-sie",
          },
        },
      ),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await rejection(client.streamGenerate("m", "hi", { maxNewTokens: 4 }).next());

    expect(error).toBeInstanceOf(ServerError);
    expect((error as ServerError).request).toEqual({ servedBy: "remote", upstream: "team-sie" });
  });

  it("an error without disclosure headers carries no request metadata", async () => {
    const response = new Response(
      JSON.stringify({ detail: { code: "INVALID_INPUT", message: "bad" } }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );

    const error = await rejection(handleError(response));

    expect(error).toBeInstanceOf(RequestError);
    expect((error as RequestError).request).toBeUndefined();
    expect((error as RequestError).requestId).toBeUndefined();
  });

  it.each([
    [header.served_by.name, "elsewhere"],
    [header.served_by.name, "Remote"],
    [header.upstream.name, "Team_SIE"],
    [header.upstream.name, "a".repeat(64)],
    [header.fallback_reason.name, "sometimes"],
    [header.fallback_error.name, "not a code"],
  ])("drops %s: %s, which is outside its contract", async (name, value) => {
    mockFetch.mockResolvedValueOnce(reply("generate", { [name]: value }));
    const client = new SIEClient("http://localhost:8080");

    expect(await call(client, "generate")).toBeUndefined();
  });
});
