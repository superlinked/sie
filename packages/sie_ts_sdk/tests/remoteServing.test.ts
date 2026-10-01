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
