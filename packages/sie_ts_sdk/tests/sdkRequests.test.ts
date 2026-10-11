/**
 * The requests this SDK sends must match the shared request fixture.
 *
 * Every case of `packages/wire-fixtures/sdk_requests.json` is issued through
 * the real client with `fetch` stubbed, and the request that reached the wire
 * is compared with the fixture. The Python SDK replays the same cases in
 * `test_sdk_requests.py`, so a call one SDK serializes differently, or cannot
 * express, fails here or there.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { packMessage, unpackMessage } from "../src/msgpack.js";
import type { SIEClientOptions } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

interface FixtureRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
  may_send_default?: Record<string, unknown>;
}

interface FixtureCase {
  id: string;
  summary: string;
  client?: { gpu?: string; remote?: "forbid" };
  request: FixtureRequest;
}

const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../wire-fixtures/sdk_requests.json", import.meta.url)),
    "utf8",
  ),
) as {
  client: { api_key: string };
  media: Record<string, string>;
  compared_headers: string[];
  cases: FixtureCase[];
};

const BASE_URL = "http://sie.test";

function media(name: string): Uint8Array {
  const encoded = fixture.media[name];
  if (encoded === undefined) throw new Error(`fixture has no media named ${name}`);
  return new Uint8Array(Buffer.from(encoded, "base64"));
}

const SCHEMA = {
  type: "object",
  properties: { vendor: { type: "string" }, total: { type: "number" } },
  required: ["vendor", "total"],
};
const QUERY = { text: "What is machine learning?" };
const DOCUMENTS = [
  { id: "doc-1", text: "Machine learning is a subfield of AI." },
  { id: "doc-2", text: "Python is a programming language." },
];
const ENTITY_TEXT = { text: "Apple was founded by Steve Jobs." };

async function drain(stream: AsyncIterable<unknown>): Promise<void> {
  for await (const _chunk of stream) {
    // The request is what is under test; the chunks are discarded.
  }
}

/** The call each fixture case stands for, spelled with this SDK's options. */
const CALLS: Record<string, (client: SIEClient) => Promise<unknown>> = {
  encode_single_text: (client) => client.encode("BAAI/bge-m3", { text: "hello" }),
  encode_batch_all_params: (client) =>
    client.encode(
      "BAAI/bge-m3",
      [
        { id: "a", text: "first" },
        { id: "b", text: "second" },
      ],
      {
        outputTypes: ["dense", "sparse"],
        instruction: "Represent the query",
        outputDtype: "float16",
        isQuery: true,
      },
    ),
  encode_profile_option: (client) =>
    client.encode(
      "lightonai/GTE-ModernColBERT-v1",
      { text: "a long document" },
      {
        outputTypes: ["multivector"],
        isQuery: false,
        adapterOptions: { profile: "long_context" },
      },
    ),
  encode_model_profile_id: (client) => client.encode("BAAI/bge-m3:sparse", { text: "hello" }),
  encode_image: (client) =>
    client.encode("google/siglip2-base-patch16-224", { images: [media("png")] }),
  encode_pool_and_gpu: (client) =>
    client.encode("BAAI/bge-m3", { text: "hello" }, { gpu: "eval-bench/l4" }),
  encode_client_defaults: (client) => client.encode("BAAI/bge-m3", { text: "hello" }),
  score_basic: (client) => client.score("BAAI/bge-reranker-v2-m3", QUERY, DOCUMENTS),
  score_instruction_and_options: (client) =>
    client.score("Qwen/Qwen3-Reranker-4B", QUERY, DOCUMENTS, {
      instruction: "Rank passages that answer the question",
      adapterOptions: { profile: "default" },
    }),
  extract_labels: (client) =>
    client.extract("urchade/gliner_multi-v2.1", ENTITY_TEXT, {
      labels: ["person", "organization"],
    }),
  extract_threshold: (client) =>
    client.extract("urchade/gliner_multi-v2.1", ENTITY_TEXT, {
      labels: ["person", "organization"],
      threshold: 0.75,
    }),
  extract_image_without_params: (client) =>
    client.extract("lightonai/LightOnOCR-2-1B", { images: [media("png")] }),
  extract_output_schema_and_instruction: (client) =>
    client.extract(
      "fastino/GLiNER2.5-Decide",
      { text: "Get me a person. This is the third time." },
      {
        outputSchema: {
          handoff: { type: "noul", instructions: "Should a person take over?" },
        },
        instruction: "Triage the support ticket.",
        adapterOptions: { overflow_policy: "truncate_text" },
      },
    ),
  extract_audio: (client) =>
    client.extract("openai/whisper-large-v3-turbo", {
      audio: { data: media("wav"), format: "wav", sampleRate: 16_000 },
    }),
  extract_document: (client) =>
    client.extract("docling", { document: { data: media("pdf"), format: "pdf" } }),
  generate_basic: (client) =>
    client.generate("Qwen/Qwen3-4B-Instruct-2507", "Write a haiku about the sea.", {
      maxNewTokens: 64,
      temperature: 0.7,
    }),
  generate_all_fields: (client) =>
    client.generate(
      "google/gemma-4-31B-it:h100-96k-hires-thinking-budget512-no-spec",
      "Extract the invoice.",
      {
        maxNewTokens: 256,
        temperature: 0,
        topP: 0.9,
        stop: ["\n\n"],
        frequencyPenalty: 0.5,
        presencePenalty: 0.25,
        grammar: { json_schema: SCHEMA, strict: true },
        seed: 7,
        logitBias: { "42": -1.5 },
        routingKey: "tenant-1",
        promptCacheKey: "invoice-v1",
        safetyIdentifier: "user-9",
        loraAdapter: "invoices",
        adapterOptions: { overall_timeout_s: 30 },
      },
    ),
  generate_image: (client) =>
    client.generate("Qwen/Qwen3-VL-8B-Instruct", "Describe the image.", {
      maxNewTokens: 32,
      images: [media("png")],
    }),
  stream_generate: (client) =>
    drain(
      client.streamGenerate("Qwen/Qwen3-4B-Instruct-2507", "Write a haiku.", {
        maxNewTokens: 64,
        temperature: 0.7,
        logprobs: true,
        topLogprobs: 2,
      }),
    ),
  chat_completions: (client) =>
    client.chatCompletions({
      model: "Qwen/Qwen3-4B-Instruct-2507",
      messages: [
        { role: "system", content: "Answer in JSON." },
        { role: "user", content: "Extract the invoice fields." },
      ],
      max_completion_tokens: 256,
      temperature: 0,
      response_format: {
        type: "json_schema",
        json_schema: { name: "invoice", strict: true, schema: SCHEMA },
      },
      schema_in_prompt: false,
    }),
  stream_chat_completions: (client) =>
    drain(
      client.streamChatCompletions({
        model: "Qwen/Qwen3-4B-Instruct-2507",
        messages: [{ role: "user", content: "Count to ten." }],
        max_completion_tokens: 64,
        stream_options: { include_usage: true },
      }),
    ),
};

function msgpackResponse(body: unknown): Response {
  return new Response(packMessage(body), {
    status: 200,
    headers: { "Content-Type": "application/msgpack" },
  });
}

/** A minimal success for the route, shaped by what the request asked for. */
function successFor(path: string, accept: string | undefined, itemCount: number): Response {
  if (path.startsWith("/v1/encode/")) {
    const items = Array.from({ length: itemCount }, () => ({
      dense: { values: new Float32Array([0.5]) },
    }));
    return msgpackResponse({ model: "m", items });
  }
  if (path.startsWith("/v1/score/")) {
    return msgpackResponse({ model: "m", scores: [{ item_id: "doc-1", score: 0.5, rank: 0 }] });
  }
  if (path.startsWith("/v1/extract/")) {
    const items = Array.from({ length: itemCount }, () => ({ entities: [] }));
    return msgpackResponse({ model: "m", items });
  }
  if (accept === "text/event-stream") {
    return new Response("data: [DONE]\n\n", {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }
  const body = path.startsWith("/v1/generate/")
    ? { model: "m", text: "ok", finish_reason: "stop" }
    : { id: "chat-1", object: "chat.completion", model: "m", choices: [] };
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** Fixture view of a decoded body: bytes named by media, `null` members dropped. */
function canonical(value: unknown): unknown {
  if (value instanceof Uint8Array) {
    const base64 = Buffer.from(value).toString("base64");
    const name = Object.keys(fixture.media).find((key) => fixture.media[key] === base64);
    return name === undefined ? { $bytes: base64 } : { $media: name };
  }
  if (Array.isArray(value)) return value.map(canonical);
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, member]) => member !== null && member !== undefined)
      .map(([key, member]) => [key, canonical(member)] as const)
      .sort(([left], [right]) => left.localeCompare(right));
    return Object.fromEntries(entries);
  }
  return value;
}

/** The fixture's expected body with base64 media references resolved. */
function expectedBody(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(expectedBody);
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    if (typeof record.$media_base64 === "string") return fixture.media[record.$media_base64];
    return canonical(
      Object.fromEntries(
        Object.entries(record).map(([key, member]) => [key, expectedBody(member)]),
      ),
    );
  }
  return value;
}

interface SentRequest {
  method: string | undefined;
  path: string;
  headers: Record<string, string>;
  body: unknown;
}

function sentRequest(): SentRequest {
  expect(mockFetch).toHaveBeenCalledTimes(1);
  const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
  const headers = Object.fromEntries(
    Object.entries(init.headers as Record<string, string>).map(([name, value]) => [
      name.toLowerCase(),
      value,
    ]),
  );
  const raw = init.body;
  const body =
    headers["content-type"] === "application/msgpack"
      ? unpackMessage(raw as Uint8Array)
      : JSON.parse(raw as string);
  return {
    method: init.method,
    path: decodeURIComponent(new URL(url).pathname),
    headers,
    body: canonical(body),
  };
}

describe("shared SDK request fixture", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockFetch.mockImplementation(async (url: string, init: RequestInit) => {
      const headers = init.headers as Record<string, string>;
      const isMsgpack = headers["Content-Type"] === "application/msgpack";
      const decoded = isMsgpack
        ? unpackMessage<{ items?: unknown[] }>(init.body as Uint8Array)
        : undefined;
      return successFor(new URL(url).pathname, headers.Accept, decoded?.items?.length ?? 1);
    });
  });

  it("has a call for every case, and no call without a case", () => {
    expect(Object.keys(CALLS).sort()).toEqual(fixture.cases.map((entry) => entry.id).sort());
  });

  it.each(fixture.cases.map((entry) => [entry.id, entry] as const))("%s", async (_id, entry) => {
    const options: SIEClientOptions = { apiKey: fixture.client.api_key, ...entry.client };
    const client = new SIEClient(BASE_URL, options);
    const call = CALLS[entry.id];
    if (call === undefined) throw new Error(`no call for fixture case ${entry.id}`);

    await call(client);
    await client.close();

    const sent = sentRequest();
    expect(sent.method).toBe(entry.request.method);
    expect(sent.path).toBe(entry.request.path);
    for (const name of fixture.compared_headers) {
      expect({ [name]: sent.headers[name] }).toEqual({ [name]: entry.request.headers[name] });
    }
    expect(sent.headers["x-sie-sdk-version"]).toBeTruthy();

    const body = sent.body as Record<string, unknown>;
    for (const [member, value] of Object.entries(entry.request.may_send_default ?? {})) {
      if (member in body) {
        expect({ [member]: body[member] }).toEqual({ [member]: value });
        delete body[member];
      }
    }
    expect(body).toEqual(expectedBody(entry.request.body));
  });
});

describe("encode and extract results", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("repeats the response's model and timing on each encode result", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({
        model: "BAAI/bge-m3",
        items: [
          { id: "a", dense: { values: new Float32Array([0.5]) } },
          { id: "b", dense: { values: new Float32Array([0.25]) } },
        ],
        timing: { total_ms: 12.5, queue_ms: 1, tokenization_ms: 2, inference_ms: 9.5 },
      }),
    );
    const client = new SIEClient(BASE_URL);

    const results = await client.encode("BAAI/bge-m3", [{ text: "first" }, { text: "second" }]);

    for (const result of results) {
      expect(result.model).toBe("BAAI/bge-m3");
      expect(result.timing).toEqual({
        totalMs: 12.5,
        queueMs: 1,
        tokenizationMs: 2,
        inferenceMs: 9.5,
      });
    }
  });

  it("leaves model and timing unset when the response carries neither", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({ items: [{ dense: { values: new Float32Array([0.5]) } }] }),
    );
    const client = new SIEClient(BASE_URL);

    const result = await client.encode("BAAI/bge-m3", { text: "hello" });

    expect(result.model).toBeUndefined();
    expect(result.timing).toBeUndefined();
  });

  it("repeats the response's model on each extract result", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({
        model: "urchade/gliner_multi-v2.1",
        items: [{ entities: [] }, { entities: [] }],
      }),
    );
    const client = new SIEClient(BASE_URL);

    const results = await client.extract(
      "urchade/gliner_multi-v2.1",
      [{ text: "first" }, { text: "second" }],
      { labels: ["person"] },
    );

    expect(results.map((result) => result.model)).toEqual([
      "urchade/gliner_multi-v2.1",
      "urchade/gliner_multi-v2.1",
    ]);
  });

  it("reads an extract result without an entities member as having none", async () => {
    mockFetch.mockResolvedValueOnce(
      msgpackResponse({ model: "docling", items: [{ data: { text: "hello" } }] }),
    );
    const client = new SIEClient(BASE_URL);

    const result = await client.extract("docling", { text: "hello" });

    expect(result.entities).toEqual([]);
    expect(result.data).toEqual({ text: "hello" });
  });
});

describe("runtime options", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockFetch.mockImplementation(async () => msgpackResponse({ model: "m", items: [{}] }));
  });

  function sentParams(): Record<string, unknown> | undefined {
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    return unpackMessage<{ params?: Record<string, unknown> }>(init.body as Uint8Array).params;
  }

  it("lets isQuery win over an is_query runtime option on encode", async () => {
    const client = new SIEClient(BASE_URL);

    await client.encode(
      "BAAI/bge-m3",
      { text: "hello" },
      { isQuery: true, adapterOptions: { is_query: false, normalize: true } },
    );

    expect(sentParams()?.options).toEqual({ is_query: true, normalize: true });
  });

  it("lets threshold win over a threshold runtime option on extract", async () => {
    const client = new SIEClient(BASE_URL);

    await client.extract(
      "urchade/gliner_multi-v2.1",
      { text: "hello" },
      { labels: ["person"], threshold: 0.75, adapterOptions: { threshold: 0.25, flat_ner: true } },
    );

    expect(sentParams()?.options).toEqual({ threshold: 0.75, flat_ner: true });
  });

  it("labels an image with an unrecognized signature as jpeg", async () => {
    const client = new SIEClient(BASE_URL);
    const bytes = new Uint8Array([1, 2, 3, 4, 5]);

    await client.encode("google/siglip2-base-patch16-224", { images: [bytes] });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sent = unpackMessage<{ items: { images: unknown[] }[] }>(init.body as Uint8Array);
    expect(sent.items[0]?.images).toEqual([{ data: bytes, format: "jpeg" }]);
  });
});

describe("base URL", () => {
  it("drops every trailing slash", () => {
    expect(new SIEClient("http://localhost:8080//").getBaseUrl()).toBe("http://localhost:8080");
  });
});
