/**
 * SIEClient tests focused on real user scenarios.
 *
 * These tests verify that users can:
 * 1. Create clients with various configurations
 * 2. Encode text and receive embeddings
 * 3. List available models
 * 4. Handle errors gracefully
 *
 * Tests use mocked fetch to simulate server responses.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { RequestError, SIEConnectionError, ServerError } from "../src/errors.js";
import { packMessage, unpackMessage } from "../src/msgpack.js";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Helper to create a mock Response
function createMockResponse(body: unknown, options: ResponseInit = {}): Response {
  const isJson = typeof body === "object" && !(body instanceof Uint8Array);
  const responseBody = isJson
    ? JSON.stringify(body)
    : body instanceof Uint8Array
      ? body
      : packMessage(body);

  return new Response(responseBody, {
    status: options.status ?? 200,
    headers: {
      "Content-Type": isJson ? "application/json" : "application/msgpack",
      ...options.headers,
    },
  });
}

// Helper to create msgpack response
function createMsgpackResponse(body: unknown, status = 200): Response {
  return new Response(packMessage(body), {
    status,
    headers: { "Content-Type": "application/msgpack" },
  });
}

describe("SIEClient construction", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should create client with minimal options", () => {
    const client = new SIEClient("http://localhost:8080");
    expect(client).toBeInstanceOf(SIEClient);
  });

  it("should expose base URL via getBaseUrl()", () => {
    const client = new SIEClient("http://localhost:8080");
    expect(client.getBaseUrl()).toBe("http://localhost:8080");
  });

  it("should return normalized URL from getBaseUrl() without trailing slash", () => {
    const client = new SIEClient("http://localhost:8080/");
    expect(client.getBaseUrl()).toBe("http://localhost:8080");
  });

  it("should normalize base URL by removing trailing slash", async () => {
    const client = new SIEClient("http://localhost:8080/");

    // Mock a simple response
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1, 0.2]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "test" });

    // Verify URL doesn't have double slashes
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/v1/encode/bge-m3",
      expect.anything(),
    );
  });

  it("should accept timeout option", () => {
    const client = new SIEClient("http://localhost:8080", { timeout: 60000 });
    expect(client).toBeInstanceOf(SIEClient);
  });

  it("should accept gpu option", async () => {
    const client = new SIEClient("http://localhost:8080", { gpu: "l4" });

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "test" });

    // Verify GPU header is set
    const fetchCall = mockFetch.mock.calls[0];
    const headers = fetchCall?.[1]?.headers as Record<string, string>;
    expect(headers["X-SIE-MACHINE-PROFILE"]).toBe("l4");
  });

  it("should accept apiKey option", async () => {
    const client = new SIEClient("http://localhost:8080", { apiKey: "sk-test-key" });

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "test" });

    // Verify Authorization header
    const fetchCall = mockFetch.mock.calls[0];
    const headers = fetchCall?.[1]?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer sk-test-key");
  });
});

describe("SIEClient.encode() - basic usage", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should encode single item and return EncodeResult", async () => {
    // User scenario: "I want to encode a single text document"
    const embedding = new Float32Array([0.1, 0.2, 0.3, 0.4]);

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: embedding } }],
      }),
    );

    const result = await client.encode("bge-m3", { text: "Hello world" });

    // Should return single result, not array
    expect(result.dense).toBeInstanceOf(Float32Array);
    expect(result.dense?.length).toBe(4);
  });

  it("should encode batch items and return EncodeResult[]", async () => {
    // User scenario: "I want to encode multiple documents efficiently"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          { id: "doc-1", dense: { values: new Float32Array([0.1]) } },
          { id: "doc-2", dense: { values: new Float32Array([0.2]) } },
          { id: "doc-3", dense: { values: new Float32Array([0.3]) } },
        ],
      }),
    );

    const results = await client.encode("bge-m3", [
      { id: "doc-1", text: "First document" },
      { id: "doc-2", text: "Second document" },
      { id: "doc-3", text: "Third document" },
    ]);

    expect(results).toHaveLength(3);
    expect(results[0]?.id).toBe("doc-1");
    expect(results[1]?.id).toBe("doc-2");
    expect(results[2]?.id).toBe("doc-3");
  });

  it("should preserve item IDs through encode cycle", async () => {
    // User scenario: "I need to match embeddings back to my source documents"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ id: "uuid-abc123", dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    const result = await client.encode("bge-m3", { id: "uuid-abc123", text: "Some text" });

    expect(result.id).toBe("uuid-abc123");
  });

  it("should construct correct URL with model in path", async () => {
    // Verify wire format: POST /v1/encode/{model}
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("BAAI/bge-m3", { text: "test" });

    // URL should have encoded model name
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/v1/encode/BAAI%2Fbge-m3",
      expect.anything(),
    );
  });

  it("should send correct Content-Type header", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "test" });

    const fetchCall = mockFetch.mock.calls[0];
    const headers = fetchCall?.[1]?.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/msgpack");
    expect(headers.Accept).toBe("application/msgpack");
  });

  it("should send correct request body format", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "Hello world" });

    // Verify request body structure
    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ items: Array<{ text: string }> }>(body);

    expect(parsed.items).toHaveLength(1);
    expect(parsed.items[0]?.text).toBe("Hello world");
  });
});

describe("SIEClient.encode() - encode options", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should pass outputTypes in params", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            dense: { values: new Float32Array([0.1]) },
            sparse: { indices: new Int32Array([0, 5]), values: new Float32Array([0.5, 0.8]) },
          },
        ],
      }),
    );

    await client.encode("bge-m3", { text: "test" }, { outputTypes: ["dense", "sparse"] });

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ params?: { output_types?: string[] } }>(body);

    expect(parsed.params?.output_types).toEqual(["dense", "sparse"]);
  });

  it("should pass instruction in params", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "What is ML?" }, { instruction: "Retrieve passages" });

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ params?: { instruction?: string } }>(body);

    expect(parsed.params?.instruction).toBe("Retrieve passages");
  });

  it("should pass isQuery as is_query in params", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    await client.encode("bge-m3", { text: "query" }, { isQuery: true });

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ params?: { is_query?: boolean } }>(body);

    expect(parsed.params?.is_query).toBe(true);
  });

  it("should allow per-request GPU override", async () => {
    const clientWithDefaultGpu = new SIEClient("http://localhost:8080", { gpu: "l4" });

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1]) } }],
      }),
    );

    // Override default GPU with a100
    await clientWithDefaultGpu.encode("bge-m3", { text: "test" }, { gpu: "a100-80gb" });

    const fetchCall = mockFetch.mock.calls[0];
    const headers = fetchCall?.[1]?.headers as Record<string, string>;
    expect(headers["X-SIE-MACHINE-PROFILE"]).toBe("a100-80gb");

    await clientWithDefaultGpu.close();
  });
});

describe("SIEClient.encode() - response parsing", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should parse dense embeddings from nested wire format", async () => {
    // Wire format: {"dense": {"values": Float32Array}}
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1, 0.2, 0.3]) } }],
      }),
    );

    const result = await client.encode("bge-m3", { text: "test" });

    expect(result.dense).toBeInstanceOf(Float32Array);
    expect(result.dense?.[0]).toBeCloseTo(0.1);
    expect(result.dense?.[1]).toBeCloseTo(0.2);
    expect(result.dense?.[2]).toBeCloseTo(0.3);
  });

  it("should parse sparse embeddings correctly", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            sparse: {
              indices: new Int32Array([0, 10, 100]),
              values: new Float32Array([0.5, 0.8, 0.3]),
            },
          },
        ],
      }),
    );

    const result = await client.encode("bge-m3", { text: "test" }, { outputTypes: ["sparse"] });

    expect(result.sparse?.indices).toBeInstanceOf(Int32Array);
    expect(result.sparse?.values).toBeInstanceOf(Float32Array);
    expect(result.sparse?.indices?.length).toBe(3);
  });

  it("should parse multivector embeddings from nested wire format", async () => {
    // Wire format: {"multivector": {"values": Float32Array[]}}
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            multivector: {
              values: [
                new Float32Array([0.1, 0.2]),
                new Float32Array([0.3, 0.4]),
                new Float32Array([0.5, 0.6]),
              ],
            },
          },
        ],
      }),
    );

    const result = await client.encode(
      "jina-colbert-v2",
      { text: "test" },
      { outputTypes: ["multivector"] },
    );

    expect(result.multivector).toHaveLength(3);
    expect(result.multivector?.[0]).toBeInstanceOf(Float32Array);
    expect(result.multivector?.[0]?.[0]).toBeCloseTo(0.1);
  });
});

describe("SIEClient.listModels()", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should list available models", async () => {
    // User scenario: "I want to see what models are available"
    mockFetch.mockResolvedValueOnce(
      createMockResponse({
        models: [
          { name: "bge-m3", loaded: true, inputs: ["text"], outputs: ["dense", "sparse"] },
          {
            name: "colpali-v1.3",
            loaded: false,
            inputs: ["text", "image"],
            outputs: ["multivector"],
          },
        ],
      }),
    );

    const models = await client.listModels();

    expect(models).toHaveLength(2);
    expect(models[0]?.name).toBe("bge-m3");
    expect(models[0]?.loaded).toBe(true);
    expect(models[1]?.inputs).toContain("image");
  });

  it("should use GET request for models endpoint", async () => {
    mockFetch.mockResolvedValueOnce(createMockResponse({ models: [] }));

    await client.listModels();

    const fetchCall = mockFetch.mock.calls[0];
    expect(fetchCall?.[1]?.method).toBe("GET");
  });

  it("should convert max_sequence_length to maxSequenceLength", async () => {
    mockFetch.mockResolvedValueOnce(
      createMockResponse({
        models: [
          {
            name: "bge-m3",
            loaded: true,
            inputs: ["text"],
            outputs: ["dense"],
            max_sequence_length: 8192,
          },
        ],
      }),
    );

    const models = await client.listModels();

    expect(models[0]?.maxSequenceLength).toBe(8192);
  });
});

describe("SIEClient error handling", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080", { timeout: 1000 });
  });

  afterEach(async () => {
    await client.close();
  });

  it("should throw RequestError for 4xx responses", async () => {
    // User scenario: "I made an invalid request"
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ code: "INVALID_MODEL", detail: "Model not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(client.encode("nonexistent-model", { text: "test" })).rejects.toThrow(
      RequestError,
    );
  });

  it("should throw ServerError for 5xx responses", async () => {
    // User scenario: "The server had an error"
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ code: "INTERNAL_ERROR", detail: "Something broke" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(client.encode("bge-m3", { text: "test" })).rejects.toThrow(ServerError);
  });

  it("should throw SIEConnectionError for fetch failures", async () => {
    // User scenario: "The server is down"
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));

    await expect(client.encode("bge-m3", { text: "test" })).rejects.toThrow(SIEConnectionError);
  });

  it("should throw SIEConnectionError for timeout", async () => {
    // User scenario: "The request took too long"
    // Create a promise that never resolves
    mockFetch.mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          // Simulate abort after timeout
          setTimeout(() => {
            const error = new Error("Aborted");
            error.name = "AbortError";
            reject(error);
          }, 10);
        }),
    );

    await expect(client.encode("bge-m3", { text: "test" })).rejects.toThrow(SIEConnectionError);
  });
});

describe("SIEClient.close()", () => {
  it("should be callable multiple times without error", async () => {
    const client = new SIEClient("http://localhost:8080");

    // Should not throw
    await client.close();
    await client.close();
  });
});

describe("Real-world usage patterns", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should support document embedding workflow", async () => {
    // User scenario: "I want to embed a batch of documents and store them"
    const client = new SIEClient("http://localhost:8080");

    // Simulate embedding 3 documents
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          { id: "doc-1", dense: { values: new Float32Array([0.1, 0.2, 0.3]) } },
          { id: "doc-2", dense: { values: new Float32Array([0.4, 0.5, 0.6]) } },
          { id: "doc-3", dense: { values: new Float32Array([0.7, 0.8, 0.9]) } },
        ],
      }),
    );

    const documents = [
      { id: "doc-1", text: "First document about AI" },
      { id: "doc-2", text: "Second document about ML" },
      { id: "doc-3", text: "Third document about NLP" },
    ];

    const embeddings = await client.encode("bge-m3", documents);

    // User can build an index from results
    const index = new Map<string, Float32Array>();
    for (const result of embeddings) {
      if (result.id && result.dense) {
        index.set(result.id, result.dense);
      }
    }

    expect(index.size).toBe(3);
    expect(index.get("doc-1")?.[0]).toBeCloseTo(0.1);

    await client.close();
  });

  it("should support query embedding with instruction", async () => {
    // User scenario: "I want to embed a search query with instruction"
    const client = new SIEClient("http://localhost:8080");

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ dense: { values: new Float32Array([0.1, 0.2, 0.3]) } }],
      }),
    );

    const queryEmbedding = await client.encode(
      "gte-qwen2-7b",
      { text: "What is machine learning?" },
      {
        instruction: "Retrieve passages that answer this question",
        isQuery: true,
      },
    );

    expect(queryEmbedding.dense).toBeInstanceOf(Float32Array);

    await client.close();
  });

  it("should support multimodal embedding", async () => {
    // User scenario: "I want to embed images for visual search"
    const client = new SIEClient("http://localhost:8080");

    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            multivector: {
              values: [new Float32Array([0.1, 0.2])],
            },
          },
        ],
      }),
    );

    const imageBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]); // JPEG magic bytes
    const result = await client.encode(
      "colpali-v1.3",
      { images: [imageBytes] },
      { outputTypes: ["multivector"] },
    );

    expect(result.multivector).toBeDefined();

    await client.close();
  });
});

describe("SIEClient.score() - reranking", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should score items against a query", async () => {
    // User scenario: "I want to rerank search results"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        scores: [
          { item_id: "doc-2", score: 0.95, rank: 0 },
          { item_id: "doc-1", score: 0.72, rank: 1 },
          { item_id: "doc-3", score: 0.45, rank: 2 },
        ],
      }),
    );

    const result = await client.score("bge-reranker-v2", { text: "What is machine learning?" }, [
      { id: "doc-1", text: "Python is a programming language" },
      { id: "doc-2", text: "Machine learning is a subset of AI" },
      { id: "doc-3", text: "The weather is nice today" },
    ]);

    expect(result.scores).toHaveLength(3);
    expect(result.scores[0]?.itemId).toBe("doc-2"); // Most relevant
    expect(result.scores[0]?.rank).toBe(0);
    expect(result.scores[0]?.score).toBeCloseTo(0.95);
  });

  it("should use correct URL with model in path", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        scores: [],
      }),
    );

    await client.score("bge-reranker-v2", { text: "query" }, [{ text: "doc" }]);

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/v1/score/bge-reranker-v2",
      expect.anything(),
    );
  });

  it("should support topK option", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        scores: [
          { item_id: "doc-1", score: 0.9, rank: 0 },
          { item_id: "doc-2", score: 0.8, rank: 1 },
        ],
      }),
    );

    await client.score("bge-reranker-v2", { text: "query" }, [{ text: "doc1" }, { text: "doc2" }], {
      topK: 2,
    });

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ top_k?: number }>(body);

    expect(parsed.top_k).toBe(2);
  });

  it("should echo query ID if provided", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        query_id: "search-42",
        scores: [{ item_id: "doc-1", score: 0.9, rank: 0 }],
      }),
    );

    const result = await client.score("bge-reranker-v2", { id: "search-42", text: "query" }, [
      { text: "doc" },
    ]);

    expect(result.queryId).toBe("search-42");
  });

  it("should allow per-request GPU override", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        scores: [],
      }),
    );

    await client.score("bge-reranker-v2", { text: "query" }, [{ text: "doc" }], {
      gpu: "a100-80gb",
    });

    const fetchCall = mockFetch.mock.calls[0];
    const headers = fetchCall?.[1]?.headers as Record<string, string>;
    expect(headers["X-SIE-MACHINE-PROFILE"]).toBe("a100-80gb");
  });
});

describe("SIEClient.extract() - NER", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("should extract entities from single item", async () => {
    // User scenario: "I want to extract named entities"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            entities: [
              { text: "Apple", label: "organization", score: 0.95, start: 0, end: 5 },
              { text: "Steve Jobs", label: "person", score: 0.92, start: 22, end: 32 },
            ],
          },
        ],
      }),
    );

    const result = await client.extract(
      "gliner-multi-v2.1",
      { text: "Apple was founded by Steve Jobs." },
      { labels: ["person", "organization"] },
    );

    expect(result.entities).toHaveLength(2);
    expect(result.entities[0]?.text).toBe("Apple");
    expect(result.entities[0]?.label).toBe("organization");
    expect(result.entities[1]?.text).toBe("Steve Jobs");
    expect(result.entities[1]?.label).toBe("person");
  });

  it("should extract entities from batch items", async () => {
    // User scenario: "I want to process many documents"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            id: "doc-1",
            entities: [{ text: "Elon Musk", label: "person", score: 0.9, start: 0, end: 9 }],
          },
          {
            id: "doc-2",
            entities: [{ text: "Google", label: "organization", score: 0.88, start: 0, end: 6 }],
          },
        ],
      }),
    );

    const results = await client.extract(
      "gliner-multi-v2.1",
      [
        { id: "doc-1", text: "Elon Musk announced..." },
        { id: "doc-2", text: "Google released a new product" },
      ],
      { labels: ["person", "organization"] },
    );

    expect(results).toHaveLength(2);
    expect(results[0]?.id).toBe("doc-1");
    expect(results[0]?.entities[0]?.text).toBe("Elon Musk");
    expect(results[1]?.id).toBe("doc-2");
    expect(results[1]?.entities[0]?.text).toBe("Google");
  });

  it("should use correct URL with model in path", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ entities: [] }],
      }),
    );

    await client.extract("gliner-multi-v2.1", { text: "test" }, { labels: ["person"] });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/v1/extract/gliner-multi-v2.1",
      expect.anything(),
    );
  });

  it("should pass labels in params", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ entities: [] }],
      }),
    );

    await client.extract(
      "gliner-multi-v2.1",
      { text: "test" },
      { labels: ["person", "organization", "location"] },
    );

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ params?: { labels?: string[] } }>(body);

    expect(parsed.params?.labels).toEqual(["person", "organization", "location"]);
  });

  it("should pass threshold option in params", async () => {
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [{ entities: [] }],
      }),
    );

    await client.extract(
      "gliner-multi-v2.1",
      { text: "test" },
      { labels: ["person"], threshold: 0.8 },
    );

    const fetchCall = mockFetch.mock.calls[0];
    const body = fetchCall?.[1]?.body as Uint8Array;
    const parsed = unpackMessage<{ params?: { threshold?: number } }>(body);

    expect(parsed.params?.threshold).toBe(0.8);
  });

  it("should return entity positions for highlighting", async () => {
    // User scenario: "I want to highlight entities in my UI"
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        items: [
          {
            entities: [
              { text: "John Smith", label: "person", score: 0.95, start: 0, end: 10 },
              { text: "Acme Corp", label: "organization", score: 0.88, start: 20, end: 29 },
            ],
          },
        ],
      }),
    );

    const result = await client.extract(
      "gliner-multi-v2.1",
      { text: "John Smith works at Acme Corp as a developer." },
      { labels: ["person", "organization"] },
    );

    // User can use positions to highlight text
    const text = "John Smith works at Acme Corp as a developer.";
    for (const entity of result.entities) {
      if (entity.start !== undefined && entity.end !== undefined) {
        const extracted = text.slice(entity.start, entity.end);
        expect(extracted).toBe(entity.text);
      }
    }
  });
});

describe("Real-world reranking workflow", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should support typical RAG reranking flow", async () => {
    // User scenario: "First retrieve candidates, then rerank"
    const client = new SIEClient("http://localhost:8080");

    // First: Get initial candidates from vector search (simulated)
    const candidates = [
      { id: "chunk-1", text: "Machine learning models require training data." },
      { id: "chunk-2", text: "Python is popular for data science." },
      { id: "chunk-3", text: "Deep learning is a subset of machine learning." },
      { id: "chunk-4", text: "The sky is blue on sunny days." },
    ];

    // Second: Rerank candidates
    mockFetch.mockResolvedValueOnce(
      createMsgpackResponse({
        model: "bge-reranker-v2",
        scores: [
          { item_id: "chunk-3", score: 0.95, rank: 0 },
          { item_id: "chunk-1", score: 0.88, rank: 1 },
          { item_id: "chunk-2", score: 0.45, rank: 2 },
          { item_id: "chunk-4", score: 0.12, rank: 3 },
        ],
      }),
    );

    const result = await client.score(
      "bge-reranker-v2",
      { text: "What is machine learning?" },
      candidates,
    );

    // Get top 2 for context
    const topChunks = result.scores.slice(0, 2).map((s) => s.itemId);
    expect(topChunks).toEqual(["chunk-3", "chunk-1"]);

    await client.close();
  });
});
