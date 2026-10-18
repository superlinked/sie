/**
 * SIEClient.systemOne: System One typed decisions over extract, checked
 * against the shared golden fixture `wire-fixtures/systemone.json`.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { RequestError } from "../src/errors.js";
import { packMessage, unpackMessage } from "../src/msgpack.js";
import {
  choiceConfidence,
  scoreConfidence,
  systemOneRequest,
  systemOneResult,
} from "../src/systemone.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const fixturesDir = fileURLToPath(new URL("../../wire-fixtures/", import.meta.url));
// biome-ignore lint/suspicious/noExplicitAny: golden fixture JSON
const FIXTURE: any = JSON.parse(readFileSync(`${fixturesDir}systemone.json`, "utf8"));
const RESPONSE_CASE = FIXTURE.responses[0];
const MODEL: string = RESPONSE_CASE.model;
// Invalid requests only the HTTP route can see: the SDK takes the model as an
// argument and sends no media fields.
const ROUTE_ONLY_INVALID_REQUESTS = new Set([
  "missing model",
  "empty model",
  "images",
  "audio",
  "videos",
]);
const QUESTIONS: Record<string, unknown> = Object.fromEntries(
  (RESPONSE_CASE.questions as string[]).map((qid) => [qid, { type: "noul" }]),
);

function extractResponse(
  item: Record<string, unknown>,
  headers: Record<string, string> = {},
): Response {
  return new Response(packMessage({ model: MODEL, items: [item] }), {
    status: 200,
    headers: { "Content-Type": "application/msgpack", ...headers },
  });
}

function expectClose(actual: unknown, expected: unknown, path = "$"): void {
  if (expected !== null && typeof expected === "object") {
    const actualObject = actual as Record<string, unknown>;
    expect(Object.keys(actualObject).sort(), path).toEqual(Object.keys(expected).sort());
    for (const [key, value] of Object.entries(expected)) {
      expectClose(actualObject[key], value, `${path}.${key}`);
    }
  } else if (typeof expected === "number" && !Number.isInteger(expected)) {
    expect(actual as number, path).toBeCloseTo(expected, 9);
  } else {
    expect(actual, path).toEqual(expected);
  }
}

describe("SIEClient.systemOne()", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockReset();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("sends one strict extract item and reshapes the answers", async () => {
    mockFetch.mockResolvedValueOnce(
      extractResponse(
        { id: "item-0", entities: [], data: RESPONSE_CASE.data },
        { "X-SIE-Units-Input-Tokens": "42" },
      ),
    );

    const result = await client.systemOne(MODEL, "I was charged twice.", QUESTIONS);

    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/v1/extract/");
    // biome-ignore lint/suspicious/noExplicitAny: decoded msgpack request body
    const body = unpackMessage(new Uint8Array(init.body as ArrayBuffer)) as Record<string, any>;
    expect(body.items).toEqual([{ text: "I was charged twice." }]);
    expect(body.params.options.overflow_policy).toBe("error");
    expect(Object.keys(body.params.output_schema)).toEqual(RESPONSE_CASE.questions);
    expectClose(result, RESPONSE_CASE.response);
  });

  it("throws RequestError when the state does not fit", async () => {
    mockFetch.mockResolvedValueOnce(
      extractResponse({
        id: "item-0",
        entities: [],
        data: {},
        error: { code: "INPUT_TOO_LONG", message: "does not fit" },
      }),
    );
    await expect(client.systemOne(MODEL, "x", QUESTIONS)).rejects.toBeInstanceOf(RequestError);
  });
});

describe("System One fixture", () => {
  it.each(FIXTURE.requests)("builds the extract call: $name", (testCase) => {
    const { item, outputSchema } = systemOneRequest(
      testCase.request.state,
      testCase.request.questions,
    );
    expect(item).toEqual(testCase.extract.item);
    expect(outputSchema).toEqual(testCase.extract.output_schema);
  });

  it.each(
    FIXTURE.invalid_requests.filter(
      // biome-ignore lint/suspicious/noExplicitAny: golden fixture JSON
      (testCase: any) => !ROUTE_ONLY_INVALID_REQUESTS.has(testCase.name),
    ),
  )("refuses: $name", (testCase) => {
    expect(() =>
      systemOneRequest(testCase.request.state, testCase.request.questions ?? {}),
    ).toThrow(RequestError);
  });

  it.each(FIXTURE.responses)("renders the response: $name", (testCase) => {
    const questions = Object.fromEntries(
      (testCase.questions as string[]).map((qid) => [qid, { type: "noul" }]),
    );
    const result = systemOneResult(
      testCase.model,
      {
        id: "item-0",
        entities: [],
        data: testCase.data,
        request: { usage: { inputTokens: testCase.input_tokens } },
      },
      questions,
    );
    expectClose(result, testCase.response);
  });

  it.each([
    { "0": 0.5, high: 0.5 },
    { "0": 0.5, "1": "half" },
  ])("refuses unreadable probabilities: %o", (probabilities) => {
    const result = { id: "item-0", entities: [], data: { q: { type: "score", probabilities } } };
    expect(() => systemOneResult(MODEL, result, { q: { type: "score" } })).toThrow(RequestError);
  });

  it("matches the confidence goldens", () => {
    for (const testCase of FIXTURE.confidence.choice) {
      expect(choiceConfidence(testCase.probabilities)).toBeCloseTo(testCase.confidence, 9);
    }
    for (const testCase of FIXTURE.confidence.score) {
      expect(scoreConfidence(testCase.probabilities)).toBeCloseTo(testCase.confidence, 9);
    }
  });
});
