/**
 * SIEClient.responses() and SIEClient.streamResponses() tests.
 *
 * All tests use a mocked `fetch`; SSE bodies are built inline as
 * `event: <type>\ndata: <json>\n\n` frames with no `[DONE]` terminator.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  ProvisioningError,
  RequestError,
  SIEConnectionError,
  SIEStreamError,
} from "../src/errors.js";
import type {
  ResponseInputItem,
  ResponseOutputItem,
  ResponseRequest,
  ResponseResult,
  ResponseStreamEvent,
  ResponseTool,
} from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

type Frame = Record<string, unknown>;

function sseResponse(frames: Frame[]): Response {
  const encoder = new TextEncoder();
  const text = frames
    .map((frame, sequence_number) => {
      const event = typeof frame.type === "string" ? `event: ${frame.type}\n` : "";
      return `${event}data: ${JSON.stringify({ ...frame, sequence_number })}\n\n`;
    })
    .join("");
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function response(
  status: ResponseResult["status"],
  output: ResponseOutputItem[] = [],
  fields: Partial<ResponseResult> = {},
): ResponseResult {
  return {
    id: "resp-req-1",
    object: "response",
    created_at: 0,
    model: "m",
    status,
    incomplete_details: null,
    error: null,
    store: false,
    parallel_tool_calls: true,
    tool_choice: "auto",
    output,
    usage:
      status === "in_progress"
        ? null
        : {
            input_tokens: 3,
            input_tokens_details: { cached_tokens: 0 },
            output_tokens: 2,
            output_tokens_details: { reasoning_tokens: 0 },
            total_tokens: 5,
          },
    ...fields,
  };
}

const opening: Frame[] = [
  { type: "response.created", response: response("in_progress") },
  { type: "response.in_progress", response: response("in_progress") },
];

function message(text: string, status: "in_progress" | "completed"): ResponseOutputItem {
  return {
    type: "message",
    id: "msg-req-1",
    role: "assistant",
    status,
    content: status === "in_progress" ? [] : [{ type: "output_text", text, annotations: [] }],
  };
}

function textEvents(...deltas: string[]): Frame[] {
  const text = deltas.join("");
  const part = { type: "output_text", text, annotations: [] };
  const location = { item_id: "msg-req-1", output_index: 0, content_index: 0 };
  return [
    { type: "response.output_item.added", output_index: 0, item: message("", "in_progress") },
    { type: "response.content_part.added", ...location, part: { ...part, text: "" } },
    ...deltas.map((delta) => ({ type: "response.output_text.delta", ...location, delta })),
    { type: "response.output_text.done", ...location, text },
    { type: "response.content_part.done", ...location, part },
    { type: "response.output_item.done", output_index: 0, item: message(text, "completed") },
  ];
}

function functionCall(status: "in_progress" | "completed", args: string): ResponseOutputItem {
  return {
    type: "function_call",
    id: "fc-req-1-0",
    call_id: "call_1",
    name: "get_weather",
    arguments: args,
    status,
  };
}

function customToolCall(status: "in_progress" | "completed", input: string): ResponseOutputItem {
  return {
    type: "custom_tool_call",
    id: "ctc-req-1-0",
    call_id: "call_2",
    name: "run_python",
    input,
    status,
  };
}

const functionCallEvents: Frame[] = [
  { type: "response.output_item.added", output_index: 0, item: functionCall("in_progress", "") },
  {
    type: "response.function_call_arguments.delta",
    item_id: "fc-req-1-0",
    output_index: 0,
    delta: '{"city":',
  },
  {
    type: "response.function_call_arguments.delta",
    item_id: "fc-req-1-0",
    output_index: 0,
    delta: '"Paris"}',
  },
  {
    type: "response.function_call_arguments.done",
    item_id: "fc-req-1-0",
    output_index: 0,
    arguments: '{"city":"Paris"}',
  },
  {
    type: "response.output_item.done",
    output_index: 0,
    item: functionCall("completed", '{"city":"Paris"}'),
  },
];

const customToolCallEvents: Frame[] = [
  { type: "response.output_item.added", output_index: 0, item: customToolCall("in_progress", "") },
  {
    type: "response.custom_tool_call_input.delta",
    item_id: "ctc-req-1-0",
    output_index: 0,
    delta: "print(1)",
  },
  {
    type: "response.custom_tool_call_input.done",
    item_id: "ctc-req-1-0",
    output_index: 0,
    input: "print(1)",
  },
  {
    type: "response.output_item.done",
    output_index: 0,
    item: customToolCall("completed", "print(1)"),
  },
];

const IMAGE = "data:image/png;base64,iVBORw0KGgo=";

const TOOLS: ResponseTool[] = [
  {
    type: "function",
    name: "get_weather",
    description: "Look up the weather",
    parameters: { type: "object", properties: { city: { type: "string" } } },
    strict: false,
  },
  { type: "custom", name: "run_python", description: "Run code", format: { type: "text" } },
];

const INPUT: ResponseInputItem[] = [
  { role: "developer", content: "Be brief." },
  {
    role: "user",
    content: [
      { type: "input_text", text: "What is this?" },
      { type: "input_image", image_url: IMAGE, detail: "low" },
    ],
  },
  { type: "function_call", call_id: "call_1", name: "get_weather", arguments: '{"city":"Paris"}' },
  { type: "function_call_output", call_id: "call_1", output: "sunny" },
  { type: "custom_tool_call", call_id: "call_2", name: "run_python", input: "print(1)" },
  {
    type: "custom_tool_call_output",
    call_id: "call_2",
    output: [
      { type: "input_text", text: "1" },
      { type: "input_image", image_url: IMAGE },
    ],
  },
];

const FULL_REQUEST: ResponseRequest = {
  model: "m",
  input: INPUT,
  instructions: "You are a helpful assistant.",
  max_output_tokens: 64,
  temperature: 0.2,
  top_p: 0.9,
  seed: -7,
  tools: TOOLS,
  tool_choice: { type: "custom", name: "run_python" },
  parallel_tool_calls: false,
  reasoning: { effort: "low" },
  text: {
    format: { type: "json_schema", name: "answer", schema: { type: "object" }, strict: true },
  },
  metadata: { tenant: "t-1" },
  prompt_cache_key: "cache-1",
  user: "user-1",
  safety_identifier: "safety-1",
};

async function collect(stream: AsyncIterable<ResponseStreamEvent>): Promise<ResponseStreamEvent[]> {
  const events: ResponseStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe("SIEClient.responses", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("POSTs every typed field to /v1/responses with stream:false", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(response("completed", [message("Hello", "completed")]), 200, {
        "x-sie-request-id": "req-1",
        "x-sie-units-output-tokens": "2",
        "x-sie-credits-debited": "9",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const result = await client.responses(FULL_REQUEST);

    expect(result.status).toBe("completed");
    expect(result.output[0]).toEqual(message("Hello", "completed"));
    expect(result.request).toEqual({ id: "req-1", usage: { outputTokens: 2 }, creditsDebited: 9 });
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/responses");
    expect(init.method).toBe("POST");
    expect(init.headers.Accept).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ ...FULL_REQUEST, stream: false });
  });

  it.each<[keyof ResponseRequest, unknown]>([
    ["tool_choice", "required"],
    ["tool_choice", { type: "function", name: "get_weather" }],
    ["reasoning", { effort: null }],
    ["text", { format: { type: "json_object" } }],
    ["instructions", null],
  ])("forwards %s unchanged", async (field, value) => {
    mockFetch.mockResolvedValueOnce(jsonResponse(response("completed")));

    const client = new SIEClient("http://localhost:8080");
    await client.responses({ model: "m", input: "Hi", [field]: value } as ResponseRequest);

    const [, init] = mockFetch.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      model: "m",
      input: "Hi",
      [field]: value,
      stream: false,
    });
  });

  it("returns function and custom tool call output items", async () => {
    const output = [
      functionCall("completed", '{"city":"Paris"}'),
      customToolCall("completed", "1"),
    ];
    mockFetch.mockResolvedValueOnce(jsonResponse(response("completed", output)));

    const client = new SIEClient("http://localhost:8080");
    const result = await client.responses({ model: "m", input: "Hi", tools: TOOLS });

    expect(result.output).toEqual(output);
  });

  it("throws RequestError when stream:true is passed", async () => {
    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.responses({ model: "m", input: "Hi", stream: true }),
    ).rejects.toBeInstanceOf(RequestError);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("rejects an unsafe seed before network I/O", async () => {
    const client = new SIEClient("http://localhost:8080");
    await expect(client.responses({ model: "m", input: "Hi", seed: 1.5 })).rejects.toThrow(
      RangeError,
    );
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("throws ProvisioningError on 503 PROVISIONING when waitForCapacity is false", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ error: { code: "PROVISIONING", message: "warming" } }, 503, {
        "Retry-After": "7",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.responses({ model: "m", input: "Hi" }, { waitForCapacity: false }),
    ).rejects.toBeInstanceOf(ProvisioningError);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it.each([
    [
      "detail",
      {
        detail: { code: "UNAUTHORIZED", message: "Missing Authorization or x-api-key header" },
      },
    ],
    [
      "openai",
      {
        error: {
          message: "Missing Authorization or x-api-key header",
          type: "authentication_error",
          param: null,
          code: "UNAUTHORIZED",
        },
      },
    ],
  ])("throws the same RequestError for a 401 in the %s envelope", async (_envelope, body) => {
    mockFetch.mockResolvedValueOnce(jsonResponse(body, 401, { "WWW-Authenticate": "Bearer" }));

    const client = new SIEClient("http://localhost:8080");
    const error = await client
      .responses({ model: "m", input: "Hi" })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(RequestError);
    expect((error as RequestError).constructor).toBe(RequestError);
    expect((error as RequestError).statusCode).toBe(401);
    expect((error as RequestError).code).toBe("UNAUTHORIZED");
    expect((error as RequestError).message).toBe("Missing Authorization or x-api-key header");
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("retries 503 MODEL_LOADING then returns the result", async () => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(
          jsonResponse({ error: { code: "MODEL_LOADING", message: "loading" } }, 503),
        )
        .mockResolvedValueOnce(jsonResponse(response("completed")));

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const promise = client.responses({ model: "m", input: "Hi" });
      await vi.advanceTimersByTimeAsync(5_000);
      const result = await promise;

      expect(result.status).toBe("completed");
      expect(mockFetch).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("SIEClient.streamResponses", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it.each<[string, Frame[], string]>([
    [
      "text",
      [
        ...opening,
        ...textEvents("Hel", "lo"),
        {
          type: "response.completed",
          response: response("completed", [message("Hello", "completed")]),
        },
      ],
      "response.completed",
    ],
    [
      "function call",
      [
        ...opening,
        ...functionCallEvents,
        {
          type: "response.completed",
          response: response("completed", [functionCall("completed", '{"city":"Paris"}')]),
        },
      ],
      "response.completed",
    ],
    [
      "custom tool call",
      [
        ...opening,
        ...customToolCallEvents,
        {
          type: "response.completed",
          response: response("completed", [customToolCall("completed", "print(1)")]),
        },
      ],
      "response.completed",
    ],
    [
      "incomplete",
      [
        ...opening,
        ...textEvents("Hel"),
        {
          type: "response.incomplete",
          response: response("incomplete", [message("Hel", "completed")], {
            incomplete_details: { reason: "max_output_tokens" },
          }),
        },
      ],
      "response.incomplete",
    ],
  ])("yields typed %s events until the terminal event", async (_name, frames, terminal) => {
    mockFetch.mockResolvedValueOnce(sseResponse(frames));

    const client = new SIEClient("http://localhost:8080");
    const events = await collect(client.streamResponses(FULL_REQUEST));

    expect(events).toEqual(frames.map((frame, sequence_number) => ({ ...frame, sequence_number })));
    expect(events.at(-1)?.type).toBe(terminal);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/responses");
    expect(init.headers.Accept).toBe("text/event-stream");
    expect(JSON.parse(init.body)).toEqual({ ...FULL_REQUEST, stream: true });
  });

  it("narrows events by type", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        ...opening,
        ...textEvents("Hel", "lo"),
        {
          type: "response.completed",
          response: response("completed", [message("Hello", "completed")]),
        },
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    let text = "";
    let usage: ResponseResult["usage"] = null;
    for await (const event of client.streamResponses({ model: "m", input: "Hi" })) {
      if (event.type === "response.output_text.delta") text += event.delta;
      if (event.type === "response.completed") usage = event.response.usage;
    }

    expect(text).toBe("Hello");
    expect(usage).toMatchObject({ total_tokens: 5 });
  });

  it("throws SIEStreamError on response.failed and never yields it", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        ...opening,
        ...textEvents("Hel"),
        {
          type: "response.failed",
          response: response("failed", [], { error: { code: "inference_error", message: "boom" } }),
        },
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const seen: string[] = [];
    const error = await (async () => {
      try {
        for await (const event of client.streamResponses({ model: "m", input: "Hi" })) {
          seen.push(event.type);
        }
      } catch (err) {
        return err;
      }
      return undefined;
    })();

    expect(error).toBeInstanceOf(SIEStreamError);
    expect((error as SIEStreamError).code).toBe("inference_error");
    expect((error as SIEStreamError).message).toBe("boom");
    expect(seen).not.toContain("response.failed");
    expect(seen).toHaveLength(opening.length + textEvents("Hel").length);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("throws SIEStreamError on response.failed without an error object", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([...opening, { type: "response.failed", response: response("failed") }]),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(collect(client.streamResponses({ model: "m", input: "Hi" }))).rejects.toThrow(
      SIEStreamError,
    );
  });

  it("throws SIEStreamError on an SSE error frame with its request id", async () => {
    mockFetch.mockResolvedValueOnce(
      sseResponse([
        ...opening,
        {
          error: { code: "first_chunk_timeout", message: "timed out", param: null },
          request_id: "req-err",
        },
      ]),
    );

    const client = new SIEClient("http://localhost:8080");
    const error = await collect(client.streamResponses({ model: "m", input: "Hi" })).catch(
      (err: unknown) => err,
    );

    expect(error).toBeInstanceOf(SIEStreamError);
    expect((error as SIEStreamError).code).toBe("first_chunk_timeout");
    expect((error as SIEStreamError).requestId).toBe("req-err");
  });

  it("throws SIEConnectionError when the stream ends before a terminal event", async () => {
    mockFetch.mockResolvedValueOnce(sseResponse([...opening, ...textEvents("Hel")]));

    const client = new SIEClient("http://localhost:8080");
    await expect(collect(client.streamResponses({ model: "m", input: "Hi" }))).rejects.toThrow(
      SIEConnectionError,
    );
  });

  it("rejects an unsafe seed before opening the stream", async () => {
    const client = new SIEClient("http://localhost:8080");
    await expect(
      collect(client.streamResponses({ model: "m", input: "Hi", seed: Number.NaN })),
    ).rejects.toThrow(RangeError);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it.each([
    ["PROVISIONING", 5_000],
    ["MODEL_LOADING", 5_000],
  ])("retries a 503 %s before the first event", async (code, delayMs) => {
    vi.useFakeTimers();
    try {
      mockFetch
        .mockResolvedValueOnce(jsonResponse({ error: { code, message: "not yet" } }, 503))
        .mockResolvedValueOnce(
          sseResponse([
            ...opening,
            { type: "response.completed", response: response("completed") },
          ]),
        );

      const client = new SIEClient("http://localhost:8080", {
        timeout: 30_000,
        provisionTimeout: 60_000,
      });
      const consume = collect(client.streamResponses({ model: "m", input: "Hi" }));
      await vi.advanceTimersByTimeAsync(delayMs);
      const events = await consume;

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(events.map((event) => event.type)).toEqual([
        "response.created",
        "response.in_progress",
        "response.completed",
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("throws ProvisioningError without retrying when the client opts out of waiting", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ error: { code: "PROVISIONING", message: "warming" } }, 503),
    );

    const client = new SIEClient("http://localhost:8080", { waitForCapacity: false });
    await expect(collect(client.streamResponses({ model: "m", input: "Hi" }))).rejects.toThrow(
      ProvisioningError,
    );
    expect(mockFetch).toHaveBeenCalledOnce();
  });
});
