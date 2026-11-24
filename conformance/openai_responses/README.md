# OpenAI Responses conformance vectors

`POST /v1/responses` is served by translating the request onto
`/v1/chat/completions` and serving it in process with the chat route of the
topology (the gateway, or `sie-server` without a gateway), then translating the
answer back. This is the shape of the Anthropic Messages endpoint
(`/v1/messages`, [#753](https://github.com/superlinked/sie/pull/753) and
[#815](https://github.com/superlinked/sie/pull/815)): every chat gate, limit, timeout and meter
applies unchanged, and errors use the OpenAI envelope
`{"error":{"message","type","param","code"}}` that chat already uses.

The files here are the executable specification for that translation
([#1013](https://github.com/superlinked/sie/issues/1013)). The gateway and `sie-server` implementations both
replay every file, so a request is accepted, refused and answered the same way
on both.

| File | Contents |
|---|---|
| `request_vectors.json` | `accepted`: a Responses body and the exact chat body it becomes. `rejected`: a Responses body and the refusal (`status`, `type`, `code`, `param`, `message`). |
| `response_vectors.json` | A chat completion, the Responses request it answers, and the Response it becomes. |
| `stream_vectors.json` | Chat SSE `data` payloads (JSON chunks and the literal `"[DONE]"`) and the Responses events they become. |
| `error_vectors.json` | A chat error answer returned before the stream opens and the answer `/v1/responses` returns. |
| `codex_requests.json` | Requests Codex CLI 0.167.0 sent, scrubbed, with the expected refusal or a summary of the translated chat body. |

Every accepted chat body must also pass the chat route's own validation.
`created_at` is the only nondeterministic field of an answer; the vectors use
`0` and tests inject the clock. Item ids derive from the request id: the
response is `resp-<request_id>`, the message item `msg-<request_id>`, and the
tool call at position `k` (0-based) `fc-<request_id>-<k>` or
`ctc-<request_id>-<k>`. In a stream, a message item that follows a tool call is
`msg-<request_id>-<output_index>`.

## Request

### Top-level fields

| Responses field | Chat body |
|---|---|
| `model` (required string) | `model` |
| `input` (required: a string, or a non-empty array of items) | `messages` |
| `instructions` (string or null) | first part of the leading system message |
| `max_output_tokens` (positive integer) | `max_completion_tokens`; omitted when absent or null |
| `temperature`, `top_p`, `seed` | same name; chat validates them |
| `stream` (boolean) | `stream`; when true also `"stream_options": {"include_usage": true}` |
| `tools` | `tools`; an empty list is omitted |
| `tool_choice` | `tool_choice` |
| `parallel_tool_calls` | `parallel_tool_calls` |
| `reasoning.effort` | `reasoning_effort`, passed as sent; chat validates it |
| `text.format` | `{"type":"text"}`: nothing. `{"type":"json_object"}`: `response_format: {"type":"json_object"}`. `{"type":"json_schema","name","schema","strict"?,"description"?}`: `response_format: {"type":"json_schema","json_schema":{"name","schema","strict"?,"description"?}}` |
| `metadata`, `prompt_cache_key`, `user`, `safety_identifier`, `service_tier` | same name, passed through |

Every chat body carries `"include_reasoning": false`, so model thinking stays
private, and `"stream"` (false when absent). An accepted field sent as `null`
is treated as absent.

Accepted with no effect and never logged: `store` false, `background` false,
`truncation: "disabled"`, `include` when every element is
`"reasoning.encrypted_content"`, `client_metadata`, `reasoning.summary`,
`reasoning.generate_summary` and `text.verbosity`.

### Refusals

A refusal is a 400 with `type: "invalid_request_error"`, a `param` naming the
field as the caller sent it, and the message `"<param>: <reason>"`.

`code: "unsupported_field"`:

- `store: true`, and a non-null `previous_response_id` or `conversation`:
  `SIE does not store responses; send the full conversation in input`.
- `background: true`: `background responses are not supported`.
- `truncation` other than `"disabled"`: `only "disabled" is supported`.
- An `include` element other than `"reasoning.encrypted_content"`: param
  `include[i]`, `'<value>' is not supported`.
- `prompt`, `max_tool_calls`, `top_logprobs`, `stream_options`, any other
  unknown top-level field, and an unknown key in `reasoning`, `text`,
  `text.format`, a tool, an input item or a content part:
  `Extra inputs are not permitted`.
- An unknown `text.format.type`: `'<type>' is not supported`.
- The input item, part, tool and `tool_choice` refusals below.

`code: "invalid_request"`: a missing `model` or `input` (`Field required`), an
empty `input` list, a value of the wrong type or range (for example
`max_output_tokens: "16"` or `0`), an unknown role, and a named `tool_choice`
whose tool is not declared.

### Input items

A string `input` becomes one user message. In an array, an object with `role`
and no `type` is a message. The accepted item types are `message`,
`function_call`, `function_call_output`, `custom_tool_call`,
`custom_tool_call_output` and `reasoning`; every item may carry `id` and
`status`, which are ignored. Any other type is refused with param
`input[i].type`: `items of type '<type>' are not supported`.

- **System text.** `instructions`, then every `system` and `developer` message
  that comes before the first item of another kind, joined with `"\n\n"`,
  form one leading chat system message. A `system` or `developer` message
  after that is translated in place as a chat `system` message.
- **Content.** `content` is a string or a list of parts: `input_text`,
  `output_text` (its `annotations` and `logprobs` are ignored), `text`, and,
  in user messages only, `input_image`. Texts are concatenated without a
  separator. A message with an image becomes a list of chat `text` and
  `image_url` parts in order, adjacent texts joined and empty texts dropped.
  `input_image` takes a base64 `data:` URL (`detail` is ignored); an http(s)
  URL or a `file_id` is refused at that field with
  `only base64 data: URLs are supported`. Other part types (`input_file`,
  `input_audio`, `refusal` and the rest) are refused with param
  `input[i].content[j].type`.
- **Roles.** `user` and `assistant` keep their role; `system` and `developer`
  become `system`. Any other role is refused.
- **Assistant output.** An assistant message and the `function_call` and
  `custom_tool_call` items next to it become one chat assistant message: the
  concatenated text (`null` when there is none and there are calls) and the
  calls in order. A `function_call` becomes
  `{"id": call_id, "type": "function", "function": {"name", "arguments"}}` with
  `arguments` as sent; a `custom_tool_call` passes its `input` as the
  arguments `{"input": <input>}`, serialized as compact JSON.
- **Tool outputs.** `function_call_output` and `custom_tool_call_output` become
  `{"role": "tool", "tool_call_id": call_id, "content": <text>}`. `output` is a
  string or a list of text parts (concatenated) and `input_image` parts. The
  images of a run of consecutive tool outputs become one chat user message
  right after that run's tool messages. When the next item is a user message,
  its content follows the images in that same message, as `/v1/messages` does.
- **Reasoning.** `reasoning` items are dropped. They do not end a run of
  assistant output or of tool outputs.

### Tools and tool_choice

- `{"type":"function","name","description"?,"parameters"?,"strict"?}` becomes
  `{"type":"function","function":{"name","description"?,"parameters"?}}`.
  `strict: false` is accepted and dropped; `strict: true` is refused at
  `tools[i].strict`: `strict function schemas are not supported`.
- `{"type":"custom","name","description"?,"format"?}` becomes a function whose
  parameters are one required string `input`. `format` is accepted and not
  enforced. A call to a custom tool comes back as a `custom_tool_call` item.
- Every other tool type (`web_search`, `web_search_preview`, `file_search`,
  `code_interpreter`, `computer_use_preview`, `mcp`, `local_shell`,
  `image_generation`, `namespace` and unknown types) is refused at
  `tools[i].type`: `'<type>' tools are not supported; send function or custom tools`.
- Tool names and the tool count are checked by the chat route (64 tools, names
  matching `^[a-zA-Z0-9_-]{1,64}$`); its error comes back with the param
  `tools[i].function.name` renamed to `tools[i].name` (see [Errors](#errors)).
- `tool_choice` `"auto"`, `"none"` and `"required"` pass through.
  `{"type":"function","name"}` and `{"type":"custom","name"}` become
  `{"type":"function","function":{"name"}}`; the name must be a declared tool.
  `{"type":"allowed_tools"}` and hosted-tool choices are refused at
  `tool_choice.type`. Without tools, `"auto"` and `"none"` are dropped.

## Buffered answer

A chat completion with the content `"Hello."` for request `r1` becomes:

```json
{
  "id": "resp-r1", "object": "response", "created_at": 0,
  "model": "m", "status": "completed",
  "incomplete_details": null, "error": null, "store": false,
  "parallel_tool_calls": true, "tool_choice": "auto",
  "output": [
    {"type": "message", "id": "msg-r1", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "Hello.", "annotations": []}]}
  ],
  "usage": {
    "input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 13
  }
}
```

`model` is the model the request named. `parallel_tool_calls` and
`tool_choice` echo the request as sent, defaulting to `true` and `"auto"`. `output` holds a `message` item with one `output_text`
part when the chat content is non-empty, then one item per tool call: a
`function_call` with the arguments string, or a `custom_tool_call` whose
`input` is the `input` string of the arguments, or the raw arguments when they
are not a JSON object with a string `input`. The finish reasons `length` and
`content_filter` make the response `incomplete` with
`incomplete_details.reason` `max_output_tokens` or `content_filter`; every
other finish reason (`stop`, `tool_calls` or any other value) completes it. Usage is `input_tokens`, `input_tokens_details.cached_tokens`,
`output_tokens`, `output_tokens_details.reasoning_tokens` and `total_tokens`
from the chat usage, with absent details counted as 0.

## Streaming

Each event is sent as `event: <type>` and `data: <JSON>`. Every `data` object
carries `type`, equal to the event name, and `sequence_number`, counting from
0. Lifecycle events (`response.created`, `response.in_progress`,
`response.completed`, `response.incomplete`, `response.failed`) carry only
`response`, the buffered-answer object, which is `in_progress` with an empty
`output` and `usage: null` until the terminal event. `output_item.added` and
`output_item.done` carry `output_index` and `item`. The other item events carry
`item_id` and `output_index`, and the text and content-part events also carry
`content_index: 0`.

1. `response.created` and `response.in_progress` open the stream.
2. The first non-empty text delta adds the message item and its `output_text`
   part; each text delta is a `response.output_text.delta`.
3. Text ends at the first tool-call delta or at the finish:
   `output_text.done`, `content_part.done`, `output_item.done`. Text that
   arrives after a tool call ends that call and opens a new message item at
   the next `output_index`, with id `msg-<request_id>-<output_index>`.
4. Each new chat tool call adds a `function_call` item with empty arguments,
   then one `function_call_arguments.delta` per non-empty arguments fragment.
   A custom tool call adds a `custom_tool_call` item; its fragments are
   buffered and sent at the end as one `custom_tool_call_input.delta` with the
   decoded input. A call ends at the next call or at the finish with
   `function_call_arguments.done` (or `custom_tool_call_input.done`) and
   `output_item.done`.
5. After the usage chunk and `[DONE]`, `response.completed` (or
   `response.incomplete` for a `length` or `content_filter` finish) carries the
   full answer; a chat stream that closes after a finish without `[DONE]` ends
   the same way. The stream ends there, without a `[DONE]` frame of its own.
6. A chat error chunk after the stream opened becomes `response.failed` with
   `status: "failed"`, `error: {"code", "message"}` from the chat error, the
   items completed so far as `output` (an item still open is left out) and
   `usage: null`, and the stream ends.
7. A chat stream that ends without a finish reason fails the same way with
   `error: {"code": "server_error", "message": "the generation stream ended without a finish reason"}`,
   and one that breaks with
   `error: {"code": "server_error", "message": "the generation stream ended unexpectedly"}`.
8. Tool-call chunks are routed by their chat `index`, as on `/v1/messages`. A
   call starts at the first chunk for an index the stream has not used; that
   chunk must carry the call `id` and function `name`, otherwise the stream
   fails with `server_error` and
   `"a tool call arrived without its id and name"`. A non-empty arguments
   fragment for a call whose item has already closed, because a later call or
   text started, fails the stream with `server_error` and
   `"tool call arguments arrived after another output item started"`.

## Errors

A chat error returned before the stream opens is returned as it is: the same
status, `Retry-After` and `x-sie-error-code` headers, and OpenAI error body.
The route adds no header of its own. The only change is the `param` when it
names a chat field, which is renamed to the Responses field the caller sent;
the message keeps the chat route's wording.

| Chat `param` | Responses `param` |
|---|---|
| `tools[i].function.<path>` | `tools[i].<path>` |
| `max_completion_tokens` | `max_output_tokens` |
| `reasoning_effort` | `reasoning.effort` |
| `response_format` | `text.format` |
| `response_format.json_schema.<path>` | `text.format.<path>` |

Every other param, for example `messages[i].content[j].image_url`, is returned
unchanged.

## Codex CLI requests

`codex_requests.json` holds requests Codex CLI 0.167.0 sent in headless runs
against a local stub; no model was called. Developer instructions and the
environment context are replaced by placeholders with their original length;
ids, UUIDs, `client_metadata` values and local paths are replaced; headers are
dropped. Tool definitions and the `input_image` data URL are as captured. With
Codex's default configuration the request carries a `namespace` tool and a
`web_search` tool and is refused at the first of them; with web search,
multi-agent and goal tools turned off, every request translates.
