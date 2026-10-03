import type { ModelInfo, WireModelInfo } from "../../src/types.js";

// Four `/v1/models` fields are emitted as an explicit null rather than omitted:
// the gateway hardcodes `last_error: null` and serializes `max_sequence_length`
// / `revision` as `Option<_>` with no skip_serializing_if; the server declares
// `last_error`, `max_sequence_length`, `revision` and `capabilities` as
// `X | None`, and pydantic emits each as null.
//
// `?:` alone only permits omission, so both contracts must admit null on each
// of the four. Reverting any of them to a bare `?: T` fails this file.
const wireWithExplicitNulls: WireModelInfo = {
  name: "BAAI/bge-m3",
  loaded: false,
  state: "available",
  inputs: ["text"],
  outputs: ["dense"],
  last_error: null,
  max_sequence_length: null,
  revision: null,
  capabilities: null,
};

const modelWithExplicitNulls: ModelInfo = {
  name: "BAAI/bge-m3",
  loaded: false,
  state: "available",
  inputs: ["text"],
  outputs: ["dense"],
  lastError: null,
  maxSequenceLength: null,
  revision: null,
  capabilities: null,
  // Required (not nullable): `toModelInfo` normalizes a missing wire value to
  // `[]`, so every `ModelInfo` carries a list. Present here only to satisfy the
  // contract — this file pins the four NULLABLE fields above.
  aliases: [],
};

const modelWithExplicitNullStreaming: ModelInfo = {
  name: "Qwen/Qwen3-4B-Instruct",
  loaded: true,
  inputs: ["text"],
  outputs: ["text"],
  aliases: [],
  capabilities: { streaming: null },
};

void wireWithExplicitNulls;
void modelWithExplicitNulls;
void modelWithExplicitNullStreaming;
