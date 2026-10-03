import type { GenerateChunk } from "../../src/types.js";

interface TaggedGenerateChunk extends GenerateChunk {
  tag: string;
}

const terminal: TaggedGenerateChunk = {
  request_id: "request-1",
  seq: 1,
  text_delta: "",
  done: true,
  tag: "completed",
};
const attestedTerminal: TaggedGenerateChunk = {
  ...terminal,
  execution_identity_sha256: "a".repeat(64),
  execution_binding_sha256: "b".repeat(64),
};

void terminal;
void attestedTerminal;
