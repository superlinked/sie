import type { SIEClient } from "../../src/client.js";
import type { GenerateOptions } from "../../src/types.js";

declare const client: SIEClient;

// Existing callers commonly retain grammar configuration in a broad record.
// This must remain source-compatible while runtime validation enforces the
// native json_schema | regex | ebnf envelope.
const legacyGrammar: Record<string, unknown> = { regex: "\\d+" };
const options: GenerateOptions = {
  maxNewTokens: 8,
  grammar: legacyGrammar,
};

void client.generate("model", "Return digits", options);
void client.streamGenerate("model", "Return digits", options);
