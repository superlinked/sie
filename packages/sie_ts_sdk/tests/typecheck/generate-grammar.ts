import type { SIEClient } from "../../src/client.js";
import type {
  EbnfGrammar,
  GenerateOptions,
  JsonSchemaGrammar,
  RegexGrammar,
} from "../../src/types.js";

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

// The server/OpenAPI/gateway contract treats nullable metadata as absent.
const nullableGrammars: [JsonSchemaGrammar, RegexGrammar, EbnfGrammar] = [
  { json_schema: { type: "object" }, label: null, strict: null },
  { regex: "\\d+", label: null, strict: null },
  { ebnf: 'root ::= "ok"', label: null, strict: null },
];

for (const grammar of nullableGrammars) {
  void client.generate("model", "Return structured output", { maxNewTokens: 8, grammar });
  void client.streamGenerate("model", "Return structured output", {
    maxNewTokens: 8,
    grammar,
  });
}
