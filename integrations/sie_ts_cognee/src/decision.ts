import type { SIEClient } from "@superlinked/sie-sdk";
import type { SourceChunk } from "./retrieval";

export class EvidenceOverflowError extends Error {
  override name = "EvidenceOverflowError";
}

export interface Decision {
  choice: string;
  probabilities: Record<string, number>;
}

function checkSize(text: string, maxChars: number): void {
  if (!Number.isSafeInteger(maxChars) || maxChars < 1)
    throw new Error("maxChars must be a positive integer.");
  if (Array.from(text).length > maxChars)
    throw new EvidenceOverflowError(
      "Evidence exceeds maxChars; select fewer sources or request review.",
    );
}

export function evidencePacket(
  query: string,
  chunks: readonly SourceChunk[],
  maxChars: number,
): string {
  if (!query.trim() || !chunks.length)
    throw new Error("A query and at least one source chunk are required.");
  const packet = `Question: ${query}\n${chunks.map((chunk) => `[${chunk.chunkId}] ${chunk.text}`).join("\n")}`;
  checkSize(packet, maxChars);
  return packet;
}

export async function decide(
  client: SIEClient,
  packet: string,
  options: {
    model: string;
    choices: Record<string, string>;
    instructions: string;
    maxChars: number;
    questionId?: string;
  },
): Promise<Decision> {
  const { choices } = options;
  const questionId = options.questionId ?? "action";
  if (!Object.hasOwn(choices, "unknown") && !Object.hasOwn(choices, "review"))
    throw new Error("Choices must include unknown or review.");
  if (!packet.trim() || !options.instructions.trim() || !questionId.trim())
    throw new Error("packet, instructions and questionId must be nonempty.");
  if (
    Object.entries(choices).some(
      ([label, criterion]) => !label.trim() || typeof criterion !== "string" || !criterion.trim(),
    )
  ) {
    throw new Error("Choice labels and criteria must be nonempty.");
  }
  checkSize(packet, options.maxChars);
  const reply = await client.extract(
    options.model,
    { id: "decision", text: packet },
    {
      outputSchema: {
        [questionId]: { type: "choice", instructions: options.instructions, criteria: choices },
      },
      adapterOptions: { overflow_policy: "error" },
      waitForCapacity: true,
    },
  );
  const answer = reply.data?.[questionId];
  if (reply.error || !answer || typeof answer !== "object" || Array.isArray(answer))
    throw new Error("Decision response is missing its question answer.");
  const { choice, probabilities } = answer as Record<string, unknown>;
  if (!probabilities || typeof probabilities !== "object" || Array.isArray(probabilities))
    throw new Error("Decision must return probabilities.");
  const values = Object.entries(probabilities);
  if (
    values.length !== Object.keys(choices).length ||
    values.some(([label]) => !Object.hasOwn(choices, label))
  ) {
    throw new Error("Decision must return probabilities for exactly the requested choices.");
  }
  const validated: Record<string, number> = Object.create(null);
  for (const [label, probability] of values) {
    if (
      typeof probability !== "number" ||
      !Number.isFinite(probability) ||
      probability < 0 ||
      probability > 1
    )
      throw new Error("Decision probabilities must be finite numbers in [0, 1].");
    validated[label] = probability;
  }
  const total = Object.values(validated).reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 1e-4) throw new Error("Choice probabilities must sum to one.");
  if (
    typeof choice !== "string" ||
    !Object.hasOwn(choices, choice) ||
    validated[choice] !== Math.max(...Object.values(validated))
  ) {
    throw new Error("Decision choice must match a highest probability label.");
  }
  return { choice, probabilities: validated };
}
