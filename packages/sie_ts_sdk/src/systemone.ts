/**
 * System One typed decisions on top of `extract`.
 *
 * `SIEClient.systemOne` answers one state against typed `noul`, `choice` and
 * `score` questions with a decision model, and returns the System One response
 * shape (`{ model, answers, usage }`) that SIE's `POST /v1/systemone` serves. It
 * runs as one `extract` call, so it waits for capacity and model loading like
 * every other SDK call. Mirrors the Python SDK's `sie_sdk.systemone`.
 */

import { RequestError } from "./errors.js";
import type { ExtractItem, ExtractResult } from "./types.js";

/** Score levels per question, as the System One API accepts. */
export const MAX_SCORE_LEVELS = 10;

/** Runtime options every System One call sends: input that does not fit the model's row is refused, not cut. */
export const SYSTEM_ONE_OPTIONS: Readonly<Record<string, unknown>> = { overflow_policy: "error" };

/** A System One state: the text to judge, a JSON object, or a list of conversation turns. */
export type SystemOneState = string | Record<string, unknown> | unknown[];

/** Options for `SIEClient.systemOne`. */
export interface SystemOneOptions {
  /** Runtime options forwarded as `params.options`, as for `extract`. */
  adapterOptions?: Record<string, unknown>;
  /** GPU type for this request */
  gpu?: string;
  /** Whether to wait for capacity */
  waitForCapacity?: boolean;
}

/** Tokens a System One call read; decision models generate none. */
export interface SystemOneUsage {
  /**
   * The metered count the response reported (SIE Serverless meters every
   * request); 0 when the deployment reports no metering.
   */
  input_tokens: number;
  output_tokens: number;
}

/** Answers to typed questions, in the System One response shape. */
export interface SystemOneResult {
  /** The decision model that answered */
  model: string;
  /**
   * One answer per question id. `noul`: `noul`, the probability of yes,
   * `answer` and `confidence`.
   * `choice`: `choice`, `confidence` and `probabilities` by option. `score`:
   * `score`, `confidence`, and `legend` and `probabilities` by level index.
   */
  answers: Record<string, Record<string, unknown>>;
  usage: SystemOneUsage;
}

function invalid(message: string): RequestError {
  return new RequestError(message, "INVALID_INPUT");
}

/**
 * The extract item and `outputSchema` for a System One call. A question may
 * omit `instructions`; a score question takes 1 to 10 levels, none null.
 */
export function systemOneRequest(
  state: unknown,
  questions: Record<string, unknown>,
): { item: ExtractItem; outputSchema: Record<string, Record<string, unknown>> } {
  let item: ExtractItem;
  if (typeof state === "string") {
    item = { text: state };
  } else if (state !== null && typeof state === "object") {
    item = { metadata: { state } };
  } else {
    throw invalid("state must be a string, an object, or an array");
  }
  const entries =
    questions !== null && typeof questions === "object" && !Array.isArray(questions)
      ? Object.entries(questions)
      : [];
  if (entries.length === 0) {
    throw invalid("questions must be a non-empty object of questions by id");
  }
  const outputSchema: Record<string, Record<string, unknown>> = {};
  for (const [qid, question] of entries) {
    if (question === null || typeof question !== "object" || Array.isArray(question)) {
      throw invalid(`question ${JSON.stringify(qid)} must be an object`);
    }
    const normalized: Record<string, unknown> = { ...(question as Record<string, unknown>) };
    if (normalized.instructions === undefined || normalized.instructions === null) {
      normalized.instructions = "";
    }
    const levels = normalized.criteria;
    if (normalized.type === "score" && Array.isArray(levels)) {
      if (levels.length === 0) {
        throw invalid(`score question ${JSON.stringify(qid)} has no levels`);
      }
      if (levels.length > MAX_SCORE_LEVELS) {
        throw invalid(
          `score question ${JSON.stringify(qid)} has ${levels.length} levels; at most ${MAX_SCORE_LEVELS} are allowed`,
        );
      }
      if (levels.some((level) => level === null || level === undefined)) {
        throw invalid(`score question ${JSON.stringify(qid)} has a null level`);
      }
    }
    outputSchema[qid] = normalized;
  }
  return { item, outputSchema };
}

/** System One choice confidence: `(p_max - 1/n) / (1 - 1/n)`, within [0, 1]. */
export function choiceConfidence(probabilities: number[]): number {
  const n = probabilities.length;
  if (n <= 1) return 1;
  const floor = 1 / n;
  return Math.min(1, Math.max(0, (Math.max(...probabilities) - floor) / (1 - floor)));
}

/**
 * System One score confidence: `max(0, 1 - spread / even)`, where `spread` is
 * the expected distance from the most likely level and `even` that of a
 * uniform distribution from the middle level.
 */
export function scoreConfidence(probabilities: number[]): number {
  const n = probabilities.length;
  if (n <= 1) return 1;
  let mode = 0;
  probabilities.forEach((p, index) => {
    if (p > (probabilities[mode] ?? 0)) mode = index;
  });
  const spread = probabilities.reduce((sum, p, index) => sum + p * Math.abs(index - mode), 0);
  const middle = (n - 1) / 2;
  let even = 0;
  for (let index = 0; index < n; index++) even += Math.abs(index - middle);
  even /= n;
  return Math.min(1, Math.max(0, 1 - spread / even));
}

function probability(qid: string, value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RequestError(
      `unreadable answer to question ${JSON.stringify(qid)}: probability ${JSON.stringify(value)} is not a number`,
      "INVALID_RESPONSE",
    );
  }
  return value;
}

function toAnswer(qid: string, answer: Record<string, unknown>): Record<string, unknown> {
  const result = { ...answer };
  const probabilities = result.probabilities;
  if (probabilities === null || typeof probabilities !== "object") return result;
  const byKey = probabilities as Record<string, unknown>;
  if (result.type === "choice") {
    result.confidence = choiceConfidence(Object.values(byKey).map((p) => probability(qid, p)));
  } else if (result.type === "score") {
    const keys = Object.keys(byKey);
    const badKey = keys.find((key) => !/^\d+$/.test(key));
    if (badKey !== undefined) {
      throw new RequestError(
        `unreadable answer to question ${JSON.stringify(qid)}: score level ${JSON.stringify(badKey)} is not an integer`,
        "INVALID_RESPONSE",
      );
    }
    const ordered = keys
      .sort((a, b) => Number(a) - Number(b))
      .map((key) => probability(qid, byKey[key]));
    result.confidence = scoreConfidence(ordered);
  }
  return result;
}

/**
 * An extract result in the System One response shape. Throws `RequestError`
 * when the model refused the state (for example, it does not fit the model's
 * window) or answered no question of the request.
 */
export function systemOneResult(
  model: string,
  result: ExtractResult,
  questions: Record<string, unknown>,
): SystemOneResult {
  if (result.error) {
    throw new RequestError(result.error.message, result.error.code, 422, undefined, "state");
  }
  const data = (result.data ?? {}) as Record<string, unknown>;
  const missing = Object.keys(questions).filter((qid) => {
    const answer = data[qid];
    return answer === null || typeof answer !== "object";
  });
  if (missing.length > 0) {
    throw new RequestError(
      `model ${JSON.stringify(model)} returned no answer for questions ${JSON.stringify(missing)}; use a decision model such as fastino/GLiNER2.5-Decide`,
      "INVALID_RESPONSE",
    );
  }
  const answers: Record<string, Record<string, unknown>> = {};
  for (const qid of Object.keys(questions)) {
    answers[qid] = toAnswer(qid, data[qid] as Record<string, unknown>);
  }
  const inputTokens = result.request?.usage?.inputTokens;
  return {
    model,
    answers,
    usage: { input_tokens: typeof inputTokens === "number" ? inputTokens : 0, output_tokens: 0 },
  };
}
