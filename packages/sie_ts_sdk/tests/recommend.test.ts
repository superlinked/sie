/**
 * Tests for the model recommendation read (`POST /v1/recommend`).
 *
 * The SDK's job here is narrow: send the task (and the target language, when
 * the caller gives one), return the gateway's answer unchanged, and map
 * failures onto the same typed errors every other method uses. `basis` and
 * `evidence_guarded` are what let a caller decide whether to trust a pick, so
 * the answer must reach the caller with every field intact.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { RequestError, ServerError } from "../src/errors.js";
import { RECOMMEND_PATH, buildRecommendBody } from "../src/internal/parsing.js";
import type { Recommendation } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const RANKED: Recommendation = {
  task: "rerank",
  label: "Rerank",
  basis: "ranked",
  shared_benchmarks: ["mteb__AskUbuntuDupQuestions"],
  fast: {
    intent: "fast",
    model: "Qwen/Qwen3-Reranker-0.6B",
    runtime_id: "Qwen/Qwen3-Reranker-0.6B",
    profile: "default",
    alias: "rerank-fast",
    available: true,
    quality_ref: "quality-evidence/rerank-fast.json",
    performance_ref: null,
    measurement_status: "verified",
    evidence_guarded: true,
  },
  best: {
    intent: "smart",
    model: "Qwen/Qwen3-Reranker-4B",
    runtime_id: "Qwen/Qwen3-Reranker-4B",
    profile: "default",
    alias: "rerank-best",
    available: true,
    quality_ref: "quality-evidence/rerank-best.json",
    performance_ref: null,
    measurement_status: "verified",
    evidence_guarded: false,
  },
};

// `best` chosen for one target language: the answer names the language and the
// per-language comparison behind the pick, next to the usual fields.
const TRANSLATION_JA: Recommendation = {
  task: "translation",
  label: "Translation",
  basis: "curated",
  shared_benchmarks: [],
  best: {
    intent: "fast",
    model: "tencent/Hy-MT2-1.8B",
    runtime_id: "tencent/Hy-MT2-1.8B",
    profile: "default",
    alias: null,
    available: true,
    evidence_guarded: false,
  },
  target_language: "ja_JP",
  language_evidence_ref: "quality-evidence/translation-by-language.json",
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function sentBody(): Record<string, unknown> {
  const init = mockFetch.mock.calls[0][1] as RequestInit;
  return JSON.parse(init.body as string);
}

describe("recommend body", () => {
  it("carries only the task when no language is given", () => {
    expect(buildRecommendBody("rerank")).toEqual({ task: "rerank" });
    expect(Object.keys(buildRecommendBody("rerank", undefined))).toEqual(["task"]);
  });

  it("adds target_language when one is given", () => {
    expect(buildRecommendBody("translation", "ja_JP")).toEqual({
      task: "translation",
      target_language: "ja_JP",
    });
  });
});

describe("client.recommend", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("posts the task to /v1/recommend and returns the answer verbatim", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(200, RANKED));
    const client = new SIEClient("http://localhost:8080");

    const answer = await client.recommend("rerank");

    expect(answer).toEqual(RANKED);
    expect(answer.basis).toBe("ranked");
    expect(answer.best?.evidence_guarded).toBe(false);
    expect(answer.fast?.evidence_guarded).toBe(true);
    expect(mockFetch.mock.calls[0][0]).toBe(`http://localhost:8080${RECOMMEND_PATH}`);
    expect((mockFetch.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect(sentBody()).toEqual({ task: "rerank" });
  });

  it("sends targetLanguage as target_language and returns the language pick", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(200, TRANSLATION_JA));
    const client = new SIEClient("http://localhost:8080");

    const answer = await client.recommend("translation", { targetLanguage: "ja_JP" });

    expect(sentBody()).toEqual({ task: "translation", target_language: "ja_JP" });
    expect(answer).toEqual(TRANSLATION_JA);
    expect(answer.best?.model).toBe("tencent/Hy-MT2-1.8B");
    expect(answer.target_language).toBe("ja_JP");
    expect(answer.language_evidence_ref).toBe("quality-evidence/translation-by-language.json");
  });

  it("raises RequestError for an unknown task", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(404, {
        detail: {
          code: "TASK_NOT_FOUND",
          message: 'unknown task "embeddings"; this release recommends for: rerank, translation',
        },
      }),
    );
    const client = new SIEClient("http://localhost:8080");

    const error = await client.recommend("embeddings").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(RequestError);
    expect((error as RequestError).statusCode).toBe(404);
    expect((error as RequestError).code).toBe("TASK_NOT_FOUND");
  });

  it("raises ServerError for a 5xx", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(503, { detail: { code: "UNAVAILABLE", message: "down" } }),
    );
    const client = new SIEClient("http://localhost:8080");

    await expect(client.recommend("rerank")).rejects.toBeInstanceOf(ServerError);
  });
});
