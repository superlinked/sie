/**
 * Integration tests for the SIE LangChain.js reranker against a running server.
 *
 * These tests require a running SIE server with the
 * jinaai/jina-reranker-v2-base-multilingual model.
 * The server is started automatically via the SDK's globalSetup.
 *
 * To run integration tests:
 *   pnpm test:integration
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SIEReranker } from "../../src/index.js";

const SERVER_INFO_FILE = resolve(
  __dirname,
  "../../../../packages/sie_ts_sdk/tests/integration/.server-info.json",
);

interface ServerInfo {
  url: string;
  pid: number;
}

function getServerUrl(): string {
  try {
    const serverInfo: ServerInfo = JSON.parse(readFileSync(SERVER_INFO_FILE, "utf-8"));
    return serverInfo.url;
  } catch {
    throw new Error("Server info file not found - globalSetup may have failed");
  }
}

describe("SIEReranker integration tests", () => {
  let reranker: SIEReranker;

  beforeAll(() => {
    reranker = new SIEReranker({
      baseUrl: getServerUrl(),
      model: "jinaai/jina-reranker-v2-base-multilingual",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await reranker.close();
  });

  it("ranks the relevant document first with distinct descending scores", async () => {
    const documents = [
      { pageContent: "The weather is sunny today.", metadata: { source: "weather" } },
      { pageContent: "Python is used for data science.", metadata: { source: "python" } },
      {
        pageContent: "Semantic search understands the meaning of a query.",
        metadata: { source: "search" },
      },
    ];

    const result = await reranker.compressDocuments(documents, "How does semantic search work?");

    expect(result).toHaveLength(3);
    expect(result[0]?.metadata.source).toBe("search");
    const scores = result.map((doc) => doc.metadata.relevance_score as number);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(new Set(scores).size).toBe(scores.length);
  });
});
