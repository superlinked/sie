/**
 * Integration tests for the SIE LlamaIndex.TS node postprocessor against a running server.
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
import { type NodeWithScore, TextNode } from "llamaindex";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SIENodePostprocessor } from "../../src/index.js";

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

describe("SIENodePostprocessor integration tests", () => {
  let postprocessor: SIENodePostprocessor;

  beforeAll(() => {
    postprocessor = new SIENodePostprocessor({
      baseUrl: getServerUrl(),
      modelName: "jinaai/jina-reranker-v2-base-multilingual",
      timeout: 60_000,
    });
  });

  afterAll(async () => {
    await postprocessor.close();
  });

  it("ranks the relevant node first with distinct descending scores", async () => {
    const texts = [
      "The weather is sunny today.",
      "Python is used for data science.",
      "Semantic search understands the meaning of a query.",
    ];
    const nodes: NodeWithScore[] = texts.map((text) => ({
      node: new TextNode({ text }),
      score: 0.5,
    }));

    const result = await postprocessor.postprocessNodes(nodes, "How does semantic search work?");

    expect(result).toHaveLength(3);
    expect(result[0]?.node.getContent()).toBe(texts[2]);
    const scores = result.map((node) => node.score as number);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(new Set(scores).size).toBe(scores.length);
  });
});
