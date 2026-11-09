import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createCognee } from "../src/config";

const native = vi.hoisted(() => ({
  failure: "" as "" | "constructor" | "setting",
  snapshots: [] as { environment: NodeJS.ProcessEnv; settings: Record<string, unknown> }[],
}));
vi.mock("@cognee/cognee-ts", () => ({
  initWithThreads: vi.fn(),
  Cognee: class {
    config: { set(key: string, value: unknown): void };
    constructor() {
      if (native.failure === "constructor") throw new Error("constructor failure");
      const snapshot = {
        environment: Object.fromEntries(
          Object.keys(inherited).map((key) => [key, process.env[key]]),
        ),
        settings: {} as Record<string, unknown>,
      };
      native.snapshots.push(snapshot);
      this.config = {
        set(key, value) {
          if (native.failure === "setting") throw new Error("setting failure");
          snapshot.settings[key] = value;
        },
      };
    }
  },
}));
const roots: string[] = [];
const inherited = {
  LLM_ARGS: " ",
  CACHE_BACKEND: "redis",
  CACHING: "false",
  AUTO_FEEDBACK: "true",
  DB_PROVIDER: "postgres",
  STORAGE_BACKEND: "s3",
  STORAGE_BUCKET_NAME: "unrelated-bucket",
  GRAPH_DATABASE_URL: "unrelated-graph",
};
function settings(maxCompletionTokens = 8192) {
  const storageRoot = mkdtempSync(join(tmpdir(), "sie-cognee-init-"));
  roots.push(storageRoot);
  return {
    llm: { baseUrl: "http://localhost:8080" },
    embedding: { baseUrl: "http://localhost:8081" },
    llmModel: `model-${maxCompletionTokens}`,
    embeddingModel: "embed",
    embeddingDimensions: 4,
    storageRoot,
    maxCompletionTokens,
  };
}
beforeEach(() => {
  native.failure = "";
  native.snapshots.length = 0;
  for (const [key, value] of Object.entries(inherited)) vi.stubEnv(key, value);
});
afterEach(() => {
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function expectRestored() {
  for (const [key, value] of Object.entries(inherited)) expect(process.env[key]).toBe(value);
}

it("captures bounded startup settings and restores the environment", async () => {
  const first = settings(8192);
  const second = settings(4096);
  await createCognee(first);
  await createCognee(second);
  expectRestored();
  expect(native.snapshots).toHaveLength(2);
  for (const [index, input] of [first, second].entries()) {
    expect(native.snapshots[index]?.environment).toMatchObject({
      LLM_ARGS: JSON.stringify({ max_tokens: input.maxCompletionTokens }),
      CACHE_BACKEND: "fs",
      CACHING: "true",
      AUTO_FEEDBACK: "false",
      DB_PROVIDER: "sqlite",
      STORAGE_BACKEND: "local",
      STORAGE_BUCKET_NAME: "",
      GRAPH_DATABASE_URL: "",
    });
    expect(native.snapshots[index]?.settings).toMatchObject({
      llm_model: input.llmModel,
      llm_max_completion_tokens: input.maxCompletionTokens,
      embedding_model_name: "embed",
      system_root_directory: input.storageRoot,
    });
  }
});
it.each(["constructor", "setting"] as const)(
  "restores environment when native %s fails",
  async (failure) => {
    native.failure = failure;
    await expect(createCognee(settings())).rejects.toThrow(`${failure} failure`);
    expectRestored();
  },
);
it("restores absent temporary environment variables by deleting them", async () => {
  for (const key of Object.keys(inherited)) vi.stubEnv(key, undefined);
  await createCognee(settings());
  for (const key of Object.keys(inherited)) expect(Object.hasOwn(process.env, key)).toBe(false);
});
