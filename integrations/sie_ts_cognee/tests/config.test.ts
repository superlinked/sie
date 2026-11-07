import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import {
  type CogneeSettings,
  cogneeConfiguration,
  compatibleUrl,
  prepareStorage,
} from "../src/config";

const roots: string[] = [];
function settings(): CogneeSettings {
  const storageRoot = mkdtempSync(join(tmpdir(), "sie-cognee-config-"));
  roots.push(storageRoot);
  return {
    llm: { baseUrl: "http://127.0.0.1:8080" },
    embedding: { baseUrl: "http://localhost:8081/v1" },
    llmModel: "google/gemma-4-E4B-it",
    embeddingModel: "BAAI/bge-m3",
    embeddingDimensions: 1024,
    storageRoot,
  };
}
afterEach(() => {
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

it("normalizes compatible paths exactly once", () => {
  expect(compatibleUrl({ baseUrl: "https://example.org/prefix/v1/v1/" })).toBe(
    "https://example.org/prefix/v1",
  );
  expect(compatibleUrl({ baseUrl: "http://[::1]:8080" })).toBe("http://[::1]:8080/v1");
});
it.each([
  "file:///tmp/test",
  "https://name:secret@example.org",
  "https://example.org?key=secret",
  "https://example.org/#secret",
  "not a url",
  "http:/example.org",
  "https:example.org",
])("rejects unsafe endpoint %s", (baseUrl) => {
  expect(() => compatibleUrl({ baseUrl })).toThrow();
});
it("uses separate explicit routes and puts all local storage below the owned directory", () => {
  const input = settings();
  const config = cogneeConfiguration(input);
  expect(config).toMatchObject({
    llmProvider: "custom",
    llmModel: input.llmModel,
    llmEndpoint: "http://127.0.0.1:8080/v1",
    embeddingEndpoint: "http://localhost:8081/v1",
    embeddingProvider: "openai",
    llmApiKey: "sie-local-no-key",
    embeddingApiKey: "sie-local-no-key",
    llmMock: false,
    dbProvider: "sqlite",
    storageBackend: "local",
    storageBucketName: "",
    migrationDbUrl: "",
    graphDatabaseUrl: "",
  });
  for (const key of [
    "dataRootDirectory",
    "cacheRootDirectory",
    "logsRootDirectory",
    "graphFilePath",
    "vectorDbUrl",
  ])
    expect(String(config[key])).toContain(input.storageRoot);
  expect(config.relationalDbUrl).toBe(`sqlite:${join(input.storageRoot, "cognee.db")}?mode=rwc`);
});
it("requires a key for remote endpoints, including localhost lookalikes", () => {
  for (const baseUrl of [
    "https://example.org",
    "http://localhost.example.org",
    "http://127.0.0.2",
  ]) {
    expect(() => cogneeConfiguration({ ...settings(), llm: { baseUrl } })).toThrow(
      /explicit API key/,
    );
  }
});
it("keeps keys out of the storage marker", () => {
  const input = {
    ...settings(),
    embedding: { baseUrl: "https://example.org", apiKey: "test-only-secret" },
  };
  prepareStorage(input);
  expect(readFileSync(join(input.storageRoot, ".sie-cognee.json"), "utf8")).not.toContain(
    "test-only-secret",
  );
});
it("refuses unowned storage, malformed markers, and embedding changes", () => {
  const input = settings();
  writeFileSync(join(input.storageRoot, "existing.txt"), "keep me");
  expect(() => prepareStorage(input)).toThrow(/empty dedicated/);
  expect(readFileSync(join(input.storageRoot, "existing.txt"), "utf8")).toBe("keep me");
  const owned = settings();
  prepareStorage(owned);
  expect(() => prepareStorage(owned)).not.toThrow();
  expect(() => prepareStorage({ ...owned, embeddingDimensions: 768 })).toThrow(/differ/);
  writeFileSync(join(owned.storageRoot, ".sie-cognee.json"), "broken");
  expect(() => prepareStorage(owned)).toThrow(/invalid/);
});
it("rejects symlink storage roots", () => {
  const input = settings();
  const parent = settings();
  const link = join(parent.storageRoot, "link");
  symlinkSync(input.storageRoot, link);
  expect(() => prepareStorage({ ...input, storageRoot: link })).toThrow(/symbolic/);
});
it("rejects incompatible model and hidden fallback/mock settings before initialization", () => {
  expect(() => cogneeConfiguration({ ...settings(), llmModel: "Qwen/Qwen3.5-4B" })).toThrow(
    /Cognee TypeScript 0.2.0/,
  );
  vi.stubEnv("MOCK_EMBEDDING", "true");
  expect(() => cogneeConfiguration(settings())).toThrow(/Unset MOCK_EMBEDDING/);
});

it.each([0, -1, 1.5, 65537, Number.NaN])(
  "rejects an invalid completion budget %s",
  (maxCompletionTokens) => {
    expect(() => cogneeConfiguration({ ...settings(), maxCompletionTokens })).toThrow(
      /maxCompletionTokens/,
    );
  },
);
it("rejects symlinks at active database paths, including dangling links", () => {
  for (const name of ["cognee.db", "graph", "vectors"]) {
    const input = settings();
    prepareStorage(input);
    symlinkSync(join(input.storageRoot, "missing-target"), join(input.storageRoot, name));
    expect(() => prepareStorage(input)).toThrow(/database paths/);
  }
});
