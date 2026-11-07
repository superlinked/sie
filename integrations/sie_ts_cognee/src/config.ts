import { lstatSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Cognee } from "@cognee/cognee-ts";

export interface Endpoint {
  baseUrl: string;
  apiKey?: string;
}

export interface CogneeSettings {
  llm: Endpoint;
  embedding: Endpoint;
  llmModel: string;
  embeddingModel: string;
  embeddingDimensions: number;
  storageRoot: string;
  /** Bounds both the native runtime and model-request concurrency. Defaults to two. */
  concurrency?: number;
  /** Maximum generation output tokens, including structured graph extraction. Default: 8192. */
  maxCompletionTokens?: number;
}

export function compatibleUrl(endpoint: Endpoint): string {
  let url: URL;
  try {
    url = new URL(endpoint.baseUrl);
  } catch {
    throw new Error("Endpoint must be an HTTP(S) URL without credentials, query or fragment.");
  }
  if (
    !/^https?:\/\/[^/]/i.test(endpoint.baseUrl) ||
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("Endpoint must be an HTTP(S) URL without credentials, query or fragment.");
  }
  url.pathname = `${url.pathname.replace(/\/+$/, "").replace(/(?:\/v1)+$/, "")}/v1`;
  return url.toString().replace(/\/$/, "");
}

function compatibleKey(endpoint: Endpoint): string {
  const url = new URL(compatibleUrl(endpoint));
  if (endpoint.apiKey?.trim()) return endpoint.apiKey;
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) return "sie-local-no-key";
  throw new Error("Remote generation and embedding endpoints require an explicit API key.");
}

const ROUTE_ENV = [
  "MOCK_LLM",
  "MOCK_EMBEDDING",
  "LLM_ARGS",
  "LLM_CASSETTE",
  "LLM_RECORD_PATH",
  "MOCK_LLM_CASSETTE",
  "COGNEE_RECORD_LLM",
  "LLM_FALLBACK_MODEL",
  "LLM_FALLBACK_PROVIDER",
  "LLM_FALLBACK_ENDPOINT",
  "LLM_FALLBACK_API_KEY",
];

/** Validated setting values. Use createCognee to apply runtime initialization and output budgets. */
export function cogneeConfiguration(settings: CogneeSettings): Record<string, unknown> {
  if (!settings.llmModel?.trim() || !settings.embeddingModel?.trim()) {
    throw new Error("Generation and embedding model IDs must be nonempty.");
  }
  if (settings.llmModel.toLowerCase().startsWith("qwen")) {
    throw new Error(
      "Cognee TypeScript 0.2.0 adds incompatible request options for Qwen generation. Use a non-Qwen model with tool-calling support.",
    );
  }
  if (!Number.isSafeInteger(settings.embeddingDimensions) || settings.embeddingDimensions < 1) {
    throw new Error("embeddingDimensions must be a positive integer.");
  }
  const concurrency = settings.concurrency ?? 2;
  if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 16) {
    throw new Error("concurrency must be an integer from 1 to 16.");
  }
  const maxCompletionTokens = settings.maxCompletionTokens ?? 8192;
  if (
    !Number.isSafeInteger(maxCompletionTokens) ||
    maxCompletionTokens < 1 ||
    maxCompletionTokens > 65536
  )
    throw new Error("maxCompletionTokens must be an integer from 1 to 65536.");
  if (!settings.storageRoot?.trim()) throw new Error("storageRoot must be nonempty.");
  for (const name of ROUTE_ENV) {
    if (process.env[name]?.trim())
      throw new Error(`Unset ${name} before configuring SIE for Cognee.`);
  }
  const root = resolve(settings.storageRoot);
  return {
    llmProvider: "custom",
    llmModel: settings.llmModel,
    llmEndpoint: compatibleUrl(settings.llm),
    llmApiKey: compatibleKey(settings.llm),
    llmMaxParallelRequests: concurrency,
    llmMaxRetries: 2,
    llmMaxCompletionTokens: maxCompletionTokens,
    llmMock: false,
    llmCassette: "",
    llmRecordPath: "",
    llmFallbackModel: "",
    llmFallbackProvider: "",
    llmFallbackEndpoint: "",
    llmFallbackApiKey: "",
    embeddingProvider: "openai",
    embeddingModel: settings.embeddingModel,
    embeddingDimensions: settings.embeddingDimensions,
    embeddingEndpoint: compatibleUrl(settings.embedding),
    embeddingApiKey: compatibleKey(settings.embedding),
    systemRootDirectory: root,
    dataRootDirectory: join(root, "data"),
    cacheRootDirectory: join(root, "cache"),
    logsRootDirectory: join(root, "logs"),
    dbProvider: "sqlite",
    relationalDbUrl: `sqlite:${join(root, "cognee.db")}?mode=rwc`,
    migrationDbUrl: "",
    storageBackend: "local",
    storageBucketName: "",
    graphDatabaseProvider: "ladybug",
    graphDatabaseUrl: "",
    graphFilePath: join(root, "graph"),
    vectorDbProvider: "lancedb",
    vectorDbUrl: join(root, "vectors"),
    cacheBackend: "fs",
    autoFeedback: false,
    enableCaching: true,
    classificationModel: settings.llmModel,
    summarizationModel: settings.llmModel,
  };
}

export function prepareStorage(settings: CogneeSettings): string {
  const root = resolve(settings.storageRoot);
  const expected = {
    format: "sie-cognee-typescript-v1",
    embeddingModel: settings.embeddingModel,
    embeddingDimensions: settings.embeddingDimensions,
    embeddingEndpoint: compatibleUrl(settings.embedding),
  };
  mkdirSync(root, { recursive: true, mode: 0o700 });
  if (lstatSync(root).isSymbolicLink())
    throw new Error("Storage root must not be a symbolic link.");
  const marker = join(root, ".sie-cognee.json");
  const names = readdirSync(root);
  if (!names.includes(".sie-cognee.json")) {
    if (names.length)
      throw new Error(
        "Use an empty dedicated storage directory; this directory is not owned by SIE for Cognee.",
      );
    writeFileSync(marker, `${JSON.stringify(expected)}\n`, { flag: "wx", mode: 0o600 });
  } else {
    let actual: unknown;
    try {
      if (lstatSync(marker).isSymbolicLink()) throw new Error("symbolic link");
      actual = JSON.parse(readFileSync(marker, "utf8"));
    } catch {
      throw new Error(
        "Storage ownership marker is invalid. Restore it or use a new storage directory.",
      );
    }
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(
        "Storage embedding settings differ. Use a new storage directory when changing the embedding model, dimensions or endpoint.",
      );
    }
  }
  for (const name of ["data", "cache", "logs"]) {
    const directory = join(root, name);
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    if (lstatSync(directory).isSymbolicLink())
      throw new Error("Storage directories must not be symbolic links.");
  }
  for (const name of ["cognee.db", "cognee.db-wal", "cognee.db-shm", "graph", "vectors"]) {
    try {
      if (lstatSync(join(root, name)).isSymbolicLink())
        throw new Error("Storage database paths must not be symbolic links.");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return root;
}

// These settings have environment support but no config.set key in Cognee 0.2.0.
const ENV_SETTINGS: Record<string, string> = {
  cacheBackend: "CACHE_BACKEND",
  autoFeedback: "AUTO_FEEDBACK",
  enableCaching: "CACHING",
  dbProvider: "DB_PROVIDER",
  storageBackend: "STORAGE_BACKEND",
  storageBucketName: "STORAGE_BUCKET_NAME",
  graphDatabaseUrl: "GRAPH_DATABASE_URL",
};

let runtimeInitialized = false;

/** Configure before importing Cognee elsewhere. This does not run inference or warm models. */
export async function createCognee(settings: CogneeSettings): Promise<Cognee> {
  const configuration = cogneeConfiguration(settings);
  prepareStorage(settings);
  process.env.COGNEE_DISABLE_DOTENV = "1";
  process.env.TELEMETRY_DISABLED = "1";
  const library = await import("@cognee/cognee-ts");
  if (!runtimeInitialized) {
    library.initWithThreads(settings.concurrency ?? 2);
    runtimeInitialized = true;
  }
  const overrides: Record<string, string> = {
    LLM_ARGS: JSON.stringify({ max_tokens: configuration.llmMaxCompletionTokens }),
  };
  for (const [key, name] of Object.entries(ENV_SETTINGS))
    overrides[name] = String(configuration[key]);
  const previous = Object.fromEntries(Object.keys(overrides).map((key) => [key, process.env[key]]));
  try {
    // No await inside this scope: each handle captures its own settings before the environment is restored.
    for (const [key, value] of Object.entries(overrides)) process.env[key] = value;
    // Passing constructor settings would discard Cognee's env-only llm_args during deserialization.
    const cognee = new library.Cognee();
    for (const [key, value] of Object.entries(configuration)) {
      if (Object.hasOwn(ENV_SETTINGS, key)) continue;
      const nativeKey =
        key === "embeddingModel"
          ? "embedding_model_name"
          : key.replace(/[A-Z]/g, (character) => `_${character.toLowerCase()}`);
      cognee.config.set(nativeKey, value);
    }
    return cognee;
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}
