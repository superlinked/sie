import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Cognee } from "@cognee/cognee-ts";
import {
  type Endpoint,
  SourceDocument,
  SourceMemory,
  compatibleUrl,
  createCognee,
} from "@superlinked/sie-cognee";
import { SIEClient } from "@superlinked/sie-sdk";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const DATA = resolve(ROOT, "../data");
export const ROLES = ["llm", "embedding", "ocr", "rerank", "decision"] as const;
export type Role = (typeof ROLES)[number];
export interface Configuration {
  models: Record<Role, string> & { embeddingDimensions: number };
  limits: Record<
    | "retrieveTopK"
    | "rerankTopK"
    | "evidenceChars"
    | "requestTimeoutMs"
    | "provisionTimeoutMs"
    | "experimentTimeoutMs"
    | "documentBytes",
    number
  >;
  decision: { query: string; instructions: string; choices: Record<string, string> };
  learning: { query: string };
}

export function endpointFor(role: Role, env: NodeJS.ProcessEnv): Endpoint {
  const base = env.SIE_BASE_URL || "https://api.superlinked.com";
  const baseUrl = env[`SIE_${role.toUpperCase()}_BASE_URL`] || base;
  compatibleUrl({ baseUrl: base });
  compatibleUrl({ baseUrl });
  const keyName = `SIE_${role.toUpperCase()}_API_KEY`;
  const apiKey = Object.hasOwn(env, keyName)
    ? env[keyName]
    : new URL(base).origin === new URL(baseUrl).origin
      ? env.SIE_API_KEY
      : undefined;
  return { baseUrl, apiKey: apiKey || undefined };
}

export interface Demo {
  config: Configuration;
  namespace: string;
  dataset: string;
  storageRoot: string;
  run: string;
  cognee: Cognee;
  memory: SourceMemory;
  native(role: Role): SIEClient;
  show(name: string, value: unknown): void;
}

export function readInput(filename: string): string {
  try {
    return readFileSync(join(DATA, filename), "utf8");
  } catch {
    throw new Error("Download the pinned inputs with pnpm fetch-data first.");
  }
}

export async function seedMemory(demo: Demo): Promise<void> {
  for (const [sourceId, filename] of [
    ["POLICY-01", "readiness-policy.txt"],
    ["CHECK-01", "projector-check.txt"],
    ["CHECK-02", "microphone-check.txt"],
  ] as const) {
    await demo.memory.add(new SourceDocument({ text: readInput(filename), sourceId, filename }));
  }
  demo.show("memory-build", await demo.memory.build());
}

export async function launch(name: string, action: (demo: Demo) => Promise<void>): Promise<void> {
  let initialized = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const config = JSON.parse(readFileSync(join(ROOT, "config.json"), "utf8")) as Configuration;
    for (const role of ROLES)
      config.models[role] = process.env[`SIE_${role.toUpperCase()}_MODEL`] || config.models[role];
    if (process.env.SIE_EMBEDDING_DIMENSIONS)
      config.models.embeddingDimensions = Number(process.env.SIE_EMBEDDING_DIMENSIONS);
    for (const value of Object.values(config.limits))
      if (!Number.isSafeInteger(value) || value < 1)
        throw new Error("All limits must be positive integers.");
    const namespace = process.env.SIE_COGNEE_NAMESPACE || "example-v1";
    if (!/^[a-z0-9][a-z0-9-]{0,49}$/.test(namespace))
      throw new Error("Namespace must contain 1–50 lowercase letters, digits or hyphens.");
    const endpoints = Object.fromEntries(
      ROLES.map((role) => [role, endpointFor(role, process.env)]),
    ) as Record<Role, Endpoint>;
    const storageRoot = join(ROOT, ".state", namespace);
    const run = join(
      ROOT,
      "runs",
      namespace,
      `${new Date().toISOString().replaceAll(":", "-")}-${randomUUID().slice(0, 8)}-${name}`,
    );
    mkdirSync(run, { recursive: true, mode: 0o700 });
    const show = (label: string, value: unknown) => {
      const rendered = JSON.stringify(value, null, 2);
      writeFileSync(join(run, `${label}.json`), `${rendered}\n`, { mode: 0o600 });
      console.log(`\n${label}\n${rendered}`);
    };
    timer = setTimeout(() => {
      console.error(
        "The example exceeded its time limit. Inspect the saved output before retrying.",
      );
      process.exit(1);
    }, config.limits.experimentTimeoutMs);
    const cognee = await createCognee({
      llm: endpoints.llm,
      embedding: endpoints.embedding,
      llmModel: config.models.llm,
      embeddingModel: config.models.embedding,
      embeddingDimensions: config.models.embeddingDimensions,
      storageRoot,
      concurrency: 2,
    });
    initialized = true;
    const dataset = `sie-ts-${namespace}`;
    const demo: Demo = {
      config,
      namespace,
      dataset,
      storageRoot,
      run,
      cognee,
      memory: new SourceMemory(cognee, { dataset, storageRoot }),
      native: (role) =>
        new SIEClient(compatibleUrl(endpoints[role]).replace(/\/v1$/, ""), {
          apiKey: endpoints[role].apiKey || "",
          timeoutMs: config.limits.requestTimeoutMs,
          provisionTimeout: config.limits.provisionTimeoutMs,
          waitForCapacity: true,
        }),
      show,
    };
    show("configuration", {
      models: config.models,
      endpoints: Object.fromEntries(ROLES.map((role) => [role, endpoints[role].baseUrl])),
      dataset,
      storageRoot,
      run,
    });
    await cognee.warm();
    await action(demo);
    console.log(`\nSaved output to ${run}`);
  } catch (error) {
    console.error(
      `${error instanceof Error ? error.name : "Error"}: The example did not complete. Check the configured model routes, input files and storage settings.`,
    );
    process.exitCode = 1;
  } finally {
    if (timer) clearTimeout(timer);
    if (initialized) (await import("@cognee/cognee-ts")).shutdown();
  }
}
