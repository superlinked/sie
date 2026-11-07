import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const DATASET = "superlinked/sie-task-evidence";
export const REVISION = "487cdc3bd0480659afd7ee88329449e6a6c8655d";
export const PREFIX = "cognee/workshop-starter/2026-11-06/v1";
export const MANIFEST_SHA256 = "81dbd18690c085a6f12e4a1772a082d92e102b9cff81616f53965e2d398c0ac8";
const BASE = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/${PREFIX}`;
const REQUIRED = [
  "access-note.png",
  "access-note.txt",
  "correction.txt",
  "microphone-check.txt",
  "projector-check.txt",
  "readiness-policy.txt",
];
const FILE_LIMIT = 1024 * 1024;
const MANIFEST_LIMIT = 64 * 1024;
const TOTAL_LIMIT = 4 * 1024 * 1024;
interface FileEntry {
  bytes: number;
  sha256: string;
}

export function checkFile(body: Uint8Array, expected: FileEntry): void {
  if (
    body.byteLength !== expected.bytes ||
    createHash("sha256").update(body).digest("hex") !== expected.sha256
  )
    throw new Error("Input size or checksum differs from the pinned dataset.");
}

export function manifestFiles(body: Uint8Array): Record<string, FileEntry> {
  if (createHash("sha256").update(body).digest("hex") !== MANIFEST_SHA256)
    throw new Error("Manifest checksum differs from the pinned inputs.");
  const manifest = JSON.parse(Buffer.from(body).toString("utf8"));
  if (
    manifest.schema_version !== 1 ||
    manifest.license !== "CC0-1.0" ||
    !manifest.files ||
    typeof manifest.files !== "object" ||
    Array.isArray(manifest.files)
  )
    throw new Error("Unexpected input manifest schema or license.");
  const files = manifest.files as Record<string, FileEntry>;
  if (JSON.stringify(Object.keys(files).sort()) !== JSON.stringify(REQUIRED))
    throw new Error("Manifest must contain exactly the six input files.");
  let total = 0;
  for (const entry of Object.values(files)) {
    if (
      !entry ||
      !Number.isSafeInteger(entry.bytes) ||
      entry.bytes <= 0 ||
      entry.bytes > FILE_LIMIT ||
      typeof entry.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(entry.sha256)
    )
      throw new Error("Invalid manifest file entry.");
    total += entry.bytes;
  }
  if (total > TOTAL_LIMIT) throw new Error("Inputs exceed the total byte limit.");
  return files;
}

async function download(name: string, limit: number): Promise<Buffer> {
  const response = await fetch(`${BASE}/${name}`, { signal: AbortSignal.timeout(60_000) });
  if (!response.ok || !response.body) throw new Error("Input download failed.");
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      size += result.value.byteLength;
      if (size > limit) throw new Error("Input download exceeds its byte limit.");
      parts.push(result.value);
    }
  } finally {
    await reader.cancel();
  }
  return Buffer.concat(parts);
}

function cached(path: string, limit: number): Buffer {
  if (statSync(path).size > limit) throw new Error("Cached input exceeds its byte limit.");
  return readFileSync(path);
}

export async function fetchInputs(directory: string): Promise<void> {
  const manifestPath = join(directory, "manifest.json");
  const body = existsSync(manifestPath)
    ? cached(manifestPath, MANIFEST_LIMIT)
    : await download("manifest.json", MANIFEST_LIMIT);
  const files = manifestFiles(body);
  const missing: string[] = [];
  for (const [name, entry] of Object.entries(files)) {
    const path = join(directory, name);
    if (existsSync(path)) checkFile(cached(path, entry.bytes), entry);
    else missing.push(name);
  }
  mkdirSync(directory, { recursive: true });
  const staging = mkdtempSync(join(directory, ".fetch-"));
  try {
    for (const name of missing) {
      const entry = files[name];
      if (!entry) throw new Error("Missing input manifest entry.");
      const data = await download(name, entry.bytes);
      checkFile(data, entry);
      writeFileSync(join(staging, name), data);
    }
    for (const name of missing) renameSync(join(staging, name), join(directory, name));
    if (!existsSync(manifestPath)) writeFileSync(manifestPath, body, { flag: "wx" });
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await fetchInputs(resolve(dirname(fileURLToPath(import.meta.url)), "../../data"));
    console.log(`Verified six CC0 inputs from ${DATASET}@${REVISION}.`);
  } catch (error) {
    console.error(
      `Input fetch failed (${error instanceof Error ? error.name : "Error"}). Check the connection or move edited cached inputs aside before retrying.`,
    );
    process.exitCode = 1;
  }
}
