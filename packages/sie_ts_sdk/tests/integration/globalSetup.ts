/**
 * Global setup for integration tests.
 *
 * Starts the SIE server before running integration tests.
 * Mirrors the pattern from packages/sie_sdk/tests/conftest.py
 *
 * Note: Uses a temp file to communicate server info to tests since globalSetup
 * runs in a separate process from the tests when using vitest's fork pool.
 */

import { execSync, spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { createConnection } from "node:net";
import { resolve } from "node:path";

const TEST_PORT = process.env.SIE_TEST_PORT ? Number.parseInt(process.env.SIE_TEST_PORT, 10) : 8081;
const SERVER_HOST = "localhost";
const SERVER_STARTUP_TIMEOUT_MS = 180_000; // 3 minutes for bundle resolution
const POLL_INTERVAL_MS = 500;

// File to communicate server info to tests (globalThis doesn't work across process boundaries)
export const SERVER_INFO_FILE = resolve(__dirname, ".server-info.json");

export interface ServerInfo {
  url: string;
  pid: number;
}

/**
 * Check if a port is open and accepting connections.
 */
function portIsOpen(host: string, port: number, timeoutMs = 1000): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection({ host, port }, () => {
      socket.end();
      resolve(true);
    });
    socket.setTimeout(timeoutMs);
    socket.on("error", () => {
      socket.destroy();
      resolve(false);
    });
    socket.on("timeout", () => {
      socket.destroy();
      resolve(false);
    });
  });
}

/**
 * Wait for the server to become available.
 */
async function waitForServer(host: string, port: number, timeoutMs: number): Promise<boolean> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    if (await portIsOpen(host, port)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  return false;
}

/**
 * Find the repository root by looking for mise.toml.
 */
function findRepoRoot(): string {
  let current = resolve(__dirname, "..", "..", "..");
  while (current !== "/") {
    try {
      const miseToml = resolve(current, "mise.toml");
      require("node:fs").accessSync(miseToml);
      return current;
    } catch {
      current = resolve(current, "..");
    }
  }
  // Fallback
  return resolve(__dirname, "..", "..", "..", "..");
}

/**
 * Find mise executable.
 */
function findMise(): string | null {
  try {
    return execSync("which mise", { encoding: "utf-8" }).trim();
  } catch {
    return null;
  }
}

export default async function globalSetup(): Promise<void> {
  console.log("\n🚀 Starting SIE server for integration tests...");

  // Check if user explicitly wants to use an external server
  const externalServerUrl = process.env.SIE_SERVER_URL;
  if (externalServerUrl) {
    console.log(`   Using external server from SIE_SERVER_URL: ${externalServerUrl}`);
    const serverInfo: ServerInfo = {
      url: externalServerUrl,
      pid: 0, // 0 indicates we didn't start this server
    };
    writeFileSync(SERVER_INFO_FILE, JSON.stringify(serverInfo));
    console.log(`✅ Using external server at ${serverInfo.url}\n`);
    return;
  }

  // Check if port is already in use - fail if so (unless explicitly allowed)
  if (await portIsOpen(SERVER_HOST, TEST_PORT)) {
    if (process.env.SIE_REUSE_SERVER === "1") {
      console.log(`   Found existing server on port ${TEST_PORT} - reusing (SIE_REUSE_SERVER=1)`);
      const serverInfo: ServerInfo = {
        url: `http://${SERVER_HOST}:${TEST_PORT}`,
        pid: 0,
      };
      writeFileSync(SERVER_INFO_FILE, JSON.stringify(serverInfo));
      console.log(`✅ Using existing server at ${serverInfo.url}\n`);
      return;
    }
    throw new Error(
      `Port ${TEST_PORT} is already in use. Either:
  1. Stop the existing server, or
  2. Set SIE_REUSE_SERVER=1 to use it, or
  3. Set SIE_SERVER_URL=http://host:port to use a specific server`,
    );
  }

  // Find mise
  const misePath = findMise();
  if (!misePath) {
    throw new Error("mise not found in PATH - required for integration tests");
  }

  // Find repo root and models directory
  const repoRoot = findRepoRoot();
  const modelsDir = resolve(repoRoot, "packages", "sie_server", "models");

  console.log(`   Repo root: ${repoRoot}`);
  console.log(`   Models dir: ${modelsDir}`);
  console.log(`   Port: ${TEST_PORT}`);

  // Start server via mise run serve
  const proc = spawn(
    misePath,
    ["run", "serve", "--", "-p", String(TEST_PORT), "--models-dir", modelsDir],
    {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    },
  );

  const serverUrl = `http://${SERVER_HOST}:${TEST_PORT}`;

  // Collect output for debugging
  let serverOutput = "";
  proc.stdout?.on("data", (data: Buffer) => {
    serverOutput += data.toString();
  });
  proc.stderr?.on("data", (data: Buffer) => {
    serverOutput += data.toString();
  });

  // Handle unexpected exit
  proc.on("exit", (code) => {
    if (code !== null && code !== 0) {
      console.error(`❌ Server exited unexpectedly with code ${code}`);
      console.error(`   Output:\n${serverOutput.slice(-2000)}`);
    }
  });

  // Wait for server to be ready
  console.log(`   Waiting for server to start (timeout: ${SERVER_STARTUP_TIMEOUT_MS / 1000}s)...`);
  const isReady = await waitForServer(SERVER_HOST, TEST_PORT, SERVER_STARTUP_TIMEOUT_MS);

  if (!isReady) {
    proc.kill("SIGTERM");
    throw new Error(
      `Server failed to start within ${SERVER_STARTUP_TIMEOUT_MS / 1000}s.\nOutput:\n${serverOutput.slice(-2000)}`,
    );
  }

  // Write server info to file for tests to read
  const serverInfo: ServerInfo = {
    url: serverUrl,
    pid: proc.pid ?? 0,
  };
  writeFileSync(SERVER_INFO_FILE, JSON.stringify(serverInfo));

  console.log(`✅ Server ready at ${serverUrl}\n`);
}
