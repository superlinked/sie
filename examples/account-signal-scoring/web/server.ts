import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SIEClient } from "@superlinked/sie-sdk";
import { config } from "../src/config.js";
import type { ScoreEvent } from "../src/events.js";
import { loadAccounts, runScore } from "../src/score.js";
import { scoreSignals } from "../src/signals.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.join(ROOT, "web", "public");
const INDEX_PATH = path.join(ROOT, config.paths.index);

function send(
  res: http.ServerResponse,
  status: number,
  body: string | Buffer,
  contentType = "text/plain",
): void {
  res.writeHead(status, { "content-type": contentType });
  res.end(body);
}

function serveFile(res: http.ServerResponse, file: string): void {
  if (!fs.existsSync(file)) return send(res, 404, "not found");
  const ext = path.extname(file).toLowerCase();
  const ct =
    {
      ".html": "text/html",
      ".css": "text/css",
      ".js": "text/javascript",
      ".json": "application/json",
      ".png": "image/png",
      ".svg": "image/svg+xml",
    }[ext] ?? "application/octet-stream";
  res.writeHead(200, { "content-type": ct });
  fs.createReadStream(file).pipe(res);
}

function setupSse(res: http.ServerResponse) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  return (event: ScoreEvent) => {
    res.write(`event: ${event.type}\n`);
    const payload = "data" in event ? event.data : null;
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };
}

function ensureIndex(): void {
  if (fs.existsSync(INDEX_PATH)) return;
  console.log("building playbook index...");
  const result = spawnSync(
    process.execPath,
    [path.join(ROOT, "node_modules/.bin/tsx"), path.join(ROOT, "src/index-build.ts")],
    { cwd: ROOT, encoding: "utf8", stdio: "inherit" },
  );
  if (result.status !== 0) throw new Error("index-build failed");
}

async function checkSie(): Promise<boolean> {
  try {
    const r = await fetch(`${config.sieUrl}/healthz`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch {
    return false;
  }
}

async function fetchRegistered(): Promise<{ ok: boolean; names: string[] }> {
  try {
    const r = await fetch(`${config.sieUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return { ok: false, names: [] };
    const json = (await r.json()) as { models?: { name: string }[] };
    return { ok: true, names: (json.models ?? []).map((m) => m.name) };
  } catch {
    return { ok: false, names: [] };
  }
}

async function handleRun(res: http.ServerResponse, accountId: string): Promise<void> {
  const push = setupSse(res);
  const account = loadAccounts().find((a) => a.id === accountId);
  if (!account) {
    push({ type: "error", data: { stage: "lookup", message: `unknown account id: ${accountId}` } });
    res.end();
    return;
  }

  if (!(await checkSie())) {
    push({ type: "error", data: { stage: "sie", message: `SIE not reachable at ${config.sieUrl}` } });
    res.end();
    return;
  }

  try {
    ensureIndex();
  } catch (e) {
    push({ type: "error", data: { stage: "index", message: e instanceof Error ? e.message : String(e) } });
    res.end();
    return;
  }

  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,
    waitForCapacity: true,
    provisionTimeout: 900_000,
  });

  try {
    await runScore(account, { client, emit: push });
  } catch (e) {
    push({ type: "error", data: { stage: "pipeline", message: e instanceof Error ? e.message : String(e) } });
  } finally {
    res.end();
  }
}

/** The pre-SIE board: accounts split into risk/opportunity by the raw score. */
function boardPayload() {
  return loadAccounts()
    .map((a) => ({ account: a, ...scoreSignals(a.signals) }))
    .sort((a, b) => b.score - a.score)
    .map((r) => ({
      id: r.account.id,
      name: r.account.name,
      domain: r.account.domain,
      owner: r.account.owner,
      arr: r.account.arr,
      renewalDays: r.account.renewalDays,
      contact: r.account.contact,
      signals: r.account.signals,
      score: r.score,
      band: r.band,
      direction: r.direction,
      reason: r.reason,
    }));
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  const p = url.pathname;

  if (p === "/" || p === "/index.html") return serveFile(res, path.join(PUBLIC_DIR, "index.html"));
  if (p.startsWith("/static/")) return serveFile(res, path.join(PUBLIC_DIR, p.slice("/static/".length)));

  if (p === "/api/health") {
    const { ok, names } = await fetchRegistered();
    return send(
      res,
      200,
      JSON.stringify({
        sie: ok,
        sieUrl: config.sieUrl,
        registeredModels: names.length,
        registered: names,
        chatModel: config.chatModel || null,
        brief: config.chatModel ? "llm" : "deterministic",
      }),
      "application/json",
    );
  }

  if (p === "/api/accounts") {
    return send(res, 200, JSON.stringify(boardPayload()), "application/json");
  }

  if (p === "/api/run") {
    const id = url.searchParams.get("id");
    if (!id) return send(res, 400, "missing id");
    return handleRun(res, id);
  }

  return send(res, 404, "not found");
});

server.listen(config.port, () => {
  const url = `http://localhost:${config.port}`;
  console.log(`account-signal-scoring ui: ${url}`);
  console.log(`brief mode: ${config.chatModel ? `llm (${config.chatModel})` : "deterministic (set SIE_CHAT_MODEL to use an LLM)"}`);
  if (process.env.OPEN_BROWSER !== "0") {
    const opener = process.platform === "darwin" ? "open" : process.platform === "win32" ? "start" : "xdg-open";
    spawnSync(opener, [url], { stdio: "ignore" });
  }
});
