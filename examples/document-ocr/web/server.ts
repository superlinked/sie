import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SIEClient } from "@superlinked/sie-sdk";
import { NER_MODELS, RECOGNITION_MODELS, STRUCTURED_MODELS, config } from "../src/config.js";
import type { PipelineEvent } from "../src/events.js";
import { runPipeline } from "../src/pipeline.js";
import type { SampleDoc } from "../src/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.resolve(__dirname, "public");
const SAMPLES_PATH = path.resolve(ROOT, config.paths.samples);
const SAMPLE_DIR = path.resolve(ROOT, config.paths.sampleDir);

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
};

function send(res: http.ServerResponse, status: number, body: string, type = "text/plain") {
  res.writeHead(status, { "content-type": type });
  res.end(body);
}

function serveFile(res: http.ServerResponse, file: string) {
  if (!fs.existsSync(file)) return send(res, 404, "not found");
  const ext = path.extname(file).toLowerCase();
  res.writeHead(200, { "content-type": MIME[ext] ?? "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
}

function setupSse(res: http.ServerResponse) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  return (event: { type: string; data?: unknown }) => {
    res.write(`event: ${event.type}\n`);
    res.write(`data: ${JSON.stringify(event.data ?? null)}\n\n`);
  };
}

async function fetchModels(): Promise<{ ok: boolean; names: string[]; cuda: boolean | null }> {
  try {
    const r = await fetch(`${config.sieUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return { ok: false, names: [], cuda: null };
    const json = (await r.json()) as { models?: { name?: string }[] };
    const names = (json.models ?? [])
      .map((model) => model.name)
      .filter((name): name is string => typeof name === "string" && name.length > 0);
    // ModelInfo and the health probes do not report a device. null is unknown,
    // not "CUDA unavailable"; only an explicit false blocks a gpuRequired model.
    return { ok: true, names, cuda: null };
  } catch {
    return { ok: false, names: [], cuda: null };
  }
}

function readJson<T>(p: string): T {
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

async function handleRun(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  sampleId: string,
  recognitionModel: string,
  structuredModel: string,
  nerModel: string,
) {
  const push = setupSse(res);
  let closed = false;
  req.on("close", () => {
    closed = true;
  });

  const samples = readJson<SampleDoc[]>(SAMPLES_PATH);
  const sample = samples.find((s) => s.id === sampleId);
  if (!sample) {
    push({ type: "error", data: { stage: "lookup", message: `unknown sample id: ${sampleId}` } });
    return res.end();
  }

  const imagePath = path.resolve(SAMPLE_DIR, sample.filename);
  if (!fs.existsSync(imagePath)) {
    push({ type: "error", data: { stage: "lookup", message: `sample image not found: ${sample.filename}` } });
    return res.end();
  }
  const imageBytes = fs.readFileSync(imagePath);

  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,            // 10 min request timeout (CPU + Rosetta is slow)
    waitForCapacity: true,        // retry while a model is warming up
    provisionTimeout: 900_000,    // 15 min ceiling on cold-load polling
  });

  try {
    await runPipeline({
      client,
      imageBytes,
      sample,
      recognitionModel,
      structuredModel,
      nerModel,
      emit: (event: PipelineEvent) => {
        if (closed) return;
        push({ type: event.type, data: "data" in event ? event.data : null });
      },
    });
  } catch (err) {
    push({ type: "error", data: { stage: "pipeline", message: (err as Error).message } });
  } finally {
    await client.close().catch(() => {});
    res.end();
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  const p = url.pathname;

  if (p === "/" || p === "/index.html") return serveFile(res, path.join(PUBLIC_DIR, "index.html"));
  if (p.startsWith("/static/")) return serveFile(res, path.join(PUBLIC_DIR, p.slice("/static/".length)));
  if (p.startsWith("/samples/")) return serveFile(res, path.join(SAMPLE_DIR, p.slice("/samples/".length)));

  if (p === "/api/health") {
    const { ok, names, cuda } = await fetchModels();
    return send(
      res,
      200,
      JSON.stringify({
        sie: ok,
        sieUrl: config.sieUrl,
        registeredModels: names.length,
        registered: names,
        cuda,
        // True only when /v1/models succeeded, including an empty list.
        // The page must not infer this from sie; that flag can be another probe.
        catalogKnown: ok,
      }),
      "application/json",
    );
  }
  if (p === "/api/models") {
    return send(
      res,
      200,
      JSON.stringify({
        recognition: RECOGNITION_MODELS,
        structured: STRUCTURED_MODELS,
        ner: NER_MODELS,
        defaults: config.defaults,
      }),
      "application/json",
    );
  }
  if (p === "/api/samples") {
    return send(res, 200, fs.readFileSync(SAMPLES_PATH, "utf8"), "application/json");
  }
  if (p === "/api/run") {
    const id = url.searchParams.get("id");
    const recognitionModel =
      url.searchParams.get("recognition") ?? config.defaults.recognition;
    const structuredModel = url.searchParams.get("structured") ?? config.defaults.structured;
    const nerModel = url.searchParams.get("ner") ?? config.defaults.ner;
    if (!id) return send(res, 400, "missing id");
    return handleRun(req, res, id, recognitionModel, structuredModel, nerModel);
  }

  return send(res, 404, "not found");
});

server.listen(config.port, () => {
  const url = `http://localhost:${config.port}`;
  console.log(`document-ocr ui: ${url}`);
  if (process.env.OPEN_BROWSER !== "0") {
    const opener =
      process.platform === "darwin"
        ? "open"
        : process.platform === "win32"
          ? "start"
          : "xdg-open";
    spawnSync(opener, [url], { stdio: "ignore" });
  }
});
