import express, { Request, Response } from "express";
import path from "path";
import fs from "fs";
import yaml from "js-yaml";
import { SIEClient } from "@superlinked/sie-sdk";
import { search, loadConfig, loadIndex, type SearchResult } from "./search.js";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");
const app = express();
const PORT = Number(process.env.PORT ?? "3000");

let client: SIEClient;
let config: ReturnType<typeof loadConfig>;
let embeddings: number[][];
let metadata: Array<{
  id: string; title: string; description: string; category: string;
  price: number; rating: number; features: string[]; image?: string;
  attributes: Record<string, string>;
}>;
let categoryCounts: [string, number][];

app.use(express.json());
app.use(express.static(path.join(ROOT, "static")));

app.get("/", (_req: Request, res: Response) => {
  res.sendFile(path.join(ROOT, "static", "index.html"));
});

app.get("/api/search", async (req: Request, res: Response): Promise<void> => {
  const q = req.query.q as string | undefined;
  const category = req.query.category as string | undefined;
  const brand = req.query.brand as string | undefined;

  if (!q) {
    res.status(400).json({ error: "Query parameter 'q' is required" });
    return;
  }

  const filters: Record<string, string> = {};
  if (category) filters.category = category;
  if (brand) filters.brand = brand;

  try {
    const results: SearchResult[] = await search(
      client,
      config,
      embeddings,
      metadata,
      q,
      Object.keys(filters).length ? filters : undefined
    );

    res.json({
      query: q,
      filters,
      results: results.map((r) => ({
        id: r.product_id,
        title: r.title,
        description: r.description,
        category: r.category,
        price: r.price,
        rating: r.rating,
        features: r.features ?? [],
        image: r.image,
        attributes: r.attributes,
        scores: {
          vector: Number(r.vector_score.toFixed(4)),
          rerank: Number((r.rerank_score ?? 0).toFixed(4)),
        },
      })),
    });
  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: "Search failed" });
  }
});

app.get("/api/categories", (_req: Request, res: Response) => {
  res.json(categoryCounts);
});

app.get("/api/stats", (_req: Request, res: Response) => {
  res.json({
    total_products: metadata.length,
    embedding_dims: embeddings[0]?.length ?? 0,
    categories: categoryCounts.length,
    models: config.models,
  });
});

async function start(): Promise<void> {
  const configData = fs.readFileSync(path.join(ROOT, "config.yaml"), "utf-8");
  config = yaml.load(configData) as ReturnType<typeof loadConfig>;

  const clusterUrl = process.env.SIE_CLUSTER_URL ?? config.cluster.url;
  const apiKey = process.env.SIE_API_KEY ?? config.cluster.api_key;

  console.log("Connecting to SIE cluster:", clusterUrl);
  const gpu = config.cluster.gpu;
  const provisionTimeout = (config.cluster.provision_timeout_s ?? 600) * 1000;
  client = new SIEClient(clusterUrl, { apiKey, gpu, waitForCapacity: true, provisionTimeout, timeout: 120_000 });

  console.log("Loading index...");
  const idx = loadIndex();
  embeddings = idx.embeddings;
  metadata = idx.metadata;
  console.log(`Loaded ${metadata.length} products, ${embeddings[0]?.length}d`);

  const counts: Record<string, number> = {};
  for (const m of metadata) counts[m.category] = (counts[m.category] ?? 0) + 1;
  categoryCounts = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  app.listen(PORT, () => {
    console.log(`\nServer at http://localhost:${PORT}`);
    console.log(`Models:`);
    console.log(`  embedder:  ${config.models.embedding}`);
    console.log(`  extractor: ${config.models.extractor}`);
    console.log(`  reranker:  ${config.models.reranker}`);
    console.log(`\nAPI:`);
    console.log(`  GET /api/search?q=...&category=...&brand=...`);
    console.log(`  GET /api/categories`);
    console.log(`  GET /api/stats`);
  });
}

process.on("SIGINT", () => {
  console.log("\nShutting down...");
  process.exit(0);
});

start().catch((err) => {
  console.error("Startup error:", err);
  process.exit(1);
});
