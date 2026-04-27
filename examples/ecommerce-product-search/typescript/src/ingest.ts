import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import { SIEClient } from "@superlinked/sie-sdk";
import type { Entity } from "@superlinked/sie-sdk";

import { fileURLToPath } from "url";

// Config matches config.yaml structure exactly
interface Config {
  cluster: {
    url: string;
    api_key: string;
    gpu: string;
    provision_timeout_s: number;
  };
  models: {
    embedding: string;
    reranker: string;
    extractor: string;
  };
  extract_labels: string[];
  search: {
    top_k_candidates: number;
    top_k_results: number;
  };
  ingest: {
    batch_size_extraction: number;
    batch_size_encoding: number;
    confidence_threshold: number;
    junk_text_max_len?: number;
  };
}

interface Product {
  id: string;
  title: string;
  description: string;
  category: string;
  price: number;
  rating: number;
  features?: string[];
  image?: string;
}

interface ExtractedProduct extends Product {
  attributes: Record<string, string>;
}

// Resolve paths relative to this file's location (works regardless of cwd)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");
const TS_DIR = path.resolve(__dirname, "..");

function loadConfig(): Config {
  const data = fs.readFileSync(path.join(ROOT, "config.yaml"), "utf-8");
  return yaml.load(data) as Config;
}

function loadProducts(): Product[] {
  const data = fs.readFileSync(path.join(ROOT, "data", "products.json"), "utf-8");
  return JSON.parse(data);
}

function buildAttributes(
  entities: Entity[],
  labels: string[],
  threshold: number,
  junkLen: number
): Record<string, string> {
  const best: Record<string, { text: string; score: number }> = {};

  for (const e of entities) {
    const text = e.text.trim();
    if (e.score < threshold) continue;
    if (text.length > junkLen && !text.includes(" ")) continue;
    if (!labels.includes(e.label)) continue;
    if (!(e.label in best) || e.score > best[e.label].score) {
      best[e.label] = { text, score: e.score };
    }
  }

  return Object.fromEntries(Object.entries(best).map(([k, v]) => [k, v.text]));
}

async function extractAttributes(
  client: SIEClient,
  products: Product[],
  config: Config
): Promise<Record<string, Record<string, string>>> {
  const model = config.models.extractor;
  const labels = config.extract_labels;
  const batchSize = config.ingest.batch_size_extraction;
  const threshold = config.ingest.confidence_threshold;
  const junkLen = config.ingest.junk_text_max_len ?? 15;
  const total = products.length;

  console.log(`\n--- Extraction (${model}) ---`);
  console.log(`  Labels: ${labels.join(", ")}`);

  const extracted: Record<string, Record<string, string>> = {};

  for (let i = 0; i < total; i += batchSize) {
    const batch = products.slice(i, Math.min(i + batchSize, total));
    const items = batch.map((p) => ({ id: p.id, text: p.description.substring(0, 512) }));
    const t0 = Date.now();

    const results = await client.extract(model, items, { labels });

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`  [${Math.min(i + batchSize, total)}/${total}] batch in ${elapsed}s`);

    results.forEach((result, idx) => {
      const productId = batch[idx].id;
      extracted[productId] = buildAttributes(result.entities, labels, threshold, junkLen);
    });
  }

  return extracted;
}

async function encodeProducts(
  client: SIEClient,
  products: Product[],
  config: Config
): Promise<number[][]> {
  const model = config.models.embedding;
  const batchSize = config.ingest.batch_size_encoding;
  const total = products.length;

  console.log(`\n--- Encoding (${model}) ---`);

  const allEmbeddings: number[][] = [];

  for (let i = 0; i < total; i += batchSize) {
    const batch = products.slice(i, Math.min(i + batchSize, total));
    const items = batch.map((p) => ({
      id: p.id,
      text: `${p.title}. ${p.description.substring(0, 512)}`,
    }));
    const t0 = Date.now();

    const results = await client.encode(model, items);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`  [${Math.min(i + batchSize, total)}/${total}] batch in ${elapsed}s`);

    // Fail loud on missing dense vectors. Silently skipping here would desync
    // embeddings from metadata (which is built positionally from products).
    for (const r of results) {
      if (!r.dense) {
        throw new Error(`encode() returned no dense vector for item ${r.id ?? "<unknown>"}`);
      }
      allEmbeddings.push(Array.from(r.dense));
    }
  }

  return allEmbeddings;
}

async function main(): Promise<void> {
  try {
    console.log("=== E-Commerce Product Search (TypeScript Ingest) ===\n");

    const config = loadConfig();
    const clusterUrl = process.env.SIE_CLUSTER_URL ?? config.cluster.url;
    const apiKey = process.env.SIE_API_KEY ?? config.cluster.api_key;

    console.log(`Cluster: ${clusterUrl}`);
    console.log(`Embedder: ${config.models.embedding}`);
    console.log(`Extractor: ${config.models.extractor}`);

    const gpu = config.cluster.gpu;
    const provisionTimeout = (config.cluster.provision_timeout_s ?? 600) * 1000;
    const client = new SIEClient(clusterUrl, { apiKey, gpu, waitForCapacity: true, provisionTimeout, timeout: 300_000 });

    console.log("\nLoading products...");
    const products = loadProducts();
    console.log(`Loaded ${products.length} products`);

    // Phase 1: Extract attributes
    const extracted = await extractAttributes(client, products, config);
    console.log(`\nExtraction complete`);

    // Phase 2: Encode embeddings
    const embeddings = await encodeProducts(client, products, config);
    console.log(`\nEncoding complete: ${embeddings.length} embeddings, ${embeddings[0]?.length}d`);

    // Build metadata with attributes
    const metadata: ExtractedProduct[] = products.map((p) => ({
      ...p,
      description: p.description.substring(0, 500),
      attributes: extracted[p.id] ?? {},
    }));

    // Save output to data-ts/ inside typescript/
    const outDir = path.join(TS_DIR, "data-ts");
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    fs.writeFileSync(path.join(outDir, "embeddings.json"), JSON.stringify(embeddings));
    console.log(`\nSaved embeddings → data-ts/embeddings.json`);

    fs.writeFileSync(path.join(outDir, "metadata.json"), JSON.stringify(metadata, null, 2));
    console.log(`Saved metadata   → data-ts/metadata.json (${metadata.length} products)`);

    console.log("\n=== Ingest Complete ===");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
