import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import { SIEClient } from "@superlinked/sie-sdk";
import { fileURLToPath } from "url";

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
  };
}

interface Metadata {
  id: string;
  title: string;
  description: string;
  category: string;
  price: number;
  rating: number;
  features: string[];
  image?: string;
  attributes: Record<string, string>;
}

export interface SearchResult {
  product_id: string;
  title: string;
  category: string;
  price: number;
  rating: number;
  description: string;
  features: string[];
  image?: string;
  attributes: Record<string, string>;
  vector_score: number;
  rerank_score?: number;
  rank: number;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");
const TS_DIR = path.resolve(__dirname, "..");

export function loadConfig(): Config {
  const data = fs.readFileSync(path.join(ROOT, "config.yaml"), "utf-8");
  return yaml.load(data) as Config;
}

export function loadIndex(): { embeddings: number[][]; metadata: Metadata[] } {
  const outDir = path.join(TS_DIR, "data-ts");
  const raw: number[][] = JSON.parse(
    fs.readFileSync(path.join(outDir, "embeddings.json"), "utf-8")
  );
  const embeddings = raw.map(normalize);
  const metadata: Metadata[] = JSON.parse(
    fs.readFileSync(path.join(outDir, "metadata.json"), "utf-8")
  );
  return { embeddings, metadata };
}

function normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n > 0 ? v.map((x) => x / n) : v;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function toResult(
  m: Metadata, vectorScore: number, rank: number, rerankScore?: number
): SearchResult {
  return {
    product_id: m.id, title: m.title, category: m.category,
    price: m.price, rating: m.rating, description: m.description,
    features: m.features, image: m.image, attributes: m.attributes,
    vector_score: vectorScore, rerank_score: rerankScore, rank,
  };
}

export async function search(
  client: SIEClient,
  config: Config,
  embeddings: number[][],
  metadata: Metadata[],
  query: string,
  filters?: Record<string, string>
): Promise<SearchResult[]> {
  const topKCandidates = config.search.top_k_candidates;
  const topKResults = config.search.top_k_results;

  // Step 1: Encode query as single item with isQuery in options
  const queryResult = await client.encode(
    config.models.embedding,
    { id: "query", text: query },
    { isQuery: true }
  );
  if (!queryResult.dense) throw new Error("Failed to encode query");

  const queryVec = normalize(Array.from(queryResult.dense));

  // Step 2: Vector similarity
  let candidates = embeddings
    .map((e, idx) => ({ idx, score: dot(queryVec, e) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topKCandidates);

  // Step 3: Attribute filters (optional)
  if (filters && Object.keys(filters).length > 0) {
    const filtered = candidates.filter((c) => {
      const m = metadata[c.idx];
      for (const [key, value] of Object.entries(filters)) {
        if (key === "category") {
          if (!m.category.toLowerCase().includes(value.toLowerCase())) return false;
        } else {
          const attrVal = m.attributes[key] ?? "";
          if (!attrVal.toLowerCase().includes(value.toLowerCase())) return false;
        }
      }
      return true;
    });
    if (filtered.length > 0) {
      candidates = filtered;
    } else {
      // No candidate matched the filter. Fall back to the unfiltered pool so the
      // user still gets results, but warn so the behavior is observable.
      console.warn(`  [filter] no matches for ${JSON.stringify(filters)}, falling back to unfiltered candidates`);
    }
  }

  // Step 4: Rerank with cross-encoder score()
  const rerankItems = candidates.map((c) => {
    const m = metadata[c.idx];
    const attrStr = Object.entries(m.attributes)
      .map(([k, v]) => `${k}: ${v}`)
      .join(", ");
    return {
      id: m.id,
      text: `${m.title}. Category: ${m.category}. ${attrStr ? attrStr + ". " : ""}${m.description}`,
    };
  });

  let scoreResult;
  try {
    scoreResult = await client.score(
      config.models.reranker,
      { id: "query", text: query },
      rerankItems
    );
  } catch (err) {
    console.warn(`  Reranking failed (${err}), falling back to vector ranking`);
    return candidates.slice(0, topKResults).map((c, i) =>
      toResult(metadata[c.idx], c.score, i + 1)
    );
  }

  const candidateById = new Map(candidates.map((c) => [metadata[c.idx].id, c]));

  return scoreResult.scores.flatMap((scored, i) => {
    const c = candidateById.get(scored.itemId);
    if (!c) return [];
    return [toResult(metadata[c.idx], c.score, i + 1, scored.score)];
  }).slice(0, topKResults);
}

// CLI demo
async function main(): Promise<void> {
  console.log("=== E-Commerce Product Search (TypeScript Search) ===\n");

  const config = loadConfig();
  const clusterUrl = process.env.SIE_CLUSTER_URL ?? config.cluster.url;
  const apiKey = process.env.SIE_API_KEY ?? config.cluster.api_key;

  const gpu = config.cluster.gpu;
  const provisionTimeout = (config.cluster.provision_timeout_s ?? 600) * 1000;
  const client = new SIEClient(clusterUrl, { apiKey, gpu, waitForCapacity: true, provisionTimeout, timeout: 120_000 });
  const { embeddings, metadata } = loadIndex();
  console.log(`Loaded index: ${metadata.length} products, ${embeddings[0]?.length}d\n`);

  const demos: Array<{ query: string; filters?: Record<string, string> }> = [
    { query: "lightweight waterproof hiking boots" },
    { query: "gold jewelry for women" },
    { query: "wireless bluetooth headphones", filters: { category: "All Electronics" } },
  ];

  for (const { query, filters } of demos) {
    const filterStr = filters ? ` (filters: ${JSON.stringify(filters)})` : "";
    console.log(`\n--- Query: "${query}"${filterStr} ---`);

    try {
      const results = await search(client, config, embeddings, metadata, query, filters);
      results.forEach((r) => {
        const attrs = Object.entries(r.attributes)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
        console.log(`  ${r.rank}. ${r.title.substring(0, 65)}`);
        console.log(`     category=${r.category}  price=$${r.price ?? "?"}  rating=${r.rating ?? "?"}`);
        console.log(`     attrs: ${attrs || "none"}`);
        console.log(`     vector=${r.vector_score.toFixed(3)}  rerank=${r.rerank_score?.toFixed(4) ?? "N/A"}`);
        console.log();
      });
    } catch (err) {
      console.error(`  Search failed: ${err}`);
    }
  }
}

// Only run CLI demo when this file is the entry point
if (process.argv[1] === __filename) {
  main();
}
