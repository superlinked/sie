// SIE-driven account scoring: turn a pile of signals into one score and a play.
//
// Pipeline:
//   0. Roll the account's signals into one deterministic 0-100 score + band.
//   1. Build a natural-language account context summary.
//   2. Extract entities with GLiNER (surfaced to the UI; not fed to the score).
//   3. Encode the summary with a small dense encoder (MiniLM).
//   4. Cosine-rank against a pre-encoded corpus of past-outcome playbooks.
//   5. Rerank the top K with a cross-encoder to pick the play that fits.
//   6. Draft the account brief (SIE chat/completions, else deterministic).

import fs from "node:fs";
import path from "node:path";
import { SIEClient } from "@superlinked/sie-sdk";
import { generateBrief } from "./brief.js";
import { config } from "./config.js";
import type { ScoreEvent } from "./events.js";
import { buildAccountSummary, scoreSignals } from "./signals.js";
import type { Account, IndexRecord, Playbook } from "./types.js";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

export function loadJson<T>(rel: string): T {
  return JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8")) as T;
}

export function loadAccounts(): Account[] {
  return loadJson<Account[]>(config.paths.accounts);
}

export function loadPlaybooks(): Playbook[] {
  return loadJson<Playbook[]>(config.paths.playbooks);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export interface ScoreRunDeps {
  client: SIEClient;
  emit: (event: ScoreEvent) => void;
}

export async function runScore(account: Account, deps: ScoreRunDeps): Promise<void> {
  const { client, emit } = deps;
  const start = Date.now();

  emit({
    type: "models",
    data: {
      encoder: config.models.encoder,
      reranker: config.models.reranker,
      extractor: config.models.extractor,
      chat: config.chatModel || "(deterministic fallback)",
    },
  });

  // 0. Deterministic signal roll-up.
  const signalScore = scoreSignals(account.signals);
  emit({ type: "signals", data: signalScore });

  const summary = buildAccountSummary(account);

  // 1. Extract entities (surface to the UI; not fed into the score).
  emit({ type: "extracting" });
  const tEx = Date.now();
  const extractOut = await client.extract(
    config.models.extractor,
    { text: summary },
    { labels: [...config.extractLabels] },
  );
  const entities = (extractOut.entities ?? []).map((e) => ({
    label: e.label ?? "",
    text: e.text ?? "",
    score: Number(e.score ?? 0),
  }));
  emit({ type: "extracted", data: { entities, ms: Date.now() - tEx } });

  // 2. Encode the account summary.
  emit({ type: "encoding" });
  const tEn = Date.now();
  const enc = await client.encode(config.models.encoder, { text: summary });
  const queryVec = enc.dense;
  if (!queryVec) throw new Error("encoder returned no dense vector for the account");
  emit({ type: "encoded", data: { dim: queryVec.length, ms: Date.now() - tEn } });

  // 3. Cosine-rank the playbook corpus to shortlist the top-K candidates.
  const playbooks = loadPlaybooks();
  const index: IndexRecord[] = loadJson(config.paths.index);
  const byId = new Map(playbooks.map((p) => [p.id, p]));
  const shortlist = index
    .map((rec) => ({ id: rec.id, score: cosine(Array.from(queryVec), rec.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, config.rerank.topK);

  // 4. Rerank the shortlist with the cross-encoder for the sharper final order.
  emit({ type: "scoring" });
  const tSc = Date.now();
  const docs = shortlist.map((c) => {
    const p = byId.get(c.id);
    return { id: c.id, text: p ? `${p.label}. ${p.summary} ${p.play}` : "" };
  });
  const scored = await client.score(config.models.reranker, { text: summary }, docs);
  const rerankById = new Map<string, number>();
  for (const s of scored.scores ?? []) rerankById.set(s.itemId, s.score);
  const hits = shortlist.map((c) => {
    const p = byId.get(c.id)!;
    return {
      id: p.id,
      label: p.label,
      direction: p.direction,
      summary: p.summary,
      outcome: p.outcome,
      play: p.play,
      score: rerankById.get(c.id) ?? 0,
    };
  });
  hits.sort((a, b) => b.score - a.score);
  const topPlaybook = hits[0] ? byId.get(hits[0].id) : undefined;
  emit({ type: "scored", data: { hits, topPlaybook: topPlaybook?.label ?? "", ms: Date.now() - tSc } });

  // 5. Draft the account brief.
  emit({ type: "briefing" });
  const tBr = Date.now();
  const brief = await generateBrief(account, topPlaybook, summary);
  emit({ type: "brief", data: { ...brief, ms: Date.now() - tBr } });

  emit({ type: "done", data: { totalMs: Date.now() - start } });
}
