// Headless scorer: rank every sample account into one board a team can act on.
//
//   npm run score           # score all accounts, print the ranked board
//   npm run score helix     # score a single account by id
//
// Requires SIE running (see compose.yml) and the playbook index built
// (`npm run index`, or the web server builds it on first run).
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { SIEClient } from "@superlinked/sie-sdk";
import { config } from "./config.js";
import type { ScoreEvent } from "./events.js";
import { loadAccounts, runScore } from "./score.js";
import { scoreSignals } from "./signals.js";
import type { AccountScore } from "./signals.js";
import type { Account } from "./types.js";

interface ScoredRow extends AccountScore {
  account: Account;
  play: string;
  briefLine: string;
}

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

function ensureIndex(): void {
  if (fs.existsSync(path.join(ROOT, config.paths.index))) return;
  console.log("building playbook index...");
  const r = spawnSync(process.execPath, [path.join(ROOT, "node_modules/.bin/tsx"), path.join(ROOT, "src/index-build.ts")], {
    cwd: ROOT,
    stdio: "inherit",
  });
  if (r.status !== 0) throw new Error("index-build failed");
}

async function scoreOne(client: SIEClient, account: Account, verbose: boolean) {
  let play = "";
  let briefLine = "";
  const emit = (e: ScoreEvent) => {
    if (e.type === "scored") play = e.data.hits[0]?.play ?? "";
    if (e.type === "brief") briefLine = e.data.summary;
    if (verbose && "data" in e) console.log(`  ${e.type}`, JSON.stringify(e.data));
  };
  await runScore(account, { client, emit });
  const s = scoreSignals(account.signals);
  return { account, ...s, play, briefLine };
}

async function main(): Promise<void> {
  const id = process.argv[2];
  const accounts = loadAccounts();
  const targets = id ? accounts.filter((a) => a.id === id) : accounts;
  if (targets.length === 0) throw new Error(`unknown account id: ${id}`);

  ensureIndex();
  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,
    waitForCapacity: true,
    provisionTimeout: 900_000,
  });

  const rows: ScoredRow[] = [];
  for (const a of targets) rows.push(await scoreOne(client, a, Boolean(id)));

  // Risk board first (descending urgency), then opportunity board.
  const risk = rows.filter((r) => r.direction === "risk").sort((a, b) => b.score - a.score);
  const opp = rows.filter((r) => r.direction === "opportunity").sort((a, b) => b.score - a.score);

  const line = (r: (typeof rows)[number]) =>
    `  [${r.band.toUpperCase().padEnd(5)}] ${String(r.score).padStart(3)}  ${r.account.name.padEnd(20)} $${r.account.arr.toLocaleString().padStart(8)}  → ${r.play}`;

  console.log("\n=== CHURN-RISK BOARD ===");
  for (const r of risk) console.log(line(r));
  console.log("\n=== EXPANSION BOARD ===");
  for (const r of opp) console.log(line(r));
  console.log("");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
