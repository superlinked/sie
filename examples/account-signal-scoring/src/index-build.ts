// One-time index build: encode the outcome-playbook corpus into dense vectors
// and dump to data/playbook_index.json. Re-run with `npm run index` after
// editing data/playbooks.json.
import fs from "node:fs";
import path from "node:path";
import { SIEClient } from "@superlinked/sie-sdk";
import { config } from "./config.js";
import type { IndexRecord, Playbook } from "./types.js";

async function main(): Promise<void> {
  const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
  const inPath = path.join(root, config.paths.playbooks);
  const outPath = path.join(root, config.paths.index);

  const playbooks: Playbook[] = JSON.parse(fs.readFileSync(inPath, "utf8"));
  console.log(`encoding ${playbooks.length} playbooks`);

  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,
    waitForCapacity: true,
    provisionTimeout: 900_000,
  });

  const items = playbooks.map((p) => ({ id: p.id, text: `${p.label}. ${p.summary} ${p.play}` }));
  const results = await client.encode(config.models.encoder, items);

  const records: IndexRecord[] = playbooks.map((p, i) => {
    const single = results[i];
    if (!single?.dense) throw new Error(`encoder returned no dense vector for ${p.id}`);
    return { id: p.id, vector: Array.from(single.dense) };
  });

  fs.writeFileSync(outPath, JSON.stringify(records, null, 2));
  console.log(`wrote ${outPath} (${records.length} vectors, dim=${records[0]?.vector.length})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
