import assert from "node:assert/strict";
import { SIEClient } from "../../packages/sie_ts_sdk/dist/index.js";

const client = new SIEClient(process.env.SIE_SERVER_URL, { timeout: 60_000 });
try {
  assert.ok((await client.listModels()).length > 0);
  const model = "sie-fake";
  const result = await client.encode(model, { id: "ts-cpu-smoke", text: "Hello world" });
  assert.equal(result.id, "ts-cpu-smoke");
  assert.ok(result.dense instanceof Float32Array);
  assert.equal(result.dense.length, 384);
  assert.ok(Array.from(result.dense).every(Number.isFinite));
  const batch = await client.encode(model, [
    { id: "one", text: "Hello" },
    { id: "two", text: "World" },
  ]);
  assert.deepEqual(batch.map((item) => item.id), ["one", "two"]);
  const scored = await client.score(model, { text: "query" }, [{ text: "one" }, { text: "two" }]);
  assert.equal(scored.scores.length, 2);
  console.log("Built TypeScript SDK CPU encode/batch/score passed.");
} finally {
  await client.close();
}
