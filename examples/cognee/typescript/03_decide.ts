import { decide, evidencePacket, rerankChunks } from "@superlinked/sie-cognee";
import { launch, seedMemory } from "./common.js";

await launch("03-decide", async (demo) => {
  await seedMemory(demo);
  const { query, instructions, choices } = demo.config.decision;
  const candidates = await demo.memory.retrieve(query, demo.config.limits.retrieveTopK);
  demo.show("retrieved-source-chunks", candidates);
  const ranked = await rerankChunks(demo.native("rerank"), query, candidates, {
    model: demo.config.models.rerank,
    topK: demo.config.limits.rerankTopK,
  });
  demo.show("reranked-chunks", ranked);
  const packet = evidencePacket(
    query,
    ranked.map((result) => result.chunk),
    demo.config.limits.evidenceChars,
  );
  demo.show("exact-evidence-packet", { text: packet });
  demo.show(
    "decision",
    await decide(demo.native("decision"), packet, {
      model: demo.config.models.decision,
      choices,
      instructions,
      maxChars: demo.config.limits.evidenceChars,
    }),
  );
});
