import { launch, seedMemory } from "./common.js";

await launch("01-memory", async (demo) => {
  await seedMemory(demo);
  demo.show(
    "source-chunks",
    await demo.memory.retrieve(demo.config.decision.query, demo.config.limits.retrieveTopK),
  );
  if (process.argv.includes("--graph")) {
    demo.show(
      "model-derived-graph-context",
      await demo.cognee.search(demo.config.decision.query, {
        datasets: [demo.dataset],
        searchType: "GRAPH_COMPLETION",
        onlyContext: true,
        saveInteraction: false,
      }),
    );
  }
});
