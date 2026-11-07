import { basename, join } from "node:path";
import { ocrPage } from "@superlinked/sie-cognee";
import { documentPage } from "./cli.js";
import { DATA, launch } from "./common.js";

await launch<ReturnType<typeof documentPage>>(
  "02-documents",
  async (demo, { file, image }) => {
    const source = await ocrPage(demo.native("ocr"), {
      model: demo.config.models.ocr,
      image,
      sourceId: `${basename(file)}:page-1`,
      filename: basename(file),
      pageNumber: 1,
      maxBytes: demo.config.limits.documentBytes,
    });
    demo.show("actual-ocr", source);
    demo.show("document-id", await demo.memory.add(source));
    demo.show("memory-build", await demo.memory.build());
    demo.show(
      "retrieved-ocr-chunks",
      await demo.memory.retrieve("Is step-free access confirmed?", demo.config.limits.retrieveTopK),
    );
  },
  (config) =>
    documentPage(process.argv.slice(2), join(DATA, "access-note.png"), config.limits.documentBytes),
);
