import { readFileSync, statSync } from "node:fs";
import { basename, extname, join, resolve } from "node:path";
import { ocrPage } from "@superlinked/sie-cognee";
import { DATA, launch } from "./common.js";

await launch("02-documents", async (demo) => {
  const args = process.argv.slice(2).filter((argument) => argument !== "--");
  if (args.length > 1) throw new Error("Supply at most one raster page.");
  const file = resolve(args[0] ?? join(DATA, "access-note.png"));
  if (![".png", ".jpg", ".jpeg", ".webp"].includes(extname(file).toLowerCase()))
    throw new Error(
      "Supply one raster page as PNG, JPEG or WebP; render PDF pages before calling this example.",
    );
  if (statSync(file).size > demo.config.limits.documentBytes)
    throw new Error("Document exceeds documentBytes.");
  const source = await ocrPage(demo.native("ocr"), {
    model: demo.config.models.ocr,
    image: readFileSync(file),
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
});
