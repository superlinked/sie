import type { SIEClient } from "@superlinked/sie-sdk";
import { NER_MODELS, RECOGNITION_MODELS, STRUCTURED_MODELS } from "./config.js";
import { structuredExtract } from "./donut.js";
import type { PipelineEvent } from "./events.js";
import { extractFields } from "./extract.js";
import { recognize } from "./ocr.js";
import type { SampleDoc, TriageResult } from "./types.js";

export type RunInput = {
  client: SIEClient;
  imageBytes: Uint8Array;
  sample: SampleDoc;
  recognitionModel: string;
  structuredModel: string;
  nerModel: string;
  emit: (event: PipelineEvent) => void;
};

function lookup<T extends { id: string }>(list: T[], id: string): T {
  const found = list.find((m) => m.id === id);
  if (!found) throw new Error(`unknown model id: ${id}`);
  return found;
}

export async function runPipeline({
  client,
  imageBytes,
  sample,
  recognitionModel,
  structuredModel,
  nerModel,
  emit,
}: RunInput): Promise<TriageResult> {
  const t0 = Date.now();

  emit({
    type: "models",
    data: { extractor: nerModel, recognition: recognitionModel, structured: structuredModel },
  });

  let markdown = "";
  let recognitionMs = 0;
  if (recognitionModel) {
    const recOpt = lookup(RECOGNITION_MODELS, recognitionModel);
    emit({ type: "recognition_start", data: { model: recognitionModel } });
    const tRec = Date.now();
    try {
      markdown = await recognize(client, recOpt.id, imageBytes, recOpt.options);
    } catch (err) {
      emit({
        type: "error",
        data: { stage: "recognition", message: `${recognitionModel} failed: ${(err as Error).message}` },
      });
      throw err;
    }
    recognitionMs = Date.now() - tRec;
  }
  emit({
    type: "recognition_done",
    data: {
      markdown: recognitionModel ? markdown : "No recognition model selected.",
      ms: recognitionMs,
    },
  });

  // Structured (Donut variants, etc.)
  let donut = { entities: [] as { label: string; text: string }[], data: undefined as unknown };
  let donutMs = 0;
  if (structuredModel) {
    const strOpt = lookup(STRUCTURED_MODELS, structuredModel);
    emit({ type: "donut_start" });
    const tDon = Date.now();
    try {
      donut = await structuredExtract(client, strOpt.id, imageBytes, strOpt.options);
    } catch (err) {
      emit({
        type: "error",
        data: { stage: "donut", message: `${structuredModel} failed: ${(err as Error).message}` },
      });
    }
    donutMs = Date.now() - tDon;
  }
  emit({ type: "donut_done", data: { entities: donut.entities, rawData: donut.data, ms: donutMs } });

  let fields: { label: string; text: string; score: number }[] = [];
  let glinerMs = 0;
  if (nerModel) {
    const nerOpt = lookup(NER_MODELS, nerModel);
    emit({ type: "gliner_start", data: { labels: sample.labels } });
    const tGli = Date.now();
    try {
      fields = await extractFields(client, nerOpt.id, markdown, sample.labels);
    } catch (err) {
      emit({
        type: "error",
        data: { stage: "gliner", message: `${nerModel} failed: ${(err as Error).message}` },
      });
    }
    glinerMs = Date.now() - tGli;
  }
  emit({ type: "gliner_done", data: { fields, ms: glinerMs } });

  const totalMs = Date.now() - t0;
  emit({ type: "done", data: { totalMs } });

  return {
    sampleId: sample.id,
    recognitionModel,
    markdown,
    donutEntities: donut.entities,
    donutData: donut.data,
    glinerFields: fields,
    timings: { recognitionMs, donutMs, glinerMs, totalMs },
  };
}
