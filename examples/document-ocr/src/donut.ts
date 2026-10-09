import { detectImageFormat, type SIEClient } from "@superlinked/sie-sdk";
import type { DonutEntity } from "./types.js";

/** Run any image-input "structured" extractor (Donut variants, etc.). */
export async function structuredExtract(
  client: SIEClient,
  model: string,
  imageBytes: Uint8Array,
  options?: Record<string, unknown>,
): Promise<{ entities: DonutEntity[]; data: unknown }> {
  const format = detectImageFormat(imageBytes);
  if (format === "unknown") throw new Error("could not detect image format");
  const result = await client.extract(
    model,
    { images: [{ data: imageBytes, format }] },
    { labels: [], adapterOptions: options },
  );
  const entities = (result.entities ?? []).map((e) => ({
    label: e.label,
    text: e.text,
  }));
  return { entities, data: result.data };
}
