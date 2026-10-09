import { detectImageFormat, type SIEClient } from "@superlinked/sie-sdk";

export async function recognize(
  client: SIEClient,
  model: string,
  imageBytes: Uint8Array,
  options?: Record<string, unknown>,
): Promise<string> {
  const format = detectImageFormat(imageBytes);
  if (format === "unknown") throw new Error("could not detect image format");
  const result = await client.extract(
    model,
    { images: [{ data: imageBytes, format }] },
    { labels: [], adapterOptions: options },
  );
  if (!result.entities || result.entities.length === 0) return "";
  const text = result.entities[0]?.text;
  return typeof text === "string" ? text : "";
}
