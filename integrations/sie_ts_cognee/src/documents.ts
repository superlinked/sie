import { createHash } from "node:crypto";
import { basename } from "node:path";
import type { SIEClient } from "@superlinked/sie-sdk";

export interface SourceMetadata {
  sourceId: string;
  filename: string;
  pageNumber: number | null;
  contentSha256: string;
}

export interface SourceDocumentInput {
  text: string;
  sourceId: string;
  filename: string;
  pageNumber?: number;
}

export class SourceDocument {
  readonly text: string;
  readonly metadata: SourceMetadata;

  constructor(input: SourceDocumentInput) {
    if (!input.text?.trim() || !input.sourceId?.trim())
      throw new Error("Source text and sourceId must be nonempty.");
    if (
      !input.filename ||
      basename(input.filename) !== input.filename ||
      /[\\\r\n]/.test(input.filename) ||
      [".", ".."].includes(input.filename)
    ) {
      throw new Error("filename must be a name without a directory path.");
    }
    if (
      input.pageNumber !== undefined &&
      (!Number.isSafeInteger(input.pageNumber) || input.pageNumber < 1)
    ) {
      throw new Error("pageNumber must be a positive integer.");
    }
    this.text = input.text;
    this.metadata = {
      sourceId: input.sourceId,
      filename: input.filename,
      pageNumber: input.pageNumber ?? null,
      contentSha256: createHash("sha256").update(input.text).digest("hex"),
    };
  }

  toDataInput(): { type: "text"; text: string } {
    return { type: "text", text: this.text };
  }
}

export async function ocrPage(
  client: SIEClient,
  options: {
    model: string;
    image: Uint8Array;
    sourceId: string;
    filename: string;
    pageNumber: number;
    maxBytes?: number;
  },
): Promise<SourceDocument> {
  const maxBytes = options.maxBytes ?? 10_000_000;
  if (
    !Number.isSafeInteger(maxBytes) ||
    maxBytes < 1 ||
    !options.image.byteLength ||
    options.image.byteLength > maxBytes
  ) {
    throw new Error("Page image is empty or exceeds maxBytes.");
  }
  const reply = await client.extract(
    options.model,
    { images: [options.image] },
    { waitForCapacity: true },
  );
  if (reply.error || !Array.isArray(reply.entities) || !reply.entities.length)
    throw new Error("OCR returned no usable text entities.");
  const texts = reply.entities.map((entity) => {
    if (!entity || typeof entity.text !== "string")
      throw new Error("OCR returned a malformed text entity.");
    return entity.text;
  });
  return new SourceDocument({
    text: texts.join("\n"),
    sourceId: options.sourceId,
    filename: options.filename,
    pageNumber: options.pageNumber,
  });
}
