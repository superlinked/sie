/**
 * Audio and video conversion utilities for the SIE TypeScript SDK.
 *
 * Media is transported as raw bytes inside msgpack with optional format
 * metadata. Inputs work in both Node.js (`Uint8Array` / `Buffer`) and browsers
 * (`ArrayBuffer` / `Blob` / `File`).
 */

/** Binary inputs accepted directly by audio and video Item fields. */
export type MediaInput = Uint8Array | ArrayBuffer | Blob | string;

/** User-facing audio input with optional wire metadata. */
export interface AudioInput {
  data: MediaInput;
  format?: string;
  sampleRate?: number;
}

/** User-facing video input with an optional format hint. */
export interface VideoInput {
  data: MediaInput;
  format?: string;
}

/** Audio shape serialized onto the generic Item wire contract. */
export interface AudioWireFormat {
  data: Uint8Array;
  format?: string;
  sample_rate?: number;
}

/** Video shape serialized onto the generic Item wire contract. */
export interface VideoWireFormat {
  data: Uint8Array;
  format?: string;
}

/** Convert bytes, browser binary objects, or base64 strings to bytes. */
export async function toMediaBytes(input: MediaInput): Promise<Uint8Array> {
  if (input instanceof Uint8Array) {
    return input;
  }

  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }

  if (typeof Blob !== "undefined" && input instanceof Blob) {
    return new Uint8Array(await input.arrayBuffer());
  }

  if (typeof input === "string") {
    const dataUrlMatch = input.match(/^data:[^;]+;base64,(.+)$/);
    return base64ToBytes(dataUrlMatch?.[1] ?? input);
  }

  throw new Error(`Unsupported media input type: ${typeof input}`);
}

/** Convert a direct or metadata-wrapped audio input to its msgpack wire shape. */
export async function toAudioWireFormat(
  input: MediaInput | AudioInput | AudioWireFormat,
): Promise<AudioWireFormat> {
  const wrapped = isWrappedMedia(input) ? input : { data: input };
  const result: AudioWireFormat = { data: await toMediaBytes(wrapped.data) };
  if (wrapped.format !== undefined) {
    result.format = wrapped.format;
  }
  const sampleRate =
    "sampleRate" in wrapped
      ? wrapped.sampleRate
      : "sample_rate" in wrapped
        ? wrapped.sample_rate
        : undefined;
  if (sampleRate !== undefined) {
    result.sample_rate = sampleRate;
  }
  return result;
}

/** Convert a direct or metadata-wrapped video input to its msgpack wire shape. */
export async function toVideoWireFormat(
  input: MediaInput | VideoInput | VideoWireFormat,
): Promise<VideoWireFormat> {
  const wrapped = isWrappedMedia(input) ? input : { data: input };
  const result: VideoWireFormat = { data: await toMediaBytes(wrapped.data) };
  if (wrapped.format !== undefined) {
    result.format = wrapped.format;
  }
  return result;
}

function isWrappedMedia(
  input: MediaInput | AudioInput | AudioWireFormat | VideoInput | VideoWireFormat,
): input is AudioInput | AudioWireFormat | VideoInput | VideoWireFormat {
  return typeof input === "object" && input !== null && "data" in input;
}

function base64ToBytes(base64: string): Uint8Array {
  if (typeof atob === "function") {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  return new Uint8Array(Buffer.from(base64, "base64"));
}
