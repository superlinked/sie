/**
 * Image handling utilities for the SIE TypeScript SDK.
 *
 * Per design.md Section 4.3, images are serialized as bytes for transport.
 * This module handles conversion from various input formats to Uint8Array.
 *
 * Supported input formats:
 * - Uint8Array (raw bytes)
 * - ArrayBuffer / Buffer (Node.js)
 * - Blob / File (browser)
 * - string (base64 or data URL)
 *
 * @example
 * ```typescript
 * import { toImageBytes } from "@superlinked/sie-sdk";
 *
 * // From file input (browser)
 * const file = document.querySelector('input[type="file"]').files[0];
 * const bytes = await toImageBytes(file);
 *
 * // From base64 string
 * const bytes = await toImageBytes(base64String);
 *
 * // From Uint8Array (passthrough)
 * const bytes = await toImageBytes(existingBytes);
 * ```
 */

/**
 * Type for all supported image input formats.
 */
export type ImageInput = Uint8Array | ArrayBuffer | Blob | string;

/**
 * Wire format for images sent to the server.
 * Per design.md Section 4.3.
 */
export interface ImageWireFormat {
  data: Uint8Array;
  format: "jpeg" | "png" | "webp";
}

/**
 * Convert various image input types to Uint8Array.
 *
 * Accepts:
 * - Uint8Array: passed through as-is
 * - ArrayBuffer / Buffer: wrapped in Uint8Array
 * - Blob / File: read as ArrayBuffer then wrapped
 * - string: decoded from base64 or data URL
 *
 * @param input - Image data in any supported format
 * @returns Image bytes as Uint8Array
 *
 * @example
 * ```typescript
 * // From base64 string
 * const bytes = await toImageBytes(base64String);
 *
 * // From file (browser)
 * const bytes = await toImageBytes(file);
 * ```
 */
export async function toImageBytes(input: ImageInput): Promise<Uint8Array> {
  // Already Uint8Array
  if (input instanceof Uint8Array) {
    return input;
  }

  // ArrayBuffer (or Buffer in Node.js)
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }

  // Blob or File (browser)
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    const buffer = await input.arrayBuffer();
    return new Uint8Array(buffer);
  }

  // Base64 string or data URL
  if (typeof input === "string") {
    // Check if it's a data URL
    const dataUrlMatch = input.match(/^data:[^;]+;base64,(.+)$/);
    if (dataUrlMatch?.[1]) {
      return base64ToBytes(dataUrlMatch[1]);
    }

    // Assume it's raw base64
    return base64ToBytes(input);
  }

  throw new Error(`Unsupported image input type: ${typeof input}`);
}

/**
 * Convert base64 string to Uint8Array.
 */
function base64ToBytes(base64: string): Uint8Array {
  // Use atob in browser, Buffer in Node.js
  if (typeof atob === "function") {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  // Node.js environment
  return new Uint8Array(Buffer.from(base64, "base64"));
}

/**
 * Convert image bytes to wire format for transport.
 *
 * Per design.md Section 4.3, images are sent as:
 * `{ data: <bytes>, format: "jpeg" | "png" | "webp" }`
 *
 * @param input - Image data in any supported format
 * @param format - Image format (defaults to "jpeg")
 * @returns Image in wire format
 */
export async function toImageWireFormat(
  input: ImageInput,
  format: "jpeg" | "png" | "webp" = "jpeg",
): Promise<ImageWireFormat> {
  const data = await toImageBytes(input);
  return { data, format };
}

/**
 * Detect image format from bytes (magic number check).
 *
 * @param bytes - Image bytes
 * @returns Detected format or "unknown"
 */
export function detectImageFormat(bytes: Uint8Array): "jpeg" | "png" | "webp" | "unknown" {
  if (bytes.length < 4) {
    return "unknown";
  }

  // JPEG: starts with FF D8 FF
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "jpeg";
  }

  // PNG: starts with 89 50 4E 47 (0x89 'PNG')
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return "png";
  }

  // WebP: starts with RIFF....WEBP
  if (
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes.length >= 12 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return "webp";
  }

  return "unknown";
}
