/**
 * Image handling tests
 */

import { describe, expect, it } from "vitest";
import { detectImageFormat, toImageBytes, toImageWireFormat } from "../src/images.js";

describe("toImageBytes", () => {
  it("passes through Uint8Array as-is", async () => {
    const input = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);
    const result = await toImageBytes(input);

    expect(result).toBe(input);
  });

  it("converts ArrayBuffer to Uint8Array", async () => {
    const buffer = new ArrayBuffer(4);
    new Uint8Array(buffer).set([0x89, 0x50, 0x4e, 0x47]);

    const result = await toImageBytes(buffer);

    expect(result).toBeInstanceOf(Uint8Array);
    expect(result[0]).toBe(0x89);
    expect(result[1]).toBe(0x50);
  });

  it("decodes base64 string", async () => {
    // "Hello" in base64
    const base64 = "SGVsbG8=";
    const result = await toImageBytes(base64);

    expect(result).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(result)).toBe("Hello");
  });

  it("decodes data URL", async () => {
    // "Hello" as data URL
    const dataUrl = "data:image/jpeg;base64,SGVsbG8=";
    const result = await toImageBytes(dataUrl);

    expect(result).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(result)).toBe("Hello");
  });

  it("handles PNG data URL correctly", async () => {
    // Small valid data
    const dataUrl = "data:image/png;base64,dGVzdA==";
    const result = await toImageBytes(dataUrl);

    expect(result).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(result)).toBe("test");
  });

  it("throws for unsupported input type", async () => {
    await expect(toImageBytes(123 as unknown as Uint8Array)).rejects.toThrow(
      "Unsupported image input type",
    );
  });
});

describe("toImageWireFormat", () => {
  it("wraps bytes in wire format with default jpeg", async () => {
    const input = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);
    const result = await toImageWireFormat(input);

    expect(result.data).toBe(input);
    expect(result.format).toBe("jpeg");
  });

  it("allows specifying format", async () => {
    const input = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const result = await toImageWireFormat(input, "png");

    expect(result.data).toBe(input);
    expect(result.format).toBe("png");
  });

  it("supports webp format", async () => {
    const input = new Uint8Array([0x52, 0x49, 0x46, 0x46]);
    const result = await toImageWireFormat(input, "webp");

    expect(result.format).toBe("webp");
  });
});

describe("detectImageFormat", () => {
  it("detects JPEG format", () => {
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
    expect(detectImageFormat(jpeg)).toBe("jpeg");
  });

  it("detects PNG format", () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]);
    expect(detectImageFormat(png)).toBe("png");
  });

  it("detects WebP format", () => {
    // RIFF....WEBP
    const webp = new Uint8Array([
      0x52,
      0x49,
      0x46,
      0x46, // RIFF
      0x00,
      0x00,
      0x00,
      0x00, // file size (placeholder)
      0x57,
      0x45,
      0x42,
      0x50, // WEBP
    ]);
    expect(detectImageFormat(webp)).toBe("webp");
  });

  it("returns unknown for unrecognized format", () => {
    const unknown = new Uint8Array([0x00, 0x01, 0x02, 0x03]);
    expect(detectImageFormat(unknown)).toBe("unknown");
  });

  it("returns unknown for too short input", () => {
    const short = new Uint8Array([0xff, 0xd8]);
    expect(detectImageFormat(short)).toBe("unknown");
  });

  it("returns unknown for empty input", () => {
    const empty = new Uint8Array([]);
    expect(detectImageFormat(empty)).toBe("unknown");
  });
});
