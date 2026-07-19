import { describe, expect, it } from "vitest";
import { toAudioWireFormat, toMediaBytes, toVideoWireFormat } from "../src/index.js";

describe("media conversion", () => {
  it("passes Uint8Array inputs through", async () => {
    const bytes = new Uint8Array([1, 2, 3]);

    expect(await toMediaBytes(bytes)).toBe(bytes);
  });

  it("reads browser Blob inputs", async () => {
    const blob = new Blob([new Uint8Array([4, 5, 6])], { type: "audio/wav" });

    expect(await toMediaBytes(blob)).toEqual(new Uint8Array([4, 5, 6]));
  });

  it("decodes base64 data URLs", async () => {
    expect(await toMediaBytes("data:audio/wav;base64,AQID")).toEqual(new Uint8Array([1, 2, 3]));
  });

  it("preserves audio format and converts sampleRate to the wire key", async () => {
    const result = await toAudioWireFormat({
      data: new Uint8Array([1, 2]),
      format: "pcm",
      sampleRate: 16_000,
    });

    expect(result).toEqual({
      data: new Uint8Array([1, 2]),
      format: "pcm",
      sample_rate: 16_000,
    });
  });

  it("accepts an already wire-shaped audio input", async () => {
    const result = await toAudioWireFormat({
      data: new Uint8Array([1, 2]),
      format: "wav",
      sample_rate: 48_000,
    });

    expect(result.sample_rate).toBe(48_000);
  });

  it("wraps direct video bytes without inventing a format", async () => {
    const result = await toVideoWireFormat(new Uint8Array([9, 8, 7]));

    expect(result).toEqual({ data: new Uint8Array([9, 8, 7]) });
  });
});
