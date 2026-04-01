/**
 * msgpack tests focused on msgpack-numpy compatibility.
 *
 * These tests verify that we can:
 * 1. Encode/decode TypedArrays in msgpack-numpy format
 * 2. Handle the wire format the SIE server uses
 * 3. Round-trip data correctly
 */

import { describe, expect, it } from "vitest";
import { packMessage, unpackMessage } from "../src/msgpack.js";

describe("msgpack basic types", () => {
  it("should round-trip simple objects", () => {
    const data = { hello: "world", count: 42 };
    const packed = packMessage(data);
    const unpacked = unpackMessage<typeof data>(packed);
    expect(unpacked).toEqual(data);
  });

  it("should handle nested objects", () => {
    const data = {
      model: "bge-m3",
      items: [{ text: "hello" }, { text: "world" }],
      options: { normalize: true },
    };
    const packed = packMessage(data);
    const unpacked = unpackMessage<typeof data>(packed);
    expect(unpacked).toEqual(data);
  });

  it("should handle null and undefined values", () => {
    const data = { a: null, b: undefined };
    const packed = packMessage(data);
    const unpacked = unpackMessage<Record<string, unknown>>(packed);
    // msgpack treats undefined as nil (null)
    expect(unpacked.a).toBeNull();
    expect(unpacked.b).toBeNull();
  });
});

describe("msgpack-numpy Float32Array", () => {
  it("should round-trip Float32Array", () => {
    const original = new Float32Array([1.5, 2.5, 3.5]);
    const packed = packMessage(original);
    const unpacked = unpackMessage<Float32Array>(packed);

    expect(unpacked).toBeInstanceOf(Float32Array);
    expect(unpacked.length).toBe(original.length);
    expect(unpacked[0]).toBeCloseTo(1.5);
    expect(unpacked[1]).toBeCloseTo(2.5);
    expect(unpacked[2]).toBeCloseTo(3.5);
  });

  it("should handle Float32Array with many elements", () => {
    const original = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      original[i] = Math.random();
    }

    const packed = packMessage(original);
    const unpacked = unpackMessage<Float32Array>(packed);

    expect(unpacked).toBeInstanceOf(Float32Array);
    expect(unpacked.length).toBe(1024);
    for (let i = 0; i < 1024; i++) {
      expect(unpacked[i]).toBeCloseTo(original[i] ?? 0);
    }
  });

  it("should handle empty Float32Array", () => {
    const original = new Float32Array(0);
    const packed = packMessage(original);
    const unpacked = unpackMessage<Float32Array>(packed);

    expect(unpacked).toBeInstanceOf(Float32Array);
    expect(unpacked.length).toBe(0);
  });
});

describe("msgpack-numpy Int32Array", () => {
  it("should round-trip Int32Array", () => {
    const original = new Int32Array([1, -2, 3, -4]);
    const packed = packMessage(original);
    const unpacked = unpackMessage<Int32Array>(packed);

    expect(unpacked).toBeInstanceOf(Int32Array);
    expect(unpacked.length).toBe(original.length);
    expect(Array.from(unpacked)).toEqual([1, -2, 3, -4]);
  });

  it("should handle large Int32Array values", () => {
    const original = new Int32Array([2147483647, -2147483648, 0, 12345678]);
    const packed = packMessage(original);
    const unpacked = unpackMessage<Int32Array>(packed);

    expect(Array.from(unpacked)).toEqual([2147483647, -2147483648, 0, 12345678]);
  });
});

describe("msgpack-numpy in nested structures", () => {
  it("should handle TypedArrays in objects (like encode response)", () => {
    // This simulates the server response format
    const data = {
      results: [
        {
          id: "doc-1",
          dense: new Float32Array([0.1, 0.2, 0.3]),
        },
      ],
    };

    const packed = packMessage(data);
    const unpacked = unpackMessage<typeof data>(packed);

    expect(unpacked.results).toHaveLength(1);
    expect(unpacked.results[0]?.id).toBe("doc-1");
    expect(unpacked.results[0]?.dense).toBeInstanceOf(Float32Array);
    expect(unpacked.results[0]?.dense[0]).toBeCloseTo(0.1);
  });

  it("should handle sparse embedding format", () => {
    const data = {
      sparse: {
        indices: new Int32Array([0, 5, 10]),
        values: new Float32Array([0.1, 0.5, 0.9]),
      },
    };

    const packed = packMessage(data);
    const unpacked = unpackMessage<typeof data>(packed);

    expect(unpacked.sparse.indices).toBeInstanceOf(Int32Array);
    expect(unpacked.sparse.values).toBeInstanceOf(Float32Array);
    expect(Array.from(unpacked.sparse.indices)).toEqual([0, 5, 10]);
    expect(unpacked.sparse.values[1]).toBeCloseTo(0.5);
  });

  it("should handle multivector format (array of Float32Arrays)", () => {
    // Multivector: each token has its own embedding
    const data = {
      multivector: [new Float32Array([1, 2]), new Float32Array([3, 4]), new Float32Array([5, 6])],
    };

    const packed = packMessage(data);
    const unpacked = unpackMessage<{ multivector: Float32Array[] }>(packed);

    expect(unpacked.multivector).toHaveLength(3);
    expect(unpacked.multivector[0]).toBeInstanceOf(Float32Array);
    expect(unpacked.multivector[1]).toBeInstanceOf(Float32Array);
    expect(unpacked.multivector[2]).toBeInstanceOf(Float32Array);
  });
});

describe("msgpack wire format efficiency", () => {
  it("should produce compact binary output", () => {
    // A Float32Array should be much smaller in msgpack than JSON
    const floats = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
      floats[i] = Math.random();
    }

    const packed = packMessage(floats);
    const jsonSize = JSON.stringify(Array.from(floats)).length;

    // msgpack with extension should be ~4 bytes per float + small overhead
    // JSON would be ~15+ bytes per float (decimal representation + comma)
    expect(packed.byteLength).toBeLessThan(jsonSize / 2);
  });

  it("should handle mixed content efficiently", () => {
    const data = {
      model: "bge-m3",
      embeddings: new Float32Array(768),
      metadata: { source: "test" },
    };

    const packed = packMessage(data);

    // Should be roughly: string overhead + 768*4 bytes + metadata overhead
    // Much smaller than JSON representation
    expect(packed.byteLength).toBeLessThan(4000);
  });
});

describe("msgpack-numpy format details", () => {
  it("should use extension type 78 for numpy arrays", () => {
    // The extension type 78 (0x4E, 'N') is used by msgpack-numpy
    const arr = new Float32Array([1.0, 2.0]);
    const packed = packMessage(arr);

    // Find extension type marker in packed data
    // Extension format: 0xc7 (ext8) or 0xc8 (ext16) or 0xc9 (ext32) followed by size and type
    // For small arrays: 0xd4-0xd8 (fixext) with type byte
    // We should see 0x4E (78) somewhere as the type
    const bytes = Array.from(packed);
    const hasNumpyType = bytes.includes(78);
    expect(hasNumpyType).toBe(true);
  });

  it("should encode dtype correctly for float32", () => {
    const arr = new Float32Array([1.0]);
    const packed = packMessage(arr);

    // The dtype '<f4' should be in the extension data
    const packedStr = new TextDecoder().decode(packed);
    expect(packedStr).toContain("<f4");
  });

  it("should encode dtype correctly for int32", () => {
    const arr = new Int32Array([1]);
    const packed = packMessage(arr);

    // The dtype '<i4' should be in the extension data
    const packedStr = new TextDecoder().decode(packed);
    expect(packedStr).toContain("<i4");
  });
});
