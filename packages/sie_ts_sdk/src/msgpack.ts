/**
 * MessagePack serialization with msgpack-numpy compatibility.
 *
 * The SIE server uses Python's msgpack-numpy library which serializes numpy arrays
 * using extension type 78 ('N'). This module provides compatible encoding/decoding.
 *
 * Wire format for numpy arrays (extension type 78):
 * - dtype string (e.g., '<f4' for float32, '<i4' for int32) terminated by '|'
 * - shape as comma-separated dimensions terminated by '|'
 * - raw array data in little-endian format
 */

import { ExtensionCodec, decode, encode } from "@msgpack/msgpack";

// msgpack-numpy extension type code (ord('N') = 78)
const EXT_TYPE_NUMPY = 78;

/**
 * Parse numpy dtype string to get byte size and TypedArray constructor
 */
function parseDtype(dtype: string): {
  size: number;
  construct: (
    buffer: ArrayBuffer,
  ) => Float32Array | Int32Array | Float64Array | Int16Array | Int8Array | Uint8Array;
} {
  // Numpy dtypes: '<f4' (float32), '<f8' (float64), '<i4' (int32), '<i2' (int16),
  // '<i1' (int8), '|u1' (uint8), '<f2' (float16)
  // Note: '<' means little-endian, '|' means not applicable (single byte)
  const typeChar = dtype.slice(-2, -1); // 'f', 'i', 'u', etc.
  const sizeChar = dtype.slice(-1); // '4', '8', '2', '1'
  const size = Number.parseInt(sizeChar, 10);

  switch (`${typeChar}${size}`) {
    case "f4":
      return { size: 4, construct: (buf) => new Float32Array(buf) };
    case "f8":
      return { size: 8, construct: (buf) => new Float64Array(buf) };
    case "f2":
      // float16 - no native JS type, decode to Float32Array with conversion
      return {
        size: 2,
        construct: (buf) => {
          const float16 = new Uint16Array(buf);
          const float32 = new Float32Array(float16.length);
          for (let i = 0; i < float16.length; i++) {
            float32[i] = float16ToFloat32(float16[i] ?? 0);
          }
          return float32;
        },
      };
    case "i4":
      return { size: 4, construct: (buf) => new Int32Array(buf) };
    case "i2":
      return { size: 2, construct: (buf) => new Int16Array(buf) };
    case "i1":
      return { size: 1, construct: (buf) => new Int8Array(buf) };
    case "u1":
      return { size: 1, construct: (buf) => new Uint8Array(buf) };
    default:
      throw new Error(`Unsupported numpy dtype: ${dtype}`);
  }
}

/**
 * Convert a float16 (IEEE 754 half-precision) value to float32.
 */
function float16ToFloat32(h: number): number {
  const sign = (h >>> 15) & 0x1;
  const exp = (h >>> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) {
      // Zero
      return sign ? -0 : 0;
    }
    // Subnormal
    const f = frac / 1024;
    return (sign ? -1 : 1) * f * 2 ** -14;
  }
  if (exp === 31) {
    // Infinity or NaN
    return frac === 0 ? (sign ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY) : Number.NaN;
  }
  // Normal
  return (sign ? -1 : 1) * (1 + frac / 1024) * 2 ** (exp - 15);
}

/**
 * Decode msgpack-numpy extension data to TypedArray
 */
function decodeNumpyArray(
  data: Uint8Array,
): Float32Array | Int32Array | Float64Array | Int16Array | Int8Array | Uint8Array {
  // Find first '|' separator (between dtype and shape)
  let dtypeEnd = 0;
  while (dtypeEnd < data.length && data[dtypeEnd] !== 0x7c) {
    // '|' = 0x7c
    dtypeEnd++;
  }

  const dtypeBytes = data.slice(0, dtypeEnd);
  const dtype = new TextDecoder().decode(dtypeBytes);

  // Find second '|' separator (between shape and data)
  let shapeEnd = dtypeEnd + 1;
  while (shapeEnd < data.length && data[shapeEnd] !== 0x7c) {
    shapeEnd++;
  }

  const shapeBytes = data.slice(dtypeEnd + 1, shapeEnd);
  const shapeStr = new TextDecoder().decode(shapeBytes);
  const shape = shapeStr.length > 0 ? shapeStr.split(",").map((s) => Number.parseInt(s, 10)) : [];

  // Remaining bytes are the array data
  const arrayData = data.slice(shapeEnd + 1);

  // Parse dtype and create TypedArray
  const { size, construct } = parseDtype(dtype);

  // Calculate total elements from shape
  const totalElements =
    shape.length > 0 ? shape.reduce((a, b) => a * b, 1) : arrayData.length / size;

  // Ensure proper alignment by copying to a new buffer
  const buffer = new ArrayBuffer(totalElements * size);
  new Uint8Array(buffer).set(arrayData.slice(0, totalElements * size));

  return construct(buffer);
}

/**
 * Encode TypedArray to msgpack-numpy extension format
 */
function encodeNumpyArray(arr: Float32Array | Int32Array): Uint8Array {
  let dtype: string;
  if (arr instanceof Float32Array) {
    dtype = "<f4";
  } else if (arr instanceof Int32Array) {
    dtype = "<i4";
  } else {
    throw new Error("Unsupported TypedArray type");
  }

  // Build wire format: dtype + '|' + shape + '|' + data
  const dtypeBytes = new TextEncoder().encode(dtype);
  const shapeBytes = new TextEncoder().encode(arr.length.toString());
  const separator = new Uint8Array([0x7c]); // '|'
  const dataBytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);

  // Concatenate all parts
  const result = new Uint8Array(dtypeBytes.length + 1 + shapeBytes.length + 1 + dataBytes.length);
  let offset = 0;
  result.set(dtypeBytes, offset);
  offset += dtypeBytes.length;
  result.set(separator, offset);
  offset += 1;
  result.set(shapeBytes, offset);
  offset += shapeBytes.length;
  result.set(separator, offset);
  offset += 1;
  result.set(dataBytes, offset);

  return result;
}

/**
 * Create extension codec compatible with msgpack-numpy
 */
function createExtensionCodec(): ExtensionCodec {
  const codec = new ExtensionCodec();

  // Register numpy array decoder (extension type 78)
  codec.register({
    type: EXT_TYPE_NUMPY,
    encode: (value: unknown): Uint8Array | null => {
      if (value instanceof Float32Array || value instanceof Int32Array) {
        return encodeNumpyArray(value);
      }
      return null;
    },
    decode: (
      data: Uint8Array,
    ): Float32Array | Int32Array | Float64Array | Int16Array | Int8Array | Uint8Array => {
      return decodeNumpyArray(data);
    },
  });

  return codec;
}

const extensionCodec = createExtensionCodec();

/**
 * Pack a message to MessagePack format (msgpack-numpy compatible)
 */
export function packMessage(data: unknown): Uint8Array {
  return encode(data, { extensionCodec });
}

/**
 * Check if an object is a msgpack-numpy array representation
 */
function isNumpyArrayMap(
  obj: unknown,
): obj is { nd: boolean; type: string; shape: number[]; data: Uint8Array } {
  if (typeof obj !== "object" || obj === null) {
    return false;
  }
  const map = obj as Record<string, unknown>;
  return (
    map.nd === true &&
    typeof map.type === "string" &&
    Array.isArray(map.shape) &&
    map.data instanceof Uint8Array
  );
}

/**
 * Convert a msgpack-numpy array map to a typed array or array of typed arrays
 *
 * For 1D arrays: returns a single typed array
 * For 2D arrays: returns an array of typed arrays (one per row)
 */
function convertNumpyArrayMap(map: {
  nd: boolean;
  type: string;
  shape: number[];
  data: Uint8Array;
}):
  | Float32Array
  | Int32Array
  | Float64Array
  | Int16Array
  | Int8Array
  | Uint8Array
  | Float32Array[]
  | Int32Array[]
  | Int8Array[] {
  const dtype = map.type;
  const arrayData = map.data;

  // Parse dtype to determine array type
  const { size, construct } = parseDtype(dtype);

  // Handle 2D arrays (e.g., multivector with shape [num_tokens, dim])
  if (map.shape.length === 2 && map.shape[0] !== undefined && map.shape[1] !== undefined) {
    const numRows = map.shape[0];
    const numCols = map.shape[1];
    const result: (
      | Float32Array
      | Int32Array
      | Float64Array
      | Int16Array
      | Int8Array
      | Uint8Array
    )[] = [];

    for (let row = 0; row < numRows; row++) {
      const offset = row * numCols * size;
      const buffer = new ArrayBuffer(numCols * size);
      new Uint8Array(buffer).set(arrayData.slice(offset, offset + numCols * size));
      result.push(construct(buffer));
    }

    return result as Float32Array[] | Int32Array[] | Int8Array[];
  }

  // 1D array (or scalar): return single typed array
  const totalElements =
    map.shape.length > 0 ? map.shape.reduce((a, b) => a * b, 1) : arrayData.length / size;

  // Ensure proper alignment by copying to a new buffer
  const buffer = new ArrayBuffer(totalElements * size);
  new Uint8Array(buffer).set(arrayData.slice(0, totalElements * size));

  return construct(buffer);
}

/**
 * Recursively convert msgpack-numpy array representations to typed arrays
 */
function convertNumpyArrays(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Check if this is a numpy array map
  if (isNumpyArrayMap(obj)) {
    return convertNumpyArrayMap(obj);
  }

  // Recursively process arrays
  if (Array.isArray(obj)) {
    return obj.map((item) => convertNumpyArrays(item));
  }

  // Don't process typed arrays - they're already converted
  if (ArrayBuffer.isView(obj)) {
    return obj;
  }

  // Recursively process objects
  if (typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = convertNumpyArrays(value);
    }
    return result;
  }

  return obj;
}

/**
 * Unpack a MessagePack message (msgpack-numpy compatible)
 *
 * Note: msgpack-numpy uses byte string keys (b'nd', b'type', b'shape', b'data') for numpy
 * array metadata. In JavaScript these become Uint8Array which need to be decoded as text.
 * After decoding, we recursively convert numpy array maps to typed arrays.
 */
export function unpackMessage<T = unknown>(data: Uint8Array): T {
  const decoded = decode(data, {
    extensionCodec,
    // Convert byte string keys (Uint8Array) to text strings
    mapKeyConverter: (key: unknown): string | number => {
      if (typeof key === "string" || typeof key === "number") {
        return key;
      }
      // msgpack-numpy uses byte string keys for numpy array metadata
      if (key instanceof Uint8Array) {
        return new TextDecoder().decode(key);
      }
      // Fallback: Convert arrays/objects to JSON string keys
      return JSON.stringify(key);
    },
  });

  // Convert any numpy array maps to typed arrays
  return convertNumpyArrays(decoded) as T;
}
