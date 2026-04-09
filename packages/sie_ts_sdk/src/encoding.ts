/**
 * Helpers for converting SIE encode results to plain JavaScript types.
 *
 * These functions bridge SDK result types (Float32Array, typed dicts) to the
 * plain `number[]` / object formats that framework integrations and vector
 * databases expect.
 *
 * @example
 * ```typescript
 * import { SIEClient, denseEmbedding, sparseEmbedding } from "@superlinked/sie-sdk";
 *
 * const client = new SIEClient("http://localhost:8080");
 * const result = await client.encode("BAAI/bge-m3", { text: "hello" });
 * const vec = denseEmbedding(result);   // number[]
 * const sp  = sparseEmbedding(result);  // { indices: number[], values: number[] }
 * ```
 */

import type { EncodeResult, SparseResult } from "./types.js";
import { toNumberArray } from "./types.js";

/** Sparse vector in `{ indices, values }` format. */
export interface SparseVector {
  indices: number[];
  values: number[];
}

/**
 * Extract the dense embedding from an encode result as `number[]`.
 *
 * @param result - An {@link EncodeResult} from `client.encode()`.
 * @param strict - If `true` (default), throw when dense is missing. If `false`, return `[]`.
 * @returns The dense vector as a plain JavaScript number array.
 */
export function denseEmbedding(result: EncodeResult, strict = true): number[] {
  const dense = result.dense;
  if (!dense) {
    if (strict) {
      throw new Error("Encode result missing dense embedding");
    }
    return [];
  }
  return toNumberArray(dense);
}

/**
 * Extract the sparse embedding from an encode result.
 *
 * @param result - An {@link EncodeResult} from `client.encode()`.
 * @returns Object with `indices: number[]` and `values: number[]`. Empty arrays if sparse is absent.
 */
export function sparseEmbedding(result: EncodeResult): SparseVector {
  const sparse = result.sparse;
  if (!sparse) {
    return { indices: [], values: [] };
  }
  return {
    indices: toNumberArray(sparse.indices),
    values: toNumberArray(sparse.values),
  };
}

/**
 * Extract the sparse embedding as a `Map<number, number>` (token index → weight).
 *
 * Useful for ChromaDB which expects sparse embeddings in this format.
 *
 * @param result - An {@link EncodeResult} from `client.encode()`.
 * @returns Map from integer token indices to float weights. Empty map if sparse is absent.
 */
export function sparseEmbeddingMap(result: EncodeResult): Map<number, number> {
  const sparse = result.sparse;
  if (!sparse) {
    return new Map();
  }
  const indices = toNumberArray(sparse.indices);
  const values = toNumberArray(sparse.values);
  const map = new Map<number, number>();
  for (let i = 0; i < indices.length; i++) {
    map.set(indices[i]!, values[i]!);
  }
  return map;
}

/**
 * Convert a raw sparse sub-object to plain JavaScript arrays.
 *
 * Unlike {@link sparseEmbedding}, this takes the sparse value itself
 * (already extracted from the result) — useful inside named-vector
 * loops where you've already pulled `result.sparse`.
 *
 * @param sparse - A {@link SparseResult} with `indices` and `values` typed arrays.
 * @returns Object with `indices: number[]` and `values: number[]`.
 */
export function normalizeSparseVector(sparse: SparseResult): SparseVector {
  return {
    indices: toNumberArray(sparse.indices),
    values: toNumberArray(sparse.values),
  };
}

/**
 * Convert a multivector (ColBERT) result to `number[][]`.
 *
 * SIE returns multivectors as `Float32Array[]`. Vector databases expect
 * nested plain JavaScript arrays.
 *
 * @param raw - Array of Float32Array token vectors.
 * @returns Nested array of number vectors.
 */
export function multivectorEmbedding(raw: Float32Array[]): number[][] {
  return raw.map((v) => toNumberArray(v));
}
