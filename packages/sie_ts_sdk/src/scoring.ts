/**
 * Client-side scoring functions (MaxSim for ColBERT)
 */

/**
 * Compute MaxSim score between a query and a document (both multivector)
 * MaxSim: for each query token, find max similarity to any document token, then sum
 */
export function maxsim(query: Float32Array[], document: Float32Array[]): number {
  if (query.length === 0 || document.length === 0) {
    return 0;
  }

  let totalScore = 0;

  for (const queryToken of query) {
    let maxSim = Number.NEGATIVE_INFINITY;

    for (const docToken of document) {
      // Dot product (assumes vectors are normalized)
      let sim = 0;
      for (let i = 0; i < queryToken.length; i++) {
        sim += (queryToken[i] ?? 0) * (docToken[i] ?? 0);
      }
      if (sim > maxSim) {
        maxSim = sim;
      }
    }

    totalScore += maxSim;
  }

  return totalScore;
}

/**
 * Compute MaxSim scores between a query and multiple documents
 */
export function maxsimDocuments(query: Float32Array[], documents: Float32Array[][]): number[] {
  return documents.map((doc) => maxsim(query, doc));
}

/**
 * Compute MaxSim scores for a batch of queries against a batch of documents
 * Returns a flattened array of scores: [q0d0, q0d1, ..., q1d0, q1d1, ...]
 */
export function maxsimBatch(queries: Float32Array[][], documents: Float32Array[][]): Float32Array {
  const scores = new Float32Array(queries.length * documents.length);
  let idx = 0;

  for (const query of queries) {
    for (const doc of documents) {
      scores[idx++] = maxsim(query, doc);
    }
  }

  return scores;
}
