/**
 * Type tests focused on real user workflows.
 *
 * These tests verify that users can:
 * 1. Create Item objects easily for common use cases
 * 2. Work with embedding results (Float32Array, sparse, multivector)
 * 3. Convert between TypedArrays and regular arrays
 * 4. Track items through the pipeline with IDs
 * 5. Work with score and extract results
 */

import { describe, expect, it } from "vitest";
import type {
  EncodeResult,
  Entity,
  ExtractResult,
  Item,
  ModelInfo,
  ScoreEntry,
  ScoreResult,
  SparseResult,
} from "../src/types.js";
import { toFloat32Array, toNumberArray } from "../src/types.js";

describe("Item creation - common user patterns", () => {
  it("creates simple text items", () => {
    // Most common case: just encoding text
    const item: Item = { text: "Hello world" };

    expect(item.text).toBe("Hello world");
    expect(item.id).toBeUndefined();
  });

  it("creates items with IDs for tracking", () => {
    // User scenario: "I need to match results back to my documents"
    const items: Item[] = [
      { id: "doc-1", text: "First document" },
      { id: "doc-2", text: "Second document" },
      { id: "doc-3", text: "Third document" },
    ];

    expect(items.map((i) => i.id)).toEqual(["doc-1", "doc-2", "doc-3"]);
  });

  it("creates items with metadata", () => {
    // User scenario: "I want to attach extra info to track through the pipeline"
    const item: Item = {
      id: "product-123",
      text: "Blue running shoes",
      metadata: {
        category: "footwear",
        price: 99.99,
        inStock: true,
      },
    };

    expect(item.metadata?.category).toBe("footwear");
    expect(item.metadata?.price).toBe(99.99);
  });

  it("creates multimodal items with images", () => {
    // User scenario: "I'm using ColPali for image+text retrieval"
    const imageBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]); // JPEG magic bytes

    const item: Item = {
      text: "A photo of a cat",
      images: [imageBytes],
    };

    expect(item.images).toHaveLength(1);
    expect(item.images?.[0]).toBeInstanceOf(Uint8Array);
  });

  it("creates items with pre-encoded multivectors for client-side scoring", () => {
    // User scenario: "I have cached embeddings, don't re-encode"
    const cachedEmbedding = [new Float32Array([0.1, 0.2, 0.3]), new Float32Array([0.4, 0.5, 0.6])];

    const item: Item = {
      multivector: cachedEmbedding,
    };

    expect(item.multivector).toHaveLength(2);
    expect(item.text).toBeUndefined(); // No text, using cached embedding
  });
});

describe("EncodeResult - working with embeddings", () => {
  it("provides dense embeddings as Float32Array", () => {
    // User scenario: "I need to store embeddings in a vector database"
    const result: EncodeResult = {
      dense: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    };

    expect(result.dense).toBeInstanceOf(Float32Array);
    expect(result.dense?.length).toBe(4);

    // User can iterate over values
    const values: number[] = [];
    for (const v of result.dense ?? []) {
      values.push(v);
    }
    expect(values).toHaveLength(4);
  });

  it("echoes back item ID for tracking", () => {
    // User scenario: "Match results to my original documents"
    const result: EncodeResult = {
      id: "doc-42",
      dense: new Float32Array([0.1, 0.2]),
    };

    expect(result.id).toBe("doc-42");
  });

  it("provides sparse embeddings for SPLADE-type models", () => {
    // User scenario: "I'm doing sparse retrieval with BM25 + SPLADE"
    const result: EncodeResult = {
      sparse: {
        indices: new Int32Array([10, 25, 100, 512]),
        values: new Float32Array([0.5, 0.8, 0.3, 0.9]),
      },
    };

    const sparse = result.sparse as SparseResult;
    expect(sparse.indices).toBeInstanceOf(Int32Array);
    expect(sparse.values).toBeInstanceOf(Float32Array);

    // User can build sparse vector for retrieval
    const sparseVector = new Map<number, number>();
    for (let i = 0; i < sparse.indices.length; i++) {
      const idx = sparse.indices[i];
      const val = sparse.values[i];
      if (idx !== undefined && val !== undefined) {
        sparseVector.set(idx, val);
      }
    }

    expect(sparseVector.get(100)).toBeCloseTo(0.3);
    expect(sparseVector.get(512)).toBeCloseTo(0.9);
  });

  it("provides multivector embeddings for ColBERT", () => {
    // User scenario: "I'm using late interaction for better quality"
    const result: EncodeResult = {
      multivector: [
        new Float32Array([0.1, 0.2, 0.3]), // token 1
        new Float32Array([0.4, 0.5, 0.6]), // token 2
        new Float32Array([0.7, 0.8, 0.9]), // token 3
      ],
    };

    expect(result.multivector).toHaveLength(3);
    expect(result.multivector?.[0]?.length).toBe(3);

    // Each token has its own embedding
    for (const tokenEmb of result.multivector ?? []) {
      expect(tokenEmb).toBeInstanceOf(Float32Array);
    }
  });

  it("provides timing info for performance analysis", () => {
    // User scenario: "I'm profiling my pipeline"
    const result: EncodeResult = {
      dense: new Float32Array([0.1]),
      timing: {
        totalMs: 15.5,
        queueMs: 1.2,
        tokenizationMs: 2.3,
        inferenceMs: 12.0,
      },
    };

    expect(result.timing?.totalMs).toBeCloseTo(15.5);
    expect(result.timing?.inferenceMs).toBeCloseTo(12.0);
  });
});

describe("Float32Array / number[] conversion", () => {
  it("converts Float32Array to number[] for JSON serialization", () => {
    // User scenario: "I need to send embeddings over an API that expects JSON"
    const embedding = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);

    const jsonSafe = toNumberArray(embedding);

    expect(Array.isArray(jsonSafe)).toBe(true);
    expect(jsonSafe).toHaveLength(5);
    expect(JSON.stringify(jsonSafe)).toBe(
      "[0.10000000149011612,0.20000000298023224,0.30000001192092896,0.4000000059604645,0.5]",
    );
  });

  it("converts number[] back to Float32Array", () => {
    // User scenario: "I loaded embeddings from JSON, need to use with SDK"
    const fromJson = [0.1, 0.2, 0.3, 0.4, 0.5];

    const embedding = toFloat32Array(fromJson);

    expect(embedding).toBeInstanceOf(Float32Array);
    expect(embedding.length).toBe(5);
  });

  it("round-trips through JSON correctly", () => {
    // User scenario: "Store in Redis, retrieve, use for comparison"
    const original = new Float32Array([0.123, 0.456, 0.789]);

    // Store as JSON
    const json = JSON.stringify(toNumberArray(original));

    // Retrieve and convert back
    const restored = toFloat32Array(JSON.parse(json) as number[]);

    // Values should be close (Float32 precision)
    for (let i = 0; i < original.length; i++) {
      const origVal = original[i] ?? 0;
      const restoredVal = restored[i] ?? 0;
      expect(restoredVal).toBeCloseTo(origVal, 5);
    }
  });
});

describe("ScoreResult - reranking results", () => {
  it("provides sorted scores for reranking", () => {
    // User scenario: "Get the top 3 most relevant documents"
    const result: ScoreResult = {
      scores: [
        { itemId: "doc-3", score: 0.95, rank: 0 },
        { itemId: "doc-1", score: 0.82, rank: 1 },
        { itemId: "doc-2", score: 0.65, rank: 2 },
      ],
    };

    expect(result.scores[0]?.itemId).toBe("doc-3"); // Most relevant
    expect(result.scores[0]?.rank).toBe(0);

    // User can easily get top K
    const top2 = result.scores.slice(0, 2);
    expect(top2.map((s) => s.itemId)).toEqual(["doc-3", "doc-1"]);
  });

  it("echoes back query ID for tracking", () => {
    // User scenario: "Match results to my search queries"
    const result: ScoreResult = {
      queryId: "search-42",
      scores: [{ itemId: "doc-1", score: 0.9, rank: 0 }],
    };

    expect(result.queryId).toBe("search-42");
  });

  it("provides score entries with all needed info", () => {
    const entry: ScoreEntry = {
      itemId: "product-123",
      score: 0.87,
      rank: 5,
    };

    expect(entry.itemId).toBe("product-123");
    expect(entry.score).toBeCloseTo(0.87);
    expect(entry.rank).toBe(5);
  });
});

describe("ExtractResult - NER results", () => {
  it("provides extracted entities with positions", () => {
    // User scenario: "I want to highlight entities in my UI"
    const result: ExtractResult = {
      id: "doc-1",
      entities: [
        { text: "John Smith", label: "person", score: 0.95, start: 0, end: 10 },
        { text: "Acme Corp", label: "organization", score: 0.88, start: 20, end: 29 },
      ],
    };

    expect(result.entities).toHaveLength(2);

    // User can use positions for highlighting
    const text = "John Smith works at Acme Corp as a developer.";
    for (const entity of result.entities) {
      const extracted = text.slice(entity.start, entity.end);
      expect(extracted).toBe(entity.text);
    }
  });

  it("groups entities by label", () => {
    // User scenario: "Show me all the people mentioned"
    const result: ExtractResult = {
      entities: [
        { text: "John", label: "person", score: 0.9, start: 0, end: 4 },
        { text: "Apple", label: "organization", score: 0.85, start: 10, end: 15 },
        { text: "Jane", label: "person", score: 0.88, start: 20, end: 24 },
      ],
    };

    // Group by label
    const byLabel = new Map<string, Entity[]>();
    for (const entity of result.entities) {
      const list = byLabel.get(entity.label) ?? [];
      list.push(entity);
      byLabel.set(entity.label, list);
    }

    expect(byLabel.get("person")?.map((e) => e.text)).toEqual(["John", "Jane"]);
    expect(byLabel.get("organization")?.map((e) => e.text)).toEqual(["Apple"]);
  });

  it("filters by confidence threshold", () => {
    // User scenario: "Only show high-confidence entities"
    const result: ExtractResult = {
      entities: [
        { text: "John", label: "person", score: 0.95, start: 0, end: 4 },
        { text: "maybe-org", label: "organization", score: 0.45, start: 10, end: 19 },
        { text: "Jane", label: "person", score: 0.88, start: 25, end: 29 },
      ],
    };

    const highConfidence = result.entities.filter((e) => e.score >= 0.8);
    expect(highConfidence.map((e) => e.text)).toEqual(["John", "Jane"]);
  });
});

describe("ModelInfo - model discovery", () => {
  it("provides model capabilities for UI", () => {
    // User scenario: "Show available models in a dropdown"
    const models: ModelInfo[] = [
      {
        name: "bge-m3",
        loaded: true,
        inputs: ["text"],
        outputs: ["dense", "sparse", "multivector"],
        dims: { dense: 1024, multivector: 1024 },
        maxSequenceLength: 8192,
      },
      {
        name: "colpali-v1.3",
        loaded: false,
        inputs: ["text", "image"],
        outputs: ["multivector"],
        dims: { multivector: 128 },
        maxSequenceLength: 2048,
      },
    ];

    // User can filter for specific capabilities
    const textOnlyModels = models.filter(
      (m) => m.inputs.includes("text") && !m.inputs.includes("image"),
    );
    expect(textOnlyModels.map((m) => m.name)).toEqual(["bge-m3"]);

    const multimodalModels = models.filter((m) => m.inputs.includes("image"));
    expect(multimodalModels.map((m) => m.name)).toEqual(["colpali-v1.3"]);

    // User can check if model is loaded (warm)
    const loadedModels = models.filter((m) => m.loaded);
    expect(loadedModels.map((m) => m.name)).toEqual(["bge-m3"]);
  });

  it("helps validate user requests", () => {
    // User scenario: "Check if model supports sparse before requesting"
    const model: ModelInfo = {
      name: "bge-m3",
      loaded: true,
      inputs: ["text"],
      outputs: ["dense", "sparse", "multivector"],
    };

    function canRequestSparse(m: ModelInfo): boolean {
      return m.outputs.includes("sparse");
    }

    function canProcessImage(m: ModelInfo): boolean {
      return m.inputs.includes("image");
    }

    expect(canRequestSparse(model)).toBe(true);
    expect(canProcessImage(model)).toBe(false);
  });
});

describe("Batch processing patterns", () => {
  it("maintains item order through encode results", () => {
    // User scenario: "I send 100 docs, I need results in same order"
    const items: Item[] = [
      { id: "a", text: "First" },
      { id: "b", text: "Second" },
      { id: "c", text: "Third" },
    ];

    // Simulated results (in real usage, from SDK)
    const results: EncodeResult[] = [
      { id: "a", dense: new Float32Array([0.1]) },
      { id: "b", dense: new Float32Array([0.2]) },
      { id: "c", dense: new Float32Array([0.3]) },
    ];

    // IDs should match original order
    for (let i = 0; i < items.length; i++) {
      expect(results[i]?.id).toBe(items[i]?.id);
    }
  });

  it("supports building a document store from results", () => {
    // User scenario: "Index documents for later retrieval"
    // Original documents would be stored elsewhere, we just have IDs
    const _documentIds = ["doc-1", "doc-2"];

    // Simulated encode results
    const results: EncodeResult[] = [
      { id: "doc-1", dense: new Float32Array([0.1, 0.2]) },
      { id: "doc-2", dense: new Float32Array([0.3, 0.4]) },
    ];

    // Build index by ID
    const embeddingIndex = new Map<string, Float32Array>();
    for (const result of results) {
      if (result.id && result.dense) {
        embeddingIndex.set(result.id, result.dense);
      }
    }

    expect(embeddingIndex.get("doc-1")?.[0]).toBeCloseTo(0.1);
    expect(embeddingIndex.get("doc-2")?.[0]).toBeCloseTo(0.3);
  });
});
