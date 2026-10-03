/**
 * The TS SDK's wire types must match the shared golden fixtures.
 *
 * Round-trips packages/wire-fixtures/*.json against the runtime witnesses the
 * types are derived from (MODEL_STATES, MODEL_INFO_WIRE_FIELDS), so drift fails
 * in CI rather than shipping. See packages/wire-fixtures/README.md.
 *
 * ModelInfo needs a second check the Python SDK does not: the Python client
 * returns the parsed body as-is, while this one re-maps it. A field can be
 * correctly declared on the type and still never reach a caller, so the last
 * test drives a full wire entry through the real client.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import type { ModelInfo } from "../src/types.js";
import { MODEL_INFO_WIRE_FIELDS, MODEL_STATES } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const fixturesDir = fileURLToPath(new URL("../../wire-fixtures/", import.meta.url));

function loadFixture(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(`${fixturesDir}${name}`, "utf8"));
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** The three top-level keys the client camelCases; everything else passes through. */
const WIRE_TO_CLIENT_KEY: Record<string, string> = {
  max_sequence_length: "maxSequenceLength",
  last_error: "lastError",
  pending_generation: "pendingGeneration",
};

/**
 * Assert the mapper carried every field in the fixture's typed set.
 *
 * Derived from `MODEL_INFO_WIRE_FIELDS` rather than a hand-written list — a
 * hand-written list is exactly what let the mapper drop fields in the first
 * place, and would let a newly typed field slip through unasserted. Shared by
 * the `getModel` and `listModels` tests for the same reason the two call sites
 * share one mapper: the original bug was two copies of the same list drifting.
 */
function expectEveryTypedFieldPresent(model: ModelInfo | undefined): void {
  expect(model).toBeDefined();
  for (const wireKey of MODEL_INFO_WIRE_FIELDS) {
    const clientKey = WIRE_TO_CLIENT_KEY[wireKey] ?? wireKey;
    expect({ key: clientKey, present: Object.hasOwn(model ?? {}, clientKey) }).toEqual({
      key: clientKey,
      present: true,
    });
  }
}

/** A `/v1/models/{model}` body with every emitted key populated. */
const FULL_WIRE_ENTRY = {
  name: "BAAI/bge-m3",
  loaded: true,
  state: "loaded",
  last_error: null,
  inputs: ["text"],
  outputs: ["dense", "sparse"],
  dims: { dense: 1024, sparse: 250002 },
  max_sequence_length: 8192,
  revision: "5617a9f61b028005a4858fdac845db406aefb181",
  profiles: { default: { is_default: true }, fp8: { is_default: false } },
  capabilities: { grammar: ["json_schema"], tools: true, code: false, sql: false, guard: false },
  pending_generation: { total: 0, groups: [] },
  routing: { policy: "remote_only", upstream_kind: "sie" },
  aliases: ["sparse-embeddings-best"],
  // OpenAI retrieve-model compat keys, merged in by the detail endpoint.
  id: "BAAI/bge-m3",
  object: "model",
  created: 1_700_000_000,
  owned_by: "sie",
};

describe("wire contract golden fixtures", () => {
  it("ModelState matches the golden fixture", () => {
    const fixture = loadFixture("model_state.json");
    const states = fixture.model_states as string[];
    expect([...MODEL_STATES].sort()).toEqual([...states].sort());
  });

  it("ModelInfo declares every typed wire field", () => {
    const fixture = loadFixture("model_info.json");
    const typed = fixture.typed as string[];
    expect([...MODEL_INFO_WIRE_FIELDS].sort()).toEqual([...typed].sort());
  });

  it("ModelInfo omits the deliberately excluded wire fields", () => {
    const fixture = loadFixture("model_info.json");
    const excluded = Object.keys(fixture.excluded as Record<string, string>);
    expect(excluded.length).toBeGreaterThan(0);
    expect(MODEL_INFO_WIRE_FIELDS.filter((field) => excluded.includes(field))).toEqual([]);
  });
});

describe("getModel wire mapping", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("surfaces every typed wire field to callers", async () => {
    // Regression: the mapper used to rebuild the object from a hardcoded
    // six-field allowlist, so state/last_error/profiles/pending_generation
    // were dropped on the floor even once the type declared them.
    mockFetch.mockResolvedValueOnce(jsonResponse(FULL_WIRE_ENTRY));

    const model = await client.getModel("BAAI/bge-m3");

    expectEveryTypedFieldPresent(model);

    // Values, so a mapper that passes the right keys with the wrong contents
    // still fails. Object-valued fields are compared whole against the fixture
    // rather than by spot-checking one member — a spot-check leaves every other
    // member of that object unpinned, which is the same gap one level down.
    expect(model.name).toBe(FULL_WIRE_ENTRY.name);
    expect(model.loaded).toBe(true);
    expect(model.state).toBe("loaded");
    expect(model.aliases).toEqual(FULL_WIRE_ENTRY.aliases);
    expect(model.routing).toEqual(FULL_WIRE_ENTRY.routing);
    expect(model.lastError).toBeNull();
    expect(model.inputs).toEqual(FULL_WIRE_ENTRY.inputs);
    expect(model.outputs).toEqual(FULL_WIRE_ENTRY.outputs);
    expect(model.dims).toEqual(FULL_WIRE_ENTRY.dims);
    expect(model.revision).toBe(FULL_WIRE_ENTRY.revision);
    expect(model.profiles).toEqual(FULL_WIRE_ENTRY.profiles);
    expect(model.capabilities).toEqual(FULL_WIRE_ENTRY.capabilities);
    expect(model.pendingGeneration).toEqual(FULL_WIRE_ENTRY.pending_generation);
    expect(model.maxSequenceLength).toBe(FULL_WIRE_ENTRY.max_sequence_length);
  });

  it("strips the OpenAI compat keys instead of leaking them onto ModelInfo", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(FULL_WIRE_ENTRY));

    const model = await client.getModel("BAAI/bge-m3");

    const fixture = loadFixture("model_info.json");
    const excluded = Object.keys(fixture.excluded as Record<string, string>);
    expect(Object.keys(model).filter((key) => excluded.includes(key))).toEqual([]);
  });

  it("passes through a field the SDK does not yet know about", async () => {
    // The mapper is total by construction (rest spread), so a newly added
    // endpoint field reaches callers before the SDK ships a type for it.
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ ...FULL_WIRE_ENTRY, some_future_field: "present" }),
    );

    const model = await client.getModel("BAAI/bge-m3");

    expect((model as Record<string, unknown>).some_future_field).toBe("present");
  });

  it("passes explicit nulls through instead of coercing them", async () => {
    // The emitters send `null` rather than omitting these four: the gateway
    // hardcodes `last_error: null` and serializes max_sequence_length/revision
    // as Option<_> with no skip_serializing_if; the server declares all four
    // as `X | None`. A null must survive the mapper as null — collapsing it to
    // undefined would hide "explicitly no value" behind "field not sent".
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        name: "m",
        loaded: false,
        state: "available",
        inputs: ["text"],
        outputs: ["dense"],
        last_error: null,
        max_sequence_length: null,
        revision: null,
        capabilities: null,
      }),
    );

    const model = await client.getModel("m");

    expect(model.lastError).toBeNull();
    expect(model.maxSequenceLength).toBeNull();
    expect(model.revision).toBeNull();
    expect(model.capabilities).toBeNull();
  });

  it("surfaces a load failure with its diagnostic detail", async () => {
    const loadError = {
      code: "GATED",
      message: "Access to model is restricted",
      attempts: 3,
      permanent: true,
    };
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ ...FULL_WIRE_ENTRY, loaded: false, state: "failed", last_error: loadError }),
    );

    const model = await client.getModel("BAAI/bge-m3");

    expect(model.state).toBe("failed");
    // Whole-object equality pins all four members of ModelLoadError, including
    // `permanent` — the field callers branch on to decide whether retrying is
    // pointless — without leaving message/attempts unasserted.
    expect(model.lastError).toEqual(loadError);
  });
});

describe("listModels wire mapping", () => {
  let client: SIEClient;

  beforeEach(() => {
    mockFetch.mockClear();
    client = new SIEClient("http://localhost:8080");
  });

  afterEach(async () => {
    await client.close();
  });

  it("normalizes a missing alias list to an empty array", async () => {
    // A single SIE server emits no `aliases` key at all. `ModelInfo` declares
    // the field as always present, so the mapper must supply `[]` rather than
    // let `undefined` reach a caller — presence alone would not catch this,
    // because Object.hasOwn is satisfied by an own property set to undefined.
    const { aliases: _omitted, ...withoutAliases } = FULL_WIRE_ENTRY;
    mockFetch.mockResolvedValueOnce(jsonResponse({ models: [withoutAliases] }));

    const [model] = await client.listModels();

    expect(model?.aliases).toEqual([]);
  });

  it("applies the same mapping as getModel", async () => {
    // listModels and getModel each had their own copy of the allowlist; they
    // now share one mapper, and this pins that they stay in step.
    mockFetch.mockResolvedValueOnce(jsonResponse({ models: [FULL_WIRE_ENTRY] }));

    const [model] = await client.listModels();

    expectEveryTypedFieldPresent(model);

    expect(model?.name).toBe(FULL_WIRE_ENTRY.name);
    expect(model?.loaded).toBe(true);
    expect(model?.state).toBe("loaded");
    expect(model?.aliases).toEqual(FULL_WIRE_ENTRY.aliases);
    expect(model?.lastError).toBeNull();
    expect(model?.inputs).toEqual(FULL_WIRE_ENTRY.inputs);
    expect(model?.outputs).toEqual(FULL_WIRE_ENTRY.outputs);
    expect(model?.dims).toEqual(FULL_WIRE_ENTRY.dims);
    expect(model?.revision).toBe(FULL_WIRE_ENTRY.revision);
    expect(model?.profiles).toEqual(FULL_WIRE_ENTRY.profiles);
    expect(model?.capabilities).toEqual(FULL_WIRE_ENTRY.capabilities);
    expect(model?.pendingGeneration).toEqual(FULL_WIRE_ENTRY.pending_generation);
    expect(model?.maxSequenceLength).toBe(FULL_WIRE_ENTRY.max_sequence_length);
    expect(model?.maxSequenceLength).toBe(8192);
  });
});
