/**
 * Jobs surface (`client.jobs`) — pure helpers + wire types.
 *
 * The jobs API is the gateway's batch class. `client.jobs.submit(...)` binds to
 * `POST /v1/jobs`; this module owns the transport-free pieces (the
 * `source → operation → sink / when` slot mapping and result decoding) so they
 * mirror the Python SDK's `sie_sdk.jobs`.
 */

import { RequestError } from "./errors.js";
import { unpackMessage } from "./msgpack.js";

/** Job lifecycle state (queued → running → terminal). */
export type JobState = "queued" | "running" | "succeeded" | "failed" | "suspended" | "cancelled";

/** Terminal states with no further transitions (job lifecycle). */
export const TERMINAL_JOB_STATES: ReadonlySet<JobState> = new Set([
  "succeeded",
  "failed",
  "suspended",
  "cancelled",
]);

/** One inline input item (the `/v1/encode` item contract: `{text}` / `{id,text}`). */
export type JobItem = { text?: string; id?: string } & Record<string, unknown>;

/**
 * A job source: inline items (a list, or a bare string = one text item) or a
 * connector `scheme://<connection>/…` URI.
 */
export type JobSource = string | Array<string | JobItem>;

/**
 * The uniform source-mapping slots (wire-shaped, mirroring the Python
 * SDK's dict): `id_field` ≈ `custom_id`, `input_field` ≈ `body.input`,
 * `carry` = source fields echoed to the sink keyed by id, `input_type` pins
 * the item shape. The sink slot rides separately as `outputField`. All
 * optional — per-connector URI params stay as aliases.
 */
export interface JobFieldMap {
  id_field?: string;
  input_field?: string;
  carry?: string[];
  input_type?: "text" | "document";
}

/** Options for `client.jobs.submit`. */
export interface SubmitJobOptions {
  /** Inline items or a connector URI (incl. `upload://<file-id>`). */
  source: JobSource;
  /** Model id (e.g. "BAAI/bge-m3"). */
  model: string;
  /** Job operation: encode | score | extract | parse | generate (default "encode"). */
  operation?: string;
  /** Sink: "return" (default), "inplace", or a connector URI. */
  sink?: string | null;
  /** Override the source connection name (default: derived from the URI). */
  connection?: string | null;
  /** Distinct connection name for the sink. */
  sinkConnection?: string | null;
  /** Uniform source mapping (connector jobs only). */
  fieldMap?: JobFieldMap | null;
  /** Sink target (≈ `response.body`; aliases PG `column` / object-store `suffix`). */
  outputField?: string | null;
  /** Trigger: "now" (default), "schedule:<cron>", or "watch:<source>". */
  when?: string | null;
  /** Encode output types (default: dense). */
  outputTypes?: string[];
  /**
   * Per-item options plus the op inputs, forwarded as-is (operation
   * matrix): score → `options.query`, extract → `options.labels` /
   * `options.output_schema`, generate → sampling (e.g. `max_new_tokens`).
   */
  options?: Record<string, unknown> | null;
}

/** The preflight reservation echoed on submit / status. */
export interface JobPreflight {
  estimated_credits?: number;
  estimate_basis?: string;
}

/** One spawned chunk's settle metadata (`output.chunks[]`; results-as-refs). */
export interface JobChunk {
  seq?: number;
  items?: number;
  state?: string;
  ref?: string | null;
  units?: number | null;
  /**
   * Exact credits committed for this chunk, or `null` until the chunk's
   * settlement is acknowledged. A job settles per chunk, so these sum to the
   * job's `settled_credits` exactly. (Wire-shape field, like the rest of this
   * interface — snake_case as the API sends it.)
   */
  credits_charged?: number | null;
  /**
   * Immutable rate-book version that rated `credits_charged`. Present exactly
   * when `credits_charged` is.
   */
  rate_book_version?: string | null;
  error?: unknown;
}

/** The `201` envelope from `POST /v1/jobs` (inline or connector job). */
export interface JobSubmitResult {
  id: string;
  object: string;
  operation: string;
  model: string;
  state: JobState;
  total_items?: number;
  chunks?: number;
  preflight?: JobPreflight;
  input_source?: string;
  source?: string;
  sink?: string;
}

/** A job's public status doc from `GET /v1/jobs/{id}` (refs, never payloads). */
export interface JobStatus {
  id: string;
  object: string;
  operation: string;
  model: string;
  state: JobState;
  total_items?: number;
  completed_items?: number;
  preflight?: JobPreflight;
  settled_credits?: number;
  created_at?: number;
  finished_at?: number | null;
  output?: { kind?: string; chunks?: JobChunk[] };
}

/** One decoded per-item result retrieved from a finished job's chunk refs. */
export interface JobResultItem {
  id: string | null;
  success: boolean | null;
  units: unknown;
  dims: number | null;
  dense: number[] | Float32Array | null;
}

/** A finished job's decoded results — the chunk refs read and unpacked. */
export interface JobResults {
  job_id: string;
  state: JobState | undefined;
  total_items: number | undefined;
  settled_credits: number | undefined;
  chunks: JobChunk[];
  retrieved: number;
  dims: number | null;
  items: JobResultItem[];
}

const SINK_RETURN = new Set(["return", "default"]);
const SINK_INPLACE = new Set(["inplace", "in_place", "in place"]);

// Internal push-to-us schemes (OUR Files store): no org connection to
// name, so no `connection`/`sink_connection` is derived from the URI.
const INTERNAL_SCHEMES = new Set(["upload"]);

// Uniform source-mapping slots (the sink slot is `output_field`).
const FIELD_MAP_KEYS = new Set(["id_field", "input_field", "carry", "input_type"]);
const INPUT_TYPES = new Set(["text", "document"]);

function isConnectorUri(value: string): boolean {
  return value.includes("://");
}

function isInternalUri(uri: string): boolean {
  return INTERNAL_SCHEMES.has(uri.split("://", 1)[0] ?? "");
}

function normItem(item: string | JobItem, index: number): JobItem {
  if (typeof item === "string") return { text: item };
  if (item !== null && typeof item === "object") return item;
  throw new RequestError(`item ${index} must be a string or an object`, "invalid_request", 400);
}

/**
 * Derive a connection name from a connector URI's authority.
 *
 * `postgres://warehouse?query=…` → `warehouse`; `s3://customer-bucket/in/` →
 * `customer-bucket`. Credentials never appear in the call — the job only names
 * the connection; the runner resolves it org-scoped.
 */
export function connectionName(uri: string): string {
  // URL can't parse custom schemes reliably; take the authority manually.
  const afterScheme = uri.split("://", 2)[1] ?? "";
  const authority = afterScheme.split(/[/?#]/, 1)[0] ?? "";
  const name = authority.trim();
  if (!name) {
    throw new RequestError(
      `connector URI ${JSON.stringify(uri)} names no connection (expected 'scheme://<connection>/…')`,
      "invalid_request",
      400,
    );
  }
  return name;
}

function resolveSource(source: JobSource, connection?: string | null): Record<string, unknown> {
  if (Array.isArray(source)) {
    if (source.length === 0) {
      throw new RequestError("inline source has no items", "invalid_request", 400);
    }
    return { items: source.map((item, i) => normItem(item, i)) };
  }
  if (isConnectorUri(source)) {
    if (isInternalUri(source)) {
      // Internal scheme (upload:// = OUR Files store): no connection.
      return connection ? { src: source, connection } : { src: source };
    }
    return { src: source, connection: connection ?? connectionName(source) };
  }
  if (typeof source === "string" && source.trim()) {
    return { items: [{ text: source }] };
  }
  throw new RequestError(
    "source must be inline items (a list/string) or a connector URI (scheme://<connection>/…)",
    "invalid_request",
    400,
  );
}

function resolveSink(
  sink: string | null | undefined,
  sourceConnection: string | undefined,
  sinkConnection: string | null | undefined,
): Record<string, unknown> {
  if (sink === null || sink === undefined || SINK_RETURN.has(sink.trim().toLowerCase())) {
    return {};
  }
  if (SINK_INPLACE.has(sink.trim().toLowerCase())) {
    return { sink: "inplace" };
  }
  if (isConnectorUri(sink)) {
    const body: Record<string, unknown> = { sink };
    if (isInternalUri(sink)) {
      // Internal scheme: OUR Files store, no connection to name.
      if (sinkConnection != null) body.sink_connection = sinkConnection;
      return body;
    }
    const resolved = sinkConnection ?? connectionName(sink);
    // Thread the sink connection when explicitly overridden or distinct from
    // the source's (the common "index my own store" case reuses the source).
    if (sinkConnection != null || resolved !== sourceConnection) {
      body.sink_connection = resolved;
    }
    return body;
  }
  throw new RequestError(
    `sink must be 'return', 'inplace', or a connector URI (got ${JSON.stringify(sink)})`,
    "invalid_request",
    400,
  );
}

/**
 * Validate + map the uniform slots onto the wire fields (`field_map` +
 * `output_field`). Only set fields ride the wire (`/v1` additive-only).
 */
function resolveFieldMap(
  fieldMap: JobFieldMap | null | undefined,
  outputField: string | null | undefined,
): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (fieldMap != null) {
    const unknown = Object.keys(fieldMap).filter((key) => !FIELD_MAP_KEYS.has(key));
    if (unknown.length > 0) {
      throw new RequestError(
        `unknown field_map key(s) ${JSON.stringify(unknown)} (known: ${[...FIELD_MAP_KEYS].join(", ")})`,
        "invalid_request",
        400,
      );
    }
    if (
      fieldMap.carry != null &&
      (!Array.isArray(fieldMap.carry) || fieldMap.carry.some((c) => typeof c !== "string" || !c))
    ) {
      throw new RequestError(
        `field_map.carry must be a list of field names (got ${JSON.stringify(fieldMap.carry)})`,
        "invalid_request",
        400,
      );
    }
    if (fieldMap.input_type != null && !INPUT_TYPES.has(fieldMap.input_type)) {
      throw new RequestError(
        `field_map.input_type must be one of ${[...INPUT_TYPES].join(", ")} (got ${JSON.stringify(fieldMap.input_type)})`,
        "invalid_request",
        400,
      );
    }
    const mapped: Record<string, unknown> = {};
    if (fieldMap.id_field != null) mapped.id_field = fieldMap.id_field;
    if (fieldMap.input_field != null) mapped.input_field = fieldMap.input_field;
    if (fieldMap.input_type != null) mapped.input_type = fieldMap.input_type;
    if (fieldMap.carry != null && fieldMap.carry.length > 0) mapped.carry = fieldMap.carry;
    if (Object.keys(mapped).length > 0) body.field_map = mapped;
  }
  if (outputField != null) {
    if (typeof outputField !== "string" || !outputField) {
      throw new RequestError(
        `output_field must be a non-empty string (got ${JSON.stringify(outputField)})`,
        "invalid_request",
        400,
      );
    }
    body.output_field = outputField;
  }
  return body;
}

function resolveWhen(when: string | null | undefined): Record<string, unknown> {
  if (when == null || when.trim() === "" || when.trim().toLowerCase() === "now") {
    return {};
  }
  const text = when.trim();
  // Slice off the prefix (not `split(":")`, which would drop a value's own colons).
  if (text.toLowerCase().startsWith("schedule:")) {
    return { when: "schedule", schedule: text.slice("schedule:".length).trim() };
  }
  if (text.toLowerCase().startsWith("watch:")) {
    return { when: "watch", watch: text.slice("watch:".length).trim() };
  }
  if (text.toLowerCase() === "schedule") {
    throw new RequestError(
      "schedule trigger needs a cron expr: when='schedule:<cron>'",
      "invalid_request",
      400,
    );
  }
  // A bare cron expression (5 whitespace-separated fields) is a schedule.
  if (text.split(/\s+/).length === 5) {
    return { when: "schedule", schedule: text };
  }
  throw new RequestError(
    `unrecognized when ${JSON.stringify(when)}: use 'now', 'schedule:<cron>', or 'watch:<source>'`,
    "invalid_request",
    400,
  );
}

/**
 * Compose the `POST /v1/jobs` body from the source/op/sink/when slots.
 *
 * A thin, pure mapping: inline `items` or connector `src`/`sink` +
 * connection name, plus an optional trigger. Only the fields that are set ride
 * the wire, so an inline submit is byte-for-byte the realtime POC body and the
 * connector body is additive (`/v1` additive-only rule).
 */
export function buildJobBody(options: SubmitJobOptions): Record<string, unknown> {
  const operation = options.operation ?? "encode";
  const body: Record<string, unknown> = { operation, model: options.model };
  const sourceFields = resolveSource(options.source, options.connection);
  Object.assign(body, sourceFields);
  Object.assign(
    body,
    resolveSink(
      options.sink,
      sourceFields.connection as string | undefined,
      options.sinkConnection,
    ),
  );
  const mappingFields = resolveFieldMap(options.fieldMap, options.outputField);
  if (Object.keys(mappingFields).length > 0 && !("src" in body)) {
    throw new RequestError(
      "fieldMap/outputField apply to connector-src jobs; an inline items job maps nothing",
      "invalid_request",
      400,
    );
  }
  Object.assign(body, mappingFields);
  Object.assign(body, resolveWhen(options.when));
  if (options.outputTypes && options.outputTypes.length > 0) {
    body.output_types = options.outputTypes;
  }
  // Per-item options + op inputs (score query / extract labels / generate
  // sampling), forwarded as-is; an empty map stays off the wire (additive).
  if (options.options && Object.keys(options.options).length > 0) {
    body.options = options.options;
  }
  return body;
}

/** The chunk-ref metadata from a job status doc (`output.chunks` refs). */
export function jobChunks(jobDoc: JobStatus): JobChunk[] {
  const raw = jobDoc.output?.chunks ?? [];
  return raw.map((chunk) => ({
    seq: chunk.seq,
    items: chunk.items,
    state: chunk.state,
    ref: chunk.ref,
    units: chunk.units,
    credits_charged: chunk.credits_charged,
    rate_book_version: chunk.rate_book_version,
    error: chunk.error ?? null,
  }));
}

function toNumberArrayLike(value: unknown): number[] | Float32Array | null {
  if (value == null) return null;
  if (value instanceof Float32Array) return value;
  if (Array.isArray(value)) return value as number[];
  if (ArrayBuffer.isView(value)) return value as unknown as Float32Array;
  return null;
}

function denseInfo(dense: unknown): {
  dims: number | null;
  vector: number[] | Float32Array | null;
} {
  if (dense == null) return { dims: null, vector: null };
  if (typeof dense === "object" && !Array.isArray(dense) && !ArrayBuffer.isView(dense)) {
    const rec = dense as Record<string, unknown>;
    let raw: unknown = null;
    for (const key of ["values", "vector", "dense"]) {
      if (rec[key] != null) {
        raw = rec[key];
        break;
      }
    }
    const vector = toNumberArrayLike(raw);
    let dims = typeof rec.dims === "number" ? rec.dims : null;
    if (dims == null && vector != null) dims = vector.length;
    return { dims, vector };
  }
  const vector = toNumberArrayLike(dense);
  return { dims: vector != null ? vector.length : null, vector };
}

/** Decode one WorkResult map (from a chunk ref) into a per-item result. */
export function decodeResultItem(result: Record<string, unknown>): JobResultItem {
  const payload = result.result_msgpack;
  let decoded: Record<string, unknown> | null = null;
  if (payload instanceof Uint8Array) {
    try {
      decoded = unpackMessage<Record<string, unknown>>(payload);
    } catch {
      decoded = null;
    }
  }
  const dense = decoded && typeof decoded === "object" ? decoded.dense : null;
  const { dims, vector } = denseInfo(dense);
  return {
    id: (result.id as string) ?? null,
    success: (result.success as boolean) ?? null,
    units: result.units ?? null,
    dims,
    dense: vector,
  };
}

/** Decode a chunk ref's msgpack `WorkResult` array into per-item results. */
export function decodeChunkBytes(raw: Uint8Array): JobResultItem[] {
  const results = unpackMessage<unknown>(raw);
  if (!Array.isArray(results)) return [];
  return results.map((r) => decodeResultItem(r as Record<string, unknown>));
}
