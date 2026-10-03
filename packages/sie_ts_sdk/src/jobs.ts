/**
 * Jobs surface (`client.jobs`) — pure helpers + wire types.
 *
 * The jobs API is the gateway's batch class. `client.jobs.submit(...)` binds to
 * `POST /v1/jobs`; this module owns the transport-free pieces (the
 * `source → operation → sink / when` slot mapping and result decoding) so they
 * mirror the Python SDK's `sie_sdk.jobs`.
 */

import { MalformedChunkError, RequestError } from "./errors.js";
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
  /** Connector-only explicit consent: inspect a plan or run it. No implicit default. */
  execution?: "plan" | "run" | null;
  /** Required retry-stable key for a connector submission; inline jobs must omit it. */
  idempotencyKey?: string | null;
  /** Immediate trigger only: "now" (default); schedule/watch are unavailable. */
  when?: string | null;
  /** Encode output types (default: dense). */
  outputTypes?: string[];
  /**
   * One job-level operation map, applied uniformly to every item and forwarded
   * as-is: score → `options.query`, extract → `options.labels` /
   * `options.output_schema`, generate → sampling (e.g. `max_new_tokens`).
   */
  options?: Record<string, unknown> | null;
}

/** Validate the gateway's connector-only `Idempotency-Key` header contract. */
export function requireConnectorIdempotencyKey(key: string | null | undefined): string {
  if (
    typeof key !== "string" ||
    key.length < 1 ||
    key.length > 256 ||
    [...key].some((character) => {
      const code = character.charCodeAt(0);
      return code < 0x20 || code > 0x7e;
    })
  ) {
    throw new RequestError(
      "connector idempotencyKey must contain 1-256 printable ASCII bytes",
      "invalid_request",
      400,
    );
  }
  return key;
}

/** The preflight reservation echoed on submit / status. */
export interface JobPreflight {
  estimated_credits?: number;
  estimate_basis?: string;
}

/** Planner proofs safe to expose to a connector caller. */
export interface ConnectorJobValidation {
  source?: string;
  identity?: string;
  sink?: string;
}

/** The bounded behavior proven for one connector plan revision. */
export interface ConnectorJobCapabilities {
  incremental_inference?: boolean;
  incremental_source_scan?: boolean;
  source_scan?: string;
  source_proof?: string;
  checkpoint_profile?: string;
  incremental_selection?: boolean;
  inference?: string;
  sink_targets?: unknown;
  snapshot?: string;
  ordering?: string;
  deletion_handling?: string;
  publication?: string;
}

/** Redacted, canonical output shape for one connector plan. */
export interface ConnectorPlanOutputShape {
  result_kind: "vector";
  output_field: "embedding";
  output_types: ["dense"];
  dimensions: number | null;
}

/** A durable inspect-first plan. It contains metadata, never source rows. */
export interface ConnectorJobPlan {
  revision?: number;
  expires_at?: number;
  executable?: boolean;
  executor_available?: boolean | null;
  executor_availability?: string;
  blocking_code?: string | null;
  rows?: number;
  mapped_bytes?: number;
  input_bytes?: number;
  eligible_count?: number;
  eligible_count_quality?: string;
  eligible_input_byte_count?: number;
  matched_checkpoint_count?: number;
  skipped_unchanged_count?: number;
  deleted_preserved_count?: number;
  output_dimensions?: number | null;
  output: ConnectorPlanOutputShape;
  cost_basis?: string;
  max_reservation_credits?: number;
  validation?: ConnectorJobValidation;
  capabilities?: ConnectorJobCapabilities;
}

/** Public checkpoint/fence position for the connector profile. */
export interface ConnectorJobCheckpoint {
  profile?: string;
  profile_version?: number;
  region?: string;
  expected_generation?: number;
  generation?: number;
  published_revision?: number;
}

/** Bounded item counters for one connector attempt. */
export interface ConnectorJobItemOutcomes {
  claimed?: number;
  dispatched?: number;
  inferred?: number;
  staged?: number;
  published?: number;
  failed?: number;
  reexecution_required?: number;
  skipped_unchanged?: number;
}

/** Public evidence for one atomic sink publication. */
export interface ConnectorJobPublication {
  attempt_ordinal?: number;
  revision?: number;
  published?: number;
  skipped_unchanged?: number;
  deleted?: number;
  failed?: number;
  reexecuted?: number;
  committed_at?: number;
}

/** One public execute/repair attempt; private dispatch authority is omitted. */
export interface ConnectorJobAttempt {
  ordinal: number;
  action: "execute" | "repair";
  state: string;
  recovery_attempt_ordinal?: number;
  outcome?: string | null;
  error_code?: string | null;
  replayed?: boolean;
  billed_credits?: number | null;
  overlap_owner?: { job_id?: string; attempt_ordinal?: number } | null;
  item_outcomes?: ConnectorJobItemOutcomes;
  publication?: ConnectorJobPublication | null;
  created_at?: number;
  finished_at?: number | null;
}

/** Public crash-recovery posture; tokens and receipt MACs never appear. */
export interface ConnectorJobRecovery {
  required?: boolean;
  state?: string | null;
  outcome?: string | null;
  error_code?: string | null;
  reexecution_required?: boolean;
  repair?: {
    expires_at?: number | null;
    attempts_used?: number | null;
    attempts_remaining?: number | null;
    attempts_max?: number | null;
  };
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

/** The fresh `201` envelope from `POST /v1/jobs` (inline or connector job). */
export interface JobSubmitResult {
  id: string;
  object: string;
  operation: string;
  model: string;
  state: JobState;
  total_items?: number;
  chunks?: number;
  preflight?: JobPreflight;
  // Connector source/sink URIs and SQL are deliberately absent from public responses.
  execution?: "plan" | "run";
  phase?: string;
  plan_revision?: number;
  plan_expires_at?: number;
  idempotency_expires_at?: number;
  plan?: ConnectorJobPlan | null;
  checkpoint?: ConnectorJobCheckpoint;
  attempt?: ConnectorJobAttempt;
  publication?: ConnectorJobPublication | null;
  recovery?: ConnectorJobRecovery;
}

/** A job's public status doc from `GET /v1/jobs/{id}` (refs, never payloads). */
export interface JobStatus {
  id: string;
  object: string;
  operation: string;
  model: string;
  state: JobState;
  execution?: "plan" | "run";
  phase?: string;
  outcome?: string | null;
  error_code?: string | null;
  plan_revision?: number | null;
  plan_expires_at?: number | null;
  idempotency_expires_at?: number | null;
  plan?: ConnectorJobPlan | null;
  checkpoint?: ConnectorJobCheckpoint;
  attempt?: ConnectorJobAttempt;
  attempts?: ConnectorJobAttempt[];
  publication?: ConnectorJobPublication | null;
  recovery?: ConnectorJobRecovery;
  total_items?: number;
  completed_items?: number;
  preflight?: JobPreflight;
  settled_credits?: number;
  created_at?: number;
  finished_at?: number | null;
  output?: { kind?: string; chunks?: JobChunk[] };
}

/**
 * Stable per-item job failure (mirrors the Python SDK's `JobItemErrorDetail`).
 *
 * Decoded from the failed item's `WorkResult` in a chunk ref: `code` is the
 * worker's `error_code` and `message` its free-text `error`. Either may be
 * absent, so both keys are optional.
 */
export interface JobItemErrorDetail {
  code?: string;
  message?: string;
}

/**
 * One decoded per-item result retrieved from a finished job's chunk refs.
 *
 * A `failed` chunk still writes a result ref carrying every item's
 * `WorkResult` — successful siblings AND the failures — so `success`
 * distinguishes them and `error` carries the failure reason when the item did
 * not succeed.
 */
export interface JobResultItem {
  /**
   * Per-item id echoed from the item's `WorkResult` (`id` or the wire's
   * `work_item_id`). Ids may be numbers, so `0` is a valid id, not absent.
   */
  id: string | number | null;
  success: boolean | null;
  units: unknown;
  dims: number | null;
  dense: number[] | Float32Array | null;
  /** Why this item failed; absent on a success (and on a bare-count chunk error). */
  error?: JobItemErrorDetail;
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
const CONNECTION_NAME_START_PATTERN = /^[A-Za-z0-9]$/;
const CONNECTION_NAME_CHAR_PATTERN = /^[A-Za-z0-9._-]$/;
const POSTGRES_SCHEMA_START_PATTERN = /^[A-Za-z_]$/;
const POSTGRES_SCHEMA_CHAR_PATTERN = /^[A-Za-z0-9_$]$/;

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
  const name = authority;
  if (!name) {
    throw new RequestError(
      `connector URI ${JSON.stringify(uri)} names no connection (expected 'scheme://<connection>/…')`,
      "invalid_request",
      400,
    );
  }
  return requireConnectionName(name);
}

/** Return a canonical named-connection path segment or fail before I/O. */
export function requireConnectionName(name: string): string {
  if (
    name.length === 0 ||
    name.length > 128 ||
    !CONNECTION_NAME_START_PATTERN.test(name[0] ?? "") ||
    Array.from(name).some((char) => !CONNECTION_NAME_CHAR_PATTERN.test(char))
  ) {
    throw new RequestError(
      "connection name must be 1-128 ASCII letters, digits, '.', '_', or '-', and start with a letter or digit",
      "invalid_request",
      400,
    );
  }
  return name;
}

/** Validate the optional Postgres source/sink namespace pair before I/O. */
export function requireConnectionSchemaPolicy(
  connectionType: string,
  sourceSchema: string | null | undefined,
  sinkSchema: string | null | undefined,
): { sourceSchema: string; sinkSchema: string } | undefined {
  const sourceMissing = sourceSchema == null;
  const sinkMissing = sinkSchema == null;
  if (sourceMissing !== sinkMissing) {
    throw new RequestError(
      "sourceSchema and sinkSchema must be supplied together",
      "invalid_request",
      400,
    );
  }
  if (sourceMissing || sinkMissing) return undefined;
  if (connectionType !== "postgres") {
    throw new RequestError(
      "sourceSchema and sinkSchema apply only to postgres connections",
      "invalid_request",
      400,
    );
  }
  const validSchema = (schema: string): boolean =>
    schema.length >= 1 &&
    schema.length <= 63 &&
    POSTGRES_SCHEMA_START_PATTERN.test(schema[0] ?? "") &&
    Array.from(schema.slice(1)).every((char) => POSTGRES_SCHEMA_CHAR_PATTERN.test(char));
  if (!validSchema(sourceSchema) || !validSchema(sinkSchema)) {
    throw new RequestError(
      "sourceSchema and sinkSchema must be canonical Postgres identifiers of at most 63 ASCII bytes",
      "invalid_request",
      400,
    );
  }
  return { sourceSchema, sinkSchema };
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
      return connection
        ? { src: source, connection: requireConnectionName(connection) }
        : { src: source };
    }
    return {
      src: source,
      connection: connection == null ? connectionName(source) : requireConnectionName(connection),
    };
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
      if (sinkConnection != null) body.sink_connection = requireConnectionName(sinkConnection);
      return body;
    }
    const resolved =
      sinkConnection == null ? connectionName(sink) : requireConnectionName(sinkConnection);
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
  throw new RequestError(
    `scheduled and watched jobs are not available; omit when or use "now" (got ${JSON.stringify(when)})`,
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
  const sinkFields = resolveSink(
    options.sink,
    sourceFields.connection as string | undefined,
    options.sinkConnection,
  );
  if (
    "items" in sourceFields &&
    (options.connection != null ||
      Object.keys(sinkFields).length > 0 ||
      options.sinkConnection != null)
  ) {
    throw new RequestError(
      "connection/sink/sinkConnection apply only to connector-src jobs; inline items return results",
      "invalid_request",
      400,
    );
  }
  Object.assign(body, sinkFields);
  if ("src" in body) {
    if (options.execution !== "plan" && options.execution !== "run") {
      throw new RequestError(
        "connector jobs require execution='plan' or execution='run'",
        "invalid_request",
        400,
      );
    }
    if (
      options.execution !== "run" &&
      (isInternalUri(String(options.source)) ||
        (typeof options.sink === "string" && isInternalUri(options.sink)))
    ) {
      throw new RequestError(
        "upload:// connector jobs are run-only; set execution='run'",
        "invalid_request",
        400,
      );
    }
    body.execution = options.execution;
  } else if (options.execution != null) {
    throw new RequestError(
      "execution applies only to connector-src jobs; inline items must omit it",
      "invalid_request",
      400,
    );
  }
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
  // One job-level operation map (score query / extract labels / generate
  // sampling), applied uniformly; an empty map stays off the wire (additive).
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

/**
 * Extract a per-item failure from a `WorkResult` map (`error`/`error_code`).
 *
 * The gateway writes every item's `WorkResult` into the chunk ref — including
 * failures, each carrying its own `error` (free text) and `error_code` — so a
 * caller can see WHY a specific item failed. Returns `undefined` when the item
 * reports no failure signal (a success, or a bare-count chunk error with no
 * per-item detail).
 */
function resultItemError(result: Record<string, unknown>): JobItemErrorDetail | undefined {
  const code = result.error_code;
  const message = result.error;
  if (code == null && message == null) return undefined;
  // Only strings reach the public shape. These values come off the wire
  // unvalidated, and this path now reads FAILED chunks too, so a non-string
  // `error_code` would otherwise be asserted straight through and violate
  // `JobItemErrorDetail` for the caller.
  const detail: JobItemErrorDetail = {};
  if (typeof code === "string") detail.code = code;
  if (typeof message === "string") detail.message = message;
  // A failure signalled only by non-string junk still counts as a failure;
  // returning `{}` keeps `error` present (truthy check) without lying about
  // the code or message.
  return detail;
}

/**
 * Decode one WorkResult map (from a chunk ref) into a per-item result.
 *
 * A failed item carries no `result_msgpack` but does carry `success: false`
 * plus an `error`/`error_code` pair, which surfaces as {@link
 * JobResultItem.error}.
 */
export function decodeResultItem(result: unknown): JobResultItem {
  const rec: Record<string, unknown> =
    result != null && typeof result === "object" ? (result as Record<string, unknown>) : {};
  const payload = rec.result_msgpack;
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
  // The wire id is `work_item_id`; older/inline results use `id`. The bare-count
  // chunk error names no ids, so a per-item id comes only from the item's own
  // WorkResult (never fabricated). `??` (not `||`) so a falsy-but-valid id — the
  // number `0` or an empty string — is preserved rather than replaced.
  // Validated, not asserted: an id of `{}` or a `success` of `"false"` would
  // otherwise be cast straight into the public shape. Anything off-contract
  // degrades to `null` rather than misrepresenting the wire.
  const rawId = rec.id ?? rec.work_item_id ?? null;
  const id = typeof rawId === "string" || typeof rawId === "number" ? rawId : null;
  const item: JobResultItem = {
    id,
    success: typeof rec.success === "boolean" ? rec.success : null,
    units: rec.units ?? null,
    dims,
    dense: vector,
  };
  const error = resultItemError(rec);
  if (error !== undefined) item.error = error;
  return item;
}

/**
 * Decode a chunk ref's msgpack `WorkResult` array into per-item results.
 *
 * @throws {MalformedChunkError} If the ref's bytes are not decodable msgpack or
 * do not decode to a list. The caller confines this (one bad chunk cannot sink
 * the whole `jobs.results()` call) and reports it as a decode fault, distinct
 * from an unpublished chunk. Per-item decoding stays defensive (see
 * {@link decodeResultItem}).
 */
export function decodeChunkBytes(raw: Uint8Array): JobResultItem[] {
  let results: unknown;
  try {
    results = unpackMessage<unknown>(raw);
  } catch (error) {
    // Normalize any decode failure to one signal.
    throw new MalformedChunkError(
      `chunk ref bytes are not decodable msgpack: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!Array.isArray(results)) {
    throw new MalformedChunkError("chunk ref bytes did not decode to a WorkResult array");
  }
  return results.map((r) => decodeResultItem(r));
}
