# Compatibility Policy

> Last Updated: 2026-03-17

## Candle checkpoint filenames: ZIP 9 migration

The Rust/Candle worker's maintained Candle 0.10.2 backport replaces yanked ZIP 7
with ZIP 9. This adoption is a **breaking minor change**, because ZIP 9 changes
how two classes of non-ASCII NPZ/PTH archive filenames are decoded:

| Filename bytes and ZIP flag | ZIP 7 name | ZIP 9 name |
| --- | --- | --- |
| Valid UTF-8 bytes `C3 A9`, UTF-8 flag (bit 11) clear | `├⌐` (CP437) | `é` (UTF-8) |
| Invalid UTF-8 byte `82`, UTF-8 flag set | `�` (replacement character) | `é` (CP437 fallback) |

ZIP 9 tries UTF-8 first regardless of the flag, then falls back to CP437.
Ordinary ASCII, correctly flagged UTF-8, and CP437 filenames whose bytes are
not valid UTF-8 retain their names. Checkpoint loading does not extract archive
entries to filesystem paths.

To retain a previous tensor/member name in an affected archive, regenerate it
with that intended name encoded as valid UTF-8 and the ZIP UTF-8 flag set. For
example, encode `├⌐` itself as UTF-8 to retain that ZIP 7 name, rather than
reusing unflagged `C3 A9` bytes. Regenerate invalid flagged filenames using the
intended Unicode name. This migration applies to ZIP-backed NPZ/PTH checkpoints;
the Safetensors format is unaffected.

This document defines SIE's compatibility commitments: what constitutes a breaking change, how deprecations are communicated, and what must be true before declaring 1.0.

---

## Current Status

All SIE packages are at **0.x pre-release** (currently 0.1.x). Per [SemVer 2.0.0](https://semver.org/), the public API is not considered stable during 0.x. However, as packages are published to PyPI, npm, and OCI registries, we make the commitments described below to provide predictability for early adopters.

---

## Versioning Scheme

### SIE Release Version

Artifacts produced by this repository share a **single version number**,
managed by [release-please](https://github.com/googleapis/release-please). An
SIE release bumps these artifacts together:

| Artifact | Registry | Example |
|----------|----------|---------|
| `sie-sdk` | PyPI | 0.1.6 |
| `sie-langchain`, `sie-llamaindex`, `sie-haystack`, `sie-dspy`, `sie-crewai`, `sie-chroma` | PyPI | 0.1.6 |
| `@superlinked/sie-sdk`, `@superlinked/sie-langchain`, `@superlinked/sie-llamaindex`, `@superlinked/sie-chroma` | npm | 0.1.6 |
| `sie-cluster` Helm chart | OCI (`ghcr.io/superlinked/charts`) | 0.1.6 |
| `sie-server`, `sie-gateway` Docker images | ghcr.io | 0.1.6 |

`sie-server` and `sie-gateway` are distributed as Docker images rather than PyPI packages.

The Alibaba Cloud, AWS, Azure, and Google Cloud Terraform modules are
independently versioned libraries. Their public repositories own their source,
compatibility policy, tags, releases, and Terraform Registry versions. A
Terraform module version does not identify an SIE, image, or Helm chart
version; consumers pin a tested combination deliberately.

**Why one SIE release version:** It simplifies compatibility reasoning for the
SDK, server, gateway, images, and Helm chart. The version skew detection (see
below) depends on this guarantee.

### SemVer Interpretation

During **0.x** (current):

- **Patch** (0.1.x → 0.1.y): Bug fixes, documentation, internal refactoring. No breaking changes.
- **Minor** (0.x.0 → 0.y.0): New features, may include breaking changes. Breaking changes are documented in the CHANGELOG with a `BREAKING CHANGES` section and accompanied by a migration guide.

After **1.0**:

- **Patch** (1.0.x → 1.0.y): Bug fixes only.
- **Minor** (1.x.0 → 1.y.0): New features, backward-compatible. Deprecated features may be removed only after the deprecation window.
- **Major** (x.0.0 → y.0.0): Breaking changes permitted.

---

## Breaking Change Definition

A **breaking change** is any modification that can cause existing working code, deployments, or integrations to fail without user action. Specifically:

### Server API (breaking)

- Removing or renaming an endpoint (e.g., removing `/v1/encode`)
- Changing the HTTP method of an endpoint
- Removing or renaming a required or optional request field
- Changing the type or semantics of an existing request or response field
- Changing the structure of a successful response body (e.g., renaming `items[].dense.values` to `items[].dense.data`)
- Removing or renaming a response header that clients depend on (e.g., `X-SIE-Server-Version`)
- Changing an error code string (e.g., renaming `MODEL_LOADING` to `MODEL_NOT_READY`)
- Changing the default wire format (currently msgpack)

### Server API (not breaking)

- Adding a new endpoint
- Adding a new optional request field with a backward-compatible default
- Adding a new field to a response body
- Adding a new response header
- Adding a new error code
- Performance improvements that do not change observable behavior
- Adding a new model to the catalog

### Python SDK (breaking)

- Removing or renaming a public method on `SIEClient` or `SIEAsyncClient` (e.g., `encode`, `score`, `extract`, `list_models`, `watch`, `create_pool`, `get_pool`, `delete_pool`, `get_capacity`, `wait_for_capacity`)
- Changing the signature of a public method in a non-backward-compatible way (removing a parameter, changing a parameter from optional to required)
- Removing or renaming a public type (`Item`, `EncodeResult`, `SparseResult`, `ScoreResult`, `ExtractResult`, `ScoreEntry`)
- Removing or renaming a public exception class (`SIEError`, `SIEConnectionError`, `RequestError`, `ServerError`, `ProvisioningError`, `PoolError`, `ModelLoadingError`, `LoraLoadingError`)
- Changing the type of a return value

### Python SDK (not breaking)

- Adding a new method to `SIEClient` / `SIEAsyncClient`
- Adding a new optional parameter to an existing method
- Adding a new type or exception class
- Adding a new field to a result type

### TypeScript SDK (breaking)

Same criteria as the Python SDK, applied to `SIEClient` and its types in `@superlinked/sie-sdk`.

### Helm Chart (breaking)

- Removing or renaming a values.yaml key (e.g., renaming `gateway.replicas` to `gateway.replicaCount`)
- Changing the default value of an existing key in a way that alters behavior
- Removing a template
- Changing label selectors on StatefulSets or Deployments (causes rolling update failures)

### Helm Chart (not breaking)

- Adding a new values.yaml key with a backward-compatible default
- Adding a new template
- Updating default image tags

### Framework Integrations (breaking)

- Removing or renaming a public class (e.g., `SIEEmbeddings` in `sie-langchain`)
- Changing constructor parameters in a non-backward-compatible way

### Environment Variables (breaking)

- Removing or renaming a documented environment variable (e.g., renaming `SIE_DEVICE` to `SIE_COMPUTE_DEVICE`)
- Changing the semantics of an existing environment variable value

### Environment Variables (not breaking)

- Adding a new environment variable with a default that preserves existing behavior
- Test-harness variables (`SIE_FAKE_MEMORY_BUDGET`, `SIE_FAKE_FAULTS`) are development tooling, not part of the compatibility surface; they may change without deprecation

### CLI (breaking)

- Removing or renaming a CLI command or flag (e.g., `sie-server serve` or the `sie-gateway` binary)

### Terraform Modules (breaking)

- Removing or renaming a variable (e.g., renaming `cluster_name` to `name`)
- Removing or renaming an output
- Changing the type of an existing variable or output
- Removing a resource that users depend on

### Terraform Modules (not breaking)

- Adding a new variable with a default value
- Adding a new output
- Updating provider version constraints

### Model Configuration (not breaking)

- Adding, removing, or modifying model configuration files (`packages/sie_server/models/*.yaml`). Model configs are operational, not part of the public API. Users may add their own model configs.

---

## Deprecation Policy

### During 0.x (current)

Deprecated features are kept for at least **2 minor versions** before removal.

**Example:** If a feature is deprecated in 0.2.0, the earliest it can be removed is 0.4.0.

### After 1.0

Deprecated features are kept for at least **2 minor versions** before removal, and removal requires a new major version.

### How Deprecations Are Communicated

1. **CHANGELOG.md** — Every deprecation is listed under a `Deprecations` section in the release notes.
2. **Runtime warnings** — Deprecated SDK methods and server parameters emit `FutureWarning` (Python) or `console.warn` (TypeScript) on use, including the version in which the feature will be removed and what to use instead. `FutureWarning` is used instead of `DeprecationWarning` so that notices are visible to application users by default (see [PEP 565](https://peps.python.org/pep-0565/)).
3. **Documentation** — Deprecated features are marked in docs with a deprecation notice, the replacement, and the planned removal version.
4. **Code comments** — Deprecated code is annotated with `# Deprecated in 0.x.0, remove in 0.y.0` (or equivalent).

---

## Scheduled changes

### Secure defaults in 0.13.0

One release adds the options and warnings below and breaks nothing. It is
called release N here. Release 0.13.0 then changes the defaults. It ships at
least two minor releases and four weeks after release N, as the deprecation
policy above requires. Until 0.13.0, each component that relies on a default
that will change logs a warning naming 0.13.0.

#### Release N: options and warnings

Nothing breaks. Every new option keeps today's behaviour by default. There are
two exceptions, both security fixes:

- `/ws/status` stops sending host identity at once, which changes the values of
  three response fields (see the `ServerInfo` item below). The fields keep
  their names and types, so clients that read them keep working.
- `sie-server` stops at startup when `SIE_AUTH_TOKENS` or `SIE_AUTH_TOKEN` is
  set but holds no token, as an unset variable passed through does, unless
  `SIE_AUTH_MODE=none`. Without this, a server meant to require a token would
  run without authentication.

The changes:

- `sie-server serve`:
  - `--host` also reads `SIE_SERVER_HOST`. The default stays `0.0.0.0`.
  - Opt-in token authentication, with the gateway's variables:
    `SIE_AUTH_MODE` (`none`, `token`, `auto`) and `SIE_AUTH_TOKENS` or
    `SIE_AUTH_TOKEN`. A configured token turns it on. Clients send
    `Authorization: Bearer <token>` or `x-api-key: <token>`; when both
    carry a token, `Authorization` is used and `x-api-key` is ignored.
    `/healthz`, `/livez` and `/readyz` stay open.
  - In token mode without a configured token, the server generates one and
    saves it with mode `0600` in `SIE_DATA_DIR`. It prints the token once when
    stderr is a terminal, and otherwise only the file's path. A container with
    a terminal counts as one, and its logs keep that print.
  - `SIE_DATA_DIR` moves from `/tmp/sie-server` to a per-user directory,
    `$XDG_DATA_HOME/sie` or `~/.local/share/sie`. Images set `/app/.cache/sie`;
    mount a volume there to keep the token across containers.
  - A server that listens beyond loopback without authentication logs a
    warning on every start.
  - A server bound to loopback without authentication logs a warning, once per
    `Host`, for a request whose `Host` does not name this machine (`localhost`,
    `127.0.0.1`, `[::1]` or the bind address). The request is still answered.
  - `/ws/status` no longer sends the OS user, working directory or process id.
    `ServerInfo.user`, `working_dir` and `pid` are always `""`, `""` and `0`,
    and are deprecated. This is an exception to the rule that a changed
    response value is breaking: the old values exposed the host to any client
    that could open the WebSocket. `ServerInfo.version` reports the installed
    version.
- Rust/Candle worker: logs a notice when it binds `0.0.0.0` only because
  `SIE_SERVER_HOST` is unset.
- Gateway:
  - Accepts `x-api-key` alongside `Authorization`. The first
    `Authorization` value decides when it carries a token, and `x-api-key` is
    then ignored, so a request that passes today still passes, including one
    where a proxy in front of the gateway adds an `x-api-key` of its own.
    Otherwise `x-api-key` decides. Repeated `x-api-key` values that carry
    different tokens get `401`.
  - Later `Authorization` values are still ignored. The gateway logs a
    warning, once per process, the first time a request that passes
    authentication repeats `Authorization` with values that carry different
    tokens, with a value it cannot read, or with a token only after a first
    value that carries none. A refused request never logs it.
  - `SIE_WORKER_AUTH_TOKEN` is the token the gateway presents to workers.
  - Logs a warning when `SIE_AUTH_MODE` is unset on a non-loopback bind.
- Telemetry: `SIE_TELEMETRY_ENABLED=1` opts in explicitly. The default stays on,
  with a notice at startup. Helm `telemetry.enabled: true` sets
  `SIE_TELEMETRY_ENABLED=1`.
- Helm, with the chart change that adds them: `gateway.auth.generateToken` and
  `workers.auth`, both off by default.

#### 0.13.0: default changes (breaking)

| Surface | Until 0.13.0 | From 0.13.0 | To keep the old behaviour |
| --- | --- | --- | --- |
| `sie-server serve` bind | `0.0.0.0` | `127.0.0.1` | `--host 0.0.0.0` or `SIE_SERVER_HOST=0.0.0.0` |
| `sie-server` beyond loopback | No authentication | A token is required. Without `SIE_AUTH_TOKEN`, one is generated, and printed once on a terminal | `SIE_AUTH_MODE=none` |
| `sie-server` on loopback without authentication | A request whose `Host` does not name this machine is answered, with a warning | It gets `400`. So does a request whose `Origin` names another site, at least for WebSocket upgrades and methods other than `GET` | Token authentication, or a setting for more allowed `Host` names, designed together with this change |
| Rust/Candle worker bind | `0.0.0.0` | `127.0.0.1` | `SIE_SERVER_HOST=0.0.0.0` (the image sets it) |
| Gateway in token mode, a repeated `Authorization` header whose values carry different tokens, include one it cannot read, or carry a token only after the first value | The later values are ignored, with a warning | `401` | Send one `Authorization` header |
| Gateway binary beyond loopback with `SIE_AUTH_MODE` unset | Starts without authentication | Refuses to start | `SIE_AUTH_MODE=none`, or `token` with `SIE_AUTH_TOKENS` |
| Telemetry | On unless disabled | Off unless `SIE_TELEMETRY_ENABLED=1` | `SIE_TELEMETRY_ENABLED=1` |
| Helm `gateway.auth.mode` | `none` | `token`, with a generated Secret (needs the chart change above) | `gateway.auth.mode=none` |
| Helm `workers.auth.enabled` (with the chart change above) | `false` | `true` | `workers.auth.enabled=false` |
| Helm `workers.networkPolicy.enabled` | `false` | `true` | `workers.networkPolicy.enabled=false` |
| Helm `telemetry.enabled` | `true` | `false` | `telemetry.enabled=true` |

`ServerInfo.user`, `working_dir` and `pid` are removed no earlier than 0.13.0.

**Why.** A server that anyone on the network can call runs inference and loads
models at its owner's cost. Telemetry should be a choice, not a default.

**How to migrate.**

Docker, before:

```bash
docker run -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cpu-default
curl http://localhost:8080/v1/models
```

Docker, after: set a token, publish on loopback, and keep the data directory.

```bash
export SIE_API_KEY=$(openssl rand -hex 24)
docker run -p 127.0.0.1:8080:8080 -e SIE_AUTH_TOKEN=$SIE_API_KEY \
  -v sie-data:/app/.cache/sie ghcr.io/superlinked/sie-server:latest-cpu-default
curl -H "Authorization: Bearer $SIE_API_KEY" http://localhost:8080/v1/models
```

Without `-e SIE_AUTH_TOKEN`, read the generated token with
`docker exec <container> cat /app/.cache/sie/auth-token`. The SDKs read
`SIE_API_KEY`.

A server reached from another machine, before:

```bash
sie-server serve
```

After:

```bash
SIE_AUTH_TOKEN=<token> sie-server serve --host 0.0.0.0
```

Helm, with the chart change that adds `workers.auth`: upgrade worker
authentication in two steps, so no worker rejects a gateway that has not rolled
yet.

```bash
helm upgrade sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster -f values.yaml \
  --set workers.auth.enabled=true --set workers.auth.allowUnauthenticated=true
# after every gateway and worker pod has restarted:
helm upgrade sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster -f values.yaml \
  --set workers.auth.enabled=true --set workers.auth.allowUnauthenticated=false
```

For the gateway token, set `gateway.auth.generateToken=true` first, give the
token to your clients, then set `gateway.auth.mode=token`. The chart README
shows how to read the generated Secret.

## Migration Guides

For every breaking change, we provide a migration guide. Migration guides are included in the CHANGELOG entry for the release and, for significant changes, as a standalone section in the release notes.

A migration guide includes:

1. **What changed** — Exact description of the breaking change.
2. **Why** — Rationale for the change.
3. **How to migrate** — Step-by-step instructions with before/after code examples.
4. **Automated migration** (when feasible) — Scripts or codemods that automate the migration.

---

## Version Skew Policy

SIE uses version negotiation headers (`X-SIE-SDK-Version`, `X-SIE-Server-Version`) to detect version mismatches between SDK clients and servers/gateways.

**Supported skew:** SDK and server/gateway must share the same **major** version and be within **1 minor version** of each other.

| SDK Version | Server Version | Status |
|-------------|----------------|--------|
| 0.1.6 | 0.1.6 | Supported (exact match) |
| 0.1.6 | 0.2.0 | Supported (1 minor version apart) |
| 0.1.6 | 0.3.0 | Warning (2+ minor versions apart) |
| 0.1.6 | 1.0.0 | Warning (different major version) |

When a skew is detected — including major-version mismatches — the SDK logs a warning. It does not block requests, but users should upgrade to avoid encountering incompatibilities.

---

## 1.0 Criteria

SIE will declare 1.0 when all of the following are true:

### API Stability

- [ ] Server API (`/v1/encode`, `/v1/score`, `/v1/extract`, `/v1/models`) has been unchanged for at least 3 minor releases
- [ ] Python SDK public interface (`SIEClient`, `SIEAsyncClient`, types, exceptions) has been unchanged for at least 3 minor releases
- [ ] TypeScript SDK public interface has been unchanged for at least 3 minor releases
- [ ] Helm chart values.yaml structure has been unchanged for at least 2 minor releases

### Quality and Correctness

- [ ] Full matrix evaluation (full model catalog) passes with >95% pass rate on GKE
- [ ] Quality targets exist and are met for all supported models
- [ ] Performance baselines documented for representative hardware (L4, A100-40GB, A100-80GB)

### Production Readiness

- [ ] At least 2 external design partners running SIE in production
- [ ] Load testing validates autoscaling behavior under stress
- [ ] Observability stack (Prometheus, Grafana, alerting) validated end-to-end
- [ ] Documentation complete: quickstart, configuration reference, deployment guides, cloud deployment guide

### Publishing

- [ ] All SIE release artifacts published to public registries (PyPI, npm, ghcr.io, OCI Helm)
- [ ] CI/CD pipeline runs tests, linting, and type checking on every PR
- [ ] Release automation produces changelogs and publishes artifacts on tag

### Multi-Cloud

- [ ] Terraform modules available for GCP (GKE), AWS (EKS), and Azure (AKS)
- [ ] Deployment validated on at least 2 cloud providers

---

## References

- [SemVer 2.0.0](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [CHANGELOG.md](CHANGELOG.md) — Release history
