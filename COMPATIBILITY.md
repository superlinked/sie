# Compatibility Policy

> Last Updated: 2026-03-17

This document defines SIE's compatibility commitments: what constitutes a breaking change, how deprecations are communicated, and what must be true before declaring 1.0.

---

## Current Status

All SIE packages are at **0.x pre-release** (currently 0.1.x). Per [SemVer 2.0.0](https://semver.org/), the public API is not considered stable during 0.x. However, as packages are published to PyPI, npm, and OCI registries, we make the commitments described below to provide predictability for early adopters.

---

## Versioning Scheme

### Mono-Version

All packages share a **single version number**, managed by [release-please](https://github.com/googleapis/release-please). A release bumps the version across every artifact simultaneously:

| Artifact | Registry | Example |
|----------|----------|---------|
| `sie-sdk` | PyPI | 0.1.6 |
| `sie-langchain`, `sie-llamaindex`, `sie-haystack`, `sie-dspy`, `sie-crewai`, `sie-chroma` | PyPI | 0.1.6 |
| `@superlinked/sie-sdk`, `@superlinked/sie-langchain`, `@superlinked/sie-llamaindex`, `@superlinked/sie-chroma` | npm | 0.1.6 |
| `sie-cluster` Helm chart | OCI (`ghcr.io/superlinked/charts`) | 0.1.6 |
| `sie-server`, `sie-router` Docker images | ghcr.io | 0.1.6 |
| `superlinked/sie/google`, `superlinked/sie/aws` Terraform modules | Terraform Registry | 0.1.6 |

`sie-server`, `sie-router`, `sie-admin`, and `sie-bench` are internal packages distributed via Docker images or used in development only — they are not published to PyPI.

**Why mono-version:** Simplifies compatibility reasoning. Users deploy matching versions of SDK, server, router, and Helm chart. The version skew detection (see below) depends on this guarantee.

**Post-1.0 plan:** Re-evaluate independent versioning after 1.0 if package release cadences diverge significantly. Until then, mono-version remains the approach.

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

- Removing or renaming a values.yaml key (e.g., renaming `router.replicas` to `router.replicaCount`)
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

### CLI (breaking)

- Removing or renaming a CLI command or flag (e.g., `sie-server serve`, `sie-router`, `sie-admin cache`)

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

## Migration Guides

For every breaking change, we provide a migration guide. Migration guides are included in the CHANGELOG entry for the release and, for significant changes, as a standalone section in the release notes.

A migration guide includes:

1. **What changed** — Exact description of the breaking change.
2. **Why** — Rationale for the change.
3. **How to migrate** — Step-by-step instructions with before/after code examples.
4. **Automated migration** (when feasible) — Scripts or codemods that automate the migration.

---

## Version Skew Policy

SIE uses version negotiation headers (`X-SIE-SDK-Version`, `X-SIE-Server-Version`) to detect version mismatches between SDK clients and servers/routers.

**Supported skew:** SDK and server/router must share the same **major** version and be within **1 minor version** of each other.

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

- [ ] Full matrix evaluation (85+ models) passes with >95% pass rate on GKE
- [ ] Quality targets exist and are met for all supported models
- [ ] Performance baselines documented for representative hardware (L4, A100-40GB, A100-80GB)

### Production Readiness

- [ ] At least 2 external design partners running SIE in production
- [ ] Load testing validates autoscaling behavior under stress
- [ ] Observability stack (Prometheus, Grafana, alerting) validated end-to-end
- [ ] Documentation complete: quickstart, configuration reference, deployment guides, cloud deployment guide

### Publishing

- [ ] All packages published to public registries (PyPI, npm, ghcr.io, OCI Helm, Terraform Registry)
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
