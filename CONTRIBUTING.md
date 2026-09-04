# Contributing to SIE

Thank you for helping improve SIE. Contributions of all sizes are welcome,
from documentation corrections and focused bug fixes to new integrations and
runtime features.

## Choose something to work on

Browse the [issue tracker](https://github.com/superlinked/sie/issues), including
issues labeled [`good first issue`](https://github.com/superlinked/sie/labels/good%20first%20issue)
and [`help wanted`](https://github.com/superlinked/sie/labels/help%20wanted).
Search open issues and pull requests before starting so that work is not
duplicated.

You can open a pull request directly for a small, focused fix. Before investing
in a broad API, dependency, architecture, or cross-package change, comment on a
relevant issue or open one to confirm the intended direction with the
maintainers.

## Set up a development checkout

Fork the repository on GitHub, then clone your fork and register this repository
as `upstream`:

```bash
git clone https://github.com/<your-user>/sie.git
cd sie
git remote add upstream https://github.com/superlinked/sie.git
git fetch upstream
git switch -c <topic-branch> upstream/main
```

The normal bootstrap requires Git, Bash, and
[`mise`](https://mise.jdx.dev/getting-started.html). From the repository root,
run:

```bash
./tools/init.sh
```

The script trusts the repository's mise configuration, installs the pinned
toolchain, synchronizes the root Python workspace, and installs the root pnpm
workspace. A clean checkout does not need package-registry credentials.

Native audio development and the root Rust workspace checks require `cmake`,
because the audio crate is a member of that workspace. After installing it,
synchronize the native Python package and its optional dependencies with the
locked Python workspace when working on that package:

```bash
mise exec -- uv sync --frozen --project . --all-packages --all-extras
```

Before starting another branch, update from upstream rather than building on a
stale local `main`:

```bash
git fetch upstream
git switch -c <next-topic-branch> upstream/main
```

## Repository map

| Path | Purpose |
| --- | --- |
| [`packages/sie_server`](packages/sie_server/README.md) | Python inference server, model adapters, model and bundle configuration, and HTTP API |
| [`packages/sie_sdk`](packages/sie_sdk/README.md) | Python client SDK and public request and response types |
| `packages/sie_config` | Python configuration service |
| [`packages/sie_mcp`](packages/sie_mcp/README.md) | Python MCP edge service, plugin, and agent skill |
| [`packages/sie_audio_prep`](packages/sie_audio_prep/README.md) | Native audio preparation extension with Python bindings |
| [`packages/sie_ts_sdk`](packages/sie_ts_sdk/README.md) | TypeScript client SDK |
| [`packages/sie_gateway`](packages/sie_gateway/README.md) | Rust gateway for routing, queuing, API compatibility, and cluster state |
| [`packages/sie_server_sidecar`](packages/sie_server_sidecar/README.md) | Rust worker sidecar |
| `packages/sie_telemetry` | Shared Rust telemetry crate |
| `packages/sie_server_rust` | Standalone Rust Candle worker with its own Cargo workspace and lock |
| [`integrations`](integrations/README.md) | Python and TypeScript framework and vector-database integrations |
| [`deploy/helm/sie-cluster`](deploy/helm/sie-cluster/README.md) and [`deploy/k8s`](deploy/k8s/README.md) | Helm chart and Kubernetes deployment resources |
| [`examples`](examples/README.md) | Runnable examples, each with its own setup and validation guidance |
| [`packages/wire-fixtures`](packages/wire-fixtures/README.md), `conformance`, and [`tests/parity`](tests/parity/README.md) | Cross-language wire, protocol, and conformance fixtures |
| `tools/mise_tasks`, `tools/ci`, and `.github/workflows` | Local developer tasks and hosted CI automation |

Follow the nearest package README when it defines additional setup, runtime, or
test requirements.

## Make a focused change

Keep implementation, tests, documentation, and compatibility updates together
in one coherent pull request. Preserve public API and wire behavior unless the
pull request clearly proposes, documents, and tests a compatibility change. See
[`COMPATIBILITY.md`](COMPATIBILITY.md) for the current compatibility contract.

The repository has distinct dependency boundaries:

- `pyproject.toml` and `uv.lock` cover the root Python workspace and its Python
  packages and integrations.
- `package.json`, `pnpm-workspace.yaml`, and `pnpm-lock.yaml` cover the
  TypeScript SDK and integrations.
- The root `Cargo.toml`, `Cargo.lock`, and `deny.toml` cover the gateway,
  sidecar, telemetry, and audio Rust crates.
- `packages/sie_server_rust/Cargo.toml` and
  `packages/sie_server_rust/Cargo.lock` define the standalone Candle worker.

When dependencies change, update the appropriate lock with its owning package
manager and include the result. Do not edit generated lock content by hand or
move a dependency across these boundaries to make a check pass.

Please also observe these repository-wide conventions:

- Use `sie_sdk.SIEClient` for examples and SIE API calls.
- Use current **gateway** terminology. Keep legacy `router` names only where a
  public wire or environment compatibility contract requires them.
- Put Python imports at module scope, except for optional dependencies, and
  keep `__init__.py` files empty.
- Keep tests deterministic and credential-free. Unit tests must not depend on
  external writes or mutable package versions.
- Do not commit secrets, generated build output, package archives, Helm
  dependency archives, or the model and bundle files temporarily staged inside
  the chart.

If an HTTP API schema changes, regenerate and commit the static specifications:

```bash
mise run openapi
```

Version bumps, changelog entries, release tags, and publication metadata are
maintainer-owned and generated by release automation. They should not be part
of an ordinary feature or fix.

## Validate your change

Run `mise tasks` to see the current task list. For code changes, the common
Python checks are:

```bash
mise run lint
mise run typecheck
mise run test
```

Then run the scoped checks for every surface your change affects. The hosted CI
workflow adds full-workspace harnesses and policy gates described below.

| Surface | Commands |
| --- | --- |
| Python package | `mise run test -- <test-path>` |
| Python integrations | `mise run test-integrations -- --python-only` |
| All Python and TypeScript integrations | `mise run test-integrations` |
| TypeScript SDK and integrations | `mise run ts -- build`, then `mise run ts -- typecheck`, `mise run ts -- lint`, and `mise run ts -- test` |
| Root Rust workspace (scoped local checks; requires `cmake`) | `mise run rust-fmt -- --check`, `mise run rust-check`, `mise run rust-clippy`, and `mise run rust-test` |
| Root Rust dependency changes | `mise run gateway-deny` in addition to the root Rust checks |
| HTTP and wire contracts | `mise exec -- python tools/check_ipc_types_parity.py`, `mise exec -- python tools/check_response_chunk_protocol.py`, and `tests/parity/run_parity.sh` |
| Helm chart | `mise run helm -- dependencies`, `mise run helm -- lint --set payloadStore.enabled=false`, and `mise run helm -- template --set payloadStore.enabled=false` |

The standalone Candle worker is outside the root Rust workspace. Validate it
directly:

```bash
mise exec -- cargo fmt --manifest-path packages/sie_server_rust/Cargo.toml -- --check
mise exec -- cargo check --manifest-path packages/sie_server_rust/Cargo.toml --locked --all-targets
mise exec -- cargo clippy --manifest-path packages/sie_server_rust/Cargo.toml --locked --all-targets -- -D warnings
mise exec -- cargo test --manifest-path packages/sie_server_rust/Cargo.toml --locked
```

For a standalone worker dependency change, also run its dependency policy:

```bash
mise exec -- cargo-deny --locked --manifest-path packages/sie_server_rust/Cargo.toml --all-features --config deny.toml check
```

For full parity with the hosted Rust test job, run the CI harness:

```bash
mise exec -- uv run --frozen --project . --no-sync python tools/ci/rust_tests.py
```

This harness starts JetStream, enables the NATS publisher regression coverage,
tests the sidecar with its `cloud-storage` feature, and tests the standalone
worker. The narrower Rust commands above remain useful while iterating on one
crate, but they do not replace this full harness.

Changes under `tools/ci`, `tools/mise_tasks`, or `.github/workflows` should run
the broader tooling checks from the Python CI job:

```bash
mise exec -- uv run --frozen --project . --no-sync ruff format --check tools/ci
mise exec -- uv run --frozen --project . --no-sync ruff check --select E,F,I,UP,B tools/ci
mise exec -- uv run --frozen --project . --no-sync ruff check --select E9,F63,F7,F82 tools/mise_tasks
mise exec -- uv run --frozen --project . --no-sync pytest -q tools/ci/tests --ignore tools/ci/tests/test_required_ci.py --ignore tools/ci/tests/test_public_tree.py
```

For workflow, required-check, or public-tree changes, also run the complete
policy gate:

```bash
mise exec -- uv run --frozen --project . pytest -q tools/ci/tests/test_required_ci.py tools/ci/tests/test_public_tree.py
mise exec -- python tools/ci/check_public_tree.py
mise exec -- actionlint -ignore '^unexpected key "queue" for "concurrency" section\. expected one of "cancel-in-progress", "group"$'
```

For documentation and examples, follow the nearest README and run that
example's own tests. There is no repository-wide documentation test task.

Helm validation creates ignored dependency archives and temporarily stages
model and bundle inputs under the chart. The repository tasks clean up the
staged inputs; do not add generated files to the pull request.

## Submit a pull request

Use a [Conventional Commit](https://www.conventionalcommits.org/) style pull
request title. In the description, include:

- the problem and the proposed solution;
- a linked issue when one exists;
- any public compatibility, security, or operational impact; and
- the exact commands and results used to validate the change.

Every pull request runs the complete credential-free CI matrix, including
linting, type checks, unit and integration tests, contracts, packaging, and CPU
container checks. `CI / Required` is the protected aggregate merge gate.
Automated review feedback may also be posted on the pull request.

All paths are owned through `.github/CODEOWNERS`, so an outside contribution is
reviewed by a member of the Core team. Before merge, required CI must pass, the
required approval must be present, and all review conversations must be
resolved. Pushing another commit dismisses earlier approvals, so request review
again after addressing feedback.

## Releases

After a pull request merges, maintainers handle version updates, generated
changelog entries, tags, and publication through the repository's release
automation.
