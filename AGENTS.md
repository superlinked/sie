# AGENTS.md — Public SIE contributor guide

## Scope

This repository is the source of truth for the open-source SIE server, SDKs,
gateway, sidecar, configuration service, MCP service, Helm chart, and public
release automation. Keep changes self-contained: a clean clone must build and
test without access to another repository, private registry, or secret.

## Commands

- Use ordinary shell tools for read-only inspection.
- Run project tasks with `mise run <task>` and pass task arguments after `--`.
- Run version-managed executables that are not tasks with `mise exec --`.
- Bootstrap a clean checkout with `./tools/init.sh`.
- Run `mise tasks` before assuming a task exists.

Every pull request runs the public lint, typecheck, unit, integration, and
package/container checks; these are not selected by changed paths. Benchmark
and quality-evaluation jobs are not part of this CI.

The main tasks are `mise run lint`, `mise run typecheck`, `mise run test`,
`mise run test-integrations`, `mise run ts -- build`, `mise run ts -- lint`,
`mise run rust-check`, `mise run rust-test`, and `mise run helm -- lint`.
Build the TypeScript workspace before checking dependent packages in isolation.
The standalone Candle worker uses
`mise exec -- cargo <command> --manifest-path packages/sie_server_rust/Cargo.toml --locked`.

## Development boundaries

- Keep the root Python and pnpm locks authoritative; standalone Rust crates keep
  their checked-in locks.
- Use `sie_sdk.SIEClient` for SIE API examples. Use current `gateway`
  terminology; legacy wire names remain only where compatibility requires it.
- Keep `__init__.py` files empty and imports at module scope except for optional
  dependencies.
- Do not commit secrets, credentials, generated Helm dependencies, staged chart
  model/bundle files, build output, or package archives.
- Trust-boundary changes to ingress, authentication, identity handling, billing,
  or wire protocols require an adversarial security review.
- Use Conventional Commits.

## Releases

Release automation is intentionally fail-closed. Package, image, and chart
builds may run without publication authority; external publication additionally
requires the protected workflow inputs and repository publishing latch described
in `RELEASE.md`. Never bypass version, tag, source-revision, or full-set
verification.

The release-please baseline is 0.7.3. Do not create a substitute baseline tag or
invent its commit SHA. Publication retries reuse the original release run and
artifacts, not a newer-main rebuild.
