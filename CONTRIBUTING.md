# Contributing to SIE

Thank you for contributing. SIE accepts focused changes that keep the public
repository independently buildable and preserve compatibility across the
Python, TypeScript, Rust, wire-contract, and Helm surfaces.

## Set up a checkout

Install [mise](https://mise.jdx.dev/getting-started.html), clone the repository,
and run:

```bash
./tools/init.sh
```

The bootstrap installs the pinned toolchain, synchronizes the root Python lock,
and installs every TypeScript workspace from the root pnpm lock. It does not
need package-registry credentials.

## Validate a change

Start with the checks that own the files you changed, then run the wider gates
before requesting review:

```bash
mise run lint
mise run typecheck
mise run test

mise run ts -- lint
mise run ts -- typecheck
mise run ts -- build
mise run ts -- test

mise run rust-fmt -- --check
mise run rust-check
mise run rust-clippy
mise run rust-test

mise exec -- python tools/check_ipc_types_parity.py
mise exec -- python tools/check_response_chunk_protocol.py

mise run helm -- dependencies
mise run helm -- lint --set payloadStore.enabled=false
mise run helm -- template --set payloadStore.enabled=false
```

The standalone Candle worker is outside the root Rust workspace:

```bash
mise exec -- cargo fmt --manifest-path packages/sie_server_rust/Cargo.toml -- --check
mise exec -- cargo check --manifest-path packages/sie_server_rust/Cargo.toml --locked --all-targets
mise exec -- cargo clippy --manifest-path packages/sie_server_rust/Cargo.toml --locked --all-targets -- -D warnings
mise exec -- cargo test --manifest-path packages/sie_server_rust/Cargo.toml --locked
```

Native release-wheel builds additionally use the repository's pinned Zig
toolchain and run on Linux x86_64. The public release workflow is the authority
for the exact manylinux asset; it is not part of the PyPI package matrix.

The public Docker task resolves the checked-in release matrix and requires the
full source commit for every build. A CPU-only Candle image can be built on a
normal Docker host without publication credentials:

```bash
mise run docker -- matrix
mise run docker -- build-service \
  --registry local \
  --version 0.7.2 \
  --service sie-server-rust-cpu \
  --source-revision <40-character-git-sha>
```

Helm dependency archives and the model/bundle files temporarily staged under
the chart are generated and ignored. Do not add them to a commit.

## Pull requests

Keep diffs minimal, explain compatibility or security implications, and include
the exact validation evidence. Use a Conventional Commit title. Tests must be
deterministic and must not depend on credentials, external writes, or mutable
package versions.

See [RELEASE.md](RELEASE.md) for the public version and publication contract.
