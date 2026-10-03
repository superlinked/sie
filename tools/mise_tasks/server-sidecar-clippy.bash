#!/usr/bin/env bash
#MISE description="Lint SIE server sidecar with clippy"

set -euo pipefail

echo "## Running clippy for SIE server sidecar"
mise exec --no-deps -- cargo clippy --locked --manifest-path packages/sie_server_sidecar/Cargo.toml --all-targets -- -D warnings
