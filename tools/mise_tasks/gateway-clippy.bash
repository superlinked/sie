#!/usr/bin/env bash
#MISE description="Lint Rust gateway with clippy"

set -euo pipefail

echo "## Running clippy for sie-gateway"
mise exec --no-deps -- cargo clippy --locked --manifest-path packages/sie_gateway/Cargo.toml --all-targets -- -D warnings
