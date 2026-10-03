#!/usr/bin/env bash
#MISE description="Lint every Rust workspace crate with clippy"

set -euo pipefail

echo "## Running clippy for the Rust workspace"
mise exec -- cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
