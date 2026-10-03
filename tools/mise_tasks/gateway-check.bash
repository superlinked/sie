#!/usr/bin/env bash
#MISE description="Check Rust gateway compiles"
#USAGE flag "-r --release" help="Run cargo check in release mode"

set -euo pipefail

ARGS=(check --locked --all-targets --manifest-path packages/sie_gateway/Cargo.toml)

if [[ "${usage_release:-}" == "true" ]]; then
    ARGS+=("--release")
fi

echo "## Running cargo check for sie-gateway"
mise exec -- cargo "${ARGS[@]}"
