#!/usr/bin/env bash
#MISE description="Build Rust gateway"
#USAGE flag "-r --release" help="Build the release binary"

set -euo pipefail

ARGS=(build --locked --manifest-path packages/sie_gateway/Cargo.toml)

if [[ "${usage_release:-}" == "true" ]]; then
    ARGS+=("--release")
fi

echo "## Building sie-gateway"
mise exec -- cargo "${ARGS[@]}"
