#!/usr/bin/env bash
#MISE description="Check SIE server sidecar compiles"
#USAGE flag "-r --release" help="Run cargo check in release mode"

set -euo pipefail

ARGS=(check --locked --all-targets --manifest-path packages/sie_server_sidecar/Cargo.toml)

if [[ "${usage_release:-}" == "true" ]]; then
    ARGS+=("--release")
fi

echo "## Running cargo check for SIE server sidecar"
mise exec -- cargo "${ARGS[@]}"
