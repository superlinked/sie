#!/usr/bin/env bash
#MISE description="Run SIE server sidecar tests"
#USAGE arg "[filter]" help="Optional cargo test filter"

set -euo pipefail

BASE_ARGS=(test --locked --manifest-path packages/sie_server_sidecar/Cargo.toml)
DEFAULT_ARGS=("${BASE_ARGS[@]}")
CLOUD_ARGS=("${BASE_ARGS[@]}" --features cloud-storage)

if [[ -n "${usage_filter:-}" ]]; then
    DEFAULT_ARGS+=("${usage_filter}")
    CLOUD_ARGS+=("${usage_filter}")
fi

echo "## Running SIE server sidecar tests"
mise exec -- cargo "${DEFAULT_ARGS[@]}"

echo "## Running SIE server sidecar tests with cloud-storage feature"
mise exec -- cargo "${CLOUD_ARGS[@]}"
