#!/usr/bin/env bash
#MISE description="Build SIE server sidecar"
#USAGE flag "-r --release" help="Build the release binary"

set -euo pipefail

ARGS=(build --locked --manifest-path packages/sie_server_sidecar/Cargo.toml)

if [[ "${usage_release:-}" == "true" ]]; then
    ARGS+=("--release")
fi

echo "## Building SIE server sidecar"
mise exec -- cargo "${ARGS[@]}"
