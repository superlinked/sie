#!/usr/bin/env bash
#MISE description="Regenerate the static OpenAPI spec"
#USAGE flag "--server-only" help="Regenerate only the Python server spec"
#USAGE flag "--gateway-only" help="Regenerate only the Rust gateway spec"
set -euo pipefail

if [[ "${usage_server_only:-}" == "true" && "${usage_gateway_only:-}" == "true" ]]; then
    echo "error: --server-only and --gateway-only are mutually exclusive" >&2
    exit 2
fi

if [[ "${usage_gateway_only:-}" != "true" ]]; then
    mise exec -- uv lock --check --project .
    mise exec -- uv run --frozen --project . --package sie-server \
        sie-server openapi --output packages/sie_server/openapi.json
fi
if [[ "${usage_server_only:-}" != "true" ]]; then
    mise exec -- cargo run --quiet --manifest-path packages/sie_gateway/Cargo.toml -- openapi --output packages/sie_gateway/openapi.json
fi
