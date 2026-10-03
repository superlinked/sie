#!/usr/bin/env bash
#MISE description="Format Rust gateway"
#USAGE flag "-c --check" help="Check formatting without writing changes (inverse of default)"

set -euo pipefail

if [[ "${usage_check:-}" == "true" ]]; then
    echo "## Checking sie-gateway formatting"
    mise exec --no-deps -- cargo fmt --manifest-path packages/sie_gateway/Cargo.toml --all --check
else
    echo "## Formatting sie-gateway"
    mise exec --no-deps -- cargo fmt --manifest-path packages/sie_gateway/Cargo.toml --all
fi
