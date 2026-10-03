#!/usr/bin/env bash
#MISE description="Format SIE server sidecar"
#USAGE flag "-c --check" help="Check formatting without writing changes (inverse of default)"

set -euo pipefail

if [[ "${usage_check:-}" == "true" ]]; then
    echo "## Checking SIE server sidecar formatting"
    mise exec --no-deps -- cargo fmt --manifest-path packages/sie_server_sidecar/Cargo.toml --all --check
else
    echo "## Formatting SIE server sidecar"
    mise exec --no-deps -- cargo fmt --manifest-path packages/sie_server_sidecar/Cargo.toml --all
fi
