#!/usr/bin/env bash
#MISE description="Run tests for every Rust workspace crate"
#USAGE arg "[filter]" help="Optional cargo test filter"

set -euo pipefail

ARGS=(test --locked --workspace)
if [[ -n "${usage_filter:-}" ]]; then
	ARGS+=("${usage_filter}")
fi

echo "## Testing the Rust workspace"
mise exec -- cargo "${ARGS[@]}"
