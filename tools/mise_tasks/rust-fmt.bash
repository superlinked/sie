#!/usr/bin/env bash
#MISE description="Format every Rust workspace crate"
#USAGE flag "-c --check" help="Check formatting without writing changes"

set -euo pipefail

ARGS=(fmt --all)
if [[ "${usage_check:-}" == "true" ]]; then
	ARGS+=(--check)
fi

echo "## Formatting the Rust workspace"
mise exec -- cargo "${ARGS[@]}"
