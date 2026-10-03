#!/usr/bin/env bash
#MISE description="Build every Rust workspace crate"
#USAGE flag "-r --release" help="Build release binaries"

set -euo pipefail

ARGS=(build --locked --workspace)
if [[ "${usage_release:-}" == "true" ]]; then
	ARGS+=(--release)
fi

echo "## Building the Rust workspace"
mise exec -- cargo "${ARGS[@]}"
