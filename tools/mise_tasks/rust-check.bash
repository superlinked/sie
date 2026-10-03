#!/usr/bin/env bash
#MISE description="Check every Rust workspace crate"
#USAGE flag "-r --release" help="Run cargo check in release mode"

set -euo pipefail

ARGS=(check --locked --workspace --all-targets)
if [[ "${usage_release:-}" == "true" ]]; then
	ARGS+=(--release)
fi

echo "## Checking the Rust workspace"
mise exec -- cargo "${ARGS[@]}"
