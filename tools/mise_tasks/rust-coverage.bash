#!/usr/bin/env bash
#MISE description="Generate test coverage for the Rust workspace"
#USAGE flag "--lcov" help="Emit LCOV output to target/llvm-cov/lcov.info"
#USAGE flag "--html" help="Emit HTML output under target/llvm-cov/html/"
#USAGE arg "[filter]" help="Optional cargo test filter"

set -euo pipefail

ARGS=(llvm-cov --locked --workspace --all-features)
if [[ "${usage_lcov:-}" == "true" ]]; then
	mkdir -p target/llvm-cov
	ARGS+=(--lcov --output-path target/llvm-cov/lcov.info)
fi
if [[ "${usage_html:-}" == "true" ]]; then
	ARGS+=(--html --output-dir target/llvm-cov/html)
fi
if [[ -n "${usage_filter:-}" ]]; then
	ARGS+=(-- "${usage_filter}")
fi

echo "## Running coverage for the Rust workspace"
mise exec cargo:cargo-llvm-cov -- cargo-llvm-cov "${ARGS[@]}"
