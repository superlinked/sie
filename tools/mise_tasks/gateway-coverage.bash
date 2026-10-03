#!/usr/bin/env bash
#MISE description="Generate test coverage report for sie-gateway with cargo-llvm-cov"
#USAGE flag "--lcov" help="Emit LCOV report to target/llvm-cov/lcov.info (in addition to the text table)"
#USAGE flag "--html" help="Emit HTML report under target/llvm-cov/html/"
#USAGE arg "[filter]" help="Optional cargo test filter"

set -euo pipefail

# cargo-llvm-cov needs the canonical `llvm-tools` rustup component, which
# mise.toml declares alongside the rust toolchain pin so `mise install`
# provisions it on every dev machine and CI runner.
#
# Default invocation prints a per-file coverage table with a TOTAL row at
# the bottom -- exactly what we want piped into CI logs, no extra flags
# needed. (`--summary-only` is intentionally not exposed: cargo-llvm-cov
# only honors it together with --json/--lcov/--cobertura, and silently
# ignores it otherwise, which is misleading.)

GATEWAY_TARGET="target"  # workspace root target directory
ARGS=(llvm-cov --locked --manifest-path packages/sie_gateway/Cargo.toml --all-features)

if [[ "${usage_lcov:-}" == "true" ]]; then
    mkdir -p "${GATEWAY_TARGET}/llvm-cov"
    ARGS+=(--lcov --output-path "${GATEWAY_TARGET}/llvm-cov/lcov.info")
fi
if [[ "${usage_html:-}" == "true" ]]; then
    ARGS+=(--html --output-dir "${GATEWAY_TARGET}/llvm-cov/html")
fi
if [[ -n "${usage_filter:-}" ]]; then
    ARGS+=(-- "${usage_filter}")
fi

echo "## Running cargo-llvm-cov for sie-gateway"
mise exec cargo:cargo-llvm-cov -- cargo-llvm-cov "${ARGS[@]}"
