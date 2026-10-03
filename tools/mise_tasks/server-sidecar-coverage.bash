#!/usr/bin/env bash
#MISE description="Run SIE server sidecar tests with cargo-llvm-cov coverage (both feature configs, merged report)"
#USAGE flag "--lcov" help="Emit LCOV report to target/llvm-cov/lcov.info (in addition to the text table)"

set -euo pipefail

# Mirrors server-sidecar-test.bash: the suite runs twice (default features and
# --features cloud-storage). --no-report keeps the raw profiles so the final
# `report` merges both configurations into one table / lcov file.

SIDECAR_MANIFEST="packages/sie_server_sidecar/Cargo.toml"
SIDECAR_TARGET="target"  # workspace root target directory

echo "## Running SIE server sidecar tests with coverage (default features)"
mise exec -- cargo-llvm-cov llvm-cov --no-report --locked --manifest-path "${SIDECAR_MANIFEST}"

echo "## Running SIE server sidecar tests with coverage (cloud-storage feature)"
mise exec -- cargo-llvm-cov llvm-cov --no-report --locked --manifest-path "${SIDECAR_MANIFEST}" --features cloud-storage

if [[ "${usage_lcov:-}" == "true" ]]; then
    mkdir -p "${SIDECAR_TARGET}/llvm-cov"
    mise exec -- cargo-llvm-cov llvm-cov report --manifest-path "${SIDECAR_MANIFEST}" --lcov --output-path "${SIDECAR_TARGET}/llvm-cov/lcov.info"
fi

echo "## SIE server sidecar coverage report"
mise exec -- cargo-llvm-cov llvm-cov report --manifest-path "${SIDECAR_MANIFEST}"
