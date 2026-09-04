#!/usr/bin/env bash
set -euo pipefail

test ! -e .venv
test ! -e node_modules
bootstrap_checksums=$(mktemp)
trap 'rm -f "$bootstrap_checksums"' EXIT
git ls-files -z '*uv.lock' '*pnpm-lock.yaml' '*Cargo.lock' | xargs -0 sha256sum > "$bootstrap_checksums"
./tools/init.sh
sha256sum --check "$bootstrap_checksums"
git diff --exit-code -- '*uv.lock' '*pnpm-lock.yaml' '*Cargo.lock'
