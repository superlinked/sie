#!/usr/bin/env bash
#MISE description="Type check Python sources with ty (Astral)"
#USAGE arg "[path]" help="Path to check (default: packages/)"

set -euo pipefail

if [[ -n "${usage_path:-}" ]]; then
    PATHS_TO_CHECK=("${usage_path}")
else
    PATHS_TO_CHECK=(
        packages/sie_config
        packages/sie_mcp
        packages/sie_sdk
        packages/sie_server
    )
fi

echo "## Running ty type checker on ${PATHS_TO_CHECK[*]}"

# Install every member of the explicitly public workspace for import checking.
# sie-audio-prep is skipped: installing it compiles the native crate (needs
# cmake) and ty cannot see into the binary module anyway.
# `uv run` has no --no-install-package, so sync explicitly first (--inexact
# preserves uv run's keep-extraneous behavior), then run without re-syncing.
mise exec --no-deps python@3.12.12 uv@0.5.31 -- uv lock --check --project .
mise exec --no-deps python@3.12.12 uv@0.5.31 -- uv sync --project . --all-packages --frozen --inexact --no-install-package sie-audio-prep
mise exec --no-deps python@3.12.12 uv@0.5.31 -- uv run --project . --no-sync ty check "${PATHS_TO_CHECK[@]}"
