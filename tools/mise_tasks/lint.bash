#!/usr/bin/env bash
#MISE description="Lint Python sources"
#USAGE flag "-f --fix" help="Fix issues instead of just checking"

set -euo pipefail

PUBLIC_PATHS=(
    packages/sie_audio_prep
    packages/sie_config
    packages/sie_mcp
    packages/sie_sdk
    packages/sie_server
    integrations
)

mise exec -- uv lock --check --project .

if [[ "${usage_fix:-}" == "true" ]]; then
    echo "## Running ruff fix and format"
    mise exec -- uv run --frozen --project . ruff check --fix "${PUBLIC_PATHS[@]}"
    mise exec -- uv run --frozen --project . ruff format "${PUBLIC_PATHS[@]}"
else
    echo "## Running ruff check and format check"
    mise exec -- uv run --frozen --project . ruff format --check "${PUBLIC_PATHS[@]}"
    mise exec -- uv run --frozen --project . ruff check "${PUBLIC_PATHS[@]}"
fi
