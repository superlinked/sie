#!/usr/bin/env bash
#MISE description="Build and exercise the full CPU-container stack"

set -euo pipefail

mise run sync
mise exec -- uv run --frozen --project . --no-sync python -m tools.ci.cpu_stack_smoke
