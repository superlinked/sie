#!/usr/bin/env bash
#MISE description="Build a Superlinked MCP plugin install pack for a hosted or self-hosted cluster"
set -euo pipefail

mise exec -- uv lock --check --project .
mise exec -- uv run --frozen --project . --package sie-mcp python -m sie_mcp.cli plugin-pack "$@"
