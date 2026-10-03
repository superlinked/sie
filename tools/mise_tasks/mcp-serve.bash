#!/usr/bin/env bash
#MISE description="Start the SIE MCP edge service"
set -euo pipefail

# Scope execution to the explicit public workspace package so its optional MCP
# dependencies and sources are installed in a fresh development environment.
mise exec -- uv lock --check --project .
mise exec -- uv run --frozen --project . --package sie-mcp python -m sie_mcp.cli serve "$@"
