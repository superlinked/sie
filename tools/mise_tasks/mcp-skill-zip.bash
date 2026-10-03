#!/usr/bin/env bash
#MISE description="Package the claude.ai Superlinked Agent Skill as an uploadable ZIP"
set -euo pipefail

mise exec -- uv lock --check --project .
mise exec -- uv run --frozen --project . --package sie-mcp python -m sie_mcp.cli skill-zip "$@"
