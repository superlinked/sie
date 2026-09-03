#!/usr/bin/env bash
#MISE description="Build and verify public SIE release images"
#USAGE arg "<command>" help="Command: matrix, expected, build-server, build-service, load, publish, verify, alias"
#USAGE arg "[args]..." help="Arguments for the selected Docker command"

set -euo pipefail

mise exec -- uv lock --check --project .
exec mise exec -- uv run --frozen --project . python -m tools.mise_tasks.docker_task "$@"
