#!/usr/bin/env bash
#MISE description="Serve a remote_only model end to end through the queue on a local kind cluster"

set -euo pipefail

mise run sync
mise exec -- uv run --frozen --project . --no-sync python -m tools.ci.kind_remote_smoke
