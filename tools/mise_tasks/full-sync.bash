#!/usr/bin/env bash
#MISE description="Run full local developer environment sync"

set -eu -o pipefail

mise run sync
mise exec -- pnpm install --frozen-lockfile
