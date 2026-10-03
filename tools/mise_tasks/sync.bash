#!/usr/bin/env bash
#MISE description="Sync dependencies on local machine"

set -eu -o pipefail

# sie-audio-prep is a maturin workspace member whose source build needs cmake
# (opusic-sys bundles libopus). Skip installing it by default so machines
# without a native toolchain can sync, test, and typecheck.
# Opt in with: mise exec -- uv sync --frozen --project . --all-packages --all-extras
# (requires cmake).
mise exec -- uv lock --check --project .
mise exec -- uv sync --project . --all-packages --all-extras --frozen --no-install-package sie-audio-prep
