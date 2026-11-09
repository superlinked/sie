#!/usr/bin/env bash
#MISE description="Sync dependencies on local machine"

set -eu -o pipefail

# sie-audio-prep is a maturin workspace member whose source build needs cmake
# (opusic-sys bundles libopus). Skip installing it by default so machines
# without a native toolchain can sync, test, and typecheck.
# Opt in with: mise exec -- uv sync --frozen --project . --all-packages --all-extras
# (requires cmake).
#
# SIE_SYNC_NO_EXTRA lists extras to leave out, separated by spaces, for example
# "flash-attn" on a machine without a GPU. CI sets it; see .github/workflows/ci.yml.
no_extra=()
# shellcheck disable=SC2086 # word splitting on purpose
for extra in ${SIE_SYNC_NO_EXTRA:-}; do
  no_extra+=(--no-extra "$extra")
done
mise exec -- uv lock --check --project .
mise exec -- uv sync --project . --all-packages --all-extras ${no_extra[@]+"${no_extra[@]}"} --frozen --no-install-package sie-audio-prep
