#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v mise >/dev/null 2>&1; then
	echo "mise is required. Install it from https://mise.jdx.dev/getting-started.html and rerun this script."
	exit 1
fi

echo "Trusting mise config..."
mise trust

echo "Installing mise tools and deps..."
# The pinned Node versions share GPG bootstrap state; install them sequentially.
mise install --jobs=1
mise run full-sync

if ! command -v cmake >/dev/null 2>&1; then
	echo "Note: cmake not found. The default sync skips the native audio package, so nothing is needed now."
	echo "To build it from source, install cmake and run: mise exec -- uv sync --frozen --project . --all-packages --all-extras"
fi

echo "Init complete."
