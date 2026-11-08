#!/usr/bin/env bash
# Prepare a runner so `pnpm install` builds its native addons locally instead
# of fetching prebuilt binaries from GitHub release assets.
#
# sharp 0.32 (pulled in by chromadb-default-embed) downloads libvips and its
# own prebuilt addon from GitHub releases unless a global libvips is present;
# then its install script refuses the download and runs `node-gyp rebuild`.
# The runner image has no node-gyp on PATH, so install it into the mise node
# tree, and point node-gyp at that tree's headers so it does not fetch them
# from nodejs.org either.
set -euo pipefail

NODE_GYP_VERSION=12.4.0

sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends libvips-dev
mise exec -- npm install --global --no-fund --no-audit "node-gyp@${NODE_GYP_VERSION}"
node_dir="$(mise where node)"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "npm_config_nodedir=${node_dir}" >> "$GITHUB_ENV"
fi
mise exec -- node-gyp --version
