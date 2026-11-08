#!/usr/bin/env bash
# Place the pinned mise binary where jdx/mise-action looks for it, so the action
# skips its own download from GitHub release assets. mise.jdx.dev is mise's
# release CDN and serves every tagged version; the hash is from the release's
# SHASUMS256.txt. The version must match the
# `version:` given to every mise-action step; tools/ci/tests check that.
set -euo pipefail

MISE_VERSION=2026.5.5
MISE_SHA256_LINUX_X64=3aaab5c05a8a94a93b42b4f581779bbd5c44ddb251e7f3639fc671ec5c6aab8a

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    target=linux-x64
    expected="$MISE_SHA256_LINUX_X64"
    ;;
  *)
    echo "mise_binary.bash: no pinned hash for $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

bin_dir="${MISE_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/mise}/bin"
mkdir -p "$bin_dir"
cdn="${MISE_CDN:-https://mise.jdx.dev}"
if ! curl -fsSL --retry 3 --retry-all-errors --retry-delay 3 \
  "${cdn}/v${MISE_VERSION}/mise-v${MISE_VERSION}-${target}" \
  -o "$bin_dir/mise.download"; then
  # Not fatal: without the binary in place, mise-action downloads it from
  # GitHub as before, so two independent sources have to fail together.
  echo "::warning::mise ${MISE_VERSION} could not be fetched from ${cdn}; mise-action will download it instead"
  rm -f "$bin_dir/mise.download"
  exit 0
fi
echo "${expected}  $bin_dir/mise.download" | sha256sum --check --quiet
chmod +x "$bin_dir/mise.download"
mv "$bin_dir/mise.download" "$bin_dir/mise"
"$bin_dir/mise" version
