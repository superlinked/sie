#!/usr/bin/env bash
set -euo pipefail

export RELEASE_ASSET_FILENAME="${AUDIO_WHEEL_FILENAME:?AUDIO_WHEEL_FILENAME is required}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/upload_native_release_asset.bash" "$@"
