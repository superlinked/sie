#!/usr/bin/env bash
#MISE description="Push one exact versioned image already loaded in Docker"
#USAGE flag "--image <image>" help="Complete immutable image reference"

set -euo pipefail

: "${usage_image:?--image is required}"

case "${usage_image}" in
  *:latest|*:latest-*)
    echo "Refusing to push a floating alias before full-set verification: ${usage_image}" >&2
    exit 2
    ;;
esac

exec docker push "${usage_image}"
