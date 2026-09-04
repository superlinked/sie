#!/usr/bin/env bash
#MISE description="Load, validate, and publish one retained release image archive"
#USAGE flag "--image <image>" help="Complete immutable image reference"
#USAGE flag "--archive-dir <path>" help="Retained image archive and provenance directory"
#USAGE flag "--version <version>" help="Stable release version without v"
#USAGE flag "--source-revision <sha>" help="Exact release source commit"
#USAGE flag "--run-id <id>" help="Original Actions run ID"

set -euo pipefail

exec mise run docker -- publish \
  --image "${usage_image:?--image is required}" \
  --archive-dir "${usage_archive_dir:?--archive-dir is required}" \
  --version "${usage_version:?--version is required}" \
  --source-revision "${usage_source_revision:?--source-revision is required}" \
  --run-id "${usage_run_id:?--run-id is required}"
