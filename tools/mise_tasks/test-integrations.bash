#!/usr/bin/env bash
#MISE description="Run integration SDK surface tests (Python + TypeScript, no server required)"
#USAGE flag "--python-only" help="Run only the ten Python framework suites"

set -euo pipefail

rc=0
# SIE_SYNC_NO_EXTRA lists extras to leave out, as in tools/mise_tasks/sync.bash.
no_extra=()
# shellcheck disable=SC2086 # word splitting on purpose
for extra in ${SIE_SYNC_NO_EXTRA:-}; do
  no_extra+=(--no-extra "$extra")
done
mise exec -- uv lock --check --project .
mise exec -- uv sync --frozen --project . --all-packages --all-extras ${no_extra[@]+"${no_extra[@]}"} --no-install-package sie-audio-prep

echo "## Python integration SDK tests"
echo ""

# Run each integration separately to avoid conftest.py conflicts
# (each integration has its own conftest with framework-specific fixtures).
# The root pyproject.toml addopts filter out @pytest.mark.integration tests
# so only fast, mocked-SIE-server tests run here.
for dir in integrations/sie_{chroma,cognee,crewai,dspy,haystack,lancedb,langchain,llamaindex,qdrant,weaviate}/tests; do
  name=$(echo "$dir" | cut -d/ -f2)
  echo "--- ${name} ---"
  if ! mise exec -- uv run --frozen --project "${dir%/tests}" --no-sync pytest -c pyproject.toml "${dir}" -q; then
    rc=1
  fi
  echo ""
done

if [[ "${usage_python_only:-}" == "true" ]]; then
  exit "$rc"
fi

echo "## TypeScript integration SDK tests"
echo ""

# Build SDK first (integrations import from dist/)
echo "--- build ---"
if ! mise exec -- pnpm run -r build; then
  echo "FAILED: TypeScript build"
  exit 1
fi
echo ""

echo "--- test ---"
if ! mise exec -- pnpm run -r test; then
  rc=1
fi

exit $rc
