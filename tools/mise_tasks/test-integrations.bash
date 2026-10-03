#!/usr/bin/env bash
#MISE description="Run integration SDK surface tests (Python + TypeScript, no server required)"
#USAGE flag "--python-only" help="Run only the nine Python framework suites"

set -euo pipefail

rc=0
mise exec -- uv lock --check --project .
mise exec -- uv sync --frozen --project . --all-packages --all-extras --no-install-package sie-audio-prep

echo "## Python integration SDK tests"
echo ""

# Run each integration separately to avoid conftest.py conflicts
# (each integration has its own conftest with framework-specific fixtures).
# The root pyproject.toml addopts filter out @pytest.mark.integration tests
# so only fast, mocked-SIE-server tests run here.
for dir in integrations/sie_{chroma,crewai,dspy,haystack,lancedb,langchain,llamaindex,qdrant,weaviate}/tests; do
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
