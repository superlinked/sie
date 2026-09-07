#!/usr/bin/env bash
#MISE description="Run tests"
#USAGE flag "-c --coverage" help="Run with coverage report"
#USAGE flag "-i --integration" help="Run integration tests (requires running server)"
#USAGE flag "-d --docker" help="Run the full CPU-container stack (slow, requires local Linux Docker)"
#USAGE flag "-l --collect-only" help="Only collect and list test names, don't run them"
#USAGE flag "-k --filter <filter>" help="Filter tests by name expression (passed to pytest -k)"
#USAGE flag "-m --model <model>" help="Run model tests for a model ID (e.g. BAAI/bge-m3) or 'all' for all models"
#USAGE arg "<path>" help="Optional path to test" default=""

set -euo pipefail

ARGS=()

if [[ "${usage_docker:-}" == "true" ]]; then
    if [[ "${usage_collect_only:-}" == "true" || "${usage_coverage:-}" == "true" || \
          "${usage_integration:-}" == "true" || -n "${usage_filter:-}" || \
          -n "${usage_model:-}" || -n "${usage_path:-}" ]]; then
        echo "ERROR: --docker delegates to 'mise run cpu-stack' and cannot be combined with pytest selectors." >&2
        exit 1
    fi
    exec mise run cpu-stack
fi

if [[ "${usage_collect_only:-}" == "true" ]]; then
    ARGS+=("--collect-only" "-q")
fi

if [[ "${usage_coverage:-}" == "true" ]]; then
    # Bare --cov defers to .coveragerc [run] source_pkgs so never-imported
    # modules stay in the denominator (a --cov=<dir> scan misses src layouts).
    ARGS+=("--cov" "--cov-report=term-missing" "--cov-report=xml")
fi

if [[ -n "${usage_filter:-}" ]]; then
    ARGS+=("-k" "${usage_filter}")
fi

if [[ "${usage_integration:-}" == "true" ]]; then
    ARGS+=("-m" "integration" "-s" "-o" "log_cli=true" "-o" "log_cli_level=INFO")
fi

if [[ -n "${usage_path:-}" ]]; then
    ARGS+=("${usage_path}")
fi

if [[ -n "${usage_model:-}" && -n "${usage_path:-}" ]]; then
    echo "ERROR: --model selects the server model suite and cannot be combined with an explicit test path." >&2
    exit 1
fi

echo "## Checking public workspace lock"
mise exec -- uv lock --check --project .

# A root-project sync installs only the root project. Tests import every public
# workspace member, so fresh environments must request them explicitly instead
# of relying on packages left in an existing environment.
echo "## Syncing public workspace"
mise exec -- uv sync --frozen --project . --all-packages --no-install-package sie-audio-prep

if [[ -n "${usage_model:-}" ]]; then
    ARGS+=("-m" "model")
    if [[ "${usage_model}" == "all" ]]; then
        ARGS+=("packages/sie_server/tests/test_all_models.py")
    else
        selected_tests=$(mise exec -- python tools/mise_tasks/model_test_selection.py "${usage_model}")
        model_node_ids=()
        while IFS= read -r node_id; do
            model_node_ids+=("${node_id}")
        done <<<"${selected_tests}"
        ARGS+=("${model_node_ids[@]}")
    fi
fi

echo "## Running public workspace tests"
if [[ ${#ARGS[@]} -eq 0 ]]; then
    mise exec -- uv run --frozen --project . --no-sync pytest -c pyproject.toml
else
    mise exec -- uv run --frozen --project . --no-sync pytest -c pyproject.toml "${ARGS[@]}"
fi
