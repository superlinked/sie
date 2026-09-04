#!/usr/bin/env bash
# Run the RunBatch IPC fixture suite against the Python adapter process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERBOSE=""
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE="-v" ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        py|python|both|all|rs|rust)
            echo "run_parity.sh: language selectors are not supported; this runner executes the Python fixture consumer." >&2
            exit 64
            ;;
        *)
            echo "run_parity.sh: unknown argument: $arg" >&2
            echo "  usage: run_parity.sh [-v]" >&2
            exit 64
            ;;
    esac
done

fixture_count="$(find "$SCRIPT_DIR" -maxdepth 1 -name 'run_batch_*.json' | wc -l | tr -d ' ')"

echo "== RunBatch IPC fixture parity =="
echo "fixtures dir: $SCRIPT_DIR"
echo "fixtures:     $fixture_count"
echo

cd "$REPO_ROOT"
args=(packages/sie_server/tests/test_parity_run_batch.py)
if [[ -n "$VERBOSE" ]]; then
    args+=("$VERBOSE")
fi

if mise run test -- "${args[@]}"; then
    echo
    echo "== parity PASSED ($fixture_count fixtures, python) =="
else
    status=$?
    echo
    echo "== parity FAILED (python exit=$status) ==" >&2
    exit "$status"
fi
