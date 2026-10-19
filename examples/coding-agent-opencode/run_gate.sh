#!/usr/bin/env bash
# Seeded gate: OpenCode works the fixture non-interactively, then pytest judges it.
#
#   SIE_BASE_URL=https://... SIE_API_KEY=sk-sie-... ./run_gate.sh   # SIE Cloud (recipes/cloud)
#   ./run_gate.sh laptop                                             # sie-server on http://localhost:8080
#   SIE_BASE_URL=https://... SIE_API_KEY=... ./run_gate.sh cluster   # your gateway (recipes/cluster)
#
# Needs opencode and uv on PATH. Writes run-output/gate-<timestamp>-<suffix>/ with the
# worked copy of the fixture, OpenCode's event log and the pytest verdict.
#
# The run is bounded. OpenCode's build agent gets GATE_STEPS steps (default 50), after which
# OpenCode tells the model to answer in text. A run still going after GATE_DEADLINE_SECONDS
# (default 1800) is stopped, and the gate fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${1:-cloud}"
CONFIG="$HERE/recipes/$RECIPE/opencode.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "unknown recipe '$RECIPE'; pick laptop, gpu, openrouter, cluster or cloud" >&2
    exit 2
fi
for tool in opencode uv; do
    if ! command -v "$tool" >/dev/null; then
        echo "$tool is not on PATH" >&2
        exit 2
    fi
done
GATE_STEPS="${GATE_STEPS:-50}"
GATE_DEADLINE_SECONDS="${GATE_DEADLINE_SECONDS:-1800}"
for var in GATE_STEPS GATE_DEADLINE_SECONDS; do
    if [[ ! "${!var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$var must be a positive whole number" >&2
        exit 2
    fi
done
if grep -q '{env:SIE_' "$CONFIG"; then
    for var in SIE_BASE_URL SIE_API_KEY; do
        if [[ -z "${!var:-}" ]]; then
            echo "$var is not set" >&2
            exit 2
        fi
    done
fi

mkdir -p "$HERE/run-output"
OUT="$(mktemp -d "$HERE/run-output/gate-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
WORK="$OUT/fixture"
cp -R "$HERE/fixture" "$WORK"
git -C "$WORK" init -q
TASK="$(cat "$WORK/README.md")"

seeded_tests() {
    (cd "$WORK" && unset PYTEST_ADDOPTS PYTEST_PLUGINS && uv tool run --quiet --isolated --no-config --python 3.12 --from 'pytest>=8' pytest -q -c pyproject.toml tests)
}

echo "== before: the seeded tests"
seeded_tests 2>&1 | tail -n 3 || true

echo "== opencode run, recipe $RECIPE, up to $GATE_STEPS steps and $GATE_DEADLINE_SECONDS s"
STEP_BUDGET="{\"agent\":{\"build\":{\"steps\":$GATE_STEPS}}}"
set -m
{
    (cd "$WORK" && OPENCODE_CONFIG="$CONFIG" OPENCODE_CONFIG_CONTENT="$STEP_BUDGET" \
        opencode run --agent build --auto --format json "$TASK" </dev/null) | tee "$OUT/opencode-events.jsonl"
} &
RUN=$!
{
    sleep "$GATE_DEADLINE_SECONDS"
    touch "$OUT/deadline-passed"
    kill -TERM -- "-$RUN"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        kill -0 -- "-$RUN" || exit 0
    done
    kill -KILL -- "-$RUN"
} >/dev/null 2>&1 &
WATCHDOG=$!
set +m
trap 'kill -- "-$RUN" "-$WATCHDOG" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
OPENCODE_STATUS=0
wait "$RUN" || OPENCODE_STATUS=$?
[[ -e "$OUT/deadline-passed" ]] || kill -- "-$WATCHDOG" 2>/dev/null || true
wait "$WATCHDOG" 2>/dev/null || true
trap - EXIT INT TERM
if [[ -e "$OUT/deadline-passed" ]]; then
    echo "GATE FAIL: opencode run passed its ${GATE_DEADLINE_SECONDS}s deadline and was stopped ($OUT)"
    exit 1
fi
if [[ "$OPENCODE_STATUS" -ne 0 ]]; then
    echo "GATE FAIL: opencode run exited $OPENCODE_STATUS ($OUT)"
    exit 1
fi

echo "== after: the seeded tests must be untouched"
if ! diff -r -x __pycache__ "$HERE/fixture/tests" "$WORK/tests" || ! diff "$HERE/fixture/pyproject.toml" "$WORK/pyproject.toml"; then
    echo "GATE FAIL: the tests or pyproject.toml were changed ($OUT)"
    exit 1
fi
if find "$WORK" -name conftest.py -not -path '*/.venv/*' | grep -q .; then
    echo "GATE FAIL: a conftest.py was added ($OUT)"
    exit 1
fi

echo "== after: pytest in a fresh environment, pinned to the seeded config and tests"
if seeded_tests 2>&1 | tee "$OUT/pytest.txt"; then
    echo "GATE PASS ($OUT)"
else
    echo "GATE FAIL ($OUT)"
    exit 1
fi
