#!/usr/bin/env bash
# Seeded gate: OpenCode works the fixture non-interactively, then pytest judges it.
#
#   SIE_BASE_URL=https://... SIE_API_KEY=sk-sie-... ./run_gate.sh   # SIE Cloud (recipes/cloud)
#   ./run_gate.sh laptop                                             # sie-server on http://localhost:8080
#   SIE_BASE_URL=https://... SIE_API_KEY=... ./run_gate.sh cluster   # your gateway (recipes/cluster)
#
# Needs opencode and uv on PATH. Writes run-output/gate-<timestamp>/ with the
# worked copy of the fixture, OpenCode's event log and the pytest verdict.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${1:-cloud}"
CONFIG="$HERE/recipes/$RECIPE/opencode.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "unknown recipe '$RECIPE'; pick laptop, cluster or cloud" >&2
    exit 2
fi
for tool in opencode uv; do
    if ! command -v "$tool" >/dev/null; then
        echo "$tool is not on PATH" >&2
        exit 2
    fi
done
if [[ "$RECIPE" != "laptop" ]]; then
    for var in SIE_BASE_URL SIE_API_KEY; do
        if [[ -z "${!var:-}" ]]; then
            echo "$var is not set" >&2
            exit 2
        fi
    done
fi

OUT="$HERE/run-output/gate-$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$OUT/fixture"
mkdir -p "$OUT"
cp -R "$HERE/fixture" "$WORK"
git -C "$WORK" init -q
TASK="$(cat "$WORK/README.md")"

seeded_tests() {
    (cd "$WORK" && uv tool run --quiet --isolated --no-config --python 3.12 --from 'pytest>=8' pytest -q -c pyproject.toml tests)
}

echo "== before: the seeded tests"
seeded_tests 2>&1 | tail -n 3 || true

echo "== opencode run, recipe $RECIPE"
OPENCODE_STATUS=0
(cd "$WORK" && OPENCODE_CONFIG="$CONFIG" opencode run --auto --format json "$TASK" </dev/null) |
    tee "$OUT/opencode-events.jsonl" || OPENCODE_STATUS="${PIPESTATUS[0]}"
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
