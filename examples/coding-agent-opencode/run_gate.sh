#!/usr/bin/env bash
# Seeded gate: OpenCode works the fixture non-interactively, then pytest judges it.
#
#   SIE_API_KEY=sk-sie-... ./run_gate.sh          # SIE Cloud (recipes/cloud)
#   ./run_gate.sh laptop                           # sie-server on http://localhost:8080
#   SIE_API_KEY=... ./run_gate.sh cluster          # your gateway (recipes/cluster)
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
if [[ "$RECIPE" != "laptop" && -z "${SIE_API_KEY:-}" ]]; then
    echo "SIE_API_KEY is not set" >&2
    exit 2
fi

OUT="$HERE/run-output/gate-$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$OUT/fixture"
mkdir -p "$OUT"
cp -R "$HERE/fixture" "$WORK"
git -C "$WORK" init -q
TASK="$(cat "$WORK/README.md")"

echo "== before: the seeded tests"
(cd "$WORK" && uv run --quiet pytest -q 2>&1 | tail -n 3) || true

echo "== opencode run, recipe $RECIPE"
(cd "$WORK" && OPENCODE_CONFIG="$CONFIG" opencode run --auto --format json "$TASK" </dev/null | tee "$OUT/opencode-events.jsonl")

echo "== after: pytest on the worked copy"
if (cd "$WORK" && uv run --quiet pytest -q 2>&1 | tee "$OUT/pytest.txt"); then
    echo "GATE PASS ($OUT)"
else
    echo "GATE FAIL ($OUT)"
    exit 1
fi
