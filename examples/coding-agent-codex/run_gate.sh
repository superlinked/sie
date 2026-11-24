#!/usr/bin/env bash
# Seeded gate: Codex CLI works the shared fixture headless through SIE's /v1/responses,
# then pytest judges it.
#
#   SIE_BASE_URL=https://... SIE_API_KEY=sk-sie-... ./run_gate.sh      # SIE Cloud (recipes/cloud.sh)
#   SIE_BASE_URL=https://... SIE_API_KEY=... SIE_MODEL=... \
#     SIE_CONTEXT_TOKENS=... ./run_gate.sh cluster  # your gateway (recipes/cluster.sh)
#   ./run_gate.sh laptop     # sie-server on this machine (recipes/laptop.sh)
#
# Needs npx and uv on PATH. The recipe runs Codex CLI 0.167.0 through npx. The fixture is the
# OpenCode example's, unchanged. Writes run-output/gate-<timestamp>-<suffix>/ with the worked
# copy of the fixture, Codex's config and JSON events, a summary of the run and the pytest
# verdict.
#
# Codex runs with its own CODEX_HOME in the run directory, so no user config, profile or
# session takes part. Its commands run in the workspace-write sandbox, which allows no network
# and writes only in the worked copy and the temporary directory, so the worked copy's
# environment is synced before the run and uv's cache for the run is inside it. A run still
# going after GATE_DEADLINE_SECONDS (default 1800) is stopped and fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="$HERE/../coding-agent-opencode/fixture"
RECIPE="${1:-cloud}"
ENV_FILE="$HERE/recipes/$RECIPE.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "unknown recipe '$RECIPE'; pick cloud, cluster or laptop" >&2
    exit 2
fi
for tool in npx uv; do
    if ! command -v "$tool" >/dev/null; then
        echo "$tool is not on PATH" >&2
        exit 2
    fi
done
GATE_DEADLINE_SECONDS="${GATE_DEADLINE_SECONDS:-1800}"
if [[ ! "$GATE_DEADLINE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GATE_DEADLINE_SECONDS must be a positive whole number" >&2
    exit 2
fi
case "$RECIPE" in
    cloud) REQUIRED=(SIE_BASE_URL SIE_API_KEY) ;;
    cluster) REQUIRED=(SIE_BASE_URL SIE_API_KEY SIE_MODEL SIE_CONTEXT_TOKENS) ;;
    *) REQUIRED=() ;;
esac
for var in ${REQUIRED[@]+"${REQUIRED[@]}"}; do
    if [[ -z "${!var:-}" ]]; then
        echo "$var is not set" >&2
        exit 2
    fi
done

mkdir -p "$HERE/run-output"
OUT="$(mktemp -d "$HERE/run-output/gate-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
WORK="$OUT/fixture"
cp -R "$FIXTURE" "$WORK"
rm -rf "$WORK/.venv" "$WORK/uv.lock"
git -C "$WORK" init -q
TASK="$(cat "$WORK/README.md")"

seeded_tests() {
    (cd "$WORK" && unset PYTEST_ADDOPTS PYTEST_PLUGINS && uv tool run --quiet --isolated --no-config --python 3.12 --from 'pytest>=8' pytest -q -c pyproject.toml tests)
}

echo "== before: the seeded tests"
seeded_tests 2>&1 | tail -n 3 || true

echo "== the worked copy's environment, for the tests Codex runs in its sandbox"
(cd "$WORK" && unset VIRTUAL_ENV && UV_LINK_MODE=copy uv sync --quiet --python 3.12)

echo "== codex run, recipe $RECIPE, up to $GATE_DEADLINE_SECONDS s"
set -m
{
    (
        cd "$WORK"
        export SIE_CODEX_HOME="$OUT/codex-home"
        source "$ENV_FILE"
        export UV_CACHE_DIR="$WORK/.uv-cache"
        codex exec --json --skip-git-repo-check --sandbox workspace-write "$TASK" </dev/null
    ) | tee "$OUT/codex-events.jsonl"
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
CODEX_STATUS=0
wait "$RUN" || CODEX_STATUS=$?
[[ -e "$OUT/deadline-passed" ]] || kill -- "-$WATCHDOG" 2>/dev/null || true
wait "$WATCHDOG" 2>/dev/null || true
trap - EXIT INT TERM
if [[ -e "$OUT/deadline-passed" ]]; then
    echo "GATE FAIL: codex run passed its ${GATE_DEADLINE_SECONDS}s deadline and was stopped ($OUT)"
    exit 1
fi
if [[ "$CODEX_STATUS" -ne 0 ]]; then
    echo "GATE FAIL: codex run exited $CODEX_STATUS ($OUT)"
    exit 1
fi

echo "== the run"
if ! uv run --quiet --no-project --python 3.12 python "$HERE/gate_verdict.py" "$OUT/codex-events.jsonl" | tee "$OUT/run-summary.json"; then
    echo "GATE FAIL: the Codex run did not finish cleanly ($OUT)"
    exit 1
fi

echo "== after: the seeded tests must be untouched"
if ! diff -r -x __pycache__ "$FIXTURE/tests" "$WORK/tests" || ! diff "$FIXTURE/pyproject.toml" "$WORK/pyproject.toml"; then
    echo "GATE FAIL: the tests or pyproject.toml were changed ($OUT)"
    exit 1
fi
if find "$WORK" -name conftest.py -not -path '*/.venv/*' -not -path '*/.uv-cache/*' | grep -q .; then
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
