#!/usr/bin/env bash
#MISE description="Mac (Apple-Silicon) end-to-end smoke: default (embed/rerank) + sglang (generation) bundles + OpenAI surfaces"
#
# The Apple-Silicon acceptance gate. The bundles are device-agnostic, so the Mac
# story is the same as Linux: `serve` (default bundle) serves embeddings +
# reranking on torch-MPS, and `mise run serve -- -b sglang` serves generation on MLX. Their
# transformers pins differ, so they are SEPARATE servers — this smoke runs each
# in turn on the same port:
#   Phase 0 (fake):    weightless surface contract (zero downloads) —
#                      /docs, embeddings (+determinism), rerank, score,
#                      generate (blocking + streaming)
#   Phase 1 (default): /docs, /v1/embeddings (+throughput), /v1/rerank, /v1/score
#   Phase 2 (sglang):  /v1/generate (blocking + streaming), /v1/chat/completions
#                      (streaming + non-streaming, +MLX tok/s)
# Tears each server down before the next and exits non-zero on any failure.
#
# This script owns only the Mac-specific server lifecycle (start/teardown of the
# two local servers, phase sequencing, MLX/MPS timeouts); the probes themselves
# are the shared endpoint-agnostic primitives in
# tools/mise_tasks/smoketest.py. MODEL_LOADING retry semantics live there (SDK
# wait_for_capacity on native surfaces, the raw-HTTP loading-retry helper on the
# OpenAI surfaces).
#
# Model overrides:
#   MAC_SMOKE_EMBED_MODEL   (default BAAI/bge-m3)
#   MAC_SMOKE_RERANK_MODEL  (default BAAI/bge-reranker-v2-m3)
#   MAC_SMOKE_GEN_MODEL     (default Qwen/Qwen3.5-4B)
#   MAC_SMOKE_PORT          (default 8090)
set -euo pipefail

PORT="${MAC_SMOKE_PORT:-8090}"
EMBED_MODEL="${MAC_SMOKE_EMBED_MODEL:-BAAI/bge-m3}"
RERANK_MODEL="${MAC_SMOKE_RERANK_MODEL:-BAAI/bge-reranker-v2-m3}"
GEN_MODEL="${MAC_SMOKE_GEN_MODEL:-Qwen/Qwen3.5-4B}"
if [[ ! "$PORT" =~ ^[0-9]+$ || ${#PORT} -gt 5 ]]; then
    echo "ERROR: MAC_SMOKE_PORT must be a decimal integer between 1 and 65535." >&2
    exit 2
fi
if (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
    echo "ERROR: MAC_SMOKE_PORT must be a decimal integer between 1 and 65535." >&2
    exit 2
fi
PORT=$((10#$PORT))
BASE="http://127.0.0.1:${PORT}"
# Cold first-run downloads multi-GB weights; give loads plenty of room.
export SIE_MODEL_READY_TIMEOUT_S="${SIE_MODEL_READY_TIMEOUT_S:-1800}"
export SIE_MLX_STARTUP_TIMEOUT_S="${SIE_MLX_STARTUP_TIMEOUT_S:-1800}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "mac-smoke is Apple-Silicon only (uname=$(uname)); skipping." >&2
    exit 0
fi

SERVER_PID=""
SERVER_LOG_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
SERVER_LOG="$(mktemp "${SERVER_LOG_ROOT%/}/mac-smoke-serve.XXXXXX")"
# Our serve chain's command-line signature (uv/python argv both contain it).
# The port is anchored so e.g. -p 80901 can't match a sweep of 8090.
SERVE_CMD_PAT="sie_server\.cli serve.*-p ${PORT}([^0-9]|\$)"
# sweep_port — reap OUR serve chain if it still holds tcp:$PORT (TERM, then KILL).
# Kills by command-line signature, never by bare port ownership, so an unrelated
# local service on $PORT is never touched; start_server surfaces occupancy loudly.
sweep_port() {
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    # Single TERM: uvicorn force-quits on a second signal, which would cut the
    # graceful model unload short. Escalate only after the wait loop.
    pkill -f "$SERVE_CMD_PAT" 2>/dev/null || true
    for _ in $(seq 1 20); do
        lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1 || return 0
        pgrep -f "$SERVE_CMD_PAT" >/dev/null 2>&1 || return 0
        sleep 0.5
    done
    pkill -KILL -f "$SERVE_CMD_PAT" 2>/dev/null || true
    sleep 0.5
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # SIGTERM lets the server unload models (terminates the MLX subprocess).
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # TERM on the mise wrapper does not reach the python server at the end of the
    # mise->bash->uv chain (the orphan keeps $PORT and answers
    # the next phase's readyz for the WRONG bundle), so reap our chain by cmdline.
    sweep_port
}

cleanup() {
    local status=$?
    stop_server
    # Keep CI failures for the workflow artifact upload. Successful runs and
    # local failures clean up their randomized, symlink-safe temp file.
    if (( status == 0 )) || [[ "${CI:-}" != "true" ]]; then
        rm -f -- "$SERVER_LOG"
    else
        echo "Preserving failed server log at $SERVER_LOG" >&2
    fi
    return "$status"
}
trap cleanup EXIT

# start_server <serve-args...> — launch `mise run serve -- <args> -p $PORT` and wait for /readyz.
start_server() {
    echo "=== Starting serve $* on :${PORT} ==="
    # Anything already listening here would fake readiness for the WRONG bundle
    # (stale server) or break the bind; fail fast and loud. lsof rather than a
    # readyz probe so a non-HTTP listener is caught too.
    if lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "FAIL: something is already listening on :${PORT} before launch" >&2
        exit 1
    fi
    # A startup DEATH is retried: each mise invocation re-syncs the rustup channel
    # over the network, and a flaked sync kills the whole serve task before python
    # even starts. A live-but-never-ready server is a real hang and fails hard.
    local attempt
    for attempt in 1 2 3; do
        mise run serve -- "$@" -p "$PORT" >"$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        for i in $(seq 1 120); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "server exited during startup (attempt ${attempt}/3):" >&2
                tail -40 "$SERVER_LOG" >&2 || true
                SERVER_PID=""
                break
            fi
            if [[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/readyz" 2>/dev/null)" == "200" ]]; then
                echo "ready after ~$((i * 2))s"
                return 0
            fi
            sleep 2
        done
        if [[ -n "$SERVER_PID" ]]; then
            echo "FAIL: server not ready after ~240s" >&2
            tail -40 "$SERVER_LOG" >&2 || true
            exit 1
        fi
        # Reap any half-started child that survived the wrapper's death and took the port.
        sweep_port
        sleep 5
    done
    echo "FAIL: server failed to start after 3 attempts" >&2
    exit 1
}

# ===========================================================================
# Phase 0 — fake bundle (weightless surface contract, zero downloads)
#
# Boots the deterministic sie-fake models and asserts
# the surface contract — endpoints up, shapes right, MODEL_LOADING retry
# handling — before any real backend is involved. Failure attribution: a red
# Phase 0 with green real phases means orchestration broke; green fake
# checks with a red real phase means the MPS/MLX backend broke, not the
# orchestration. Chat-completions is intentionally absent here: weightless
# models have no chat-template tokenizer.
# ===========================================================================
start_server -b fake

mise run smoketest -- "$BASE" -t 120 \
    --check-docs \
    --model sie-fake --score-model sie-fake --rerank-model sie-fake \
    --openai-embeddings --embeddings-determinism --skip-extract \
    --generate --generate-model sie-fake --generate-mode native --generate-streaming \
    --generate-tokens 24

stop_server
echo "=== Phase 0 (fake) passed — any later failure is a real-backend issue ==="

# ===========================================================================
# Phase 1 — embeddings + reranking (default bundle, torch-MPS)
# ===========================================================================
start_server -b default

# First touch of /v1/embeddings blocks on the lazy load, so the smoketest's
# request timeout must absorb it.
mise run smoketest -- "$BASE" -t "$SIE_MODEL_READY_TIMEOUT_S" \
    --check-docs \
    --model "$EMBED_MODEL" --score-model "$RERANK_MODEL" --rerank-model "$RERANK_MODEL" \
    --openai-embeddings --embeddings-batch 32 --skip-extract

stop_server

# ===========================================================================
# Phase 2 — generation (sglang bundle, MLX subprocess)
# ===========================================================================
start_server -b sglang

# First touch of the MLX gen model is a non-blocking lazy load (subprocess
# spin-up + weight download): 503 MODEL_LOADING until ready, retried inside
# the smoketest primitives.
mise run smoketest -- "$BASE" -t "$SIE_MLX_STARTUP_TIMEOUT_S" \
    --only-generate --generate-model "$GEN_MODEL" --generate-mode both --generate-streaming \
    --generate-tokens 48

stop_server

echo ""
echo "============================================"
echo "  mac-smoke: ALL CHECKS PASSED"
echo "============================================"
