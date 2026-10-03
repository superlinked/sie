#!/usr/bin/env bash
#MISE description="Run Rust gateway tests"
#USAGE arg "[filter]" help="Optional cargo test filter"

set -euo pipefail

NATS_CONTAINER=""
cleanup() {
    if [[ -n "$NATS_CONTAINER" ]]; then
        docker rm -f "$NATS_CONTAINER" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# The publisher's >1 MiB wire/cleanup regression must traverse real JetStream,
# not silently skip in the normal gate. Respect an operator-supplied NATS_URL;
# otherwise create one exact, ephemeral, memory-backed NATS 2.12 container.
if [[ -z "${NATS_URL:-}" ]]; then
    command -v docker >/dev/null || {
        echo "gateway-test requires Docker or an explicit NATS_URL for mandatory JetStream coverage" >&2
        exit 1
    }
    NATS_CONTAINER="sie-gateway-test-nats-$$"
    docker run --rm -d \
        --name "$NATS_CONTAINER" \
        -p 127.0.0.1::4222 \
        "nats:2.12-alpine@sha256:b270f5e2428354c0335612694d7dd2fb588148e567a5757fdff325ef9c9332e6" \
        -js >/dev/null
    NATS_PORT="$(docker port "$NATS_CONTAINER" 4222/tcp | sed -n 's/.*://p' | head -1)"
    if [[ -z "$NATS_PORT" ]]; then
        echo "failed to resolve ephemeral NATS port" >&2
        exit 1
    fi
    NATS_READY=false
    # Cold Docker hosts can take several seconds to start the pinned image.
    # Preserve the exact log readiness check with a twelve-second budget.
    for _ in $(seq 1 120); do
        if docker logs "$NATS_CONTAINER" 2>&1 | grep -q "Server is ready"; then
            NATS_READY=true
            break
        fi
        sleep 0.1
    done
    if [[ "$NATS_READY" != true ]]; then
        echo "ephemeral NATS did not become ready" >&2
        exit 1
    fi
    export NATS_URL="nats://127.0.0.1:${NATS_PORT}"
fi
export SIE_RUN_NATS_PUBLISHER_TEST=1

BASE_ARGS=(test --locked --manifest-path packages/sie_gateway/Cargo.toml)
DEFAULT_ARGS=("${BASE_ARGS[@]}")
CLOUD_ARGS=("${BASE_ARGS[@]}" --features cloud-storage)

if [[ -n "${usage_filter:-}" ]]; then
    DEFAULT_ARGS+=("${usage_filter}")
    CLOUD_ARGS+=("${usage_filter}")
fi

echo "## Running sie-gateway tests"
mise exec -- cargo "${DEFAULT_ARGS[@]}"

echo "## Running sie-gateway tests with cloud-storage feature"
mise exec -- cargo "${CLOUD_ARGS[@]}"
