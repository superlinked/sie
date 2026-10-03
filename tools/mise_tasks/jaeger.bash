#!/usr/bin/env bash
#MISE description="Start Jaeger for local tracing"
#USAGE flag "-s --stop" help="Stop Jaeger container" default="false"

set -euo pipefail

CONTAINER_NAME="sie-jaeger"
JAEGER_IMAGE="jaegertracing/all-in-one:1.76.0@sha256:ab6f1a1f0fb49ea08bcd19f6b84f6081d0d44b364b6de148e1798eb5816bacac"

# Helper to run docker commands, using sg docker if needed for group permissions
run_docker() {
    if docker info &>/dev/null; then
        docker "$@"
    elif sg docker -c "docker info" &>/dev/null; then
        local command="docker"
        local arg
        for arg in "$@"; do
            printf -v command '%s %q' "$command" "$arg"
        done
        sg docker -c "$command"
    else
        echo "Error: Cannot connect to Docker daemon."
        echo ""
        echo "Please ensure Docker is installed and you have permissions:"
        echo "  1. Install Docker: sudo apt-get install docker.io"
        echo "  2. Add user to docker group: sudo usermod -aG docker \$USER"
        echo "  3. Apply group: newgrp docker (or log out and back in)"
        exit 1
    fi
}

if [[ "${usage_stop}" == "true" ]]; then
    echo "## Stopping Jaeger..."
    run_docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    run_docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Jaeger stopped"
    exit 0
fi

# Check if already running
if run_docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "## Jaeger already running"
    echo "UI: http://localhost:16686"
    echo "OTLP gRPC: localhost:4317"
    echo ""
    echo "To stop: mise run jaeger -- --stop"
    exit 0
fi

# Remove stopped container if exists
run_docker rm "${CONTAINER_NAME}" 2>/dev/null || true

echo "## Starting Jaeger..."
run_docker run -d --name "${CONTAINER_NAME}" \
    -p 16686:16686 \
    -p 4317:4317 \
    -p 4318:4318 \
    "$JAEGER_IMAGE"

echo ""
echo "Jaeger started successfully!"
echo "UI: http://localhost:16686"
echo "OTLP gRPC: localhost:4317"
echo "OTLP HTTP: localhost:4318"
echo ""
echo "Start server with tracing: mise run serve -- --tracing"
echo "To stop Jaeger: mise run jaeger -- --stop"
