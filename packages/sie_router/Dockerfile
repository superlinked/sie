# syntax=docker/dockerfile:1.6
# SIE Router - Stateless request router for elastic cloud deployments
# Lightweight image (~150MB) - no GPU dependencies
#
# Build:
#   docker build -t sie-router:latest -f packages/sie_router/Dockerfile .
#
# Run:
#   docker run -p 8080:8080 sie-router:latest --kubernetes

# =============================================================================
# Stage 1: Builder - install dependencies with pip
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy sie-sdk first (dependency of sie-router)
COPY packages/sie_sdk /tmp/sie_sdk

# Copy sie-router files
COPY packages/sie_router/pyproject.toml ./
COPY packages/sie_router/src src/

# Create venv and install everything with pip (simpler, no workspace issues with uv)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    && .venv/bin/pip install \
        -e /tmp/sie_sdk \
        -e ".[kubernetes]"

# Clean up to reduce image size
RUN find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find .venv -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find .venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find .venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Stage 2: Runtime - minimal image
# =============================================================================
FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash sie

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=sie:sie /app/.venv /app/.venv

# Copy source code (pip -e creates editable install with .pth pointing to src)
COPY --from=builder --chown=sie:sie /app/src /app/src

# Copy sie-sdk source (installed as editable in /tmp)
COPY --from=builder --chown=sie:sie /tmp/sie_sdk/src /tmp/sie_sdk/src

# Copy model and bundle configs (required for ModelRegistry)
# Router needs these to know which models exist and their bundle mappings
COPY --chown=sie:sie packages/sie_server/bundles /app/bundles
COPY --chown=sie:sie packages/sie_server/models /app/models

# Set up PATH and model config paths
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIE_BUNDLES_DIR=/app/bundles \
    SIE_MODELS_DIR=/app/models

USER sie

# Default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Default command - Kubernetes discovery mode
ENTRYPOINT ["sie-router", "serve"]
CMD ["--port", "8080", "--kubernetes"]
