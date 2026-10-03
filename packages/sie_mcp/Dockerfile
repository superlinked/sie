# syntax=docker/dockerfile:1.6
# SIE MCP edge — hosted MCP server exposing SIE document jobs.
# Lightweight image: a sie_sdk client, no GPU/model dependencies.
#
# Build (from repo root):
#   docker build -t sie-mcp:latest -f packages/sie_mcp/Dockerfile .
#
# Run:
#   docker run -p 8088:8088 \
#     -e SIE_BASE_URL=https://your-gateway \
#     -e SIE_MCP_CONNECTOR_SECRETS=your-secret \
#     sie-mcp:latest

# =============================================================================
# Dependency builder: install the edge and its dependencies
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy sie-sdk first (a dependency of sie-mcp); installing it editable satisfies
# the sie-sdk requirement so `pip install -e .` resolves without hitting PyPI.
COPY packages/sie_sdk /tmp/sie_sdk

# Copy sie-mcp files
COPY packages/sie_mcp/pyproject.toml ./
COPY packages/sie_mcp/src src/

# Create venv and install sie-sdk + sie-mcp (pulls mcp, starlette, uvicorn,
# typer, jsonschema, pypdfium2, pillow from PyPI).
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    && .venv/bin/pip install \
        -e /tmp/sie_sdk \
        -e "."

# Clean up to reduce image size
RUN find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find .venv -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find .venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find .venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Runtime image: minimal MCP edge
# =============================================================================
FROM python:3.12-slim AS runtime

# Create non-root user
RUN useradd --create-home --shell /bin/bash sie

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=sie:sie /app/.venv /app/.venv

# Copy source code (pip -e creates editable installs with .pth pointing to src)
COPY --from=builder --chown=sie:sie /app/src /app/src

# Copy the plugin dir (SKILL.md etc.) so the in-container `sie-mcp skill-zip`
# helper can find it (cli.py resolves it at /app/plugin/SKILL.md).
COPY --chown=sie:sie packages/sie_mcp/plugin /app/plugin

# Copy sie-sdk source (installed as editable in /tmp)
COPY --from=builder --chown=sie:sie /tmp/sie_sdk/src /tmp/sie_sdk/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER sie

# Default MCP edge port (sie_mcp.cli serve --port default)
EXPOSE 8088

# Health check — /healthz is unauthenticated and does not require the cluster.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/healthz')" || exit 1

# Default command — the connector-secret / cluster URL are supplied via env
# (SIE_BASE_URL, SIE_MCP_CONNECTOR_SECRETS, SIE_MCP_PUBLIC_URL, ...).
ENTRYPOINT ["sie-mcp", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8088"]
