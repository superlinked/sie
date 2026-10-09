# Extends the public sie-server image with the regulatory-rag example's
# custom code:
#   - Custom Stablebridge pruner/highlighter adapter under
#     sie_server.adapters.stablebridge_pruner.
#   - Two extra model YAMLs:
#       * answerdotai/ModernBERT-base with a "us-regulatory" LoRA profile
#         that activates sugiv/modernbert-us-stablecoin-encoder weights.
#       * sugiv/stablebridge-pruner-highlighter served by the custom adapter.
#
# Profile options["lora_id"] is already copied onto worker options["lora"]
# by sie-server, so this image does not patch the encode route.
#
# Build:  docker build -t sie-regulatory ./server-plugin
# Run:    docker run --platform linux/amd64 --gpus all -p 8080:8080 sie-regulatory
#         (--platform flag only needed on Apple Silicon; linux/amd64 runs under Rosetta)
#
# The base image already has sie-server installed and runnable, so this
# Dockerfile just layers the customization on top.

ARG SIE_TAG=latest-cpu-default
FROM ghcr.io/superlinked/sie-server:${SIE_TAG}

USER root

# Drop in the custom adapter.
COPY adapters/stablebridge_pruner /app/src/sie_server/adapters/stablebridge_pruner

# Register the custom model configs.
COPY models/*.yaml /app/models/

# Base image already exposes 8080 and sets the sie-server CLI as ENTRYPOINT.
CMD ["serve", "--port", "8080"]
