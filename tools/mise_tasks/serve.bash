#!/usr/bin/env bash
#MISE description="Start the SIE server"
set -euo pipefail

# Tag local sie-server heartbeats as development traffic so they
# don't land in the "unknown" bucket on the telemetry dashboard. Devs can
# override by exporting SIE_DEPLOYMENT_ENV before invoking mise run serve.
export SIE_DEPLOYMENT_ENV="${SIE_DEPLOYMENT_ENV:-development}"
mise exec -- uv lock --check --project .

# Inspect a copy of every argument so dependency resolution is independent of
# option order and the original argv reaches the server unchanged.
SERVER_ARGS=("$@")
BUNDLE=""
MODELS=""
for ((i = 0; i < ${#SERVER_ARGS[@]}; i++)); do
    arg="${SERVER_ARGS[$i]}"
    case "$arg" in
        -b|--bundle)
            if ((i + 1 >= ${#SERVER_ARGS[@]})) || [[ "${SERVER_ARGS[$((i + 1))]}" == -* ]]; then
                echo "ERROR: $arg requires a bundle name." >&2
                exit 2
            fi
            BUNDLE="${SERVER_ARGS[$((i + 1))]}"
            i=$((i + 1))
            ;;
        --bundle=*)
            BUNDLE="${arg#*=}"
            if [[ -z "$BUNDLE" ]]; then
                echo "ERROR: --bundle requires a bundle name." >&2
                exit 2
            fi
            ;;
        -m|--models)
            if ((i + 1 >= ${#SERVER_ARGS[@]})) || [[ "${SERVER_ARGS[$((i + 1))]}" == -* ]]; then
                echo "ERROR: $arg requires one or more model names." >&2
                exit 2
            fi
            MODELS="${SERVER_ARGS[$((i + 1))]}"
            i=$((i + 1))
            ;;
        --models=*)
            MODELS="${arg#*=}"
            if [[ -z "$MODELS" ]]; then
                echo "ERROR: --models requires one or more model names." >&2
                exit 2
            fi
            ;;
    esac
done

# Bundle dependency overlays must use the same CUDA wheel channel as the
# checked-in worker image. Gemma declares ``platform: cuda13`` and needs
# SGLang's prerelease flash-attn-4 dependency plus the cu130 torch/kernel
# indexes; without these flags an otherwise-correct CUDA 13 workstation fails
# dependency resolution before the server can start.
UV_BUNDLE_ARGS=()
if [[ -n "$BUNDLE" && -f "packages/sie_server/bundles/$BUNDLE.yaml" ]]; then
    BUNDLE_PLATFORM="$({
        mise exec -- python -c '
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?m)^platform:\s*([A-Za-z0-9_-]+)\s*$", text)
print(match.group(1) if match else "cuda12")
' "packages/sie_server/bundles/$BUNDLE.yaml"
    })"
    if [[ "$BUNDLE_PLATFORM" == "cuda13" ]]; then
        UV_BUNDLE_ARGS=(
            --index-strategy unsafe-best-match
            --prerelease=allow
            --extra-index-url https://download.pytorch.org/whl/cu130
            --extra-index-url https://docs.sglang.ai/whl/cu130/
        )
    fi
fi

# Device selection is automatic (cli.py picks mps on Apple Silicon). Bundles are
# device-agnostic: `default` serves embeddings/reranking (the flash adapters swap
# to torch-MPS on non-CUDA, flash-attn is linux-marked), and `sglang` serves
# generation (SGLangGenerationAdapter.create_for_device swaps to an MLX subprocess
# on non-CUDA). So on Mac, `mise run serve` gives embed/rerank and
# `mise run serve -- -b sglang` gives generation — same bundles as Linux.
if [[ "$(uname)" == "Darwin" && -z "$BUNDLE" && -z "$MODELS" ]]; then
    echo "Apple Silicon (mps): serving the 'default' bundle (embed/rerank). For generation use -b sglang."
fi

# Resolve dependencies using sie-server resolve-deps
REQS_FILE=$(mktemp)
trap 'rm -f "$REQS_FILE"' EXIT

if [[ -n "$BUNDLE" ]]; then
    mise exec -- uv run --frozen --project . --package sie-server \
        sie-server resolve-deps -b "$BUNDLE" > "$REQS_FILE"
elif [[ -n "$MODELS" ]]; then
    mise exec -- uv run --frozen --project . --package sie-server \
        sie-server resolve-deps -m "$MODELS" > "$REQS_FILE"
else
    mise exec -- uv run --frozen --project . --package sie-server \
        sie-server resolve-deps -b default > "$REQS_FILE"
fi

# Run with extra deps if any were resolved
if [[ -s "$REQS_FILE" ]]; then
    echo "Syncing adapter dependencies..."
    mise exec -- uv run --frozen --project . --package sie-server \
        ${UV_BUNDLE_ARGS[@]+"${UV_BUNDLE_ARGS[@]}"} \
        --with-requirements "$REQS_FILE" python -m sie_server.cli serve ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
else
    mise exec -- uv run --frozen --project . --package sie-server python -m sie_server.cli serve ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
fi
